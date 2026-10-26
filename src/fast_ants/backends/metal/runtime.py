"""Device execution and an explicit record of every host/device boundary."""

from collections import Counter
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import numpy as np

HEADER = Path(__file__).with_name('common.metal').read_text()
DEFAULT_OPTIMIZATIONS=('parallel_prefix','parallel_max','parallel_range','parallel_norms','histogram_rows',
                       'indexed_mi','parallel_cc','cached_fixed_bins','cached_cubic','histogram_channels','gaussian_sections','paired_cc','fused_value_sampling',
                       'direct_field_convolution','direct_mi_validity','simd_mi_histograms',
                       'streamed_mi_derivative','packed_mi_controls','streamed_mi_fixed_plan',
                       'parallel_moments','batched_inverse','device_mi_control','batch_mi_search',
                       'tiled_cc','fused_scaled_compose','cached_preparation','streamed_output_qa',
                       'fused_mi_finalize','cc_window_stats','overlapped_inverse',
                       'deferred_syn_controls','async_control_prefetch')
# Paired statistics passed device contracts and warmed whole-pipeline A/B.
# Legacy scalar-order convergence replay remains an explicit diagnostic.
# Fixed-row SIMD value/derivative streams passed whole-registration A/B and
# saved-output QA. Legacy inline-cubic, grouped and inverse/CC candidates stay
# opt-in; fewer buffers alone did not establish a whole-pipeline gain for them.
# Parallel moments, independently stopped paired inversions and private GPU MI
# finalization/trial batching passed targeted contracts and scoped pipeline A/B.
# Shared-halo CC, scaled composition, bounded preprocessing reuse and streamed
# CPU output QA passed targeted contracts and the pipeline source comparison.
# Fused log preparation, centered separable CC and independent-work async
# submission passed targeted GPU checks and warmed native ABBA comparison.
AVAILABLE_OPTIMIZATIONS=frozenset((*DEFAULT_OPTIMIZATIONS,'reference_cc_convergence','value_inline_cubic',
                                   'grouped_fixed_bins','fused_inverse_norms',
                                   'streamed_mi_derivative','packed_mi_controls','inline_cc_gradient',
                                   'streamed_mi_fixed_plan','parallel_moments','batched_inverse',
                                   'device_mi_control','batch_mi_search','tiled_cc',
                                   'fused_scaled_compose','cached_preparation','streamed_output_qa',
                                   'fused_mi_finalize','cc_window_stats','overlapped_inverse',
                                   'deferred_syn_controls','async_control_prefetch'))


@lru_cache(None)
def _kernel(name, names, source, outputs):
    return mx.fast.metal_kernel(
        name='fastants_'+name, input_names=list(names),
        output_names=['out'+str(i) for i in range(outputs)],
        source=source, header=HEADER, ensure_row_contiguous=True,
        compile_options={'math_mode': 'safe'})


class MetalContext:
    """Custom kernels only; no implicit CPU fallback or native registration."""

    def __init__(self,optimizations=None):
        if not mx.metal.is_available():
            raise RuntimeError('The Metal backend requires an accessible Apple GPU')
        self.kernels = Counter()
        self.control_guards = Counter()
        self.serial_cc_guard = False
        self.defer_cc_score = False
        self.async_submissions = Counter()
        self.transfers = {}
        self._meta_cache = {}
        self.optimizations=frozenset(DEFAULT_OPTIMIZATIONS if optimizations is None else optimizations)
        if not self.optimizations<=AVAILABLE_OPTIMIZATIONS:
            raise ValueError('Unknown Metal optimization: '+str(self.optimizations-AVAILABLE_OPTIMIZATIONS))

    def _record(self, direction, reason, array):
        key = direction+':'+reason
        entry = self.transfers.setdefault(key, {'calls': 0, 'bytes': 0, 'largest_array_bytes': 0})
        entry['calls'] += 1
        entry['bytes'] += array.nbytes
        entry['largest_array_bytes'] = max(entry['largest_array_bytes'], array.nbytes)

    def upload(self, values, reason='small_geometry', dtype=np.float32):
        values = np.ascontiguousarray(values, dtype=dtype)
        if values.dtype not in (np.dtype('float32'), np.dtype('uint32'), np.dtype('int32'), np.dtype('bool')):
            raise ValueError('Metal storage is FP32, uint32 or Boolean')
        self._record('host_to_device', reason, values)
        return mx.array(values)

    def control(self, values, reason, maximum_elements=32):
        if values.size > maximum_elements:
            raise ValueError('Control download exceeds declared bound: '+reason)
        return self.download(values, 'control_'+reason)

    def download(self, values, reason='final_output'):
        mx.eval(values)
        result = np.array(values, copy=True)
        self._record('device_to_host', reason, result)
        return result

    def call(self, name, source, inputs, shapes, count, dtypes=None):
        if count < 1:
            raise ValueError('Kernel dispatch must have positive work count')
        arrays = list(inputs.values())
        if any(not isinstance(a, mx.array) or a.dtype not in (mx.float32,mx.uint32,mx.int32,mx.bool_) for a in arrays):
            raise ValueError('Only explicit FP32/integer device arrays may enter Metal kernels')
        self.kernels[name] += 1
        result = _kernel(name, tuple(inputs), source, len(shapes))(
            inputs=arrays, grid=(count,1,1), threadgroup=(min(count,256),1,1),
            output_shapes=[tuple(s) for s in shapes],
            output_dtypes=dtypes or [mx.float32]*len(shapes), stream=mx.gpu)
        return result[0] if len(result)==1 else result

    def meta(self, *values):
        # Normalize/check uint32 conversion before lookup, just as upload did.
        # These small immutable kernel inputs recur thousands of times within
        # one registration. Bound cache residency rather than retaining every
        # shape encountered by a long-lived context.
        metadata=np.ascontiguousarray(values,dtype=np.uint32)
        key=(metadata.shape,metadata.tobytes())
        cached=self._meta_cache.get(key)
        if cached is not None:return cached
        result=self.upload(metadata,'index_metadata',np.uint32)
        if len(self._meta_cache)<128:self._meta_cache[key]=result
        return result

    def finish(self, *arrays):
        if arrays:
            mx.eval(*arrays)
        mx.synchronize(mx.gpu)

    def submit(self,*arrays,reason='independent_work'):
        """Enqueue independent device work while bounded host setup continues."""
        mx.async_eval(*arrays)
        self.async_submissions[reason]+=1

    def report(self):
        return {'device':mx.device_info(), 'optimizations':sorted(self.optimizations),'math_mode':'safe; FP contraction and reassociation disabled in custom source',
                'kernel_dispatches':dict(self.kernels), 'transfers':self.transfers,'scalar_control_guards':dict(self.control_guards),
                'async_submissions':dict(self.async_submissions),
                'cpu_fallbacks':[], 'dense_arithmetic':'ordinary FP32; no software-expanded precision', 'software_expanded_precision':False,
                'timing':'caller must evaluate and synchronize before stopping the clock'}
