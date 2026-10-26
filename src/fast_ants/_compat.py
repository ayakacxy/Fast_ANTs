"""Temporary module aliases for pre-reorganization imports (0.1 series).

Production code imports canonical modules. Remove this bridge only in a
documented breaking release; it does not dispatch numerical operations.
"""
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys

ALIASES = {'fast_ants.geometry': 'fast_ants.core.types',
 'fast_ants.transforms': 'fast_ants.core.transforms',
 'fast_ants.transform_io': 'fast_ants.io.transforms',
 'fast_ants.nifti_stream': 'fast_ants.io.nifti',
 'fast_ants.linear_models': 'fast_ants.pipeline.models64',
 'fast_ants.linear_models32': 'fast_ants.pipeline.models32',
 'fast_ants.optimizers': 'fast_ants.pipeline.optimizer64',
 'fast_ants.optimizers32': 'fast_ants.pipeline.optimizer32',
 'fast_ants.mattes_control': 'fast_ants.pipeline.mattes64',
 'fast_ants.mattes_control_float1': 'fast_ants.pipeline.mattes32',
 'fast_ants.float1_policy': 'fast_ants.pipeline.float1_policy',
 'fast_ants.preprocessing': 'fast_ants.pipeline.preprocessing',
 'fast_ants.pyramid': 'fast_ants.pipeline.pyramid',
 'fast_ants.sampling_plan': 'fast_ants.pipeline.sampling_plan',
 'fast_ants.masks': 'fast_ants.pipeline.masks',
 'fast_ants.registration': 'fast_ants.backends.cpu.fp64.registration',
 'fast_ants.linear_registration': 'fast_ants.backends.cpu.fp64.linear',
 'fast_ants.syn': 'fast_ants.backends.cpu.fp64.syn',
 'fast_ants.fields': 'fast_ants.backends.cpu.fp64.fields',
 'fast_ants.filtering': 'fast_ants.backends.cpu.fp64.filtering',
 'fast_ants.sampling': 'fast_ants.backends.cpu.fp64.sampling',
 'fast_ants.gradients': 'fast_ants.backends.cpu.fp64.gradients',
 'fast_ants.metrics': 'fast_ants.backends.cpu.fp64.metrics.mattes',
 'fast_ants.cc': 'fast_ants.backends.cpu.fp64.metrics.cc',
 'fast_ants.registration32': 'fast_ants.backends.cpu.fp32.registration',
 'fast_ants.linear_registration32': 'fast_ants.backends.cpu.fp32.linear',
 'fast_ants.metrics32': 'fast_ants.backends.cpu.fp32.metrics.mattes',
 'fast_ants.precision32': 'fast_ants.backends.cpu.fp32.precision',
 'fast_ants.registration_metal': 'fast_ants.backends.metal.registration',
 'fast_ants.backends.cpu.native32': 'fast_ants.core.arithmetic32',
 'fast_ants.backends.cpu.native64': 'fast_ants.core.arithmetic64',
 'fast_ants.backends.cpu.small_matrix': 'fast_ants.core.linalg',
 'fast_ants.backends.cpu.geometry': 'fast_ants.backends.cpu.common.geometry',
 'fast_ants.backends.cpu.reductions': 'fast_ants.backends.cpu.common.reductions',
 'fast_ants.backends.cpu.output_qa': 'fast_ants.quality.output_kernels',
 'fast_ants.backends.cpu.cc': 'fast_ants.backends.cpu.fp64.metrics.cc_kernels',
 'fast_ants.backends.cpu.cc_pair': 'fast_ants.backends.cpu.fp64.metrics.cc_pair',
 'fast_ants.backends.cpu.mattes': 'fast_ants.backends.cpu.fp64.metrics.mattes_kernels',
 'fast_ants.backends.cpu.linear_fused': 'fast_ants.backends.cpu.fp64.metrics.linear_fused',
 'fast_ants.backends.cpu.fields': 'fast_ants.backends.cpu.fp64.field_kernels',
 'fast_ants.backends.cpu.local_fields64': 'fast_ants.backends.cpu.fp64.local_fields',
 'fast_ants.backends.cpu.pull64': 'fast_ants.backends.cpu.fp64.pull',
 'fast_ants.backends.cpu.sampling': 'fast_ants.backends.cpu.fp64.sampling_kernels',
 'fast_ants.backends.cpu.filtering': 'fast_ants.backends.cpu.fp64.filter_kernels',
 'fast_ants.backends.cpu.dense_parallel32': 'fast_ants.backends.cpu.fp32.dense_kernels',
 'fast_ants.backends.cpu.gaussian_sections32': 'fast_ants.backends.cpu.fp32.gaussian_sections',
 'fast_ants.backends.cpu.inverse32': 'fast_ants.backends.cpu.fp32.inverse',
 'fast_ants.backends.cpu.norms32': 'fast_ants.backends.cpu.fp32.norms',
 'fast_ants.backends.cpu.mattes32': 'fast_ants.backends.cpu.fp32.metrics.mattes_kernels',
 'fast_ants.backends.cpu.value_mi32': 'fast_ants.backends.cpu.fp32.metrics.value',
 'fast_ants.backends.cpu.derivative_mi32': 'fast_ants.backends.cpu.fp32.metrics.derivative',
 'fast_ants.backends.cpu.cc_pair32': 'fast_ants.backends.cpu.fp32.metrics.cc_pair',
 'fast_ants.backends.cpu.cc_queue_pair32': 'fast_ants.backends.cpu.fp32.metrics.cc_queue',
 'fast_ants.backends.metal.mattes': 'fast_ants.backends.metal.metrics.mattes.metric',
 'fast_ants.backends.metal.fused_value32': 'fast_ants.backends.metal.metrics.mattes.sampling',
 'fast_ants.backends.metal.value_mi32': 'fast_ants.backends.metal.metrics.mattes.sampling',
 'fast_ants.backends.metal.derivative_mi32': 'fast_ants.backends.metal.metrics.mattes.derivative',
 'fast_ants.backends.metal.grouped_mattes32': 'fast_ants.backends.metal.metrics.mattes.histogram',
 'fast_ants.backends.metal.histogram_controls32': 'fast_ants.backends.metal.metrics.mattes.histogram',
 'fast_ants.backends.metal.metric_finalize32': 'fast_ants.backends.metal.metrics.mattes.finalize',
 'fast_ants.backends.metal.metric_finalize_fused32': 'fast_ants.backends.metal.metrics.mattes.finalize_fused',
 'fast_ants.backends.metal.cc_gradient32': 'fast_ants.backends.metal.metrics.cc.gradient',
 'fast_ants.backends.metal.cc_tiled32': 'fast_ants.backends.metal.metrics.cc.tiled',
 'fast_ants.backends.metal.cc_window32': 'fast_ants.backends.metal.metrics.cc.window',
 'fast_ants.backends.metal.deferred_cc32': 'fast_ants.backends.metal.metrics.cc.control',
 'fast_ants.backends.metal.paired_inverse32': 'fast_ants.backends.metal.fields.inverse',
 'fast_ants.backends.metal.inverse_overlap32': 'fast_ants.backends.metal.fields.overlap',
 'fast_ants.backends.metal.scaled_compose32': 'fast_ants.backends.metal.fields.composition',
 'fast_ants.backends.metal.gaussian_sections32': 'fast_ants.backends.metal.gaussian_sections',
 'fast_ants.backends.metal.geometry32': 'fast_ants.backends.metal.geometry',
 'fast_ants.backends.metal.moments32': 'fast_ants.backends.metal.moments'}

class _AliasLoader(Loader):
    def __init__(self,target):self.target=target
    def create_module(self,spec):return import_module(self.target)
    def exec_module(self,module):pass

class _AliasFinder(MetaPathFinder):
    _fast_ants_compat=True
    def find_spec(self,fullname,path=None,target=None):
        canonical=ALIASES.get(fullname)
        return None if canonical is None else spec_from_loader(fullname,_AliasLoader(canonical))

def install():
    if not any(getattr(finder,"_fast_ants_compat",False) for finder in sys.meta_path):
        sys.meta_path.append(_AliasFinder())
