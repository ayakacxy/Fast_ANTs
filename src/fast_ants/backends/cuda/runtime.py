"""Device execution and an explicit record of every host/device boundary."""

from collections import Counter, OrderedDict
from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np

HEADER = Path(__file__).with_name('common.cuh').read_text()
DTYPES = {np.dtype('float32'): 'float', np.dtype('uint32'): 'unsigned int',
          np.dtype('int32'): 'int', np.dtype('bool'): 'bool'}
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true')


@lru_cache(128)
def _kernel(name, declarations, source, outputs):
    arguments = [f'const {kind}* {key}' for key,kind in declarations]
    arguments += [f'{kind}* out{i}' for i,kind in enumerate(outputs)]
    arguments.append('const unsigned int nwork')
    code = HEADER+'\nextern "C" __global__ void fastants_'+name+'('+', '.join(arguments)+''') {
        unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=nwork)return;
        '''+source+'\n}\n'
    return cp.RawKernel(code, 'fastants_'+name, options=OPTIONS)


class CUDAContext:
    """Custom FP32 kernels with explicit transfers and no CPU fallback.

    Operations use the device and current stream captured at construction.
    External device arrays must be ready on that stream before use.
    """

    def __init__(self):
        try:
            if cp.cuda.runtime.getDeviceCount()<1:
                raise RuntimeError('No CUDA devices found')
            self.device = cp.cuda.Device()
            self.stream = cp.cuda.get_current_stream()
        except cp.cuda.runtime.CUDARuntimeError as error:
            raise RuntimeError('The CUDA backend requires an accessible NVIDIA GPU') from error
        self.kernels = Counter()
        self.transfers = {}
        self._meta_cache = {}
        self._geometry_cache = OrderedDict()
        self._geometry_hits = 0
        self._geometry_misses = 0

    def _record(self,direction,reason,array):
        key=direction+':'+reason
        entry=self.transfers.setdefault(key,{'calls':0,'bytes':0,'largest_array_bytes':0})
        entry['calls']+=1
        entry['bytes']+=array.nbytes
        entry['largest_array_bytes']=max(entry['largest_array_bytes'],array.nbytes)

    def require(self,values):
        if not isinstance(values,cp.ndarray) or values.dtype not in DTYPES:
            raise ValueError('Only explicit FP32/integer device arrays may enter CUDA kernels')
        if not values.flags.c_contiguous or values.device.id!=self.device.id:
            raise ValueError('CUDA arrays must be contiguous and on the context device')
        return values

    def upload(self,values,reason='small_geometry',dtype=np.float32):
        if isinstance(values,cp.ndarray):
            raise ValueError('Upload requires host values; reuse device arrays explicitly')
        values=np.ascontiguousarray(values,dtype=dtype)
        if values.dtype not in DTYPES:
            raise ValueError('CUDA storage is FP32, uint32, int32 or Boolean')
        with self.device,self.stream:
            result=cp.asarray(values,blocking=True)
        self._record('host_to_device',reason,values)
        return result

    def download(self,values,reason='final_output'):
        self.require(values)
        with self.device,self.stream:
            result=cp.asnumpy(values,stream=self.stream,blocking=True)
        self._record('device_to_host',reason,result)
        return result

    def geometry(self,values,reason='cached_geometry',dtype=np.float32):
        """Reuse small immutable operator inputs within this context.

        Dense inputs and public uploads keep independent storage. These
        internal arrays enter kernels through const pointers; keys include
        exact dtype, shape and bytes, with at most 128 cached entries.
        """
        if isinstance(values,cp.ndarray):
            raise ValueError('Geometry cache requires explicit host values')
        values=np.ascontiguousarray(values,dtype=dtype)
        if values.size>128:return self.upload(values,reason,dtype)
        key=(values.dtype.str,values.shape,values.tobytes())
        cached=self._geometry_cache.get(key)
        if cached is not None:
            self._geometry_cache.move_to_end(key);self._geometry_hits+=1
            return cached
        self._geometry_misses+=1
        result=self.upload(values,reason,dtype)
        if len(self._geometry_cache)>=128:self._geometry_cache.popitem(last=False)
        self._geometry_cache[key]=result
        return result

    def control(self,values,reason,maximum_elements=32):
        if values.size>maximum_elements:
            raise ValueError('Control download exceeds declared bound: '+reason)
        return self.download(values,'control_'+reason)

    def call(self,name,source,inputs,shapes,count,dtypes=None):
        if count<1 or count>np.iinfo(np.uint32).max:
            raise ValueError('Kernel dispatch requires a positive uint32 work count')
        arrays=[self.require(value) for value in inputs.values()]
        dtypes=[np.dtype(dtype) for dtype in (dtypes or [np.float32]*len(shapes))]
        if len(dtypes)!=len(shapes) or any(dtype not in DTYPES for dtype in dtypes):
            raise ValueError('CUDA output shapes and supported dtypes must match')
        declarations=tuple((key,DTYPES[value.dtype]) for key,value in inputs.items())
        with self.device,self.stream:
            result=[cp.empty(tuple(shape),dtype=dtype) for shape,dtype in zip(shapes,dtypes)]
            kernel=_kernel(name,declarations,source,tuple(DTYPES[dtype] for dtype in dtypes))
            kernel(((count+255)//256,),(256,),(*arrays,*result,np.uint32(count)),stream=self.stream)
        self.kernels[name]+=1
        return result[0] if len(result)==1 else result

    def meta(self,*values):
        if any(int(value)!=value or value<0 or value>np.iinfo(np.uint32).max for value in values):
            raise ValueError('CUDA index metadata must contain uint32 values')
        key=tuple(values)
        cached=self._meta_cache.get(key)
        if cached is not None:return cached
        result=self.upload(values,'index_metadata',np.uint32)
        if len(self._meta_cache)<128:self._meta_cache[key]=result
        return result

    def finish(self,*arrays):
        for array in arrays:self.require(array)
        with self.device:
            self.stream.synchronize()

    def report(self):
        properties=cp.cuda.runtime.getDeviceProperties(self.device.id)
        name=properties['name']
        if isinstance(name,bytes):name=name.decode()
        return {'device':{'id':self.device.id,'name':name,
                          'compute_capability':self.device.compute_capability},
                'cupy':cp.__version__,'kernel_dispatches':dict(self.kernels),
                'transfers':{key:dict(value) for key,value in self.transfers.items()},
                'geometry_cache':{'hits':self._geometry_hits,'misses':self._geometry_misses,
                                  'entries':len(self._geometry_cache),'maximum_entries':128},
                'math_mode':'ordinary FP32; FMA contraction disabled; precise division and square root',
                'cpu_fallbacks':[],'software_expanded_precision':False,
                'timing':'caller must synchronize before stopping the clock'}
