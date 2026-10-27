"""Private streaming writer for this library's unscaled NIfTI-1 arrays.

The caller builds the complete NiBabel image and its geometry/intent exactly
as before. A C-contiguous storage view must expose that image's Fortran-order
payload in on-disk order. NiBabel still writes the header. An available pigz
compresses gzip output with two workers; otherwise NiBabel selects the opener.
The gzip level defaults to 0 and can be selected explicitly from 0 through 9.
All unsupported images/layouts continue through ``nib.save``.
"""

from contextlib import contextmanager
import os
import shutil
import subprocess
import sys
import zlib

import nibabel as nib
from nibabel.openers import ImageOpener
import numpy as np


_CHUNK_BYTES=1024*1024


def _gzip_backend():
    requested=os.environ.get('FAST_ANTS_GZIP_BACKEND','auto').lower()
    if requested not in ('auto','gzip','pigz'):
        raise ValueError('FAST_ANTS_GZIP_BACKEND must be auto, gzip or pigz')
    executable=shutil.which('pigz') if requested!='gzip' else None
    if requested=='pigz' and executable is None:
        raise RuntimeError('FAST_ANTS_GZIP_BACKEND=pigz requires pigz on PATH')
    return ('pigz',executable) if executable is not None else ('gzip',None)


def _gzip_level():
    try:level=int(os.environ.get('FAST_ANTS_GZIP_LEVEL','0'))
    except ValueError:
        raise ValueError('FAST_ANTS_GZIP_LEVEL must be an integer from 0 through 9') from None
    if not 0<=level<=9:
        raise ValueError('FAST_ANTS_GZIP_LEVEL must be an integer from 0 through 9')
    return level


def gzip_backend_info():
    """Describe the selected gzip implementation for runtime provenance."""
    backend,executable=_gzip_backend()
    version=zlib.ZLIB_RUNTIME_VERSION
    if executable is not None:
        result=subprocess.run([executable,'--version'],capture_output=True,text=True,check=True)
        version=(result.stdout or result.stderr).strip()
    return {'backend':backend,'executable':executable,'version':version,
            'compression_level':_gzip_level(),'compression_threads_per_file':2 if backend=='pigz' else 1}


@contextmanager
def _output_stream(filename):
    compressed=filename.lower().endswith('.nii.gz')
    backend,executable=_gzip_backend() if compressed else ('gzip',None)
    level=_gzip_level() if compressed else 1
    if backend!='pigz':
        options={'compresslevel':level} if compressed else {}
        with ImageOpener(filename,'wb',**options) as output:yield output
        return
    # Stream directly to the destination. Explicit options and a clean pigz
    # environment keep compression level, worker count and gzip header stable.
    environment={key:value for key,value in os.environ.items() if key not in ('GZIP','PIGZ')}
    with open(filename,'wb') as destination, subprocess.Popen(
            [executable,'-'+str(level),'-n','-p','2','-c'],stdin=subprocess.PIPE,
            stdout=destination,stderr=subprocess.PIPE,env=environment) as process:
        try:
            yield process.stdin
            process.stdin.close();process.stdin=None
            _,error=process.communicate()
            if process.returncode:
                raise OSError('pigz failed writing '+filename+': '+error.decode(errors='replace').strip())
        except BaseException:
            if process.poll() is None:process.kill()
            raise


def _save_nifti_storage(nifti, path, storage):
    """Save a freshly built image, streaming a proven on-disk-order view.

    This is deliberately narrower than a general NIfTI writer. It performs no
    dtype conversion, scaling, axis conversion or finite-value scan.
    Scalar ZYX and vector CZYX storage are paired by the callers
    with XYZ and XYZ1C image views respectively.
    """
    filename=str(path)
    data=nifti.dataobj
    if (type(nifti) is not nib.Nifti1Image
            or not filename.lower().endswith(('.nii','.nii.gz'))
            or not isinstance(storage,np.ndarray) or not storage.flags.c_contiguous
            or not isinstance(data,np.ndarray) or not data.flags.f_contiguous
            or not storage.dtype.isnative or storage.dtype.kind not in 'biuf'
            or data.dtype!=storage.dtype or data.nbytes!=storage.nbytes
            or data.__array_interface__['data'][0]!=storage.__array_interface__['data'][0]):
        return nib.save(nifti,path)

    nifti.update_header()
    header=nifti.header.copy()
    slope,intercept=float(header['scl_slope']),float(header['scl_inter'])
    unscaled=((np.isnan(slope) and np.isnan(intercept)) or
              (slope==1. and intercept==0.))
    if (len(header.extensions) or header.get_data_dtype()!=storage.dtype
            or header.endianness!=('<' if sys.byteorder=='little' else '>')
            or header.get_data_offset() not in (0,352)
            or not unscaled
            or nifti.get_data_dtype()!=storage.dtype):
        return nib.save(nifti,path)

    # Same-type, unscaled NiBabel array writing stores slope=1/intercept=0.
    # Keep the caller's header untouched, including qform, sform and intent.
    header.set_slope_inter(1.,0.)
    with memoryview(storage).cast('B') as payload, _output_stream(filename) as output:
        # Nifti1Header writes its 348 bytes and the four extension flag bytes,
        # and chooses the normal 352-byte data offset when it was initially 0.
        header.write_to(output)
        # Compatible images have no extensions and a 352-byte data offset;
        # writing is sequential because a compressor's stdin cannot seek.
        for start in range(0,len(payload),_CHUNK_BYTES):
            output.write(payload[start:start+_CHUNK_BYTES])
