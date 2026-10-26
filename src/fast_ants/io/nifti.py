"""Private streaming writer for this library's unscaled NIfTI-1 arrays.

The caller builds the complete NiBabel image and its geometry/intent exactly
as before. A C-contiguous storage view must expose that image's Fortran-order
payload in on-disk order. NiBabel still writes the header and selects the
opener/compression defaults; only payload-sized ``tobytes`` copies are avoided.
All unsupported images/layouts continue through ``nib.save``.
"""

import sys

import nibabel as nib
from nibabel.openers import ImageOpener
import numpy as np


_CHUNK_BYTES=1024*1024


def _save_nifti_storage(nifti, path, storage):
    """Save a freshly built image, streaming a proven on-disk-order view.

    This is deliberately narrower than a general NIfTI writer. It performs no
    dtype conversion, scaling, axis conversion, finite-value scan or compression
    configuration. Scalar ZYX and vector CZYX storage are paired by the callers
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
    with memoryview(storage).cast('B') as payload, ImageOpener(filename,'wb') as output:
        # Nifti1Header writes its 348 bytes and the four extension flag bytes,
        # and chooses the normal 352-byte data offset when it was initially 0.
        header.write_to(output)
        output.seek(header.get_data_offset())
        for start in range(0,len(payload),_CHUNK_BYTES):
            output.write(payload[start:start+_CHUNK_BYTES])
