"""Streaming output has the same decoded NIfTI header/payload as NiBabel."""

import gzip
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from fast_ants.core.types import Grid, Image
from fast_ants.io.images import load_image, load_images, save_image
from fast_ants.io.nifti import _save_nifti_storage,gzip_backend_info
from fast_ants.core.fields import DisplacementField
from fast_ants.io.transforms import save_field


def decoded(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rb') as stream:
            return stream.read()
    return Path(path).read_bytes()


class NiftiStreamTests(unittest.TestCase):
    def grid(self):
        angle=.21
        matrix=np.eye(4)
        matrix[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],
            [np.sin(angle),np.cos(angle),0],[0,0,-1]])@np.diag([.7,1.3,2.1])
        matrix[:3,3]=[17.,-33.,9.]
        return Grid((3,4,5),matrix)

    def test_explicit_gzip_pigz_and_automatic_fallback(self):
        grid=self.grid()
        scalar=Image(np.arange(60,dtype=np.float32).reshape(grid.shape_zyx)*.125,grid)
        vector=DisplacementField(np.arange(180,dtype=np.float32).reshape(3,*grid.shape_zyx)*.013,grid)
        backends=('gzip','pigz') if shutil.which('pigz') else ('gzip',)
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ):
                os.environ.pop('FAST_ANTS_GZIP_LEVEL',None)
                self.assertEqual(gzip_backend_info()['compression_level'],0)
            for item,save in ((scalar,save_image),(vector,save_field)):
                oracle=None
                for backend in backends:
                    output=Path(temporary)/(backend+'.nii.gz')
                    with self.subTest(backend=backend,kind=type(item).__name__), patch.dict(
                            os.environ,{'FAST_ANTS_GZIP_BACKEND':backend}), patch(
                            'fast_ants.io.nifti._CHUNK_BYTES',17):
                        save(item,output)
                        self.assertEqual(gzip_backend_info()['backend'],backend)
                        actual=decoded(output)
                        if oracle is None:oracle=actual
                        self.assertEqual(actual,oracle)
                with patch.dict(os.environ,{'FAST_ANTS_GZIP_BACKEND':'auto'}), patch(
                        'fast_ants.io.nifti.shutil.which',return_value=None):
                    output=Path(temporary)/'fallback.nii.gz'
                    save(item,output)
                    self.assertEqual(gzip_backend_info()['backend'],'gzip')
                    self.assertEqual(decoded(output),oracle)

    def test_explicit_gzip_level_preserves_decoded_products(self):
        grid=self.grid()
        scalar=Image(np.arange(60,dtype=np.float32).reshape(grid.shape_zyx)*.125,grid)
        vector=DisplacementField(np.arange(180,dtype=np.float32).reshape(3,*grid.shape_zyx)*.013,grid)
        backends=('gzip','pigz') if shutil.which('pigz') else ('gzip',)
        with tempfile.TemporaryDirectory() as temporary:
            for item,save in ((scalar,save_image),(vector,save_field)):
                oracle=None
                for backend in backends:
                    for level in (0,1,9):
                        output=Path(temporary)/(backend+'-'+str(level)+'.nii.gz')
                        with self.subTest(backend=backend,level=level,kind=type(item).__name__), patch.dict(
                                os.environ,{'FAST_ANTS_GZIP_BACKEND':backend,'FAST_ANTS_GZIP_LEVEL':str(level)}), patch(
                                'fast_ants.io.nifti._CHUNK_BYTES',17):
                            save(item,output)
                            self.assertEqual(gzip_backend_info()['compression_level'],level)
                            actual=decoded(output)
                            if oracle is None:oracle=actual
                            self.assertEqual(actual,oracle)
                with patch.dict(os.environ,{'FAST_ANTS_GZIP_BACKEND':'auto','FAST_ANTS_GZIP_LEVEL':'0'}), patch(
                        'fast_ants.io.nifti.shutil.which',return_value=None):
                    output=Path(temporary)/'fallback-level0.nii.gz'
                    save(item,output)
                    self.assertEqual(gzip_backend_info()['backend'],'gzip')
                    self.assertEqual(gzip_backend_info()['compression_level'],0)
                    self.assertEqual(decoded(output),oracle)
                for level in ('-1','10','1.5','invalid'):
                    with self.subTest(level=level), patch.dict(os.environ,{'FAST_ANTS_GZIP_LEVEL':level}):
                        with self.assertRaisesRegex(ValueError,'FAST_ANTS_GZIP_LEVEL'):
                            gzip_backend_info()
                        with self.assertRaisesRegex(ValueError,'FAST_ANTS_GZIP_LEVEL'):
                            save(item,Path(temporary)/'invalid.nii.gz')

    def test_scalar_float_and_integer_decoded_bytes_match_original_writer(self):
        grid=self.grid()
        with tempfile.TemporaryDirectory() as temporary:
            for dtype in (np.float32,np.float64,np.int16):
                for suffix in ('.nii','.nii.gz'):
                    with self.subTest(dtype=dtype,suffix=suffix):
                        values=(np.arange(60).reshape(grid.shape_zyx)-31).astype(dtype)
                        if np.dtype(dtype).kind=='f':values*=dtype(.125)
                        image=Image(values,grid)
                        reference=nib.Nifti1Image(values.transpose(2,1,0),grid.affine_ras,dtype=dtype)
                        reference.set_qform(grid.affine_ras,1)
                        reference.set_sform(grid.affine_ras,1)
                        reference.header.set_xyzt_units('mm')
                        oracle=Path(temporary)/('oracle'+suffix)
                        output=Path(temporary)/('actual'+suffix)
                        nib.save(reference,oracle)
                        # Force multiple chunks without allocating large test arrays.
                        with patch('fast_ants.io.nifti._CHUNK_BYTES',17), patch(
                                'fast_ants.io.nifti.nib.save',side_effect=AssertionError('unexpected fallback')):
                            save_image(image,output)
                        self.assertEqual(decoded(output),decoded(oracle))
                        np.testing.assert_array_equal(np.asarray(nib.load(output).dataobj),
                                                      values.transpose(2,1,0))
                        np.testing.assert_array_equal(image.data,values)
            # Independent reads preserve order, explicit policies, and the
            # scaled-integer FP32 boundary of each original load_image call.
            first=Path(temporary)/'first.nii.gz'
            second=Path(temporary)/'scaled.nii'
            save_image(Image(np.arange(60,dtype=np.float32).reshape(grid.shape_zyx),grid),first)
            raw=np.arange(60,dtype=np.int16).reshape(grid.shape_xyz)
            scaled=nib.Nifti1Image(raw,grid.affine_ras)
            scaled.set_sform(grid.affine_ras,1)
            scaled.header.set_xyzt_units('mm')
            scaled.header.set_slope_inter(1.835897445678711,.125)
            nib.save(scaled,second)
            for policy in ('nifti','itk'):
                for dtype in (np.float32,np.float64):
                    options={'geometry_policy':'itk_scanner','scaling_policy':policy}
                    actual=load_images((second,first),dtype,**options)
                    expected=tuple(load_image(path,dtype,**options) for path in (second,first))
                    self.assertIsInstance(actual,tuple)
                    for got,wanted in zip(actual,expected):
                        np.testing.assert_array_equal(got.data,wanted.data)
                        np.testing.assert_array_equal(got.grid.index_to_lps,wanted.grid.index_to_lps)
                        self.assertEqual(got.data.dtype,wanted.data.dtype)
                        self.assertEqual(got.metadata,wanted.metadata)
            self.assertEqual(load_images(()),())
            single=load_images((first,))
            self.assertEqual(len(single),1)
            np.testing.assert_array_equal(single[0].data,load_image(first).data)
            with self.assertRaises(FileNotFoundError):
                load_images((Path(temporary)/'missing.nii',first))

    def test_lps_vector_decoded_bytes_dtype_conversion_and_unsupported_fallback(self):
        grid=self.grid()
        values=np.arange(180,dtype=np.float64).reshape(3,*grid.shape_zyx)*.013-1.3
        field=DisplacementField(values,grid)
        with tempfile.TemporaryDirectory() as temporary:
            for dtype in (np.float32,np.float64):
                for suffix in ('.nii','.nii.gz'):
                    with self.subTest(dtype=dtype,suffix=suffix):
                        old_values=values.transpose(3,2,1,0)[:,:,:,None,:].astype(dtype,copy=False)
                        reference=nib.Nifti1Image(old_values,grid.affine_ras,dtype=dtype)
                        reference.header.set_intent('vector')
                        reference.header.set_xyzt_units('mm')
                        reference.set_qform(grid.affine_ras,1)
                        reference.set_sform(grid.affine_ras,1)
                        oracle=Path(temporary)/('oracle'+suffix)
                        output=Path(temporary)/('actual'+suffix)
                        nib.save(reference,oracle)
                        with patch('fast_ants.io.nifti._CHUNK_BYTES',19), patch(
                                'fast_ants.io.nifti.nib.save',side_effect=AssertionError('unexpected fallback')):
                            save_field(field,output,dtype)
                        self.assertEqual(decoded(output),decoded(oracle))
                        loaded=nib.load(output)
                        self.assertEqual(loaded.shape,(*grid.shape_xyz,1,3))
                        self.assertEqual(loaded.header.get_intent()[0],'vector')
                        np.testing.assert_array_equal(np.asarray(loaded.dataobj),old_values)
            storage=np.zeros(grid.shape_zyx,np.float32)
            def fresh():
                return nib.Nifti1Image(storage.transpose(2,1,0),grid.affine_ras)
            extended=fresh()
            extended.header.extensions.append(nib.nifti1.Nifti1Extension(6,b'test extension'))
            scaled=fresh();scaled.header.set_slope_inter(2.,.5)
            mismatched=fresh();mismatched.set_data_dtype(np.float64)
            for image,path,raw in ((fresh(),'unsupported.img',storage),
                    (extended,'extended.nii',storage),(scaled,'scaled.nii',storage),
                    (mismatched,'dtype.nii',storage),(fresh(),'layout.nii',storage[:,:,::-1]),
                    (fresh(),'different.nii',storage.copy())):
                with self.subTest(fallback=path), patch('fast_ants.io.nifti.nib.save') as fallback:
                    _save_nifti_storage(image,path,raw)
                    fallback.assert_called_once_with(image,path)
        np.testing.assert_array_equal(field.data,values)


if __name__=='__main__':unittest.main()
