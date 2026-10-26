"""Prepare a larger timing fixture from the same three existing MRI subjects."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import json
from pathlib import Path
import nibabel as nib
import numpy as np
from validation.common.p00_common import PROJECT, block_mean, sha256


def main():
    out=PROJECT/'validation/data/p11-half';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted((PROJECT/'validation/data/public').glob('*.nii.gz')):
        raw=nib.load(path);values,affine=block_mean(raw.get_fdata(dtype=np.float32),raw.affine,2)
        scale=float(values.max());values/=np.float32(scale)
        image=nib.Nifti1Image(values,affine);image.set_qform(affine,1);image.set_sform(affine,1);image.header.set_xyzt_units('mm')
        target=out/path.name;nib.save(image,target)
        rows.append({'source':str(path.relative_to(PROJECT)),'source_sha256':sha256(path),'output':str(target.relative_to(PROJECT)),
                     'sha256':sha256(target),'shape_xyz':values.shape,'intensity_divisor':scale})
    if len(rows)!=3:raise AssertionError('No new MRI subjects allowed')
    (out/'manifest.json').write_text(json.dumps({'operation':'2x2x2 block mean; crop high edge remainder; block center geometry; divide by maximum',
        'scope':'P11 scaling diagnostic only, not a new frozen P12 quality set','subjects':rows},indent=2)+'\n')
    print('Prepared larger fixtures from the same',len(rows),'subjects',flush=True)


if __name__=='__main__':main()
