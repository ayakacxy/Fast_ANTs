"""Assess saved outputs by registration quality, without refitting or a map-equality gate.

Both the independent and official maps use the same CPU sampler for QA. NCC
uses a fixed-image intensity foreground, not a clinical brain segmentation.
Analytic landmarks have known truth; real MRI landmarks are coarse exploratory
annotations. These metrics inform review and do not certify anatomical accuracy.
"""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION

import argparse
import json
from pathlib import Path
import sys

from validation.benchmarks.measure_fp64_cpu import PROJECT, digest, import_implementation, load_reference_image, write_json
from validation.quality.verify_float1_alignment import ARTIFACTS, hashes


def deformation_regularity(forward,inverse):
    """Saved-field QA in both directions, retaining the original forward keys.

    The inverse field has its own derivative stencil evaluation; a regular
    forward field alone does not establish a regular saved inverse. Each
    Jacobian volume is released after its summary, before evaluating the
    other direction or the existing inverse-consistency check.
    """
    import numpy as np
    from fast_ants.fields import jacobian_determinant,inverse_consistency

    def jacobian_summary(field):
        jacobian=jacobian_determinant(field)
        return (float(jacobian.min()),int((jacobian<=0).sum()),
                bool(np.isfinite(field.data).all()),bool(np.isfinite(jacobian).all()))

    minimum,count,field_finite,jacobian_finite=jacobian_summary(forward)
    inverse_minimum,inverse_count,inverse_finite,inverse_jacobian_finite=jacobian_summary(inverse)
    return {'jacobian_minimum':minimum,'nonpositive_jacobian_points':count,
            'inverse_jacobian_minimum':inverse_minimum,
            'nonpositive_inverse_jacobian_points':inverse_count,
            'forward_field_finite':field_finite,'inverse_field_finite':inverse_finite,
            'forward_jacobian_finite':jacobian_finite,'inverse_jacobian_finite':inverse_jacobian_finite,
            'inverse_consistency':inverse_consistency(forward,inverse)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True,help='Saved verify_float1_alignment report.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report_path=args.report.resolve();out=args.output.resolve()
    if out.exists():parser.error('Use a new output directory')
    report=json.loads(report_path.read_text());source=Path(report['source'])
    if report['source_sha256']!=hashes(source):raise ValueError('Source snapshot no longer matches registration report')
    if not report.get('source_unchanged_during_run'):raise ValueError('Registration source changed during fitting')
    if not report.get('native_provenance'):raise ValueError('Verified reference receipt required')
    for name,expected in report['input_sha256'].items():
        if digest(name)!=expected:raise ValueError('Registration input has changed: '+name)
    api,_=import_implementation(source,1)
    import numpy as np
    from fast_ants.transforms import AffineTransform
    case=report['case'];fixed,moving=[load_reference_image(api,case[s]) for s in ('fixed','moving')]
    own=Path(report.get('independent_directory',report_path.parent/report['backend']))
    native=Path(report['native_provenance']['receipt']).parent
    inputs={}
    pinned_own=report.get('saved_artifact_sha256')
    if pinned_own:
        for name,expected in pinned_own.items():
            if digest(own/name)!=expected:raise ValueError('Saved independent artifact changed: '+name)
    def load_map(directory,official):
        names=ARTIFACTS[:3] if official else ('affine.mat','warp.nii.gz','inverse-warp.nii.gz')
        for name in names:
            inputs[str(directory/name)]=digest(directory/name)
            if official and inputs[str(directory/name)]!=report['native_provenance']['artifacts'][name]:
                raise ValueError('Official transform artifact changed: '+name)
        a,fw,iv=api.load_affine(directory/names[0]),api.load_field(directory/names[1]),api.load_field(directory/names[2])
        if not fw.grid.same_geometry(fixed.grid) or not iv.grid.same_geometry(fixed.grid):
            raise ValueError('Saved field geometry mismatch')
        return api.CompositeTransform((fw,a)),fw,iv
    own_map,own_fw,own_iv=load_map(own,False)
    native_map,native_fw,native_iv=load_map(native,True)
    f=fixed.data.astype(np.float64);m=moving.data.astype(np.float64)
    fmin,fspan=float(f.min()),float(np.ptp(f));mmin,mspan=float(m.min()),float(np.ptp(m))
    if min(fspan,mspan)<=0:raise ValueError('Nonconstant images required for intensity quality metrics')
    mask=f>fmin+.05*fspan
    if mask.sum()<8:raise ValueError('Too few fixed foreground samples')
    target=(f[mask]-fmin)/fspan
    def similarity(transform):
        warped=api.apply_image(moving,fixed,transform).data.astype(np.float64)
        values=(warped[mask]-mmin)/mspan
        x=target-target.mean();y=values-values.mean()
        norm=np.sqrt(np.sum(x*x)*np.sum(y*y))
        return {'foreground_ncc':None if norm==0 else float(np.sum(x*y)/norm),
                'foreground_normalized_mse':float(np.mean((values-target)**2)),
                'foreground_points':int(mask.sum()),'finite':bool(np.isfinite(warped).all())}
    maps={'before_identity':AffineTransform.identity(),'independent':own_map,'official_float1':native_map}
    metrics={name:similarity(t) for name,t in maps.items()}
    # New fitting reports retain the exact case/annotations used to fit. Older
    # reports already contain those fields too; avoid reading an unpinned
    # mutable global manifest when assessing a custom pair or direction.
    annotations=case
    if annotations.get('fixed_landmarks_lps_mm'):
        points,truth=[np.asarray(annotations[s+'_landmarks_lps_mm'],np.float64) for s in ('fixed','moving')]
        if report['swap'] and not report.get('annotations_already_directional',False):points,truth=truth,points
        for name,t in maps.items():
            distance=np.linalg.norm(t.map_points(points)-truth,axis=1)
            metrics[name]['landmark_error_mm']={'rms':float(np.sqrt(np.mean(distance**2))),
                'mean':float(distance.mean()),'maximum':float(distance.max()),'per_point':distance.tolist()}
    regularity={}
    for name,fw,iv in (('independent',own_fw,own_iv),('official_float1',native_fw,native_iv)):
        regularity[name]=deformation_regularity(fw,iv)
    out.mkdir(parents=True)
    assessment={'status':'quality_metrics_recorded_not_full_acceptance','scope':__doc__,
        'registration_report':str(report_path),'registration_report_sha256':digest(report_path),
        'artifact_sha256':inputs,'metrics':metrics,'regularity':regularity,
        'independent_artifact_integrity':'verified_against_fitting_report' if pinned_own else 'historical_report_without_fit_time_output_hashes',
        'landmark_scope':'analytic known truth' if case['kind']=='analytic' else 'coarse non-expert MRI annotations; exploratory only',
        'similarity_domain':'fixed intensity > minimum + 5% of fixed range, same points for all maps',
        'reference_deformation_differences_are_diagnostic':True,
        'sampler':'same native CPU double QA sampler for all saved transforms; not part of the FP32/Metal fitting backend'}
    write_json(out/'report.json',assessment)
    print(json.dumps({'metrics':metrics,'regularity':regularity},indent=2),flush=True)


if __name__=='__main__':main()
