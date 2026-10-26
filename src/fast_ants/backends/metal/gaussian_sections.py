"""FP32 two-section Gaussian row filter; no dense CPU fallback.

ITK transfer coefficients: NumFOCUS, Apache-2.0. The two damped sinusoid
components are evaluated separately rather than forming a fourth-order state.
Difference inputs drive the state and output is input plus its correction,
which preserves constants without an accumulated DC component.
"""
import numpy as np
from fast_ants.backends.metal.runtime import mx


_SOURCE='''uint line=thread_position_in_grid.x;uint L=meta[0],base=line*L;
  for(uint section=0;section<2;section++){
    uint c=section*4;float previous=0,previous2=0,old_delta=0,old=data[base];
    for(uint i=0;i<L;i++){
      float current=data[base+i],delta=current-old;
      float value=coeff[c+2]*delta;
      value=metal::fma(coeff[c+3],old_delta,value);
      value=metal::fma(-coeff[c],previous,value);
      value=metal::fma(-coeff[c+1],previous2,value);
      out1[(line*2+section)*L+i]=value;
      previous2=previous;previous=value;old_delta=delta;old=current;
    }
  }
  float reverse1=0,reverse2=0,reverse12=0,reverse22=0,old_delta=0,old=data[base+L-1];
  for(int i=int(L)-1;i>=0;i--){
    float current=data[base+i],delta=current-old;
    float value1=coeff[2]*delta;
    value1=metal::fma(coeff[3],old_delta,value1);
    value1=metal::fma(-coeff[0],reverse1,value1);
    value1=metal::fma(-coeff[1],reverse12,value1);
    float value2=coeff[6]*delta;
    value2=metal::fma(coeff[7],old_delta,value2);
    value2=metal::fma(-coeff[4],reverse2,value2);
    value2=metal::fma(-coeff[5],reverse22,value2);
    reverse12=reverse1;reverse1=value1;reverse22=reverse2;reverse2=value2;
    old_delta=delta;old=current;
    float correction=(out1[line*2*L+i]+value1)+(out1[(line*2+1)*L+i]+value2);
    out0[base+i]=current+correction;
  }'''


def section_lines32(ctx,lines,coefficients):
    """Filter FP32 device rows with a small (2,4) host coefficient table.

    The output uses one device dispatch and two forward-state scratch rows
    per input row. Image data is never downloaded. Empty arrays return without
    dispatch. The coefficient table must come from coefficients32(sigma).
    """
    if not isinstance(lines,mx.array) or lines.dtype!=mx.float32 or lines.ndim!=2:
        raise ValueError('Section filter requires two-dimensional FP32 device rows')
    if not isinstance(coefficients,np.ndarray) or coefficients.dtype!=np.float32 or coefficients.shape!=(2,4):
        raise ValueError('Section filter requires a (2,4) float32 coefficient table')
    if not np.isfinite(coefficients).all():raise ValueError('Finite section coefficients required')
    count,length=lines.shape
    if not lines.size:return lines
    output,_=ctx.call('gaussian_sections32',_SOURCE,
        {'data':lines,'coeff':ctx.upload(coefficients,'gaussian_section_coefficients'),'meta':ctx.meta(length)},
        [lines.shape,(count,2,length)],count)
    return output
