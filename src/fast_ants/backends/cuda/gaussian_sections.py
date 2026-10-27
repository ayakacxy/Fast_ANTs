"""FP32 two-section Gaussian row filter; no dense CPU fallback.

ITK transfer coefficients: NumFOCUS, Apache-2.0. The two damped sinusoid
components are evaluated separately rather than forming a fourth-order state.
Difference inputs drive the state and output is input plus its correction,
which preserves constants without an accumulated DC component.
"""
import numpy as np
from fast_ants.backends.cuda.runtime import cp


_SOURCE='''unsigned int line=i,L=meta[0],S=meta[1],N=meta[2];
  unsigned int base=(line/S)*(L*S)+line%S;
  float previous1=0,previous2=0,previous12=0,previous22=0,old_delta=0,old=data[base];
  for(unsigned int p=0;p<L;p++){
    float current=data[base+p*S],delta=current-old;
    float value1=coeff[2]*delta;
    value1=fmaf(coeff[3],old_delta,value1);
    value1=fmaf(-coeff[0],previous1,value1);
    value1=fmaf(-coeff[1],previous12,value1);
    float value2=coeff[6]*delta;
    value2=fmaf(coeff[7],old_delta,value2);
    value2=fmaf(-coeff[4],previous2,value2);
    value2=fmaf(-coeff[5],previous22,value2);
    out1[p*N+line]=value1;out1[(L+p)*N+line]=value2;
    previous12=previous1;previous1=value1;previous22=previous2;previous2=value2;
    old_delta=delta;old=current;
  }
  float reverse1=0,reverse2=0,reverse12=0,reverse22=0;old_delta=0;old=data[base+(L-1)*S];
  for(int p=int(L)-1;p>=0;p--){
    float current=data[base+p*S],delta=current-old;
    float value1=coeff[2]*delta;
    value1=fmaf(coeff[3],old_delta,value1);
    value1=fmaf(-coeff[0],reverse1,value1);
    value1=fmaf(-coeff[1],reverse12,value1);
    float value2=coeff[6]*delta;
    value2=fmaf(coeff[7],old_delta,value2);
    value2=fmaf(-coeff[4],reverse2,value2);
    value2=fmaf(-coeff[5],reverse22,value2);
    reverse12=reverse1;reverse1=value1;reverse22=reverse2;reverse2=value2;
    old_delta=delta;old=current;
    float correction=(out1[p*N+line]+value1)+(out1[(L+p)*N+line]+value2);
    out0[base+p*S]=current+correction;
  }'''


def _coefficients32(coefficients):
    if not isinstance(coefficients,np.ndarray) or coefficients.dtype!=np.float32 or coefficients.shape!=(2,4):
        raise ValueError('Section filter requires a (2,4) float32 coefficient table')
    if not np.isfinite(coefficients).all():raise ValueError('Finite section coefficients required')


def _section32(ctx,values,coefficients,length,stride):
    count=values.size//length
    output,_=ctx.call('gaussian_sections32',_SOURCE,
        {'data':values,'coeff':ctx.geometry(coefficients,'gaussian_section_coefficients'),
         'meta':ctx.meta(length,stride,count)},[values.shape,(2,length,count)],count)
    return output


def section_lines32(ctx,lines,coefficients):
    """Filter FP32 device rows with the established sequential recurrence.

    Pixel-major working storage keeps adjacent CUDA lanes on adjacent lines.
    Each line retains the same multiply/FMA order and constant extension;
    image data is never downloaded. Empty arrays return without dispatch.
    """
    if not isinstance(lines,cp.ndarray) or lines.dtype!=cp.float32 or lines.ndim!=2:
        raise ValueError('Section filter requires two-dimensional FP32 device rows')
    _coefficients32(coefficients)
    count,length=lines.shape
    if not lines.size:return lines
    ctx.require(lines)
    with ctx.device,ctx.stream:
        values=cp.ascontiguousarray(lines.T)
    result=_section32(ctx,values,coefficients,length,count)
    with ctx.device,ctx.stream:
        return cp.ascontiguousarray(result.T)


def section_axis32(ctx,values,coefficients,axis):
    """Filter native z/y axes directly; transpose only contiguous x rows.

    Forward states use pixel-major storage independently of input strides.
    The result has independent contiguous storage in the original axis order.
    """
    if not isinstance(values,cp.ndarray) or values.dtype!=cp.float32 or values.ndim!=3:
        raise ValueError('Section axis filter requires three-dimensional FP32 device values')
    if axis not in (0,1,2):raise ValueError('Section axis must lie in [0,2]')
    _coefficients32(coefficients);ctx.require(values)
    if not values.size:return values
    if axis==2:return section_lines32(ctx,values.reshape(-1,values.shape[2]),coefficients).reshape(values.shape)
    stride=int(np.prod(values.shape[axis+1:]))
    return _section32(ctx,values,coefficients,values.shape[axis],stride)
