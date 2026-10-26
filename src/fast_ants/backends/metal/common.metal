// Independently implemented ordinary FP32 operators, Apache-2.0.
// ITK recurrence/registration rules: Copyright NumFOCUS, see licenses/LICENSE-ITK.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline float2 compensated(float2 a, float v) {
    float corrected = v - a.y;
    float updated = a.x + corrected;
    return float2(updated, (updated - a.x) - corrected);
}
inline bool inside(float3 p, int3 n) {
    return all(p >= float3(-0.5f)) && all(p < float3(n) - 0.5f);
}
inline uint idx(int x, int y, int z, int3 n) { return uint((z*n.y+y)*n.x+x); }
inline int3 xyz(uint i, int3 n) { return int3(i%uint(n.x), (i/uint(n.x))%uint(n.y), i/uint(n.x*n.y)); }
inline bool edge(int3 p, int3 n) { return any(p==int3(0)) || any(p==n-1); }
inline float sample1(device const float* a, float3 p, int3 n, float anchor=0.0f) {
    p=metal::clamp(p,float3(0),float3(n-1));
    int3 lo=int3(metal::floor(p)), hi=metal::min(lo+1,n-1);
    float3 d=p-float3(lo);
    float aa=(a[idx(lo.x,lo.y,lo.z,n)]-anchor)+(a[idx(hi.x,lo.y,lo.z,n)]-a[idx(lo.x,lo.y,lo.z,n)])*d.x;
    float bb=(a[idx(lo.x,hi.y,lo.z,n)]-anchor)+(a[idx(hi.x,hi.y,lo.z,n)]-a[idx(lo.x,hi.y,lo.z,n)])*d.x;
    float cc=(a[idx(lo.x,lo.y,hi.z,n)]-anchor)+(a[idx(hi.x,lo.y,hi.z,n)]-a[idx(lo.x,lo.y,hi.z,n)])*d.x;
    float dd=(a[idx(lo.x,hi.y,hi.z,n)]-anchor)+(a[idx(hi.x,hi.y,hi.z,n)]-a[idx(lo.x,hi.y,hi.z,n)])*d.x;
    float l=aa+(bb-aa)*d.y, h=cc+(dd-cc)*d.y;
    return l+(h-l)*d.z;
}

// MLX can bind tiny image inputs through the constant address space. Keep the
// same interpolant available there as for ordinary device-backed volumes.
inline float sample1(constant const float* a, float3 p, int3 n, float anchor=0.0f) {
    p=metal::clamp(p,float3(0),float3(n-1));
    int3 lo=int3(metal::floor(p)), hi=metal::min(lo+1,n-1);
    float3 d=p-float3(lo);
    float aa=(a[idx(lo.x,lo.y,lo.z,n)]-anchor)+(a[idx(hi.x,lo.y,lo.z,n)]-a[idx(lo.x,lo.y,lo.z,n)])*d.x;
    float bb=(a[idx(lo.x,hi.y,lo.z,n)]-anchor)+(a[idx(hi.x,hi.y,lo.z,n)]-a[idx(lo.x,hi.y,lo.z,n)])*d.x;
    float cc=(a[idx(lo.x,lo.y,hi.z,n)]-anchor)+(a[idx(hi.x,lo.y,hi.z,n)]-a[idx(lo.x,lo.y,hi.z,n)])*d.x;
    float dd=(a[idx(lo.x,hi.y,hi.z,n)]-anchor)+(a[idx(hi.x,hi.y,hi.z,n)]-a[idx(lo.x,hi.y,hi.z,n)])*d.x;
    float l=aa+(bb-aa)*d.y, h=cc+(dd-cc)*d.y;
    return l+(h-l)*d.z;
}

inline float cubic(float argument) {
    float a=metal::abs(argument);
    if(a<1.0f)return metal::fma(a*a,metal::fma(3.0f,a,-6.0f),4.0f)/6.0f;
    if(a<2.0f)return (2.0f-a)*(2.0f-a)*(2.0f-a)/6.0f;
    return 0.0f;
}
inline float cubic_derivative(float argument) {
    float a=metal::abs(argument),sign=argument>=0?1.0f:-1.0f;
    if(a<1.0f)return sign*a*metal::fma(1.5f,a,-2.0f);
    if(a<2.0f)return -0.5f*sign*(2.0f-a)*(2.0f-a);
    return 0.0f;
}

struct LocalAxis { int lo; int hi; float fraction; bool valid; };
inline LocalAxis local_axis(int base,float offset,int size) {
    int whole=int(metal::floor(offset));float f=offset-float(whole);int b=base+whole;
    bool valid=!(b < -1 || (b==-1&&f<0.5f) || b>=size || (b==size-1&&f>=0.5f));
    if(b<0){b=0;f=0;}if(b>=size-1){b=size-1;f=0;}
    return {b,metal::min(b+1,size-1),f,valid};
}
inline bool local_inside(int3 b,float3 p,int3 n) {
    return local_axis(b.x,p.x,n.x).valid&&local_axis(b.y,p.y,n.y).valid&&local_axis(b.z,p.z,n.z).valid;
}
inline float local_at(device const float* data,int3 base,float3 p,int3 n,float anchor=0.0f) {
    LocalAxis x=local_axis(base.x,p.x,n.x),y=local_axis(base.y,p.y,n.y),z=local_axis(base.z,p.z,n.z);
    float a=metal::fma(data[idx(x.hi,y.lo,z.lo,n)]-data[idx(x.lo,y.lo,z.lo,n)],x.fraction,data[idx(x.lo,y.lo,z.lo,n)]-anchor);
    float b=metal::fma(data[idx(x.hi,y.hi,z.lo,n)]-data[idx(x.lo,y.hi,z.lo,n)],x.fraction,data[idx(x.lo,y.hi,z.lo,n)]-anchor);
    float c=metal::fma(data[idx(x.hi,y.lo,z.hi,n)]-data[idx(x.lo,y.lo,z.hi,n)],x.fraction,data[idx(x.lo,y.lo,z.hi,n)]-anchor);
    float d=metal::fma(data[idx(x.hi,y.hi,z.hi,n)]-data[idx(x.lo,y.hi,z.hi,n)],x.fraction,data[idx(x.lo,y.hi,z.hi,n)]-anchor);
    float lo=metal::fma(b-a,y.fraction,a),hi=metal::fma(d-c,y.fraction,c);
    return metal::fma(hi-lo,z.fraction,lo);
}
