// Ordinary FP32 image interpolation with ITK half-voxel support and edge clamping.
// Kernels are compiled with --fmad=false to retain the CPU FP32 operation order.

__device__ __forceinline__ float2 compensated(float2 a,float v) {
    float corrected=v-a.y,updated=a.x+corrected;
    return make_float2(updated,(updated-a.x)-corrected);
}

__device__ __forceinline__ bool edge(int3 p,int3 n) {
    return p.x==0||p.y==0||p.z==0||p.x==n.x-1||p.y==n.y-1||p.z==n.z-1;
}

__device__ __forceinline__ bool inside(float3 p,int3 n) {
    return p.x>=-0.5f&&p.x<float(n.x)-0.5f
        &&p.y>=-0.5f&&p.y<float(n.y)-0.5f
        &&p.z>=-0.5f&&p.z<float(n.z)-0.5f;
}

__device__ __forceinline__ unsigned int idx(int x,int y,int z,int3 n) {
    return (static_cast<unsigned int>(z)*n.y+y)*n.x+x;
}

__device__ __forceinline__ int3 xyz(unsigned int i,int3 n) {
    unsigned int nx=n.x,ny=n.y;
    return make_int3(i%nx,(i/nx)%ny,i/(nx*ny));
}

__device__ __forceinline__ float sample1(const float* a,float3 p,int3 n,float anchor=0.0f) {
    p.x=fminf(fmaxf(p.x,0.0f),float(n.x-1));
    p.y=fminf(fmaxf(p.y,0.0f),float(n.y-1));
    p.z=fminf(fmaxf(p.z,0.0f),float(n.z-1));
    int3 lo=make_int3(int(floorf(p.x)),int(floorf(p.y)),int(floorf(p.z)));
    int3 hi=make_int3(min(lo.x+1,n.x-1),min(lo.y+1,n.y-1),min(lo.z+1,n.z-1));
    float3 d=make_float3(p.x-float(lo.x),p.y-float(lo.y),p.z-float(lo.z));
    float aa=(a[idx(lo.x,lo.y,lo.z,n)]-anchor)+(a[idx(hi.x,lo.y,lo.z,n)]-a[idx(lo.x,lo.y,lo.z,n)])*d.x;
    float bb=(a[idx(lo.x,hi.y,lo.z,n)]-anchor)+(a[idx(hi.x,hi.y,lo.z,n)]-a[idx(lo.x,hi.y,lo.z,n)])*d.x;
    float cc=(a[idx(lo.x,lo.y,hi.z,n)]-anchor)+(a[idx(hi.x,lo.y,hi.z,n)]-a[idx(lo.x,lo.y,hi.z,n)])*d.x;
    float dd=(a[idx(lo.x,hi.y,hi.z,n)]-anchor)+(a[idx(hi.x,hi.y,hi.z,n)]-a[idx(lo.x,hi.y,hi.z,n)])*d.x;
    float l=aa+(bb-aa)*d.y,h=cc+(dd-cc)*d.y;
    return l+(h-l)*d.z;
}

__device__ __forceinline__ float cubic(float argument) {
    float a=fabsf(argument);
    if(a<1.0f)return __fmaf_rn(a*a,__fmaf_rn(3.0f,a,-6.0f),4.0f)/6.0f;
    if(a<2.0f)return (2.0f-a)*(2.0f-a)*(2.0f-a)/6.0f;
    return 0.0f;
}

__device__ __forceinline__ float cubic_derivative(float argument) {
    float a=fabsf(argument),sign=argument>=0?1.0f:-1.0f;
    if(a<1.0f)return sign*a*__fmaf_rn(1.5f,a,-2.0f);
    if(a<2.0f)return -0.5f*sign*(2.0f-a)*(2.0f-a);
    return 0.0f;
}

struct LocalAxis { int lo; int hi; float fraction; bool valid; };

__device__ __forceinline__ LocalAxis local_axis(int base,float offset,int size) {
    // Reject nonfinite or out-of-range offsets before converting to an integer.
    if(!isfinite(offset)||offset<=-2147483520.0f||offset>=2147483520.0f)
        return {0,0,0.0f,false};
    int whole=int(floorf(offset));float fraction=offset-float(whole);
    long long index=static_cast<long long>(base)+whole;
    bool valid=!(index < -1||(index==-1&&fraction<0.5f)||index>=size||
                 (index==size-1&&fraction>=0.5f));
    if(index<0){index=0;fraction=0;}
    if(index>=size-1){index=size-1;fraction=0;}
    int lo=int(index);
    return {lo,min(lo+1,size-1),fraction,valid};
}

__device__ __forceinline__ bool local_inside(int3 base,float3 offset,int3 n) {
    return local_axis(base.x,offset.x,n.x).valid&&local_axis(base.y,offset.y,n.y).valid
        &&local_axis(base.z,offset.z,n.z).valid;
}

__device__ __forceinline__ float local_at(const float* data,int3 base,float3 p,int3 n,float anchor=0.0f) {
    LocalAxis x=local_axis(base.x,p.x,n.x),y=local_axis(base.y,p.y,n.y),z=local_axis(base.z,p.z,n.z);
    float a=__fmaf_rn(data[idx(x.hi,y.lo,z.lo,n)]-data[idx(x.lo,y.lo,z.lo,n)],x.fraction,data[idx(x.lo,y.lo,z.lo,n)]-anchor);
    float b=__fmaf_rn(data[idx(x.hi,y.hi,z.lo,n)]-data[idx(x.lo,y.hi,z.lo,n)],x.fraction,data[idx(x.lo,y.hi,z.lo,n)]-anchor);
    float c=__fmaf_rn(data[idx(x.hi,y.lo,z.hi,n)]-data[idx(x.lo,y.lo,z.hi,n)],x.fraction,data[idx(x.lo,y.lo,z.hi,n)]-anchor);
    float d=__fmaf_rn(data[idx(x.hi,y.hi,z.hi,n)]-data[idx(x.lo,y.hi,z.hi,n)],x.fraction,data[idx(x.lo,y.hi,z.hi,n)]-anchor);
    float lo=__fmaf_rn(b-a,y.fraction,a),hi=__fmaf_rn(d-c,y.fraction,c);
    return __fmaf_rn(hi-lo,z.fraction,lo);
}
