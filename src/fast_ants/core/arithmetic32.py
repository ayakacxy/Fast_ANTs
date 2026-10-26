"""One native FP32 fused multiply-add; no software-expanded arithmetic."""

from numba import types
from numba.extending import intrinsic
from llvmlite import ir


@intrinsic
def fma32(typingctx,a,b,c):
    if any(t!=types.float32 for t in (a,b,c)):
        raise TypeError('Native FMA requires FP32 operands')
    def codegen(context,builder,signature,args):
        function=builder.module.globals.get('llvm.fma.f32')
        if function is None:
            function=ir.Function(builder.module,ir.FunctionType(ir.FloatType(),[ir.FloatType()]*3),name='llvm.fma.f32')
        return builder.call(function,args)
    return types.float32(a,b,c),codegen
