"""One hardware FP64 fused multiply-add, with its rounding boundary explicit."""

from numba import types
from numba.extending import intrinsic
from llvmlite import ir


@intrinsic
def fma64(typingctx, a, b, c):
    if any(t != types.float64 for t in (a, b, c)):
        raise TypeError("Native FP64 FMA requires FP64 operands")

    def codegen(context, builder, signature, args):
        function = builder.module.globals.get("llvm.fma.f64")
        if function is None:
            function = ir.Function(builder.module,
                ir.FunctionType(ir.DoubleType(), [ir.DoubleType()]*3), name="llvm.fma.f64")
        return builder.call(function, args)

    return types.float64(a, b, c), codegen
