"""Four-bin SIMD updates for the FP32 Mattes Parzen histogram.

The cubic B-spline formulas follow ITK (NumFOCUS, Apache-2.0). Four adjacent
bins are independent Kahan accumulators, so vector lanes preserve each bin's
scalar operation order. LLVM lowers the portable vectors to the host ISA.
"""

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic


_FLOAT = ir.FloatType()
_VECTOR = ir.VectorType(_FLOAT, 4)


def _constant(value):
    return ir.Constant(_VECTOR, [ir.Constant(_FLOAT, value)] * 4)


def _splat(builder, value):
    vector = builder.insert_element(ir.Constant(_VECTOR, ir.Undefined), value,
                                    ir.Constant(ir.IntType(32), 0))
    mask = ir.Constant(ir.VectorType(ir.IntType(32), 4), [0, 0, 0, 0])
    return builder.shuffle_vector(vector, vector, mask)


def _function(builder, name, arguments):
    function = builder.module.globals.get(name)
    if function is None:
        function = ir.Function(builder.module, ir.FunctionType(_VECTOR, arguments), name=name)
    return function


def _load(builder, data, offset):
    pointer = builder.bitcast(builder.gep(data, [offset]), _VECTOR.as_pointer())
    return builder.load(pointer, align=4)


def _store(builder, data, offset, value):
    pointer = builder.bitcast(builder.gep(data, [offset]), _VECTOR.as_pointer())
    builder.store(value, pointer, align=4)


def _add(builder, total, correction, value):
    adjusted = builder.fsub(value, correction)
    updated = builder.fadd(total, adjusted)
    return updated, builder.fsub(builder.fsub(updated, total), adjusted)


@intrinsic
def parzen_add4(typingctx, joint, correction, fixed_bin, moving_bin, fraction):
    """Accumulate four weights and return their four scalar-order derivatives."""
    if (not isinstance(joint, types.Array) or joint.dtype != types.float32
            or joint.ndim != 2 or joint.layout != 'C'
            or correction != joint or fraction != types.float32):
        raise TypeError('Parzen SIMD needs contiguous FP32 histogram buffers')
    result_type = types.UniTuple(types.float32, 4)

    def codegen(context, builder, signature, arguments):
        histogram = context.make_array(signature.args[0])(context, builder, arguments[0])
        compensation = context.make_array(signature.args[1])(context, builder, arguments[1])
        bins = cgutils.unpack_tuple(builder, histogram.shape, 2)[1]
        fi = context.cast(builder, arguments[2], signature.args[2], types.intp)
        mi = context.cast(builder, arguments[3], signature.args[3], types.intp)
        start = builder.sub(builder.add(builder.mul(fi, bins), mi), ir.Constant(bins.type, 1))
        offsets = ir.Constant(_VECTOR, [ir.Constant(_FLOAT, x) for x in (-1., 0., 1., 2.)])
        argument = builder.fsub(offsets, _splat(builder, arguments[4]))
        absolute = builder.call(_function(builder, 'llvm.fabs.v4f32', [_VECTOR]), [argument])
        fma = _function(builder, 'llvm.fma.v4f32', [_VECTOR] * 3)
        inner = builder.call(fma, [_constant(3.), absolute, _constant(-6.)])
        inner = builder.call(fma, [builder.fmul(absolute, absolute), inner, _constant(4.)])
        inner = builder.fdiv(inner, _constant(6.))
        tail = builder.fsub(_constant(2.), absolute)
        outer = builder.fdiv(builder.fmul(builder.fmul(tail, tail), tail), _constant(6.))
        below_one = builder.fcmp_ordered('<', absolute, _constant(1.))
        below_two = builder.fcmp_ordered('<', absolute, _constant(2.))
        weights = builder.select(below_one, inner,
                                 builder.select(below_two, outer, _constant(0.)))
        total, remaining = _add(builder, _load(builder, histogram.data, start),
                                _load(builder, compensation.data, start), weights)
        _store(builder, histogram.data, start, total)
        _store(builder, compensation.data, start, remaining)

        # Unused derivative arithmetic is eliminated from value-only calls.
        sign = builder.select(builder.fcmp_ordered('>=', argument, _constant(0.)),
                              _constant(1.), _constant(-1.))
        inner_d = builder.fmul(builder.fmul(sign, absolute),
            builder.call(fma, [_constant(1.5), absolute, _constant(-2.)]))
        outer_d = builder.fmul(builder.fmul(builder.fmul(_constant(-.5), sign), tail), tail)
        gradient = builder.select(below_one, inner_d,
                                  builder.select(below_two, outer_d, _constant(0.)))
        values = [builder.extract_element(gradient, ir.Constant(ir.IntType(32), i))
                  for i in range(4)]
        return context.make_tuple(builder, result_type, values)

    return result_type(joint, correction, fixed_bin, moving_bin, fraction), codegen


@intrinsic
def derivative_add4(typingctx, derivative, correction, intensity, fixed_bin, moving_bin, channel, weight):
    """Update four contiguous parameter channels without reordering a sum."""
    if (not isinstance(derivative, types.Array) or derivative.dtype != types.float32
            or derivative.ndim != 3 or derivative.layout != 'C'
            or correction != derivative or not isinstance(intensity, types.Array)
            or intensity.dtype != types.float32 or intensity.ndim != 1
            or intensity.layout != 'C' or weight != types.float32):
        raise TypeError('Derivative SIMD needs contiguous FP32 buffers')

    def codegen(context, builder, signature, arguments):
        target = context.make_array(signature.args[0])(context, builder, arguments[0])
        compensation = context.make_array(signature.args[1])(context, builder, arguments[1])
        values = context.make_array(signature.args[2])(context, builder, arguments[2])
        _, bins, parameters = cgutils.unpack_tuple(builder, target.shape, 3)
        fi, mi, first = [context.cast(builder, arguments[i], signature.args[i], types.intp)
                         for i in (3, 4, 5)]
        offset = builder.add(builder.mul(builder.add(builder.mul(fi, bins), mi), parameters), first)
        terms = builder.fmul(_load(builder, values.data, first), _splat(builder, arguments[6]))
        total, remaining = _add(builder, _load(builder, target.data, offset),
                                _load(builder, compensation.data, offset), terms)
        _store(builder, target.data, offset, total)
        _store(builder, compensation.data, offset, remaining)
        return context.get_dummy_value()

    return types.void(derivative, correction, intensity, fixed_bin, moving_bin, channel, weight), codegen
