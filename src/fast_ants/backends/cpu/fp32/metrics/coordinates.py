"""Four independent dense sample mappings using portable LLVM vectors."""

from llvmlite import ir
from numba import types
from numba.extending import intrinsic

from fast_ants.backends.cpu.fp32.metrics.parzen import _VECTOR, _splat, _load, _store, _function


@intrinsic
def map_integer4(typingctx, points, first, integer_linear, residual_linear,
                 integer_offset, residual_offset, origin, bases, offsets):
    """Map four integer-plan points; caller guarantees contiguous buffers."""
    def codegen(context, builder, signature, arguments):
        arrays = {i: context.make_array(signature.args[i])(context, builder, arguments[i])
                  for i in (0, 2, 3, 4, 5, 6, 7, 8)}
        n = context.cast(builder, arguments[1], signature.args[1], types.intp)
        index_type = n.type
        def index(value):
            return ir.Constant(index_type, value)
        start = builder.mul(n, index(3))
        a = _load(builder, arrays[0].data, start)
        b = _load(builder, arrays[0].data, builder.add(start, index(4)))
        c = _load(builder, arrays[0].data, builder.add(start, index(8)))
        def shuffle(left, right, lanes):
            mask = ir.Constant(ir.VectorType(ir.IntType(32), 4), lanes)
            return builder.shuffle_vector(left, right, mask)
        coordinates = [shuffle(shuffle(a, b, [0, 3, 6, 0]), c, [0, 1, 2, 5]),
                       shuffle(shuffle(a, b, [1, 4, 7, 0]), c, [0, 1, 2, 6]),
                       shuffle(shuffle(a, b, [2, 5, 0, 0]), c, [0, 1, 4, 7])]
        integer_vectors, floating_vectors = [], []
        int32_vector = ir.VectorType(ir.IntType(32), 4)
        int64_vector = ir.VectorType(ir.IntType(64), 4)
        for axis in range(3):
            center = builder.load(builder.gep(arrays[6].data, [index(axis)]))
            relative = builder.fsub(coordinates[axis], _splat(builder, center))
            # Retain the scalar int64 conversion before the float32 roundtrip.
            whole = builder.fptosi(relative, int64_vector)
            integer_vectors.append(builder.trunc(whole, int32_vector))
            floating_vectors.append(builder.sitofp(whole, _VECTOR))
        fma = _function(builder, 'llvm.fma.v4f32', [_VECTOR] * 3)
        def splat_int(value):
            vector = builder.insert_element(ir.Constant(int32_vector, ir.Undefined), value,
                                             ir.Constant(ir.IntType(32), 0))
            return shuffle(vector, vector, [0, 0, 0, 0])
        for component in range(3):
            base = splat_int(builder.load(builder.gep(arrays[4].data, [index(component)])))
            offset = _splat(builder, builder.load(builder.gep(arrays[5].data, [index(component)])))
            for axis in range(3):
                position = index(component * 3 + axis)
                integer = builder.load(builder.gep(arrays[2].data, [position]))
                residual = builder.load(builder.gep(arrays[3].data, [position]))
                base = builder.add(base, builder.mul(splat_int(integer), integer_vectors[axis]))
                offset = builder.call(fma, [_splat(builder, residual), floating_vectors[axis], offset])
            pointer = builder.bitcast(builder.gep(arrays[7].data, [index(component * 4)]),
                                      int32_vector.as_pointer())
            builder.store(base, pointer, align=4)
            _store(builder, arrays[8].data, index(component * 4), offset)
        return context.get_dummy_value()

    return types.void(points, first, integer_linear, residual_linear, integer_offset,
                      residual_offset, origin, bases, offsets), codegen


def integer_plan_kind(points, mapping, integer_points):
    if not integer_points:
        return 0
    return 2 if points.flags.c_contiguous and all(a.flags.c_contiguous for a in mapping) else 1
