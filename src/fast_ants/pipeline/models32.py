"""Float-state rigid/affine models for the ANTs ``--float 1`` candidate path.

Adapted from ITK 5.4.7, NumFOCUS, Apache-2.0: Euler3DTransform.hxx
(ComputeMatrix and ComputeJacobianWithRespectToParameters), and
MatrixOffsetTransformBase.hxx (ComputeOffset, ComputeTranslation, Compose).

Parameters, centers, matrices, offsets and mapped points are float32. Euler
Jacobian coefficients use native double temporaries after float trigonometry,
as in ITK. No expanded floating-point representation is used. This does not
claim bitwise agreement with a particular C++ libm/FMA implementation.

The decomposed ``derivatives`` interface is compatible with existing CPU/Metal
metrics, which round its coefficients on upload. ``jacobian`` additionally
preserves the upstream point-subtraction and final Jacobian cast boundaries.
Composed transforms retain their component sequence as well as a collapsed
float affine. The default ``transform().map_points`` remains the collapsed
storage/QA view. ``evaluation_transform().map_points`` explicitly selects the
upstream per-member rounding; metric and scale callers must opt into that view.
"""

from dataclasses import dataclass, field
import numpy as np
from numba import njit

from fast_ants.core.types import points_array
from fast_ants.core.transforms import AffineTransform
from fast_ants.core.arithmetic32 import fma32


def _array32(value, shape, name):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(value, dtype=np.float32, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@njit(cache=True, fastmath=False)
def _matmul32(left, right):
    """Float dot products with the native contraction used by the CPU oracle."""
    output = np.zeros((left.shape[0], right.shape[1]), np.float32)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for k in range(left.shape[1]):
                output[i, j] = fma32(left[i, k], right[k, j], output[i, j])
    return output


@njit(cache=True, fastmath=False)
def _offset32(matrix, translation, center):
    offset = np.empty(3, np.float32)
    for i in range(3):
        offset[i] = np.float32(translation[i]+center[i])
        for j in range(3):
            offset[i] = fma32(-matrix[i, j], center[j], offset[i])
    return offset


@njit(cache=True, fastmath=False)
def _translation32(matrix, offset, center):
    translation = np.empty(3, np.float32)
    for i in range(3):
        translation[i] = np.float32(offset[i]-center[i])
        for j in range(3):
            translation[i] = fma32(matrix[i, j], center[j], translation[i])
    return translation


@njit(cache=True, fastmath=False)
def _map32(points, matrix):
    output = np.empty(points.shape, np.float32)
    for n in range(points.shape[0]):
        for i in range(3):
            value = np.float32(0)
            for j in range(3):
                value = fma32(matrix[i, j], points[n, j], value)
            output[n, i] = np.float32(value + matrix[i, 3])
    return output


@dataclass(frozen=True, eq=False)
class AffineTransform32(AffineTransform):
    """AffineTransform-compatible float state with explicit offset semantics."""

    _translation_state: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        # Reuse the public shape/invertibility checks, then own float state.
        super().__post_init__()
        matrix = _array32(self.matrix, (4, 4), "Affine matrix")
        center = _array32(self.center, (3,), "Transform center")
        if abs(float(np.linalg.det(matrix[:3, :3].astype(np.float64)))) < 1e-12:
            raise ValueError("Affine transform becomes singular after float32 conversion")
        translation = (_translation32(matrix[:3, :3], matrix[:3, 3], center)
                       if self._translation_state is None else
                       _array32(self._translation_state, (3,), "Translation"))
        for array in (matrix, center, translation):
            array.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "_translation_state", translation)

    @classmethod
    def centered(cls, linear, translation, center):
        linear = _array32(linear, (3, 3), "Linear matrix")
        translation = _array32(translation, (3,), "Translation")
        center = _array32(center, (3,), "Transform center")
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = linear
        matrix[:3, 3] = _offset32(linear, translation, center)
        return cls(matrix, center, translation)

    @property
    def translation(self):
        return self._translation_state

    def map_points(self, points_lps):
        points = points_array(points_lps)
        values = _array32(points, points.shape, "LPS points")
        return _map32(values.reshape(-1, 3), self.matrix).reshape(points.shape)

    def inverse(self):
        # Small host inverse, rounded at the transform-state boundary. NumPy's
        # inverse is not an assertion of ITK/VNL's exact inverse rounding.
        inverse = np.linalg.inv(self.matrix[:3, :3].astype(np.float64)).astype(np.float32)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = inverse
        matrix[:3, 3] = -_matmul32(inverse, self.matrix[:3, 3:4])[:, 0]
        return AffineTransform32(matrix, self.center)


@dataclass(frozen=True, eq=False)
class SequentialAffineEvaluation32:
    """Explicit per-member computation view; not a serializable single affine.

    ``matrix`` is the collapsed storage matrix for compatibility/inspection,
    while ``map_points`` applies ``components`` in their given order. Consumers
    implementing their own point kernels must use the components, not this
    matrix, to reproduce this view's rounding. ``save_affine`` rejects this
    wrapper; callers must explicitly select ``collapsed`` for single-file IO.
    """

    collapsed: AffineTransform32
    components: tuple

    @property
    def matrix(self):
        return self.collapsed.matrix

    @property
    def center(self):
        return self.collapsed.center

    @property
    def translation(self):
        return self.collapsed.translation

    def map_points(self, points_lps):
        points = points_array(points_lps)
        result = _array32(points, points.shape, "LPS points")
        for component in self.components:
            result = component.map_points(result)
        return result

    def inverse(self):
        return SequentialAffineEvaluation32(
            self.collapsed.inverse(), tuple(component.inverse() for component in reversed(self.components)))


@dataclass(frozen=True, eq=False)
class AffineTransformSequence32(AffineTransform32):
    """Collapsed affine output retaining a separate ordered evaluation sequence.

    Components are in *point application* order, the reverse of ITK's stored
    CompositeTransform queue. Flattening this tuple removes container nesting
    only; it does not combine component matrices or their rounding operations.
    The inherited ``map_points`` and ``inverse`` keep the existing single-affine
    output/QA semantics. Use ``as_evaluation_transform`` for sequential mapping
    and inverse mapping. A saved single affine cannot retain that sequence.
    """

    components: tuple = field(default_factory=tuple, repr=False)

    def __post_init__(self):
        super().__post_init__()
        components = tuple(self.components)
        if not components or any(not isinstance(t, AffineTransform32) for t in components):
            raise ValueError("A float affine sequence requires finite float affine components")
        if any(isinstance(t, AffineTransformSequence32) for t in components):
            raise ValueError("Components must be unrolled without collapsing their matrices")
        object.__setattr__(self, "components", components)

    def as_evaluation_transform(self):
        return SequentialAffineEvaluation32(self, self.components)

    def map_points_sequential(self, points_lps):
        return self.as_evaluation_transform().map_points(points_lps)


@njit(cache=True, fastmath=False)
def _native_trig32(angles):
    # Keep float arguments inside compiled code so NumPy's scalar/vector math
    # approximation does not replace the native float libm calls used by ITK.
    values = np.empty(6, np.float32)
    for axis in range(3):
        values[2*axis] = np.cos(angles[axis])
        values[2*axis+1] = np.sin(angles[axis])
    return values


def _trig32(angles):
    angles = _array32(angles, (3,), "Euler angles")
    return tuple(_native_trig32(angles))


def rotation32(angles):
    """Default ITK Z-X-Y rotation, with float trig and float matrix products."""
    cx, sx, cy, sy, cz, sz = _trig32(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
    return _matmul32(_matmul32(rz, rx), ry)


def _euler_coefficients(angles):
    # std::cos(float) returns float; ITK then stores it in const double.
    cx, sx, cy, sy, cz, sz = (float(v) for v in _trig32(angles))
    # Layout is [parameter, output component, centered point component].
    return np.array([
        [[-sz*cx*sy, sz*sx, sz*cx*cy],
         [cz*cx*sy, -cz*sx, -cz*cx*cy], [sx*sy, cx, -sx*cy]],
        [[-cz*sy-sz*sx*cy, 0, cz*cy-sz*sx*sy],
         [-sz*sy+cz*sx*cy, 0, sz*cy+cz*sx*sy], [-cx*cy, 0, -cx*sy]],
        [[-sz*cy-cz*sx*sy, -cz*cx, -sz*sy+cz*sx*cy],
         [cz*cy-sz*sx*sy, -sz*cx, cz*sy+sz*sx*cy], [0, 0, 0]],
    ], np.float64)


@dataclass
class LinearModel32:
    kind: str
    center: np.ndarray

    def __post_init__(self):
        if self.kind not in ("Rigid", "Affine"):
            raise ValueError("Linear model must be Rigid or Affine")
        self.center = _array32(self.center, (3,), "Model center")
        self.center.setflags(write=False)

    @property
    def parameter_count(self):
        return 6 if self.kind == "Rigid" else 12

    def identity_parameters(self):
        return (np.zeros(6, np.float32) if self.kind == "Rigid" else
                np.concatenate((np.eye(3, dtype=np.float32).ravel(), np.zeros(3, np.float32))))

    def _parameters(self, parameters):
        return _array32(parameters, (self.parameter_count,), "Transform parameters")

    def transform(self, parameters):
        p = self._parameters(parameters)
        matrix = rotation32(p[:3]) if self.kind == "Rigid" else p[:9].reshape(3, 3)
        return AffineTransform32.centered(matrix, p[-3:], self.center)

    def parameters_from_affine(self, transform):
        if self.kind != "Affine":
            raise ValueError("Only an affine model accepts a general existing affine")
        matrix = _array32(transform.matrix, (4, 4), "Affine matrix")
        translation = _translation32(matrix[:3, :3], matrix[:3, 3], self.center)
        return np.concatenate((matrix[:3, :3].ravel(), translation))

    def derivatives(self, parameters):
        p = self._parameters(parameters)
        # Keep only these bounded coefficient arrays in double, as in ITK's
        # Euler Jacobian. Existing metric upload casts them to ordinary FP32.
        linear = np.zeros((self.parameter_count, 3, 3), np.float64)
        translation = np.zeros((self.parameter_count, 3), np.float64)
        if self.kind == "Rigid":
            linear[:3] = _euler_coefficients(p[:3])
        else:
            for i in range(9):
                linear[i, i//3, i%3] = 1
        translation[-3:] = np.eye(3)
        return linear, translation

    def jacobian(self, parameters, points_lps):
        """Return [point, output, parameter], rounding point-center before double work."""
        p = self._parameters(parameters)
        points = points_array(points_lps)
        local = np.subtract(_array32(points, points.shape, "LPS points"), self.center, dtype=np.float32)
        coefficients, translation = LinearModel32.derivatives(self, p)
        jacobian = np.empty((*points.shape[:-1], 3, self.parameter_count), np.float32)
        for parameter in range(self.parameter_count):
            for c in range(3):
                value = np.zeros(points.shape[:-1], np.float64)
                for axis in range(3):
                    value += coefficients[parameter, c, axis]*local[..., axis].astype(np.float64)
                jacobian[..., c, parameter] = value+translation[parameter, c]
        return jacobian

    def intensity_derivatives(self, parameters, points, moving_gradients):
        jacobian = self.jacobian(parameters, points)
        gradients = _array32(moving_gradients, jacobian.shape[:-1], "Moving gradients")
        output = np.zeros((*jacobian.shape[:-2], self.parameter_count), np.float32)
        for c in range(3):
            output += np.multiply(jacobian[..., c, :], gradients[..., c, None], dtype=np.float32)
        return output


@dataclass
class ComposedLinearModel32(LinearModel32):
    """Incremental transform followed by a frozen float affine base."""

    base: AffineTransform | None = None

    def __post_init__(self):
        super().__post_init__()
        # Keep an existing float object's translation and component state. In
        # particular, creating a new stage must not collapse previous stages.
        if self.base is not None and not isinstance(self.base, AffineTransform32):
            self.base = AffineTransform32(self.base.matrix, self.base.center)

    def transform(self, parameters):
        local = super().transform(parameters)
        if self.base is None:
            return local
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = _matmul32(self.base.matrix[:3, :3], local.matrix[:3, :3])
        matrix[:3, 3] = np.add(_matmul32(self.base.matrix[:3, :3], local.matrix[:3, 3:4])[:, 0],
                               self.base.matrix[:3, 3], dtype=np.float32)
        components = ((local,)+self.base.components if isinstance(self.base, AffineTransformSequence32)
                      else (local, self.base))
        return AffineTransformSequence32(matrix, self.center, components=components)

    def evaluation_transform(self, parameters):
        """Explicit float composite view for metric/physical-scale evaluation.

        With initialize-transforms-per-stage=0, ANTs appends each completed
        stage. Its FlattenTransformQueue unrolls containers but preserves
        members; collapse-output-transforms is applied when writing transforms.
        Within a level, each call replaces the active local transform and
        retains only the fixed previous-stage base, so members do not accumulate
        with optimizer iterations.
        """
        transform = self.transform(parameters)
        return (transform.as_evaluation_transform() if isinstance(transform, AffineTransformSequence32)
                else transform)

    def map_points_sequential(self, parameters, points_lps):
        return self.evaluation_transform(parameters).map_points(points_lps)

    def derivatives(self, parameters):
        linear, translation = super().derivatives(parameters)
        if self.base is not None:
            base = self.base.matrix[:3, :3]
            linear = np.stack([_matmul32(base, d.astype(np.float32)) for d in linear])
            translation = _matmul32(base, translation.astype(np.float32).T).T
        return linear, translation

    def jacobian(self, parameters, points_lps):
        local = super().jacobian(parameters, points_lps)
        if self.base is None:
            return local
        output = np.zeros_like(local)
        for c in range(3):
            for axis in range(3):
                output[..., c, :] += np.multiply(self.base.matrix[c, axis], local[..., axis, :], dtype=np.float32)
        return output
