"""Independent two-pass oracle for the optional FP32 paired-CC plane queue.

The oracle starts from already quantized FP32 inputs, then forms each clipped
window directly in double precision. It does not reuse the queue, local
anchors, plane grouping, Chan merge, or the implementation's score reduction.
"""

import unittest

import numpy as np
from numba import get_num_threads, set_num_threads

from fast_ants.backends.cpu.fp32.metrics.cc_queue import _work32, cc_pair32, cc_queue_pair32


def oracle(fixed, moving, fixed_gradient, moving_gradient, valid, radius):
    nz, ny, nx = fixed.shape
    forward = np.zeros((3, nz, ny, nx), np.float64)
    reverse = np.zeros((3, nz, ny, nx), np.float64)
    scores = np.zeros(fixed.shape, np.float64)
    # Convert only after input quantization. No FP64 image path is the target.
    f, m = fixed.astype(np.float64), moving.astype(np.float64)
    epsilon = float(np.finfo(np.float32).eps)
    centres = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not valid[z, y, x]:
                    continue
                window = (slice(max(0, z - radius), min(nz, z + radius + 1)),
                          slice(max(0, y - radius), min(ny, y + radius + 1)),
                          slice(max(0, x - radius), min(nx, x + radius + 1)))
                mask = valid[window]
                fs, ms = f[window][mask], m[window][mask]
                if not fs.size:
                    continue
                mean_f, mean_m = fs.mean(), ms.mean()
                df, dm = fs - mean_f, ms - mean_m
                ff, mm, fm = np.dot(df, df), np.dot(dm, dm), np.dot(df, dm)
                product = ff * mm
                score = 1.
                if abs(product) > epsilon:
                    score = fm * fm / product
                if ff > epsilon and mm > epsilon:
                    centre_f, centre_m = f[z, y, x] - mean_f, m[z, y, x] - mean_m
                    fw = 2. * fm / product * (centre_f - fm / mm * centre_m)
                    rw = 2. * fm / product * (centre_m - fm / ff * centre_f)
                    forward[:, z, y, x] = fw * moving_gradient[:, z, y, x].astype(np.float64)
                    reverse[:, z, y, x] = rw * fixed_gradient[:, z, y, x].astype(np.float64)
                scores[z, y, x] = -score
                centres += 1
    value = scores.sum(dtype=np.float64) / centres if centres else 0.
    return value, centres, forward, reverse, scores


class CCQueuePair32Tests(unittest.TestCase):
    def setUp(self):
        self.threads = get_num_threads()
        set_num_threads(1)

    def tearDown(self):
        set_num_threads(self.threads)

    def fixtures(self, shape=(3, 4, 9)):
        z, y, x = np.indices(shape, dtype=np.float64)
        fixed = .45 + .17 * np.sin(.7 * x) + .13 * np.cos(.6 * y) + .03 * z
        moving = .37 + .19 * np.sin(.7 * x + .18) + .11 * np.cos(.6 * y - .13) + .025 * z + .003 * x * y
        return ((fixed.astype(np.float32), moving.astype(np.float32)),
                ((np.float64(1e6) + fixed * 4).astype(np.float32),
                 (np.float64(-2e6) + moving * 8).astype(np.float32)),
                ((np.float64(4096) + fixed * .02).astype(np.float32),
                 (np.float64(-8192) + moving * .04).astype(np.float32)),
                (fixed.astype(np.float32), (-moving).astype(np.float32)))

    def masks(self, shape):
        full = np.ones(shape, np.bool_)
        holes = full.copy()
        holes[1:, 1:3, 2:7] = False
        # Empty x planes and invalid anchors exercise the queue's zero-count
        # merges without changing which clipped windows contain valid centres.
        holes[:, :, 0] = False
        holes[:, :, 4] = False
        sparse = np.zeros(shape, np.bool_)
        sparse[0, 0, 0] = sparse[-1, -1, -1] = sparse[1, 2, 3] = True
        return full, holes, sparse, np.zeros(shape, np.bool_)

    def check_oracle(self, inputs, radius, update_atol=3e-6, update_rtol=1e-4):
        expected = oracle(*inputs, radius)
        actual = _work32(*inputs, radius)
        self.assertEqual(actual[1], expected[1])
        self.assertAlmostEqual(float(actual[0]), expected[0], delta=3e-6)
        for index in (2, 3, 4):
            self.assertEqual(actual[index].dtype, np.float32)
            self.assertTrue(np.isfinite(actual[index]).all())
        np.testing.assert_allclose(actual[4], expected[4], atol=3e-6, rtol=3e-6)
        np.testing.assert_allclose(actual[2], expected[2], atol=update_atol, rtol=update_rtol)
        np.testing.assert_allclose(actual[3], expected[3], atol=update_atol, rtol=update_rtol)
        return actual

    def test_normalized_large_dc_masks_clipped_boundaries_and_radius(self):
        rng = np.random.default_rng(20261025)
        shape = (3, 4, 9)
        fixed_gradient = rng.uniform(-.2, .2, size=(3, *shape)).astype(np.float32)
        moving_gradient = rng.uniform(-.2, .2, size=(3, *shape)).astype(np.float32)
        for case, (fixed, moving) in enumerate(self.fixtures(shape)):
            for radius in (0, 1, 2, 12):
                for mask_id, valid in enumerate(self.masks(shape)):
                    with self.subTest(case=case, radius=radius, mask=mask_id):
                        # Low-contrast data can have large update magnitudes;
                        # relative error remains bounded independently of DC.
                        self.check_oracle((fixed, moving, fixed_gradient, moving_gradient, valid), radius,
                                          update_atol=2e-4 if case == 2 else 3e-6)

    def test_degenerate_variances_and_score_update_epsilon_branches(self):
        shape = (2, 3, 7)
        z, y, x = np.indices(shape)
        ramp = (x + y * .3 + z * .2).astype(np.float32)
        constant = np.full(shape, np.float32(1e6), np.float32)
        gradients = np.ones((3, *shape), np.float32)
        valid = np.ones(shape, np.bool_)
        for fixed, moving in ((constant, constant.copy()), (constant, ramp), (ramp, constant),
                              (ramp * np.float32(2e-4), (ramp + np.sin(x).astype(np.float32)) * np.float32(2e-4))):
            for radius in (0, 1, 8):
                with self.subTest(radius=radius, fixed_constant=bool(np.ptp(fixed) == 0),
                                  moving_constant=bool(np.ptp(moving) == 0)):
                    self.check_oracle((fixed, moving, gradients, gradients, valid), radius,
                                      update_atol=3e-3, update_rtol=2e-4)
        value, count, forward, reverse = cc_queue_pair32(constant, constant, gradients, gradients, valid, 2)
        self.assertEqual(value, -1.)
        self.assertEqual(count, constant.size)
        np.testing.assert_array_equal(forward, np.zeros_like(forward))
        np.testing.assert_array_equal(reverse, np.zeros_like(reverse))

    def test_large_dc_valid_region_ignores_invalid_anchor_outliers(self):
        shape = (3, 4, 13)
        z, y, x = np.indices(shape, dtype=np.float64)
        fixed_valid = (1e6 + .25 * np.sin(.7 * x) + .0625 * y + .125 * z).astype(np.float32)
        moving_valid = (-2e6 + .5 * np.sin(.7 * x + .23) + .125 * y + .25 * z).astype(np.float32)
        valid = np.ones(shape, np.bool_)
        valid[:, :, 0] = False
        valid[0, :, 1:4] = False  # First valid yz sample varies between planes.
        valid[:, 1, 6] = False
        gradients = np.full((3, *shape), np.float32(.1), np.float32)
        fixed_gradient = gradients.copy()
        fixed_gradient[1] = np.float32(-.15)
        for fixed_outlier, moving_outlier in ((0., 0.), (1e20, -1e20), (-1e20, 1e20)):
            fixed, moving = fixed_valid.copy(), moving_valid.copy()
            fixed[~valid] = np.float32(fixed_outlier)
            moving[~valid] = np.float32(moving_outlier)
            for radius in (1, 2, 5):
                with self.subTest(fixed_outlier=fixed_outlier, moving_outlier=moving_outlier, radius=radius):
                    actual = self.check_oracle((fixed, moving, fixed_gradient, gradients, valid), radius)
                    np.testing.assert_array_equal(actual[2][:, ~valid], np.zeros((3, int((~valid).sum())), np.float32))
                    np.testing.assert_array_equal(actual[3][:, ~valid], np.zeros((3, int((~valid).sum())), np.float32))

    def test_single_voxel_axes_strides_readonly_and_thread_consistency(self):
        rng = np.random.default_rng(71)
        for shape in ((1, 1, 17), (1, 7, 11), (3, 4, 13)):
            source_f = rng.uniform(.1, .9, size=(shape[0], shape[1], shape[2] * 2)).astype(np.float32)
            source_m = rng.uniform(.1, .9, size=source_f.shape).astype(np.float32)
            fixed, moving = source_f[:, :, ::2], source_m[:, :, ::2]
            fixed_gradient = rng.uniform(-.3, .3, size=(3, *shape)).astype(np.float32)
            moving_gradient = rng.uniform(-.3, .3, size=(3, *shape)).astype(np.float32)
            valid = rng.random(shape) > .2
            inputs = (fixed, moving, fixed_gradient, moving_gradient, valid)
            saved = tuple(value.copy() for value in inputs)
            for value in inputs:
                value.setflags(write=False)
            for radius in (0, 1, 3):
                set_num_threads(1)
                first = self.check_oracle(inputs, radius)
                if self.threads > 1:
                    set_num_threads(2)
                    second = _work32(*inputs, radius)
                    self.assertEqual(first[0], second[0])
                    self.assertEqual(first[1], second[1])
                    for index in (2, 3, 4):
                        np.testing.assert_array_equal(first[index].view(np.uint32), second[index].view(np.uint32))
                for value, original in zip(inputs, saved):
                    np.testing.assert_array_equal(value, original)
                    self.assertFalse(np.shares_memory(first[2], value))
                    self.assertFalse(np.shares_memory(first[3], value))
                self.assertFalse(np.shares_memory(first[2], first[3]))

    def test_public_alias_and_empty_inputs(self):
        self.assertIs(cc_pair32, cc_queue_pair32)
        for shape in ((0, 3, 4), (2, 0, 4), (2, 3, 0)):
            data = np.empty(shape, np.float32)
            gradient = np.empty((3, *shape), np.float32)
            valid = np.empty(shape, np.bool_)
            value, count, forward, reverse = cc_pair32(data, data, gradient, gradient, valid, 2)
            self.assertEqual((value, count), (0., 0))
            self.assertEqual(forward.shape, gradient.shape)
            self.assertEqual(reverse.shape, gradient.shape)
            self.assertEqual(forward.dtype, np.float32)
            self.assertEqual(reverse.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
