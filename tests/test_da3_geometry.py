import unittest
import numpy as np
from scripts.da3.geometry import pair_diagnostics, project, unproject
from scripts.failure_mining.visibility import pair_visibility_layers


class GeometryDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[4., 0., 3.], [0., 4., 2.], [0., 0., 1.]])
        self.e = np.eye(4)
        self.depth = np.full((5, 7), 2.)

    def test_pose_and_projection_conventions(self):
        e = np.array([[0., -1., 0., 1.], [1., 0., 0., 2.], [0., 0., 1., 3.], [0., 0., 0., 1.]])
        world = unproject(self.depth, self.k, e)
        np.testing.assert_allclose(world[2, 3], [-2., 1., -1.])
        uv, z = project(world, self.k, e)
        np.testing.assert_allclose(uv[2, 3], [3., 2.])
        np.testing.assert_allclose(z, self.depth)

    def test_translated_cameras_observing_same_plane(self):
        target = np.eye(4)
        target[0, 3] = -0.5
        diag = pair_diagnostics(self.depth, self.depth, self.k, self.k, self.e, target)
        self.assertFalse(diag['projectable'][:, 0].any())
        self.assertTrue(diag['projectable'][:, 1:].all())
        np.testing.assert_allclose(diag['relative_z_residual'][:, 1:], 0, atol=1e-7)
        np.testing.assert_allclose(diag['roundtrip_pixels'][:, 1:], 0, atol=1e-7)

    def test_signed_depth_disagreement_is_not_hidden(self):
        diag = pair_diagnostics(self.depth, self.depth / 2, self.k, self.k, self.e, self.e)
        np.testing.assert_allclose(diag['relative_z_residual'], 1.)
        np.testing.assert_allclose(diag['roundtrip_pixels'], 0.)

    def test_invalid_depth_is_not_a_zero_error(self):
        target = self.depth.copy()
        target[2, 3] = np.nan
        diag = pair_diagnostics(self.depth, target, self.k, self.k, self.e, self.e)
        self.assertFalse(diag['projectable'][2, 3])
        self.assertTrue(np.isnan(diag['relative_z_residual'][2, 3]))

    def test_visibility_layers_separate_depth_order_without_tolerance(self):
        same = pair_visibility_layers(self.depth, self.depth, self.k, self.k, self.e, self.e)
        self.assertTrue(same['source_zbuffer_winner'].all())
        self.assertTrue(same['source_depth_not_behind'].all())
        self.assertFalse(same['target_depth_closer'].any())

        closer_target = pair_visibility_layers(
            self.depth, self.depth / 2, self.k, self.k, self.e, self.e
        )
        self.assertTrue(closer_target['target_depth_closer'].all())
        self.assertFalse(closer_target['source_depth_not_behind'].any())
        np.testing.assert_allclose(closer_target['relative_z_nearest'], 1.0)

    def test_visibility_layers_apply_a_deterministic_source_zbuffer(self):
        source_depth = np.array([[1.0, 2.0]])
        target_depth = np.array([[3.0, 3.0]])
        source_k = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        target_k = np.array([[0.4, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        layers = pair_visibility_layers(
            source_depth, target_depth, source_k, target_k, self.e, self.e
        )
        np.testing.assert_array_equal(layers['rasterizable'], [[True, True]])
        np.testing.assert_array_equal(layers['source_zbuffer_winner'], [[True, False]])
        np.testing.assert_array_equal(layers['source_depth_not_behind'], [[True, False]])


if __name__ == '__main__':
    unittest.main()
