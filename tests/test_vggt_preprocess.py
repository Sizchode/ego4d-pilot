import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from scripts.vggt.preprocess import (
    derive_evaluation_homographies,
    derive_preprocess_metadata,
    load_and_preprocess_with_metadata,
    normalize_to_evaluation_canvas,
)

try:
    import torch  # noqa: F401
    from vggt.utils.load_fn import load_and_preprocess_images  # noqa: F401

    _VGGT_AVAILABLE = True
except Exception:
    _VGGT_AVAILABLE = False

try:
    import cv2  # noqa: F401
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False


class TestVGGTPreprocessingMetadata(unittest.TestCase):
    def test_crop_landscape_and_tall_images_matches_upstream_shape_rules(self):
        shape, transforms, records = derive_preprocess_metadata([(1000, 500), (500, 1000)], "crop")
        self.assertEqual(shape, (518, 518))
        self.assertEqual(transforms.shape, (2, 3, 3))
        self.assertEqual(records[0]["resized_wh"], [518, 252])
        self.assertEqual(records[0]["batch_pad_top"], 133)
        self.assertEqual(records[1]["resized_wh"], [518, 1036])
        self.assertEqual(records[1]["crop_top"], 259)
        np.testing.assert_allclose(transforms[0][0, 0], 0.518)
        np.testing.assert_allclose(transforms[0][1, 1], 252 / 500)
        np.testing.assert_allclose(transforms[0][1, 2], 133)
        np.testing.assert_allclose(transforms[1][1, 2], -259)

    def test_pad_preserves_full_image_with_centered_padding(self):
        shape, transforms, records = derive_preprocess_metadata([(1000, 500), (500, 1000)], "pad")
        self.assertEqual(shape, (518, 518))
        self.assertEqual(records[0]["resized_wh"], [518, 252])
        self.assertEqual(records[0]["prepad_top"], 133)
        self.assertEqual(records[1]["resized_wh"], [252, 518])
        self.assertEqual(records[1]["prepad_left"], 133)
        np.testing.assert_allclose(transforms[0][1, 2], 133)
        np.testing.assert_allclose(transforms[1][0, 2], 133)

    def test_rejects_empty_or_invalid_sizes(self):
        with self.assertRaises(ValueError):
            derive_preprocess_metadata([], "crop")
        with self.assertRaises(ValueError):
            derive_preprocess_metadata([(0, 10)], "crop")

    @unittest.skipUnless(_VGGT_AVAILABLE, "VGGT runtime is unavailable")
    def test_metadata_shape_matches_pinned_official_loader(self):
        with TemporaryDirectory() as directory:
            paths = []
            for index, size in enumerate([(1000, 500), (500, 1000), (641, 479)]):
                path = Path(directory) / f"{index}.png"
                Image.new("RGB", size, (index * 50, 10, 20)).save(path)
                paths.append(path)
            for mode in ("crop", "pad"):
                images, transforms, records = load_and_preprocess_with_metadata(paths, mode)
                self.assertEqual(tuple(images.shape[-2:]), tuple(records[0]["final_hw"]))
                self.assertEqual(transforms.shape, (3, 3, 3))

    def test_canvas_normalization_preserves_camera_rays(self):
        intrinsics = np.array([[[4.0, 0.0, 4.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]])
        original_to_processed = np.array([[[0.8, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 1.0]]])
        processed_to_evaluation, _ = derive_evaluation_homographies(
            [(10, 10)], original_to_processed, (10, 10)
        )
        normalized_intrinsics = processed_to_evaluation[0] @ intrinsics[0]
        expected_intrinsics = np.array([[5.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(normalized_intrinsics, expected_intrinsics)
        processed_pixel = np.array([6.0, 3.0, 1.0])
        evaluation_pixel = processed_to_evaluation[0] @ processed_pixel
        ray_before = np.linalg.inv(intrinsics[0]) @ processed_pixel
        ray_after = np.linalg.inv(normalized_intrinsics) @ evaluation_pixel
        np.testing.assert_allclose(ray_before, ray_after, atol=1e-6)

    @unittest.skipUnless(_CV2_AVAILABLE, "local OpenCV runtime is unavailable")
    def test_canvas_warp_marks_unobserved_crop_as_invalid(self):
        depth = np.ones((1, 4, 8), dtype=np.float32)
        confidence = np.full_like(depth, 2.0)
        images = np.zeros((1, 4, 8, 3), dtype=np.float32)
        intrinsics = np.array([[[4.0, 0.0, 4.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]])
        # The processed tensor is the center half of the original image height.
        original_to_processed = np.array([[[0.8, 0.0, 0.0], [0.0, 0.8, -2.0], [0.0, 0.0, 1.0]]])
        result = normalize_to_evaluation_canvas(
            depth, confidence, images, intrinsics,
            [(10, 10)], original_to_processed, (10, 10),
        )
        self.assertTrue(np.all(result["depth"][0, :2] == 0))
        self.assertTrue(np.all(result["depth"][0, 7:] == 0))
        self.assertGreater(np.count_nonzero(result["depth"]), 0)


if __name__ == "__main__":
    unittest.main()
