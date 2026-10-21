import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from scripts.common.evaluation_canvas import derive_evaluation_homographies
from scripts.pi3x.preprocess import (
    derive_preprocess_metadata,
    stage_and_load_images,
    target_size_from_first_image,
)

try:
    import pi3  # noqa: F401

    _PI3_AVAILABLE = True
except Exception:
    _PI3_AVAILABLE = False


class TestPi3XPreprocessingMetadata(unittest.TestCase):
    def test_target_size_obeys_patch_and_pixel_limits(self):
        for size in [(1920, 1080), (1080, 1920), (640, 480), (512, 512)]:
            width, height = target_size_from_first_image(size, 255000)
            self.assertEqual(width % 14, 0)
            self.assertEqual(height % 14, 0)
            self.assertLessEqual(width * height, 255000)

    def test_direct_resize_transform(self):
        transforms, records = derive_preprocess_metadata([(1000, 500), (500, 1000)], (700, 350))
        np.testing.assert_allclose(transforms[0], np.diag([0.7, 0.7, 1.0]))
        np.testing.assert_allclose(transforms[1], np.diag([1.4, 0.35, 1.0]))
        self.assertEqual(records[1]["scale_xy"], [1.4, 0.35])

    def test_evaluation_canvas_preserves_camera_rays(self):
        transforms, _ = derive_preprocess_metadata([(1000, 500)], (700, 350))
        processed_to_evaluation, _ = derive_evaluation_homographies(
            [(1000, 500)], transforms, (518, 518)
        )
        k = np.array([[400.0, 0.0, 350.0], [0.0, 400.0, 175.0], [0.0, 0.0, 1.0]])
        pixel = np.array([360.0, 170.0, 1.0])
        k_out = processed_to_evaluation[0] @ k
        pixel_out = processed_to_evaluation[0] @ pixel
        np.testing.assert_allclose(np.linalg.inv(k) @ pixel, np.linalg.inv(k_out) @ pixel_out)

    @unittest.skipUnless(_PI3_AVAILABLE, "Pi3X runtime is unavailable")
    def test_metadata_shape_matches_pinned_official_loader(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            paths = []
            for index, size in enumerate([(1000, 500), (800, 600), (500, 1000)]):
                path = inputs / f"source_{index}.png"
                Image.new("RGB", size, (index * 40, 10, 20)).save(path)
                paths.append(path)
            images, transforms, records, original_wh, target_wh = stage_and_load_images(
                paths, root / "stage", pixel_limit=255000
            )
            self.assertEqual(images.shape[0], 3)
            self.assertEqual(tuple(images.shape[-2:]), (target_wh[1], target_wh[0]))
            self.assertEqual(transforms.shape, (3, 3, 3))
            self.assertEqual(records[0]["original_wh"], list(original_wh[0]))


if __name__ == "__main__":
    unittest.main()
