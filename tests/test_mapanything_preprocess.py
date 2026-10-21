import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from scripts.common.evaluation_canvas import derive_evaluation_homographies
from scripts.mapanything.preprocess import derive_preprocess_metadata

try:
    from mapanything.utils.cropping import (
        crop_resize_if_necessary,
        rescale_image_and_other_optional_info,
    )
    from scripts.mapanything.preprocess import load_images_with_metadata

    _MAPANYTHING_AVAILABLE = True
except Exception:
    _MAPANYTHING_AVAILABLE = False


class TestMapAnythingPreprocessingMetadata(unittest.TestCase):
    def test_rescale_then_center_crop_matches_upstream_formula(self):
        transforms, records = derive_preprocess_metadata([(1000, 500), (500, 1000)], (518, 392))
        self.assertEqual(records[0]["resized_wh"], [784, 392])
        self.assertEqual(records[0]["crop_left"], 133)
        self.assertEqual(records[0]["crop_top"], 0)
        self.assertEqual(records[1]["resized_wh"], [518, 1036])
        self.assertEqual(records[1]["crop_top"], 322)
        np.testing.assert_allclose(transforms[0][0, 0], 0.78400001)
        for transform, record, (width, height) in zip(
            transforms, records, [(1000, 500), (500, 1000)], strict=True
        ):
            scale = record["continuous_scale"]
            resized_width, resized_height = record["resized_wh"]
            expected_x = (
                0.5 * scale - 0.5 * (width * scale - resized_width)
                - 0.5 - record["crop_left"]
            )
            expected_y = (
                0.5 * scale - 0.5 * (height * scale - resized_height)
                - 0.5 - record["crop_top"]
            )
            np.testing.assert_allclose(transform[:2, 2], [expected_x, expected_y])

    def test_rejects_invalid_sizes(self):
        with self.assertRaises(ValueError):
            derive_preprocess_metadata([], (518, 392))
        with self.assertRaises(ValueError):
            derive_preprocess_metadata([(0, 12)], (518, 392))
        with self.assertRaises(ValueError):
            derive_preprocess_metadata([(12, 12)], (0, 392))

    def test_evaluation_canvas_preserves_camera_rays(self):
        transforms, _ = derive_preprocess_metadata([(1000, 500)], (518, 392))
        processed_to_evaluation, _ = derive_evaluation_homographies(
            [(1000, 500)], transforms, (518, 518)
        )
        processed_intrinsics = np.array(
            [[300.0, 0.0, 259.0], [0.0, 300.0, 196.0], [0.0, 0.0, 1.0]]
        )
        output_intrinsics = processed_to_evaluation[0] @ processed_intrinsics
        processed_pixel = np.array([300.0, 180.0, 1.0])
        output_pixel = processed_to_evaluation[0] @ processed_pixel
        before = np.linalg.inv(processed_intrinsics) @ processed_pixel
        after = np.linalg.inv(output_intrinsics) @ output_pixel
        np.testing.assert_allclose(before, after, atol=1e-7)

    @unittest.skipUnless(_MAPANYTHING_AVAILABLE, "MapAnything runtime is unavailable")
    def test_metadata_matches_pinned_official_preprocessing(self):
        with TemporaryDirectory() as directory:
            paths = []
            for index, size in enumerate([(1000, 500), (1000, 500), (1000, 500)]):
                path = Path(directory) / f"{index}.png"
                Image.new("RGB", size, (index * 40, 10, 20)).save(path)
                paths.append(path)
            views, transforms, records, original_wh, target_wh = load_images_with_metadata(
                paths, resolution_set=518
            )
            self.assertEqual(len(views), len(paths))
            self.assertEqual(original_wh, [(1000, 500)] * 3)
            self.assertEqual(tuple(views[0]["img"].shape[-2:]), (target_wh[1], target_wh[0]))

            center_intrinsics = np.array(
                [[400.0, 0.0, 500.0], [0.0, 400.0, 250.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            source_image = Image.new("RGB", (1000, 500))
            resized_image, _, resized_intrinsics, _ = rescale_image_and_other_optional_info(
                image=source_image,
                output_resolution=target_wh,
                camera_intrinsics=center_intrinsics,
            )
            expected_intrinsics = resized_intrinsics.copy()
            expected_intrinsics[0, 2] -= records[0]["crop_left"]
            expected_intrinsics[1, 2] -= records[0]["crop_top"]
            processed_image = crop_resize_if_necessary(source_image, resolution=target_wh)[0]
            self.assertEqual(processed_image.size, target_wh)
            self.assertEqual(resized_image.size, tuple(records[0]["resized_wh"]))
            np.testing.assert_allclose(
                expected_intrinsics, transforms[0] @ center_intrinsics, atol=1e-5
            )
            self.assertEqual(records[0]["target_wh"], list(target_wh))


if __name__ == "__main__":
    unittest.main()
