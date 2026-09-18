"""Deterministic geometry and fusion tests for surface_mapping.py."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from surface_mapping import (  # noqa: E402
    EllipseFit,
    PreparedView,
    ShapeFit,
    SurfaceMappingConfig,
    _project_maps,
    accumulate_rotations,
    defect_area_statistics,
    ellipse_moments,
    estimate_pair_rotation,
    fit_spheroid,
    fruit_to_uv,
    fused_grade,
    mapping_status,
    oblate_depth,
    rodrigues_degrees,
    sphere_depth,
    uv_to_fruit,
)


def circle_mask(size: int = 128, radius: int = 48) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    return np.where((x - size / 2) ** 2 + (y - size / 2) ** 2 <= radius**2, 255, 0).astype(np.uint8)


def prepared_view(rgb: np.ndarray, mask: np.ndarray, frame: int, defects: dict[str, np.ndarray] | None = None) -> PreparedView:
    return PreparedView(
        track_id=1,
        frame_id=frame,
        timestamp="",
        bbox={"x": 0, "y": 0, "width": rgb.shape[1], "height": rgb.shape[0]},
        transform={"offsetX": 0, "offsetY": 0, "scaleX": 1, "scaleY": 1},
        rgb=rgb,
        mask=mask,
        safe_mask=cv2.erode(mask, np.ones((3, 3), np.uint8)),
        detection_confidence=1.0,
        focus_score=100.0,
        mask_quality=1.0,
        defect_maps=defects or {},
        defect_class_ids={name: 1 for name in (defects or {})},
        crop_path="",
        quality=1.0,
    )


def render_textured_sphere(rotation: np.ndarray, size: int = 128, radius: float = 48.0) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    x = xx - size / 2
    y = yy - size / 2
    z, valid = sphere_depth(x, y, radius)
    camera = np.stack((x, y, z), axis=-1)
    fruit = np.einsum("ij,hwj->hwi", rotation.T, camera) / radius
    texture = (
        126.0
        + 34.0 * np.sin(12.0 * fruit[..., 0] + 5.0 * fruit[..., 2])
        + 27.0 * np.cos(15.0 * fruit[..., 1] - 3.0 * fruit[..., 0])
        + 18.0 * np.sin(21.0 * fruit[..., 2] * fruit[..., 1])
    )
    texture = np.clip(texture, 0, 255).astype(np.uint8)
    rgb = np.stack((texture, texture, texture), axis=-1)
    rgb[~valid] = 18
    return rgb, valid.astype(np.uint8) * 255


def world_defect_map(mask: np.ndarray, basis: np.ndarray, radius: float, longitude: float = math.pi / 2) -> np.ndarray:
    height, width = mask.shape
    yy, xx = np.mgrid[:height, :width]
    x = xx - width / 2
    y = yy - height / 2
    z, valid = sphere_depth(x, y, radius)
    camera = np.stack((x, y, z), axis=-1)
    fruit = np.einsum("ij,hwj->hwi", basis.T, camera)
    latitude, mapped_longitude = fruit_to_uv(fruit, (radius, radius, radius))
    angular = np.angle(np.exp(1j * (mapped_longitude - longitude)))
    return (valid & (np.abs(latitude) < math.radians(9)) & (np.abs(angular) < math.radians(9))).astype(np.float32)


class SurfaceMappingTests(unittest.TestCase):
    def test_ellipse_moments_recover_center_and_axes(self) -> None:
        mask = np.zeros((180, 220), np.uint8)
        cv2.ellipse(mask, (113, 79), (58, 31), 24, 0, 360, 255, -1)
        fit = ellipse_moments(mask, stride=1)
        self.assertTrue(np.allclose(fit.center, [113, 79], atol=0.6))
        self.assertAlmostEqual(fit.semi_major, 58, delta=1.0)
        self.assertAlmostEqual(fit.semi_minor, 31, delta=1.0)

    def test_sphere_and_oblate_depth_lie_on_surface(self) -> None:
        x = np.array([0.0, 12.0, -18.0])
        y = np.array([0.0, -9.0, 14.0])
        z, valid = sphere_depth(x, y, 50.0)
        self.assertTrue(np.all(valid))
        self.assertTrue(np.allclose(x * x + y * y + z * z, 2500.0, atol=1e-6))

        radii = (55.0, 38.0, 55.0)
        pose = rodrigues_degrees(17.0, -4.0, "y")
        z, valid = oblate_depth(x, y, pose, radii)
        points = np.column_stack((x, y, z))
        fruit = (pose.T @ points.T).T
        equation = np.sum(np.square(fruit / np.asarray(radii)), axis=1)
        self.assertTrue(np.all(valid))
        self.assertTrue(np.allclose(equation, 1.0, atol=1e-6))

    def test_rodrigues_and_accumulation_order(self) -> None:
        first = rodrigues_degrees(11.0, 0.0, "y")
        second = rodrigues_degrees(7.0, 0.0, "y")
        accumulated = accumulate_rotations([first, second])
        point = np.array([0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(accumulated[2] @ point, second @ first @ point, atol=1e-8))
        expected = rodrigues_degrees(18.0, 0.0, "y")
        self.assertTrue(np.allclose(accumulated[2], expected, atol=1e-7))

    def test_synthetic_texture_recovers_known_rotation(self) -> None:
        known = 16.0
        source_rgb, source_mask = render_textured_sphere(np.eye(3))
        target_rotation = rodrigues_degrees(known, 0.0, "y")
        target_rgb, target_mask = render_textured_sphere(target_rotation)
        source = prepared_view(source_rgb, source_mask, 1)
        target = prepared_view(target_rgb, target_mask, 2)
        config = SurfaceMappingConfig(
            shape_model="sphere",
            downsample_factor=2,
            texture_percentile=90.0,
            minimum_texture_points=18,
            maximum_texture_points=100,
            rotation_search_step_degrees=2.0,
            maximum_step_degrees=24.0,
            secondary_axis_limit_degrees=4.0,
            alignment_confidence_threshold=0.02,
        )
        shape = fit_spheroid([source_mask, target_mask], config)
        estimate = estimate_pair_rotation(source, target, shape, 0, config)
        self.assertTrue(estimate.reliable, estimate.failure_reason)
        self.assertAlmostEqual(estimate.primary_degrees, known, delta=2.5)

    def test_uv_view_round_trip(self) -> None:
        latitude = np.deg2rad(np.array([-50.0, -12.0, 0.0, 31.0, 58.0]))
        longitude = np.deg2rad(np.array([-170.0, -40.0, 15.0, 88.0, 179.0]))
        radii = (61.0, 43.0, 61.0)
        fruit = uv_to_fruit(latitude, longitude, radii)
        basis = rodrigues_degrees(23.0, -7.0, "y")
        view = np.einsum("ij,nj->ni", basis, fruit)
        recovered_fruit = np.einsum("ji,nj->ni", basis, view)
        recovered_latitude, recovered_longitude = fruit_to_uv(recovered_fruit, radii)
        difference = np.angle(np.exp(1j * (recovered_longitude - longitude)))
        self.assertTrue(np.allclose(recovered_latitude, latitude, atol=1e-10))
        self.assertTrue(np.allclose(difference, 0.0, atol=1e-10))

    def _fused_fixture(self, duplicate: bool = False) -> dict[str, object]:
        size, radius = 112, 40.0
        mask = circle_mask(size, int(radius))
        rgb = np.full((size, size, 3), 130, np.uint8)
        basis = np.eye(3)
        defect = world_defect_map(mask, basis, radius)
        views = [prepared_view(rgb, mask, 1, {"scar": defect})]
        bases = [basis]
        centers = np.array([math.pi / 2])
        if duplicate:
            views.append(prepared_view(rgb.copy(), mask.copy(), 2, {"scar": defect.copy()}))
            bases.append(basis.copy())
            centers = np.array([math.pi / 2, math.pi / 2])
        fit = EllipseFit(np.array([size / 2, size / 2]), radius, radius, np.eye(2))
        shape = ShapeFit("sphere", radius, radius, radius, (radius, radius, radius), [fit] * len(views), [0.0] * len(views), 0)
        config = SurfaceMappingConfig(
            minimum_mapping_views=2,
            maximum_mapping_views=4,
            maximum_map_width=300,
            pre_roll_degrees=45,
            post_roll_degrees=45,
        )
        return _project_maps(views, bases, centers, [1.0] * max(0, len(views) - 1), shape, config, False)

    def test_overlapping_defect_is_fused_once(self) -> None:
        projected = self._fused_fixture(duplicate=True)
        stats, total, _ = defect_area_statistics(projected, 0.5)
        self.assertGreater(stats["scar"]["observed_surface_ratio"], 0.0)
        self.assertAlmostEqual(total, stats["scar"]["observed_surface_ratio"], places=8)
        grade, class_id, confidence = fused_grade(projected, 0.5, "Class 1", 0.4)
        self.assertEqual((grade, class_id), ("Class 2", 1))
        self.assertGreater(confidence, 0.5)

    def test_more_overlapping_views_do_not_double_defect_area(self) -> None:
        single = self._fused_fixture(duplicate=False)
        duplicate = self._fused_fixture(duplicate=True)
        single_stats, _, _ = defect_area_statistics(single, 0.5)
        duplicate_stats, _, _ = defect_area_statistics(duplicate, 0.5)
        self.assertAlmostEqual(
            single_stats["scar"]["target_surface_ratio"],
            duplicate_stats["scar"]["target_surface_ratio"],
            delta=0.002,
        )

    def test_unobserved_surface_remains_false(self) -> None:
        projected = self._fused_fixture(duplicate=False)
        observed = projected["observed"]
        self.assertTrue(np.any(observed))
        self.assertTrue(np.any(~observed))
        self.assertTrue(np.all(projected["rgb"][~observed] == 0))

    def test_large_angular_gap_is_partial(self) -> None:
        config = SurfaceMappingConfig()
        status, reasons = mapping_status(352.0, 73.0, 0.95, 0.80, 0.90, config)
        self.assertEqual(status, "partial")
        self.assertTrue(any("angular gap" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
