"""Software-only spheroid surface fusion for rotating naartjie tracks.

The implementation follows the geometry and constrained texture registration in
Albiol et al. (Sensors 2021, 2022).  Camera coordinates are x right, y down and
z towards the camera.  ``V`` stores fruit-axis unit vectors as columns in camera
coordinates, so ``p_camera = V @ p_fruit``.  The default fruit coordinates use
radii ``(A, B, A)``: y is the short oblate axis and x/z are equal major axes.

Unknown map pixels are never synthesized.  They remain black/zero and are false
in ``surface_observed_mask.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


EPSILON = 1.0e-8


@dataclass
class SurfaceMappingConfig:
    enabled: bool = True
    shape_model: str = "oblate"
    robust_shape_fit: bool = False
    moment_coordinate_stride: int = 4
    spherical_tolerance: float = 0.06
    rotation_primary_axis: str = "y"
    rotation_direction: int = 1
    preferred_step_degrees: float = 24.0
    maximum_step_degrees: float = 40.0
    rotation_search_step_degrees: float = 1.0
    secondary_axis_limit_degrees: float = 10.0
    downsample_factor: int = 4
    texture_percentile: float = 97.0
    maximum_texture_points: int = 80
    minimum_texture_points: int = 20
    minimum_valid_point_fraction: float = 0.45
    alignment_confidence_threshold: float = 0.18
    minimum_reliable_pair_ratio: float = 0.65
    latitude_limit_degrees: float = 60.0
    pre_roll_degrees: float = 45.0
    post_roll_degrees: float = 45.0
    target_coverage_degrees: float = 350.0
    minimum_confident_coverage: float = 300.0
    maximum_map_width: int = 1600
    minimum_observed_area_ratio: float = 0.55
    minimum_mapping_views: int = 8
    maximum_mapping_views: int = 20
    minimum_detection_confidence: float = 0.25
    minimum_focus_score: float = 35.0
    minimum_mask_area_ratio: float = 0.35
    maximum_mask_area_ratio: float = 0.92
    maximum_secondary_component_ratio: float = 0.12
    silhouette_border_pixels: int = 3
    morphological_closing_pixels: int = 5
    maximum_center_shift_ratio: float = 0.22
    maximum_scale_change_ratio: float = 0.30
    defect_probability_threshold: float = 0.50
    diagnostic: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SurfaceMappingConfig":
        value = value or {}
        known = {item.name for item in cls.__dataclass_fields__.values()}
        converted = {key: item for key, item in value.items() if key in known}
        direction = converted.get("rotation_direction", 1)
        if isinstance(direction, str):
            converted["rotation_direction"] = -1 if direction.lower() in {"negative", "reverse", "clockwise", "-1"} else 1
        config = cls(**converted)
        config.validate()
        return config

    def validate(self) -> None:
        if self.shape_model not in {"oblate", "sphere"}:
            raise ValueError("shape_model must be 'oblate' or 'sphere'")
        if self.rotation_primary_axis not in {"x", "y"}:
            raise ValueError("rotation_primary_axis must be 'x' or 'y'")
        if self.rotation_direction not in {-1, 1}:
            raise ValueError("rotation_direction must resolve to -1 or +1")
        if self.moment_coordinate_stride < 1 or self.downsample_factor < 1:
            raise ValueError("coordinate and downsample strides must be positive")
        if not (0.0 < self.texture_percentile < 100.0):
            raise ValueError("texture_percentile must be between 0 and 100")
        if self.minimum_texture_points < 4 or self.maximum_texture_points < self.minimum_texture_points:
            raise ValueError("texture point limits are invalid")
        if not (0.0 < self.minimum_valid_point_fraction <= 1.0):
            raise ValueError("minimum_valid_point_fraction must be between zero and one")
        if self.minimum_mapping_views < 2 or self.maximum_mapping_views < self.minimum_mapping_views:
            raise ValueError("mapping view limits are invalid")
        if self.maximum_step_degrees <= 0.0 or self.rotation_search_step_degrees <= 0.0:
            raise ValueError("rotation search limits must be positive")
        if not (0.0 < self.latitude_limit_degrees < 90.0):
            raise ValueError("latitude_limit_degrees must be between 0 and 90")
        if self.maximum_map_width < 64:
            raise ValueError("maximum_map_width must be at least 64")


@dataclass
class EllipseFit:
    center: np.ndarray
    semi_major: float
    semi_minor: float
    eigenvectors: np.ndarray


@dataclass
class ShapeFit:
    model: str
    a: float
    b: float
    radius: float
    radii: tuple[float, float, float]
    view_fits: list[EllipseFit]
    elevation_angles: list[float]
    reference_index: int


@dataclass
class PreparedView:
    track_id: int
    frame_id: int
    timestamp: str
    bbox: dict[str, float]
    transform: dict[str, float]
    rgb: np.ndarray
    mask: np.ndarray
    safe_mask: np.ndarray
    detection_confidence: float
    focus_score: float
    mask_quality: float
    defect_maps: dict[str, np.ndarray]
    defect_class_ids: dict[str, int]
    crop_path: str
    quality: float
    rejection_reason: str = ""


@dataclass
class PairRotation:
    source_frame: int
    target_frame: int
    matrix: np.ndarray
    primary_degrees: float
    secondary_degrees: float
    angle_degrees: float
    error: float
    valid_points: int
    valid_fraction: float
    error_margin: float
    confidence: float
    reliable: bool
    failure_reason: str = ""
    error_grid: np.ndarray | None = field(default=None, repr=False)
    source_points: np.ndarray | None = field(default=None, repr=False)
    target_points: np.ndarray | None = field(default=None, repr=False)

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value["matrix"] = self.matrix.tolist()
        value.pop("error_grid", None)
        value.pop("source_points", None)
        value.pop("target_points", None)
        return value


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def decode_rle(mask: dict[str, Any], target_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Decode the runner's alternating, row-major, background-first RLE."""
    width = int(mask.get("width", 0))
    height = int(mask.get("height", 0))
    if width <= 0 or height <= 0:
        return np.zeros(target_shape or (0, 0), np.uint8)
    flat = np.zeros(width * height, np.uint8)
    cursor = 0
    foreground = False
    for raw_count in mask.get("counts", []):
        count = max(0, int(raw_count))
        end = min(flat.size, cursor + count)
        if foreground:
            flat[cursor:end] = 255
        cursor = end
        foreground = not foreground
        if cursor >= flat.size:
            break
    decoded = flat.reshape(height, width)
    if target_shape and decoded.shape != target_shape:
        decoded = cv2.resize(decoded, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return decoded


def ellipse_moments(mask: np.ndarray, stride: int = 4) -> EllipseFit:
    """Fit the paper's projected ellipse from foreground coordinate covariance."""
    if mask.ndim != 2:
        raise ValueError("ellipse mask must be two-dimensional")
    ys, xs = np.nonzero(mask[::stride, ::stride] > 0)
    if xs.size < 12:
        raise ValueError("mask has too few foreground coordinates")
    coordinates = np.column_stack((xs.astype(np.float64) * stride, ys.astype(np.float64) * stride))
    center = coordinates.mean(axis=0)
    centered = coordinates - center
    covariance = centered.T @ centered / float(coordinates.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[1] <= EPSILON:
        raise ValueError("mask covariance has invalid eigenvalues")
    return EllipseFit(
        center=center,
        semi_major=2.0 * math.sqrt(float(eigenvalues[0])),
        semi_minor=2.0 * math.sqrt(float(eigenvalues[1])),
        eigenvectors=eigenvectors,
    )


def _continuous_eigenvectors(fits: list[EllipseFit]) -> None:
    for index in range(1, len(fits)):
        for axis in range(2):
            if float(np.dot(fits[index - 1].eigenvectors[:, axis], fits[index].eigenvectors[:, axis])) < 0.0:
                fits[index].eigenvectors[:, axis] *= -1.0
        if np.linalg.det(fits[index].eigenvectors) < 0.0:
            fits[index].eigenvectors[:, 1] *= -1.0


def fit_spheroid(masks: Iterable[np.ndarray], config: SurfaceMappingConfig) -> ShapeFit:
    fits = [ellipse_moments(mask, config.moment_coordinate_stride) for mask in masks]
    if len(fits) < 2:
        raise ValueError("at least two masks are required for shape fitting")
    _continuous_eigenvectors(fits)
    major = np.asarray([item.semi_major for item in fits], dtype=np.float64)
    minor = np.asarray([item.semi_minor for item in fits], dtype=np.float64)
    if config.robust_shape_fit and len(fits) >= 5:
        ordered = np.sort(major)
        trim = max(1, int(len(ordered) * 0.1))
        a = float(np.mean(ordered[trim:-trim])) if len(ordered) > 2 * trim else float(np.mean(ordered))
        b = float(np.percentile(minor, 10.0))
    else:
        a = float(np.mean(major))
        b = float(np.min(minor))
    if not math.isfinite(a) or not math.isfinite(b) or a <= EPSILON or b <= EPSILON:
        raise ValueError("shape radii are invalid")
    if b > a:
        a, b = b, a
    radius = float(np.mean((major + minor) * 0.5))
    spherical = config.shape_model == "sphere" or (a - b) / a < config.spherical_tolerance
    if spherical:
        return ShapeFit("sphere", radius, radius, radius, (radius, radius, radius), fits, [0.0] * len(fits), 0)

    denominator = max(EPSILON, a * a - b * b)
    cosines = np.sqrt(np.clip((minor * minor - b * b) / denominator, 0.0, 1.0))
    magnitudes = np.arccos(cosines)
    # The sign is ambiguous in a single silhouette.  Choose the signed branch
    # that varies most smoothly in the configured roller direction; extrema
    # inherit the branch with the smaller second difference.
    signed: list[float] = []
    direction = float(config.rotation_direction)
    for index, magnitude in enumerate(magnitudes):
        slope = 0.0
        if index > 0:
            slope += minor[index] - minor[index - 1]
        if index + 1 < len(minor):
            slope += minor[index + 1] - minor[index]
        candidates = [float(magnitude), -float(magnitude)]
        if index == 0:
            chosen = candidates[0] * direction
        else:
            trend_sign = direction if slope >= 0.0 else -direction
            preferred = abs(float(magnitude)) * trend_sign
            chosen = min(candidates, key=lambda item: abs(item - preferred) + 0.25 * abs(item - signed[-1]))
            if abs(slope) < EPSILON and len(signed) >= 2:
                chosen = min(candidates, key=lambda item: abs(item - 2.0 * signed[-1] + signed[-2]))
        signed.append(chosen)
    reference = int(np.argmin(minor))
    return ShapeFit("oblate", a, b, radius, (a, b, a), fits, signed, reference)


def view_pose_hint(shape: ShapeFit, index: int) -> np.ndarray:
    """Build the silhouette-derived fruit-to-camera pose for one view."""
    fit = shape.view_fits[index]
    major = np.array([fit.eigenvectors[0, 0], fit.eigenvectors[1, 0], 0.0], dtype=np.float64)
    major /= np.linalg.norm(major)
    if shape.model == "sphere":
        short = np.array([fit.eigenvectors[0, 1], fit.eigenvectors[1, 1], 0.0], dtype=np.float64)
    else:
        cosine = math.cos(shape.elevation_angles[index])
        sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
        minor_2d = fit.eigenvectors[:, 1]
        short = np.array([minor_2d[0] * sine, minor_2d[1] * sine, cosine], dtype=np.float64)
    short -= major * float(np.dot(major, short))
    short /= max(EPSILON, np.linalg.norm(short))
    equal = np.cross(major, short)
    equal /= max(EPSILON, np.linalg.norm(equal))
    basis = np.column_stack((major, short, equal))
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0
    return basis


def reference_basis(shape: ShapeFit) -> np.ndarray:
    """Return a right-handed fruit-to-camera basis at the reference view."""
    return view_pose_hint(shape, shape.reference_index)


def sphere_depth(x: np.ndarray, y: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    discriminant = radius * radius - x * x - y * y
    valid = discriminant >= -1.0e-5
    return np.sqrt(np.maximum(discriminant, 0.0)), valid


def oblate_depth(
    x: np.ndarray,
    y: np.ndarray,
    pose: np.ndarray,
    radii: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the visible (+z) ray intersection with a posed spheroid."""
    inverse_squared = np.diag(1.0 / np.square(np.asarray(radii, dtype=np.float64)))
    # pose has fruit axes in columns.  P = pose.T has those axes as rows,
    # exactly matching Q = P.T @ diag(1/r^2) @ P from the paper.
    p_rows = pose.T
    q = p_rows.T @ inverse_squared @ p_rows
    c1 = 2.0 * (q[0, 2] * x + q[1, 2] * y)
    c0 = q[0, 0] * x * x + 2.0 * q[0, 1] * x * y + q[1, 1] * y * y - 1.0
    discriminant = c1 * c1 - 4.0 * q[2, 2] * c0
    valid = discriminant >= -1.0e-7
    z = (-c1 + np.sqrt(np.maximum(discriminant, 0.0))) / max(EPSILON, 2.0 * q[2, 2])
    valid &= z > 0.0
    return z, valid


def reconstruct_depth(
    x: np.ndarray,
    y: np.ndarray,
    model: str,
    radii: tuple[float, float, float],
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if model == "sphere":
        return sphere_depth(x, y, float(np.mean(radii)))
    return oblate_depth(x, y, pose, radii)


def rodrigues_degrees(primary: float, secondary: float, primary_axis: str = "y") -> np.ndarray:
    vector = np.zeros(3, np.float64)
    if primary_axis == "x":
        vector[0], vector[1] = primary, secondary
    else:
        vector[1], vector[0] = primary, secondary
    return cv2.Rodrigues(np.deg2rad(vector))[0]


def normalize_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    normalized = u @ vt
    if np.linalg.det(normalized) < 0.0:
        u[:, -1] *= -1.0
        normalized = u @ vt
    return normalized


def accumulate_rotations(pair_rotations: Iterable[np.ndarray]) -> list[np.ndarray]:
    accumulated = [np.eye(3, dtype=np.float64)]
    for pair in pair_rotations:
        candidate = np.asarray(pair, dtype=np.float64) @ accumulated[-1]
        if np.linalg.norm(candidate.T @ candidate - np.eye(3), ord="fro") > 1.0e-6 or abs(np.linalg.det(candidate) - 1.0) > 1.0e-6:
            candidate = normalize_rotation(candidate)
        accumulated.append(candidate)
    return accumulated


def preprocess_texture(rgb: np.ndarray, mask: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    output_size = (max(1, width // factor), max(1, height // factor))
    green = cv2.resize(rgb[:, :, 1], output_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    blurred = cv2.GaussianBlur(green, (0, 0), sigmaX=1.25, sigmaY=1.25)
    high_pass = green - blurred
    small_mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
    return high_pass, small_mask


def select_texture_points(
    high_pass: np.ndarray,
    mask: np.ndarray,
    percentile: float,
    maximum: int,
    minimum: int,
    border: int,
) -> tuple[np.ndarray, np.ndarray]:
    kernel_size = _odd_kernel(max(1, border * 2 + 1))
    interior = cv2.erode(mask, np.ones((kernel_size, kernel_size), np.uint8)) > 0
    values = np.abs(high_pass[interior])
    if values.size < minimum:
        return np.empty((0, 2), np.float64), np.empty((0,), np.float32)
    threshold = float(np.percentile(values, percentile))
    ys, xs = np.nonzero(interior & (np.abs(high_pass) >= threshold))
    if xs.size < minimum:
        return np.empty((0, 2), np.float64), np.empty((0,), np.float32)
    strengths = np.abs(high_pass[ys, xs])
    order = np.argsort(strengths)[::-1][:maximum]
    points = np.column_stack((xs[order], ys[order])).astype(np.float64)
    return points, high_pass[ys[order], xs[order]].astype(np.float32)


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        image,
        x.astype(np.float32).reshape(1, -1),
        y.astype(np.float32).reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).reshape(-1)


def _pair_error(
    source_3d: np.ndarray,
    source_values: np.ndarray,
    target_high_pass: np.ndarray,
    target_mask: np.ndarray,
    target_center: np.ndarray,
    rotation: np.ndarray,
    minimum_points: int,
    minimum_fraction: float,
) -> tuple[float, int, float, np.ndarray]:
    projected = (rotation @ source_3d.T).T
    x = projected[:, 0] + target_center[0]
    y = projected[:, 1] + target_center[1]
    xi = np.rint(x).astype(np.int32)
    yi = np.rint(y).astype(np.int32)
    valid = projected[:, 2] > 0.0
    valid &= (xi >= 0) & (xi < target_mask.shape[1]) & (yi >= 0) & (yi < target_mask.shape[0])
    in_bounds = np.nonzero(valid)[0]
    if in_bounds.size:
        valid[in_bounds] &= target_mask[yi[in_bounds], xi[in_bounds]] > 0
    count = int(np.count_nonzero(valid))
    fraction = count / max(1, source_3d.shape[0])
    if count < minimum_points or fraction < minimum_fraction:
        return math.inf, count, fraction, np.column_stack((x[valid], y[valid]))
    sampled = _bilinear(target_high_pass, x[valid], y[valid])
    raw = float(np.mean(np.abs(source_values[valid] - sampled)))
    penalized = raw * (1.0 + 1.5 * (1.0 - fraction))
    return penalized, count, fraction, np.column_stack((x[valid], y[valid]))


def estimate_pair_rotation(
    source: PreparedView,
    target: PreparedView,
    shape: ShapeFit,
    source_shape_index: int,
    config: SurfaceMappingConfig,
    maximum_primary: float | None = None,
) -> PairRotation:
    factor = config.downsample_factor
    source_hp, source_mask = preprocess_texture(source.rgb, source.safe_mask, factor)
    target_hp, target_mask = preprocess_texture(target.rgb, target.safe_mask, factor)
    points, source_values = select_texture_points(
        source_hp,
        source_mask,
        config.texture_percentile,
        config.maximum_texture_points,
        config.minimum_texture_points,
        max(1, config.silhouette_border_pixels // factor),
    )
    if points.shape[0] < config.minimum_texture_points:
        return PairRotation(source.frame_id, target.frame_id, np.eye(3), 0.0, 0.0, 0.0, math.inf, 0, 0.0, 0.0, 0.0, False, "textureless source view")

    source_fit = shape.view_fits[source_shape_index]
    source_center = source_fit.center / factor
    target_fit = ellipse_moments(target.mask, config.moment_coordinate_stride)
    target_center = target_fit.center / factor
    coordinates = points - source_center
    scaled_radii = tuple(item / factor for item in shape.radii)
    pose = view_pose_hint(shape, source_shape_index)
    z, depth_valid = reconstruct_depth(coordinates[:, 0], coordinates[:, 1], shape.model, scaled_radii, pose)
    points = points[depth_valid]
    source_values = source_values[depth_valid]
    coordinates = coordinates[depth_valid]
    z = z[depth_valid]
    source_3d = np.column_stack((coordinates, z))
    if source_3d.shape[0] < config.minimum_texture_points:
        return PairRotation(source.frame_id, target.frame_id, np.eye(3), 0.0, 0.0, 0.0, math.inf, 0, 0.0, 0.0, 0.0, False, "too few reconstructable texture points")

    limit = maximum_primary if maximum_primary is not None else config.maximum_step_degrees
    step = config.rotation_search_step_degrees
    primary_values = np.arange(0.0, limit + step * 0.25, step) * config.rotation_direction
    secondary_values = np.arange(-config.secondary_axis_limit_degrees, config.secondary_axis_limit_degrees + step * 0.25, step)
    errors = np.full((len(primary_values), len(secondary_values)), np.inf, np.float64)
    support: dict[tuple[int, int], tuple[int, float]] = {}
    for row, primary in enumerate(primary_values):
        for column, secondary in enumerate(secondary_values):
            rotation = rodrigues_degrees(float(primary), float(secondary), config.rotation_primary_axis)
            error, count, fraction, _ = _pair_error(
                source_3d, source_values, target_hp, target_mask, target_center, rotation,
                config.minimum_texture_points, config.minimum_valid_point_fraction,
            )
            errors[row, column] = error
            support[(row, column)] = (count, fraction)
    if not np.any(np.isfinite(errors)):
        return PairRotation(source.frame_id, target.frame_id, np.eye(3), 0.0, 0.0, 0.0, math.inf, 0, 0.0, 0.0, 0.0, False, "no rotation candidate has enough valid correspondences", errors)

    best_index = np.unravel_index(int(np.nanargmin(errors)), errors.shape)
    coarse_primary = float(primary_values[best_index[0]])
    coarse_secondary = float(secondary_values[best_index[1]])
    half = step * 0.5
    fine_primary = coarse_primary + np.arange(-2, 3) * half
    fine_secondary = coarse_secondary + np.arange(-2, 3) * half
    fine_errors = np.full((5, 5), np.inf, np.float64)
    fine_support: dict[tuple[int, int], tuple[int, float]] = {}
    for row, primary in enumerate(fine_primary):
        if config.rotation_direction * primary < 0.0 or abs(primary) > limit + step:
            continue
        for column, secondary in enumerate(fine_secondary):
            if abs(secondary) > config.secondary_axis_limit_degrees + step:
                continue
            rotation = rodrigues_degrees(float(primary), float(secondary), config.rotation_primary_axis)
            error, count, fraction, _ = _pair_error(
                source_3d, source_values, target_hp, target_mask, target_center, rotation,
                config.minimum_texture_points, config.minimum_valid_point_fraction,
            )
            fine_errors[row, column] = error
            fine_support[(row, column)] = (count, fraction)
    fine_index = np.unravel_index(int(np.nanargmin(fine_errors)), fine_errors.shape)
    best_primary = float(fine_primary[fine_index[0]])
    best_secondary = float(fine_secondary[fine_index[1]])
    best_error = float(fine_errors[fine_index])
    best_count, best_fraction = fine_support[fine_index]

    # Appendix-B-style local quadratic fit.  Use it only for a convex,
    # well-conditioned minimum whose sub-grid displacement stays local.
    samples: list[list[float]] = []
    values: list[float] = []
    for row in range(5):
        for column in range(5):
            if not math.isfinite(float(fine_errors[row, column])):
                continue
            dx = float(fine_primary[row] - best_primary)
            dy = float(fine_secondary[column] - best_secondary)
            samples.append([dx * dx, dy * dy, dx * dy, dx, dy, 1.0])
            values.append(float(fine_errors[row, column]))
    if len(samples) >= 9:
        coefficients, *_ = np.linalg.lstsq(np.asarray(samples), np.asarray(values), rcond=None)
        hessian = np.array([[2.0 * coefficients[0], coefficients[2]], [coefficients[2], 2.0 * coefficients[1]]])
        gradient = np.array([coefficients[3], coefficients[4]])
        eigenvalues = np.linalg.eigvalsh(hessian)
        condition = np.linalg.cond(hessian)
        if np.all(eigenvalues > EPSILON) and condition < 1.0e4:
            offset = -np.linalg.solve(hessian, gradient)
            if np.all(np.abs(offset) <= step):
                refined_primary = best_primary + float(offset[0])
                refined_secondary = best_secondary + float(offset[1])
                refined_rotation = rodrigues_degrees(refined_primary, refined_secondary, config.rotation_primary_axis)
                refined_error, refined_count, refined_fraction, _ = _pair_error(
                    source_3d, source_values, target_hp, target_mask, target_center, refined_rotation,
                    config.minimum_texture_points, config.minimum_valid_point_fraction,
                )
                if refined_error <= best_error:
                    best_primary, best_secondary = refined_primary, refined_secondary
                    best_error, best_count, best_fraction = refined_error, refined_count, refined_fraction

    finite_errors = np.sort(fine_errors[np.isfinite(fine_errors)])
    second = float(finite_errors[1]) if finite_errors.size > 1 else best_error
    margin = max(0.0, second - best_error)
    contrast = float(np.mean(np.abs(source_values))) + EPSILON
    distinctiveness = min(1.0, margin / max(best_error, EPSILON) * 8.0)
    confidence = float(np.clip(math.exp(-best_error / contrast) * best_fraction * (0.35 + 0.65 * distinctiveness), 0.0, 1.0))
    matrix = rodrigues_degrees(best_primary, best_secondary, config.rotation_primary_axis)
    transformed = (matrix @ source_3d.T).T
    target_x = transformed[:, 0] + target_center[0]
    target_y = transformed[:, 1] + target_center[1]
    target_xi = np.rint(target_x).astype(np.int32)
    target_yi = np.rint(target_y).astype(np.int32)
    correspondence_valid = transformed[:, 2] > 0.0
    correspondence_valid &= (target_xi >= 0) & (target_xi < target_mask.shape[1])
    correspondence_valid &= (target_yi >= 0) & (target_yi < target_mask.shape[0])
    bounded = np.nonzero(correspondence_valid)[0]
    if bounded.size:
        correspondence_valid[bounded] &= target_mask[target_yi[bounded], target_xi[bounded]] > 0
    valid_source = points[correspondence_valid]
    target_points = np.column_stack((target_x[correspondence_valid], target_y[correspondence_valid]))
    reliable = confidence >= config.alignment_confidence_threshold and abs(best_primary) <= config.maximum_step_degrees + step
    reason = "" if reliable else "ambiguous texture alignment"
    angle = float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(matrix)[0])))
    return PairRotation(
        source.frame_id, target.frame_id, matrix, best_primary, best_secondary, angle,
        best_error, best_count, best_fraction, margin, confidence, reliable, reason,
        fine_errors, valid_source, target_points,
    )


def _grabcut_mask(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    if width < 12 or height < 12:
        raise ValueError("crop is too small for GrabCut")
    inset = max(2, int(round(min(width, height) * 0.035)))
    rectangle = (inset, inset, max(2, width - inset * 2), max(2, height - inset * 2))
    labels = np.zeros((height, width), np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), labels, rectangle, background, foreground, 4, cv2.GC_INIT_WITH_RECT)
    return np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def clean_fruit_mask(mask: np.ndarray, closing_pixels: int) -> tuple[np.ndarray, int, float]:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        raise ValueError("GrabCut returned no foreground component")
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    largest_label = int(order[0] + 1)
    largest_area = float(areas[order[0]])
    secondary_ratio = float(areas[order[1]] / largest_area) if len(order) > 1 and largest_area > 0 else 0.0
    cleaned = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    # Flood-fill the exterior, then add every enclosed background island.
    inverse = cv2.bitwise_not(cleaned)
    flooded = inverse.copy()
    flood_mask = np.zeros((cleaned.shape[0] + 2, cleaned.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 0)
    cleaned = cv2.bitwise_or(cleaned, flooded)
    kernel_size = _odd_kernel(closing_pixels)
    if kernel_size > 1:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    return cleaned, int(count - 1), secondary_ratio


def _focus_score(rgb: np.ndarray) -> float:
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return slug[:80] or "unknown"


def _natural_path_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _prepare_view(record: dict[str, Any], config: SurfaceMappingConfig) -> PreparedView:
    path = Path(str(record.get("cropPath", ""))).resolve()
    encoded = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode crop {path.name}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw = _grabcut_mask(rgb)
    mask, component_count, secondary_ratio = clean_fruit_mask(raw, config.morphological_closing_pixels)
    height, width = mask.shape
    area_ratio = float(np.count_nonzero(mask)) / mask.size
    border = max(1, config.silhouette_border_pixels)
    touches = bool(np.any(mask[:border]) or np.any(mask[-border:]) or np.any(mask[:, :border]) or np.any(mask[:, -border:]))
    detection_confidence = float(record.get("detectionConfidence", 0.0))
    focus = float(record.get("focusScore", _focus_score(rgb)))
    reasons: list[str] = []
    if touches:
        reasons.append("fruit touches crop boundary")
    if area_ratio < config.minimum_mask_area_ratio or area_ratio > config.maximum_mask_area_ratio:
        reasons.append(f"implausible mask area ratio {area_ratio:.3f}")
    if component_count > 1 and secondary_ratio > config.maximum_secondary_component_ratio:
        reasons.append(f"multiple large mask components ({secondary_ratio:.3f})")
    if focus < config.minimum_focus_score:
        reasons.append(f"focus score {focus:.1f} below threshold")
    if detection_confidence < config.minimum_detection_confidence:
        reasons.append("detection confidence below threshold")
    erode_size = _odd_kernel(max(1, config.silhouette_border_pixels * 2 + 1))
    safe_mask = cv2.erode(mask, np.ones((erode_size, erode_size), np.uint8))
    defect_maps: dict[str, np.ndarray] = {}
    defect_class_ids: dict[str, int] = {}
    for instance in record.get("defectInstances", []):
        name = str(instance.get("label") or f"class_{int(instance.get('classId', -1))}")
        defect_class_ids[name] = int(instance.get("classId", -1))
        confidence = float(instance.get("confidence", 0.0))
        decoded = decode_rle(instance.get("mask", {}), (height, width)).astype(np.float32) / 255.0
        probability = decoded * np.clip(confidence, 0.0, 1.0)
        defect_maps[name] = np.maximum(defect_maps.get(name, np.zeros_like(probability)), probability)
    quality = float(np.clip(detection_confidence, 0.0, 1.0) * np.clip(focus / max(config.minimum_focus_score * 2.0, 1.0), 0.25, 1.0))
    quality *= float(np.clip(1.0 - secondary_ratio, 0.0, 1.0))
    return PreparedView(
        track_id=int(record.get("trackId", -1)),
        frame_id=int(record.get("frameId", -1)),
        timestamp=str(record.get("timestamp", "")),
        bbox=dict(record.get("originalBoundingBox", {})),
        transform=dict(record.get("cropTransform", {})),
        rgb=rgb,
        mask=mask,
        safe_mask=safe_mask,
        detection_confidence=detection_confidence,
        focus_score=focus,
        mask_quality=float(np.clip(1.0 - abs(area_ratio - 0.65) - secondary_ratio, 0.0, 1.0)),
        defect_maps=defect_maps,
        defect_class_ids=defect_class_ids,
        crop_path=str(path),
        quality=quality,
        rejection_reason="; ".join(reasons),
    )


def _reject_geometric_outliers(views: list[PreparedView], config: SurfaceMappingConfig) -> tuple[list[PreparedView], list[dict[str, Any]]]:
    accepted: list[PreparedView] = []
    rejected: list[dict[str, Any]] = []
    previous_center: np.ndarray | None = None
    previous_scale: float | None = None
    for view in views:
        if view.rejection_reason:
            rejected.append({"frame": view.frame_id, "reason": view.rejection_reason})
            continue
        fit = ellipse_moments(view.mask, config.moment_coordinate_stride)
        normalized_center = fit.center / np.asarray([view.mask.shape[1], view.mask.shape[0]], dtype=np.float64)
        scale = math.sqrt(float(np.count_nonzero(view.mask)))
        reason = ""
        if previous_center is not None and np.linalg.norm(normalized_center - previous_center) > config.maximum_center_shift_ratio:
            reason = "fruit centre changed implausibly"
        if previous_scale is not None and abs(scale / previous_scale - 1.0) > config.maximum_scale_change_ratio:
            reason = "fruit scale changed implausibly"
        if reason:
            rejected.append({"frame": view.frame_id, "reason": reason})
            continue
        accepted.append(view)
        previous_center, previous_scale = normalized_center, scale
    return accepted, rejected


def _limit_views(views: list[PreparedView], config: SurfaceMappingConfig) -> list[PreparedView]:
    preferred_count = int(round(config.target_coverage_degrees / config.preferred_step_degrees)) + 1
    selected_limit = min(
        config.maximum_mapping_views,
        max(config.minimum_mapping_views, preferred_count),
    )
    if len(views) <= selected_limit:
        return views
    # Preserve temporal coverage.  For each evenly spaced bin choose its best
    # quality view rather than resizing or retaining every frame in memory.
    edges = np.linspace(0, len(views), selected_limit + 1, dtype=int)
    selected: list[PreparedView] = []
    for left, right in zip(edges[:-1], edges[1:]):
        bucket = views[left:max(left + 1, right)]
        selected.append(max(bucket, key=lambda item: item.quality))
    return sorted(selected, key=lambda item: item.frame_id)


def _align_views(
    views: list[PreparedView],
    shape: ShapeFit,
    config: SurfaceMappingConfig,
) -> tuple[list[PreparedView], list[np.ndarray], list[PairRotation], list[dict[str, Any]]]:
    active = [views[0]]
    rotations: list[np.ndarray] = []
    pairs: list[PairRotation] = []
    rejected: list[dict[str, Any]] = []
    source_index = 0
    while source_index < len(views) - 1:
        target_index = source_index + 1
        pair = estimate_pair_rotation(views[source_index], views[target_index], shape, source_index, config)
        pairs.append(pair)
        if pair.reliable:
            active.append(views[target_index])
            rotations.append(pair.matrix)
            source_index = target_index
            continue
        if source_index + 2 < len(views):
            bypass_index = source_index + 2
            bypass = estimate_pair_rotation(views[source_index], views[bypass_index], shape, source_index, config)
            pairs.append(bypass)
            if bypass.reliable and abs(bypass.primary_degrees) <= config.maximum_step_degrees:
                rejected.append({"frame": views[target_index].frame_id, "reason": "bypassed unreliable adjacent alignment"})
                active.append(views[bypass_index])
                rotations.append(bypass.matrix)
                source_index = bypass_index
                continue
        rejected.append({
            "frame": views[target_index].frame_id,
            "reason": pair.failure_reason or "unreliable adjacent alignment",
        })
        rejected.extend(
            {"frame": view.frame_id, "reason": "alignment chain terminated after an earlier failure"}
            for view in views[target_index + 1:]
        )
        break
    return active, accumulate_rotations(rotations), pairs, rejected


def _view_bases(shape: ShapeFit, accumulated: list[np.ndarray], active: list[PreparedView], original: list[PreparedView]) -> list[np.ndarray]:
    original_indices = {id(view): index for index, view in enumerate(original)}
    active_original_indices = [original_indices[id(view)] for view in active]
    # Failed/bypassed alignments can remove the shape fit's original reference.
    # The propagation reference must itself be an active view, so choose the
    # active silhouette with the smallest projected minor semi-axis.
    reference_active = min(
        range(len(active)),
        key=lambda index: shape.view_fits[active_original_indices[index]].semi_minor,
    )
    reference_original = active_original_indices[reference_active]
    base_reference = view_pose_hint(shape, reference_original)
    g_reference = accumulated[reference_active]
    bases: list[np.ndarray] = []
    for transform in accumulated:
        reference_to_view = transform @ g_reference.T
        basis = normalize_rotation(reference_to_view @ base_reference)
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1.0e-5) or not math.isclose(float(np.linalg.det(basis)), 1.0, abs_tol=1.0e-5):
            raise ValueError("accumulated fruit basis is not a proper rotation")
        bases.append(basis)
    return bases


def _unwrap_centers(bases: list[np.ndarray], pairs: list[PairRotation], direction: int) -> np.ndarray:
    camera_center = np.array([0.0, 0.0, 1.0])
    raw = np.asarray([math.atan2(*(lambda p: (p[2], p[0]))(basis.T @ camera_center)) for basis in bases])
    unwrapped = np.unwrap(raw)
    for index in range(1, len(unwrapped)):
        if direction * (unwrapped[index] - unwrapped[index - 1]) <= 1.0e-5:
            reliable = [item for item in pairs if item.reliable]
            estimate = abs(reliable[min(index - 1, len(reliable) - 1)].primary_degrees) if reliable else 0.0
            unwrapped[index] = unwrapped[index - 1] + direction * math.radians(max(estimate, 0.01))
    return unwrapped


def _best_single_revolution(
    views: list[PreparedView],
    bases: list[np.ndarray],
    centers: np.ndarray,
    direction: int,
    pairs: list[PairRotation],
) -> tuple[list[PreparedView], list[np.ndarray], np.ndarray]:
    directional = centers * direction
    if directional[-1] - directional[0] <= 2.0 * math.pi + 1.0e-6:
        return views, bases, centers
    best = (0, 1)
    best_score = -math.inf
    confidence_by_transition = {
        (pair.source_frame, pair.target_frame): pair.confidence
        for pair in pairs if pair.reliable
    }
    for left in range(len(views)):
        right = left + 1
        while right < len(views) and directional[right] - directional[left] <= 2.0 * math.pi + 1.0e-6:
            right += 1
        alignment_score = sum(
            confidence_by_transition.get((views[index].frame_id, views[index + 1].frame_id), 0.0)
            for index in range(left, right - 1)
        )
        score = (
            sum(view.quality for view in views[left:right])
            + alignment_score
            + 0.05 * (directional[right - 1] - directional[left])
        )
        if score > best_score:
            best_score = score
            best = (left, right)
    left, right = best
    return views[left:right], bases[left:right], centers[left:right]


def uv_to_fruit(latitude: np.ndarray, longitude: np.ndarray, radii: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = radii
    return np.stack((
        rx * np.cos(latitude) * np.cos(longitude),
        ry * np.sin(latitude),
        rz * np.cos(latitude) * np.sin(longitude),
    ), axis=-1)


def fruit_to_uv(points: np.ndarray, radii: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    normalized = points / np.asarray(radii, dtype=np.float64)
    latitude = np.arcsin(np.clip(normalized[..., 1], -1.0, 1.0))
    longitude = np.arctan2(normalized[..., 2], normalized[..., 0])
    return latitude, longitude


def surface_area_weights(latitude: np.ndarray, longitude: np.ndarray, radii: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = radii
    d_latitude = np.stack((
        -rx * np.sin(latitude) * np.cos(longitude),
        ry * np.cos(latitude),
        -rz * np.sin(latitude) * np.sin(longitude),
    ), axis=-1)
    d_longitude = np.stack((
        -rx * np.cos(latitude) * np.sin(longitude),
        np.zeros_like(latitude),
        rz * np.cos(latitude) * np.cos(longitude),
    ), axis=-1)
    return np.linalg.norm(np.cross(d_latitude, d_longitude), axis=-1)


def _bracketing_views(longitudes: np.ndarray, centers: np.ndarray, complete: bool, pre: float, post: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(centers)
    sorted_centers = centers[order]
    base = sorted_centers[0]
    query = ((longitudes - base) % (2.0 * math.pi)) + base
    if complete:
        extended = np.concatenate((sorted_centers, [sorted_centers[0] + 2.0 * math.pi]))
        extended_order = np.concatenate((order, [order[0]]))
        right = np.searchsorted(extended, query, side="right")
        right = np.clip(right, 1, len(extended) - 1)
        left = right - 1
        valid = np.ones(query.shape, dtype=bool)
    else:
        wrapped_pre = query >= sorted_centers[0] + 2.0 * math.pi - pre
        valid = ((query >= sorted_centers[0]) & (query <= sorted_centers[-1] + post)) | wrapped_pre
        right = np.searchsorted(sorted_centers, query, side="right")
        right = np.clip(right, 0, len(sorted_centers) - 1)
        left = np.clip(right - 1, 0, len(sorted_centers) - 1)
        before = query <= sorted_centers[0]
        after = query >= sorted_centers[-1]
        left[before] = right[before] = 0
        left[after] = right[after] = len(sorted_centers) - 1
        left[wrapped_pre] = right[wrapped_pre] = 0
        extended = sorted_centers
        extended_order = order
    left_centers = extended[left]
    right_centers = extended[right]
    span = right_centers - left_centers
    same = np.abs(span) < EPSILON
    weight_right = np.where(same, 0.0, (query - left_centers) / np.where(same, 1.0, span))
    weight_left = 1.0 - weight_right
    return extended_order[left], extended_order[right], weight_left, valid


def _project_maps(
    views: list[PreparedView],
    bases: list[np.ndarray],
    centers: np.ndarray,
    pair_confidences: list[float],
    shape: ShapeFit,
    config: SurfaceMappingConfig,
    geometrically_complete: bool,
) -> dict[str, Any]:
    rmax = max(shape.radii)
    natural_width = max(64, int(math.ceil(2.0 * math.pi * rmax)))
    width = min(config.maximum_map_width, natural_width)
    latitude_span = math.radians(config.latitude_limit_degrees * 2.0)
    height = max(24, int(round(width * latitude_span / (2.0 * math.pi))))
    base = float(np.min(centers))
    longitude_axis = base + np.arange(width, dtype=np.float64) * (2.0 * math.pi / width)
    latitude_axis = np.linspace(-math.radians(config.latitude_limit_degrees), math.radians(config.latitude_limit_degrees), height)
    longitude, latitude = np.meshgrid(longitude_axis, latitude_axis)
    fruit_points = uv_to_fruit(latitude, longitude, shape.radii)
    left, right, left_weight, longitude_valid = _bracketing_views(
        longitude_axis,
        centers,
        geometrically_complete,
        math.radians(config.pre_roll_degrees),
        math.radians(config.post_roll_degrees),
    )
    right_weight = 1.0 - left_weight
    rgb_sum = np.zeros((height, width, 3), np.float64)
    weights = np.zeros((height, width), np.float64)
    confidence_sum = np.zeros((height, width), np.float64)
    all_classes = sorted({name for view in views for name in view.defect_maps})
    class_ids = {name: next((view.defect_class_ids[name] for view in views if name in view.defect_class_ids), -1) for name in all_classes}
    defect_sum = {name: np.zeros((height, width), np.float64) for name in all_classes}
    view_confidence = np.asarray([
        view.quality * (np.mean(pair_confidences[max(0, index - 1):min(len(pair_confidences), index + 1)]) if pair_confidences else 1.0)
        for index, view in enumerate(views)
    ], dtype=np.float64)

    for index, (view, basis) in enumerate(zip(views, bases)):
        column_weight = np.zeros(width, np.float64)
        column_weight[left == index] += left_weight[left == index]
        column_weight[right == index] += right_weight[right == index]
        column_weight *= longitude_valid
        if not np.any(column_weight > 0.0):
            continue
        projected = np.einsum("ij,hwj->hwi", basis, fruit_points)
        fit = ellipse_moments(view.mask, config.moment_coordinate_stride)
        map_x = (projected[..., 0] + fit.center[0]).astype(np.float32)
        map_y = (projected[..., 1] + fit.center[1]).astype(np.float32)
        inside = cv2.remap(view.safe_mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        valid = inside & (projected[..., 2] > 0.0) & longitude_valid[np.newaxis, :] & (column_weight[np.newaxis, :] > 0.0)
        contribution = column_weight[np.newaxis, :] * valid
        sampled_rgb = cv2.remap(view.rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rgb_sum += sampled_rgb * contribution[..., np.newaxis]
        confidence_sum += view_confidence[index] * contribution
        for name in all_classes:
            probability = view.defect_maps.get(name)
            if probability is None:
                continue
            sampled = cv2.remap(probability, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            defect_sum[name] += sampled * contribution
        weights += contribution

    observed = weights > EPSILON
    rgb_map = np.zeros_like(rgb_sum, np.uint8)
    rgb_map[observed] = np.clip(rgb_sum[observed] / weights[observed, np.newaxis], 0, 255).astype(np.uint8)
    confidence_map = np.zeros_like(weights, np.float32)
    confidence_map[observed] = (confidence_sum[observed] / weights[observed]).astype(np.float32)
    defect_maps: dict[str, np.ndarray] = {}
    for name, values in defect_sum.items():
        probability = np.zeros_like(weights, np.float32)
        probability[observed] = np.clip(values[observed] / weights[observed], 0.0, 1.0).astype(np.float32)
        defect_maps[name] = probability
    area_weights = surface_area_weights(latitude, longitude, shape.radii)
    return {
        "rgb": rgb_map,
        "observed": observed,
        "confidence": confidence_map,
        "defects": defect_maps,
        "class_ids": class_ids,
        "area_weights": area_weights,
        "latitude": latitude,
        "longitude": longitude,
        "resolution": [width, height],
    }


def _largest_wrapped_region(mask: np.ndarray, weights: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    tiled = np.concatenate((mask, mask, mask), axis=1).astype(np.uint8)
    _, labels = cv2.connectedComponents(tiled, connectivity=8)
    middle = labels[:, mask.shape[1]:mask.shape[1] * 2]
    largest = 0.0
    for label in np.unique(middle[mask]):
        if label == 0:
            continue
        largest = max(largest, float(np.sum(weights[(middle == label) & mask])))
    return largest


def defect_area_statistics(projected: dict[str, Any], threshold: float) -> tuple[dict[str, dict[str, float]], float, float]:
    observed = projected["observed"]
    weights = projected["area_weights"]
    target_area = float(np.sum(weights))
    observed_area = float(np.sum(weights[observed]))
    observed_ratio = observed_area / max(EPSILON, target_area)
    per_class: dict[str, dict[str, float]] = {}
    combined = np.zeros_like(observed)
    for name, probability in projected["defects"].items():
        positive = observed & (probability >= threshold)
        combined |= positive
        area = float(np.sum(weights[positive]))
        largest = _largest_wrapped_region(positive, weights)
        per_class[name] = {
            "observed_surface_ratio": area / max(EPSILON, observed_area),
            "target_surface_ratio": area / max(EPSILON, target_area),
            "largest_region_observed_ratio": largest / max(EPSILON, observed_area),
        }
    total = float(np.sum(weights[combined])) / max(EPSILON, observed_area)
    return per_class, total, observed_ratio


def mapping_status(
    coverage: float,
    maximum_gap: float,
    reliable_ratio: float,
    observed_ratio: float,
    alignment_confidence: float,
    config: SurfaceMappingConfig,
) -> tuple[str, list[str]]:
    complete = (
        coverage >= config.target_coverage_degrees
        and coverage >= config.minimum_confident_coverage
        and maximum_gap <= config.maximum_step_degrees
        and reliable_ratio >= config.minimum_reliable_pair_ratio
        and observed_ratio >= config.minimum_observed_area_ratio
        and alignment_confidence >= config.alignment_confidence_threshold
    )
    reasons: list[str] = []
    if coverage < config.minimum_confident_coverage:
        reasons.append(f"coverage {coverage:.1f} degrees is below confident threshold")
    if coverage < config.target_coverage_degrees:
        reasons.append(f"coverage {coverage:.1f} degrees is below target coverage")
    if maximum_gap > config.maximum_step_degrees:
        reasons.append(f"maximum angular gap {maximum_gap:.1f} degrees exceeds limit")
    if reliable_ratio < config.minimum_reliable_pair_ratio:
        reasons.append("insufficient reliable adjacent rotations")
    if observed_ratio < config.minimum_observed_area_ratio:
        reasons.append("observed surface area is below threshold")
    if alignment_confidence < config.alignment_confidence_threshold:
        reasons.append("mean alignment confidence is below threshold")
    return ("complete" if complete else "partial"), reasons


def fused_grade(
    projected: dict[str, Any],
    threshold: float,
    fallback_grade: Any,
    fallback_confidence: float,
) -> tuple[Any, int | None, float]:
    """Apply the existing worst-class rule to non-duplicated fused evidence."""
    observed = projected["observed"]
    weights = projected["area_weights"]
    candidates: list[tuple[int, float]] = []
    for name, probability in projected["defects"].items():
        class_id = int(projected["class_ids"].get(name, -1))
        positive = observed & (probability >= threshold)
        if class_id < 0 or class_id > 2 or not np.any(positive):
            continue
        confidence = float(np.sum(probability[positive] * weights[positive]) / max(EPSILON, np.sum(weights[positive])))
        candidates.append((class_id, confidence))
    if not candidates:
        return fallback_grade, None, fallback_confidence
    worst = max(class_id for class_id, _ in candidates)
    confidences = [confidence for class_id, confidence in candidates if class_id == worst]
    return f"Class {worst + 1}", worst, float(np.mean(confidences))


def _save_png(path: Path, rgb_or_grey: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = rgb_or_grey
    if value.ndim == 3:
        value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), value):
        raise OSError(f"could not save {path.name}")


def _save_diagnostics(output: Path, views: list[PreparedView], pairs: list[PairRotation], config: SurfaceMappingConfig) -> None:
    if not config.diagnostic:
        return
    diagnostic = output / "diagnostic"
    diagnostic.mkdir(parents=True, exist_ok=True)
    for view in views:
        _save_png(diagnostic / f"cleaned_mask_frame_{view.frame_id}.png", view.mask)
        high_pass, small_mask = preprocess_texture(view.rgb, view.safe_mask, config.downsample_factor)
        points, _ = select_texture_points(high_pass, small_mask, config.texture_percentile, config.maximum_texture_points, config.minimum_texture_points, 1)
        canvas = cv2.cvtColor(cv2.resize(view.rgb, (high_pass.shape[1], high_pass.shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        for point in points.astype(int):
            cv2.circle(canvas, tuple(point), 1, (0, 0, 255), -1)
        cv2.imwrite(str(diagnostic / f"texture_points_frame_{view.frame_id}.png"), canvas)
    by_frame = {view.frame_id: view for view in views}
    for pair in pairs:
        grid = pair.error_grid if pair.error_grid is not None else np.full((5, 5), np.inf, np.float64)
        if grid is not None:
            finite_mask = np.isfinite(grid)
            normalized = np.full(grid.shape, 255, np.uint8)
            if np.any(finite_mask):
                finite = grid[finite_mask]
                normalized[finite_mask] = 0
                if finite.max() > finite.min():
                    normalized[finite_mask] = np.clip(
                        (grid[finite_mask] - finite.min()) / (finite.max() - finite.min()) * 254.0,
                        0, 254,
                    ).astype(np.uint8)
            _save_png(diagnostic / f"rotation_error_{pair.source_frame}_{pair.target_frame}.png", normalized)
        source = by_frame.get(pair.source_frame)
        target = by_frame.get(pair.target_frame)
        if source is None or target is None:
            continue
        source_small = cv2.resize(source.rgb, (source.rgb.shape[1] // config.downsample_factor, source.rgb.shape[0] // config.downsample_factor), interpolation=cv2.INTER_AREA)
        target_small = cv2.resize(target.rgb, (target.rgb.shape[1] // config.downsample_factor, target.rgb.shape[0] // config.downsample_factor), interpolation=cv2.INTER_AREA)
        height = max(source_small.shape[0], target_small.shape[0])
        canvas = np.zeros((height, source_small.shape[1] + target_small.shape[1], 3), np.uint8)
        canvas[:source_small.shape[0], :source_small.shape[1]] = cv2.cvtColor(source_small, cv2.COLOR_RGB2BGR)
        canvas[:target_small.shape[0], source_small.shape[1]:] = cv2.cvtColor(target_small, cv2.COLOR_RGB2BGR)
        source_points = pair.source_points if pair.source_points is not None else np.empty((0, 2))
        target_points = pair.target_points if pair.target_points is not None else np.empty((0, 2))
        for source_point, target_point in zip(source_points[:100], target_points[:100]):
            start = tuple(np.rint(source_point).astype(int))
            end = tuple(np.rint(target_point + np.array([source_small.shape[1], 0])).astype(int))
            cv2.line(canvas, start, end, (80, 220, 120), 1, cv2.LINE_AA)
        cv2.imwrite(str(diagnostic / f"correspondences_{pair.source_frame}_{pair.target_frame}.png"), canvas)


def _fallback_metadata(request: dict[str, Any], captured: int, rejected: list[dict[str, Any]], reasons: list[str], timings: dict[str, float]) -> dict[str, Any]:
    for stage in (
        "view_preprocessing",
        "shape_fitting",
        "pairwise_rotation_estimation",
        "map_projection",
        "defect_area_calculation",
    ):
        timings.setdefault(stage, 0.0)
    return {
        "track_id": int(request.get("trackId", -1)),
        "session_id": str(request.get("sessionId", "")),
        "model_type": str(request.get("config", {}).get("shape_model", "oblate")),
        "estimated_radii": None,
        "status": "fallback",
        "coverage_degrees": 0.0,
        "coverage_ratio": 0.0,
        "observed_area_ratio": 0.0,
        "map_resolution": None,
        "latitude_range_degrees": [
            -float(request.get("config", {}).get("latitude_limit_degrees", 60.0)),
            float(request.get("config", {}).get("latitude_limit_degrees", 60.0)),
        ],
        "captured_views": captured,
        "accepted_views": 0,
        "rejected_views": len(rejected),
        "accepted_frame_numbers": [],
        "rejected_frames": rejected,
        "pairwise_rotations": [],
        "reliable_pairs": 0,
        "unreliable_pairs": 0,
        "mean_step_degrees": 0.0,
        "standard_deviation_step_degrees": 0.0,
        "inter_view_step_degrees": [],
        "missing_gap_degrees": [],
        "alignment_confidence": 0.0,
        "defects": {},
        "total_known_defect_coverage": 0.0,
        "total_known_defect_observed_coverage": 0.0,
        "total_known_defect_target_coverage": 0.0,
        "grade": request.get("fallbackGrade"),
        "grade_confidence": float(request.get("fallbackGradeConfidence", 0.0)),
        "incomplete_evidence": True,
        "failure_reasons": reasons,
        "timings_ms": timings,
    }


def map_surface_request(request: dict[str, Any]) -> dict[str, Any]:
    """Finalize one track and persist its fused maps and metadata."""
    total_started = time.perf_counter()
    config = SurfaceMappingConfig.from_dict(request.get("config"))
    output = Path(str(request.get("outputDirectory", ""))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = list(request.get("views", []))
    timings: dict[str, float] = {}
    rejected: list[dict[str, Any]] = [dict(item) for item in request.get("captureRejections", [])]

    preprocessing_started = time.perf_counter()
    prepared: list[PreparedView] = []
    for record in records:
        try:
            prepared.append(_prepare_view(record, config))
        except Exception as error:
            rejected.append({"frame": int(record.get("frameId", -1)), "reason": str(error)})
    prepared.sort(key=lambda item: item.frame_id)
    prepared, geometric_rejections = _reject_geometric_outliers(prepared, config)
    rejected.extend(geometric_rejections)
    eligible = prepared
    prepared = _limit_views(eligible, config)
    selected_ids = {id(view) for view in prepared}
    rejected.extend(
        {"frame": view.frame_id, "reason": "not selected: preferred angular spacing or mapping view limit"}
        for view in eligible if id(view) not in selected_ids
    )
    timings["view_preprocessing"] = (time.perf_counter() - preprocessing_started) * 1000.0
    if len(prepared) < config.minimum_mapping_views:
        metadata = _fallback_metadata(
            request, len(records), rejected,
            [f"only {len(prepared)} usable views; {config.minimum_mapping_views} required"], timings,
        )
        metadata["accepted_frame_numbers"] = [item.frame_id for item in prepared]
        metadata["accepted_views"] = len(prepared)
        timings["total_finalisation"] = (time.perf_counter() - total_started) * 1000.0
        (output / "surface_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    shape_started = time.perf_counter()
    shape = fit_spheroid([view.mask for view in prepared], config)
    timings["shape_fitting"] = (time.perf_counter() - shape_started) * 1000.0
    rotation_started = time.perf_counter()
    active, accumulated, pairs, alignment_rejections = _align_views(prepared, shape, config)
    rejected.extend(alignment_rejections)
    timings["pairwise_rotation_estimation"] = (time.perf_counter() - rotation_started) * 1000.0
    reliable_all = [item for item in pairs if item.reliable]
    if len(active) < config.minimum_mapping_views or not reliable_all:
        metadata = _fallback_metadata(
            request, len(records), rejected,
            ["too few continuously aligned views"], timings,
        )
        metadata["pairwise_rotations"] = [item.metadata() for item in pairs]
        metadata["model_type"] = shape.model
        metadata["estimated_radii"] = {
            "x": shape.radii[0], "y": shape.radii[1], "z": shape.radii[2], "A": shape.a, "B": shape.b,
        }
        metadata["accepted_frame_numbers"] = [item.frame_id for item in active]
        metadata["accepted_views"] = len(active)
        metadata["reliable_pairs"] = len(reliable_all)
        metadata["unreliable_pairs"] = len(pairs) - len(reliable_all)
        if active:
            # A geometrically untrusted map is still useful in diagnostics, but
            # remains explicitly fallback/incomplete and retains unknown pixels.
            projection_started = time.perf_counter()
            bases = _view_bases(shape, accumulated, active, prepared)
            centers = _unwrap_centers(bases, reliable_all, config.rotation_direction)
            projected = _project_maps(active, bases, centers, [item.confidence for item in reliable_all], shape, config, False)
            timings["map_projection"] = (time.perf_counter() - projection_started) * 1000.0
            area_started = time.perf_counter()
            per_class, total_defect, observed_ratio = defect_area_statistics(projected, config.defect_probability_threshold)
            timings["defect_area_calculation"] = (time.perf_counter() - area_started) * 1000.0
            _save_png(output / "surface_rgb.png", projected["rgb"])
            _save_png(output / "surface_observed_mask.png", projected["observed"].astype(np.uint8) * 255)
            _save_png(output / "surface_defect_overlay.png", projected["rgb"])
            confidence_u16 = np.round(np.clip(projected["confidence"], 0.0, 1.0) * 65535.0).astype(np.uint16)
            _save_png(output / "surface_alignment_confidence.png", confidence_u16)
            metadata.update({
                "observed_area_ratio": observed_ratio,
                "map_resolution": {"width": projected["resolution"][0], "height": projected["resolution"][1]},
                "defects": per_class,
                "total_known_defect_coverage": total_defect,
                "total_known_defect_observed_coverage": total_defect,
                "total_known_defect_target_coverage": total_defect * observed_ratio,
                "rgb_map_url": f"/api/pipeline/results/{int(request.get('trackId', -1))}/surface_rgb.png",
                "defect_overlay_url": f"/api/pipeline/results/{int(request.get('trackId', -1))}/surface_defect_overlay.png",
                "observed_mask_url": f"/api/pipeline/results/{int(request.get('trackId', -1))}/surface_observed_mask.png",
                "alignment_confidence_url": f"/api/pipeline/results/{int(request.get('trackId', -1))}/surface_alignment_confidence.png",
            })
        timings["total_finalisation"] = (time.perf_counter() - total_started) * 1000.0
        (output / "surface_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _save_diagnostics(output, prepared, pairs, config)
        return metadata

    bases = _view_bases(shape, accumulated, active, prepared)
    centers = _unwrap_centers(bases, reliable_all, config.rotation_direction)
    aligned_active = active
    active, bases, centers = _best_single_revolution(active, bases, centers, config.rotation_direction, reliable_all)
    selected_frames = {view.frame_id for view in active}
    rejected.extend(
        {"frame": view.frame_id, "reason": "outside selected best 360-degree interval"}
        for view in aligned_active if view.frame_id not in selected_frames
    )
    reliable = [item for item in reliable_all if item.source_frame in selected_frames and item.target_frame in selected_frames]
    first_selected_frame = active[0].frame_id
    last_selected_frame = active[-1].frame_id
    selected_pair_attempts = [
        item for item in pairs
        if first_selected_frame <= item.source_frame <= last_selected_frame
        and first_selected_frame <= item.target_frame <= last_selected_frame
    ]
    if config.rotation_direction < 0:
        centers = -centers
        bases = [basis @ np.diag([-1.0, 1.0, -1.0]) for basis in bases]
    steps = np.abs(np.diff(np.rad2deg(centers)))
    coverage = min(360.0, float(abs(np.rad2deg(centers[-1] - centers[0]))))
    closing_gap = max(0.0, 360.0 - coverage)
    maximum_gap = max([closing_gap, *steps.tolist()]) if len(active) > 1 else 360.0
    missing_gaps = [gap for gap in [*steps.tolist(), closing_gap] if gap > config.preferred_step_degrees]
    reliable_ratio = len(reliable) / max(1, len(selected_pair_attempts))
    alignment_confidence = float(np.mean([item.confidence for item in reliable]))
    geometrically_complete = coverage >= config.target_coverage_degrees and maximum_gap <= config.maximum_step_degrees

    projection_started = time.perf_counter()
    projected = _project_maps(active, bases, centers, [item.confidence for item in reliable], shape, config, geometrically_complete)
    timings["map_projection"] = (time.perf_counter() - projection_started) * 1000.0
    area_started = time.perf_counter()
    per_class, total_defect, observed_ratio = defect_area_statistics(projected, config.defect_probability_threshold)
    timings["defect_area_calculation"] = (time.perf_counter() - area_started) * 1000.0

    status, reasons = mapping_status(
        coverage, maximum_gap, reliable_ratio, observed_ratio, alignment_confidence, config,
    )
    complete = status == "complete"
    grade, grade_class_id, grade_confidence = fused_grade(
        projected,
        config.defect_probability_threshold,
        request.get("fallbackGrade"),
        float(request.get("fallbackGradeConfidence", 0.0)),
    ) if complete else (
        request.get("fallbackGrade"), None, float(request.get("fallbackGradeConfidence", 0.0)),
    )

    _save_png(output / "surface_rgb.png", projected["rgb"])
    _save_png(output / "surface_observed_mask.png", projected["observed"].astype(np.uint8) * 255)
    confidence_u16 = np.round(np.clip(projected["confidence"], 0.0, 1.0) * 65535.0).astype(np.uint16)
    _save_png(output / "surface_alignment_confidence.png", confidence_u16)
    np.save(output / "surface_alignment_confidence.npy", projected["confidence"].astype(np.float32), allow_pickle=False)
    overlay = projected["rgb"].astype(np.float32)
    palette = [(255, 70, 80), (255, 190, 60), (90, 190, 255), (190, 105, 255), (80, 225, 150)]
    probability_files: dict[str, str] = {}
    for index, (name, probability) in enumerate(projected["defects"].items()):
        slug = _safe_slug(name)
        filename = f"surface_defect_{slug}.npy"
        np.save(output / filename, probability.astype(np.float32), allow_pickle=False)
        probability_files[name] = filename
        alpha = (probability * projected["observed"] * 0.62)[..., np.newaxis]
        colour = np.asarray(palette[index % len(palette)], np.float32)
        overlay = overlay * (1.0 - alpha) + colour * alpha
    _save_png(output / "surface_defect_overlay.png", np.clip(overlay, 0, 255).astype(np.uint8))
    _save_diagnostics(output, prepared, pairs, config)

    track_id = int(request.get("trackId", -1))
    base_url = f"/api/pipeline/results/{track_id}"
    probability_map_urls = {name: f"{base_url}/{filename}" for name, filename in probability_files.items()}
    metadata: dict[str, Any] = {
        "track_id": track_id,
        "session_id": str(request.get("sessionId", "")),
        "model_type": shape.model,
        "estimated_radii": {"x": shape.radii[0], "y": shape.radii[1], "z": shape.radii[2], "A": shape.a, "B": shape.b},
        "captured_views": len(records),
        "accepted_views": len(active),
        "rejected_views": len(rejected),
        "accepted_frame_numbers": [item.frame_id for item in active],
        "rejected_frames": rejected,
        "pairwise_rotations": [item.metadata() for item in pairs],
        "reliable_pairs": len(reliable_all),
        "unreliable_pairs": len(pairs) - len(reliable_all),
        "selected_reliable_pair_ratio": reliable_ratio,
        "coverage_degrees": coverage,
        "coverage_ratio": coverage / 360.0,
        "maximum_gap_degrees": maximum_gap,
        "observed_area_ratio": observed_ratio,
        "map_resolution": {"width": projected["resolution"][0], "height": projected["resolution"][1]},
        "latitude_range_degrees": [-config.latitude_limit_degrees, config.latitude_limit_degrees],
        "mean_step_degrees": float(np.mean(steps)) if steps.size else 0.0,
        "standard_deviation_step_degrees": float(np.std(steps)) if steps.size else 0.0,
        "inter_view_step_degrees": steps.tolist(),
        "missing_gap_degrees": missing_gaps,
        "alignment_confidence": alignment_confidence,
        "defects": per_class,
        "total_known_defect_coverage": total_defect,
        "total_known_defect_observed_coverage": total_defect,
        "total_known_defect_target_coverage": total_defect * observed_ratio,
        "defect_denominators": {"observed": "reliably observed target band", "target": "complete configured latitude band"},
        "probability_files": probability_files,
        "probability_map_urls": probability_map_urls,
        "grade": grade,
        "grade_class_id": grade_class_id,
        "grade_confidence": grade_confidence,
        "status": status,
        "incomplete_evidence": not complete,
        "failure_reasons": reasons,
        "rgb_map_url": f"{base_url}/surface_rgb.png",
        "defect_overlay_url": f"{base_url}/surface_defect_overlay.png",
        "observed_mask_url": f"{base_url}/surface_observed_mask.png",
        "alignment_confidence_url": f"{base_url}/surface_alignment_confidence.png",
        "timings_ms": timings,
    }
    timings["total_finalisation"] = (time.perf_counter() - total_started) * 1000.0
    (output / "surface_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _diagnostic_request(input_folder: Path, output_folder: Path, config_path: Path | None) -> dict[str, Any]:
    configuration: dict[str, Any] = {"diagnostic": True, "minimum_mapping_views": 2}
    if config_path:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        configuration.update(loaded.get("surface_mapping", loaded))
        configuration["diagnostic"] = True
    views = []
    paths = sorted([*input_folder.glob("*.jpg"), *input_folder.glob("*.jpeg"), *input_folder.glob("*.png")], key=_natural_path_key)
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        views.append({
            "trackId": 0,
            "frameId": index + 1,
            "timestamp": "",
            "cropPath": str(path.resolve()),
            "detectionConfidence": 1.0,
            "focusScore": _focus_score(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            "originalBoundingBox": {"x": 0, "y": 0, "width": width, "height": height},
            "cropTransform": {"offsetX": 0, "offsetY": 0, "scaleX": 1, "scaleY": 1},
            "defectInstances": [],
        })
    return {
        "trackId": 0,
        "sessionId": "diagnostic",
        "outputDirectory": str(output_folder.resolve()),
        "fallbackGrade": None,
        "fallbackGradeConfidence": 0.0,
        "config": configuration,
        "views": views,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose software-only naartjie surface mapping")
    parser.add_argument("--input-folder", required=True, type=Path, help="Folder containing time-ordered JPG/PNG crops")
    parser.add_argument("--output-folder", required=True, type=Path, help="Folder for diagnostic maps and overlays")
    parser.add_argument("--config", type=Path, help="Pipeline JSON or a surface_mapping JSON object")
    args = parser.parse_args()
    result = map_surface_request(_diagnostic_request(args.input_folder, args.output_folder, args.config))
    print(json.dumps({
        "status": result["status"],
        "coverage_degrees": result["coverage_degrees"],
        "accepted_views": result["accepted_views"],
        "failure_reasons": result["failure_reasons"],
        "timings_ms": result["timings_ms"],
    }, indent=2))
    return 0 if result["status"] in {"complete", "partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
