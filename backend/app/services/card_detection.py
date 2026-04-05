"""
Card boundary detection and perspective normalization using OpenCV.

Pipeline:
1. Decode image from upload bytes.
2. Downscale very large inputs (keeps contour stable, preserves aspect).
3. Bilateral + blur to reduce noise while keeping edges.
4. Canny + morphological close, then **several** ``approxPolyDP`` epsilons per contour.
5. Prefer quads that are large and rectangular but **not** the entire photo frame
   (screenshots / UI borders).
6. Optional second Canny (lower thresholds) if the first pass finds nothing usable.

Future extension points:
- Replace contour heuristic with a learned keypoint or segmentation model (see ``app.ml``).
- Multi-card scenes: rank candidates and return N warped crops.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.core.config import settings

_MIN_ASPECT = 0.52
_MAX_ASPECT = 0.88
_MIN_AREA_FRAC = 0.045
# Ignore quads covering more than this fraction of the image when smaller candidates exist
_MAX_AREA_FRAC = 0.93


def decode_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw file bytes to a BGR ``numpy`` array (OpenCV native order)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image: unsupported or corrupt file.")
    return image


def _resize_for_detection(image: np.ndarray, max_dim: int = 1800) -> tuple[np.ndarray, float]:
    """Return possibly resized image and scale factor to map coords back to original."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _quad_aspect_ratio(ordered: np.ndarray) -> float:
    """Width / height of the quadrilateral using average edge lengths."""
    (tl, tr, br, bl) = ordered
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    width = 0.5 * (width_top + width_bottom)
    height = 0.5 * (height_left + height_right)
    if height < 1e-6:
        return 0.0
    return width / height


def _score_quad(ordered: np.ndarray, image_area: float) -> float:
    area = float(cv2.contourArea(ordered))
    if area < _MIN_AREA_FRAC * image_area:
        return -1.0
    ar = _quad_aspect_ratio(ordered)
    if ar < _MIN_ASPECT or ar > _MAX_ASPECT:
        return -1.0
    hull = cv2.convexHull(ordered)
    hull_area = float(cv2.contourArea(hull))
    rectangularity = area / hull_area if hull_area > 1e-6 else 0.0
    ar_bonus = 1.0 - min(abs(ar - 0.71) / 0.35, 0.3)
    return area * rectangularity * (0.78 + 0.22 * ar_bonus)


def _quads_from_edges(work: np.ndarray, lo: int, hi: int) -> list[tuple[np.ndarray, float, float]]:
    """Return [(ordered, score, area_frac), ...] for valid quads."""
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 64, 64)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, lo, hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w = work.shape[:2]
    image_area = float(h * w)
    out: list[tuple[np.ndarray, float, float]] = []

    for cnt in contours[:35]:
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-3:
            continue
        for eps in (0.017, 0.024, 0.032, 0.045):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            pts = approx.reshape(4, 2).astype(np.float32)
            ordered = _order_quad_points(pts)
            sc = _score_quad(ordered, image_area)
            if sc < 0:
                continue
            af = float(cv2.contourArea(ordered)) / image_area
            out.append((ordered, sc, af))
            break
    return out


def find_card_quad(image: np.ndarray) -> tuple[np.ndarray | None, str]:
    """
    Find a 4-point polygon approximating the card in ``image`` (BGR).

    Returns (4x2 float32 points in **original** image coordinates, or None), confidence.
    """
    work, scale = _resize_for_detection(image)
    h, w = work.shape[:2]
    image_area = float(h * w)

    candidates = _quads_from_edges(work, 50, 145)
    if len([c for c in candidates if c[2] < _MAX_AREA_FRAC]) < 1:
        candidates.extend(_quads_from_edges(work, 32, 95))

    if not candidates:
        return None, "low"

    smaller = [(o, s, a) for o, s, a in candidates if a < _MAX_AREA_FRAC]
    if smaller:
        candidates = smaller

    candidates.sort(key=lambda t: -t[1])
    best_ordered, best_sc, area_frac = candidates[0]

    if scale != 1.0:
        best_ordered = (best_ordered / scale).astype(np.float32)

    full_area = float(image.shape[0] * image.shape[1])
    coverage = cv2.contourArea(best_ordered) / full_area

    hull = cv2.convexHull(best_ordered)
    rectangularity = cv2.contourArea(best_ordered) / max(float(cv2.contourArea(hull)), 1e-6)

    if coverage > 0.24 and rectangularity > 0.88:
        confidence = "high"
    elif coverage > 0.11:
        confidence = "medium"
    else:
        confidence = "low"

    return best_ordered, confidence


def fallback_full_frame_quad(h: int, w: int) -> np.ndarray:
    """Use image corners when no card contour is found (degraded mode)."""
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def deskew_warped_card(bgr: np.ndarray, max_deg: float = 2.5) -> np.ndarray:
    """
    Correct small in-plane rotation left after perspective warp (imprecise quad).

    Uses Canny + probabilistic Hough on a **border band** only, keeps segments that are
    nearly horizontal, and applies the median angle as an affine rotation.
    """
    h, w = bgr.shape[:2]
    if h < 64 or w < 64:
        return bgr

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    band = max(4, min(h // 12, w // 12, 48))
    edge = np.zeros_like(gray)
    edge[:band, :] = gray[:band, :]
    edge[-band:, :] = gray[-band:, :]
    edge[:, :band] = np.maximum(edge[:, :band], gray[:, :band])
    edge[:, -band:] = np.maximum(edge[:, -band:], gray[:, -band:])

    edges = cv2.Canny(edge, 45, 130)
    min_len = max(band * 2, int(0.08 * min(h, w)))
    lines = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 180.0,
        threshold=max(24, min_len // 2),
        minLineLength=min_len,
        maxLineGap=10,
    )
    if lines is None:
        return bgr

    angles: list[float] = []
    for ln in lines[:, 0]:
        x1, y1, x2, y2 = ln
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1.0:
            continue
        if abs(dy) > abs(dx) * 0.35:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        while ang > 90.0:
            ang -= 180.0
        while ang < -90.0:
            ang += 180.0
        if abs(ang) < max_deg * 1.6:
            angles.append(ang)

    if len(angles) < 5:
        return bgr

    median_ang = float(np.median(np.array(angles, dtype=np.float32)))
    if abs(median_ang) < 0.06 or abs(median_ang) > max_deg:
        return bgr

    ctr = (w * 0.5, h * 0.5)
    m = cv2.getRotationMatrix2D(ctr, median_ang, 1.0)
    return cv2.warpAffine(
        bgr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_card_to_rectangle(
    image: np.ndarray,
    quad: np.ndarray,
    target_height: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Apply perspective transform so ``quad`` becomes a flat rectangle."""
    ordered = _order_quad_points(quad.astype(np.float32))
    ar = _quad_aspect_ratio(ordered)
    if ar <= 0:
        ar = 0.7
    out_h = int(target_height)
    out_w = max(1, int(round(out_h * ar)))

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    return warped, (out_w, out_h)


def extract_normalized_card(image: np.ndarray, target_height: int) -> tuple[np.ndarray, str]:
    """
    Detect card (or fallback), warp to top-down view.

    Returns ``(warped_bgr, detection_confidence)``.
    """
    quad, conf = find_card_quad(image)
    if quad is None:
        h, w = image.shape[:2]
        quad = fallback_full_frame_quad(h, w)
        conf = "low"
    warped, _ = warp_card_to_rectangle(image, quad, target_height=target_height)
    if settings.warp_deskew:
        warped = deskew_warped_card(warped)
    return warped, conf
