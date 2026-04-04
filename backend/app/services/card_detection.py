"""
Card boundary detection and perspective normalization using OpenCV.

Pipeline:
1. Decode image from upload bytes.
2. Downscale very large inputs for stable contour finding.
3. Canny edges + morphological close to connect card outline.
4. Select the largest plausible quadrilateral contour as the card.
5. Apply a perspective transform to a top-down rectangle.

Future extension points:
- Replace contour heuristic with a learned keypoint or segmentation model (see ``app.ml``).
- Multi-card scenes: rank candidates and return N warped crops.
"""

from __future__ import annotations

import cv2
import numpy as np

# Reasonable bounds for trading-card-like aspect ratio (width / height)
_MIN_ASPECT = 0.55
_MAX_ASPECT = 0.85


def decode_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw file bytes to a BGR ``numpy`` array (OpenCV native order)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image — unsupported or corrupt file.")
    return image


def _resize_for_detection(image: np.ndarray, max_dim: int = 1600) -> tuple[np.ndarray, float]:
    """Return possibly resized image and scale factor applied to map coords back to original."""
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
    """
    Order four points as: top-left, top-right, bottom-right, bottom-left.

    ``pts`` shape (4, 2).
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
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


def find_card_quad(image: np.ndarray) -> tuple[np.ndarray | None, str]:
    """
    Find a 4-point polygon approximating the card in ``image`` (BGR).

    Returns (4x2 float32 points in **original** image coordinates, or None), confidence label.
    """
    work, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w = work.shape[:2]
    image_area = float(h * w)
    best: np.ndarray | None = None
    best_score = -1.0
    confidence = "low"

    for cnt in contours[:25]:
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        ordered = _order_quad_points(pts)
        ar = _quad_aspect_ratio(ordered)
        if ar < _MIN_ASPECT or ar > _MAX_ASPECT:
            continue
        area = cv2.contourArea(ordered)
        if area < 0.05 * image_area:
            continue
        # Prefer large contours with reasonable rectangularity
        rect_area = cv2.contourArea(ordered)
        hull = cv2.convexHull(ordered)
        hull_area = cv2.contourArea(hull)
        rectangularity = rect_area / hull_area if hull_area > 1e-6 else 0.0
        score = area * rectangularity
        if score > best_score:
            best_score = score
            best = ordered

    if best is None:
        return None, confidence

    # Map coordinates back to original resolution
    if scale != 1.0:
        best = (best / scale).astype(np.float32)

    # Confidence heuristic from rectangularity and coverage
    coverage = cv2.contourArea(best) / float(image.shape[0] * image.shape[1])
    if coverage > 0.25 and best_score > 0:
        confidence = "high"
    elif coverage > 0.12:
        confidence = "medium"
    else:
        confidence = "low"

    return best, confidence


def fallback_full_frame_quad(h: int, w: int) -> np.ndarray:
    """Use image corners when no card contour is found (degraded mode)."""
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def warp_card_to_rectangle(
    image: np.ndarray,
    quad: np.ndarray,
    target_height: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Apply perspective transform so ``quad`` becomes a flat rectangle.

    Returns (warped BGR image, (out_w, out_h)).
    """
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
    return warped, conf
