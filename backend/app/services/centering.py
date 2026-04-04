"""
Centering measurement (Phase 1: heuristic / stub-quality).

Interprets the **warped, top-down** card crop: outer boundary is the image frame; the
printed inner frame (artwork border) is approximated via edge density scans from each side.
This is intentionally simple — replace with sub-pixel border models or ML segmentation later.

Future hooks:
- ``measure_defects`` / surface maps can consume the same ``warped`` tensor alongside these margins.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.schemas import AnalyzeCardResponse, CenteringMetrics, EstimatedGrades


def _edge_strength_map(gray: np.ndarray) -> np.ndarray:
    """Sobel magnitude as a float32 map in [0, ~1]."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m = float(mag.max()) or 1.0
    return mag / m


def _scan_margin_from_left(strength: np.ndarray, threshold_ratio: float = 0.35) -> int:
    """Walk columns left→right until mean edge strength exceeds a fraction of global max column mean."""
    h, w = strength.shape
    col_mean = strength.mean(axis=0)
    peak = float(col_mean.max()) or 1.0
    thresh = peak * threshold_ratio
    for x in range(w):
        if col_mean[x] >= thresh:
            return x
    return 0


def _scan_margin_from_right(strength: np.ndarray, threshold_ratio: float = 0.35) -> int:
    col_mean = strength.mean(axis=0)
    peak = float(col_mean.max()) or 1.0
    thresh = peak * threshold_ratio
    w = strength.shape[1]
    for x in range(w - 1, -1, -1):
        if col_mean[x] >= thresh:
            return w - 1 - x
    return 0


def _scan_margin_from_top(strength: np.ndarray, threshold_ratio: float = 0.35) -> int:
    row_mean = strength.mean(axis=1)
    peak = float(row_mean.max()) or 1.0
    thresh = peak * threshold_ratio
    h = strength.shape[0]
    for y in range(h):
        if row_mean[y] >= thresh:
            return y
    return 0


def _scan_margin_from_bottom(strength: np.ndarray, threshold_ratio: float = 0.35) -> int:
    row_mean = strength.mean(axis=1)
    peak = float(row_mean.max()) or 1.0
    thresh = peak * threshold_ratio
    h = strength.shape[0]
    for y in range(h - 1, -1, -1):
        if row_mean[y] >= thresh:
            return h - 1 - y
    return 0


def measure_centering_margins_px(warped_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """
    Estimate left, right, top, bottom border thickness in pixels.

    Uses blurred grayscale + Sobel edge energy; scans from each side for the first
    strong edge band (inner print boundary). Falls back to mild defaults if the map is flat.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    strength = _edge_strength_map(gray)

    left = _scan_margin_from_left(strength)
    right = _scan_margin_from_right(strength)
    top = _scan_margin_from_top(strength)
    bottom = _scan_margin_from_bottom(strength)

    h, w = gray.shape[:2]
    # Guardrails: margins should not exceed half the dimension
    left = int(np.clip(left, 0, w // 2 - 1))
    right = int(np.clip(right, 0, w // 2 - 1))
    top = int(np.clip(top, 0, h // 2 - 1))
    bottom = int(np.clip(bottom, 0, h // 2 - 1))

    if left + right >= w or top + bottom >= h:
        # Flat edge map — neutral split
        return w // 4, w // 4, h // 4, h // 4

    return left, right, top, bottom


def margins_to_ratio_string(left: int, right: int) -> str:
    """Format ``'55/45'`` style string from integer margin counts."""
    total = left + right
    if total <= 0:
        return "50/50"
    lp = int(round(100.0 * left / total))
    rp = 100 - lp
    return f"{lp}/{rp}"


def margins_to_ratio_string_tb(top: int, bottom: int) -> str:
    total = top + bottom
    if total <= 0:
        return "50/50"
    tp = int(round(100.0 * top / total))
    bp = 100 - tp
    return f"{tp}/{bp}"


def _centering_deviation(left: int, right: int, top: int, bottom: int) -> float:
    """
    Scalar “how far from perfect” score in percentage points (0 = perfect 50/50 both axes).

    Uses worst-axis imbalance: max(|L-R|/sum, |T-B|/sum) scaled to 0–100 style deviation.
    """
    lr_total = left + right
    tb_total = top + bottom
    lr_dev = abs(left - right) / lr_total if lr_total else 0.0
    tb_dev = abs(top - bottom) / tb_total if tb_total else 0.0
    return float(max(lr_dev, tb_dev) * 100.0)


def approximate_psa_from_centering(deviation_pct: float) -> float:
    """
    Map centering deviation to a coarse PSA-like 1–10 (whole numbers).

    This is **not** official — PSA uses human graders and holistic criteria.
    """
    if deviation_pct <= 2.0:
        return 10.0
    if deviation_pct <= 4.5:
        return 9.0
    if deviation_pct <= 7.0:
        return 8.0
    if deviation_pct <= 10.0:
        return 7.0
    if deviation_pct <= 14.0:
        return 6.0
    if deviation_pct <= 20.0:
        return 5.0
    if deviation_pct <= 28.0:
        return 4.0
    if deviation_pct <= 38.0:
        return 3.0
    if deviation_pct <= 50.0:
        return 2.0
    return 1.0


def approximate_bgs_from_centering(deviation_pct: float) -> float:
    """Slightly finer steps with half-point spacing vs. PSA heuristic."""
    base = approximate_psa_from_centering(deviation_pct)
    if base >= 10:
        return 10.0
    # nudge toward common BGS reporting with .5 steps
    fractional = 0.5 if (deviation_pct % 3.0) > 1.5 else 0.0
    return min(9.5, max(1.0, base + fractional))


def approximate_cgc_from_centering(deviation_pct: float) -> float:
    """CGC-style whole-number proxy; tracks PSA mapping closely for MVP."""
    return approximate_psa_from_centering(deviation_pct)


def build_analyze_response(
    warped_bgr: np.ndarray,
    detection_confidence: str,
) -> AnalyzeCardResponse:
    """Run stub centering + grade estimates on a normalized card image."""
    left, right, top, bottom = measure_centering_margins_px(warped_bgr)
    lr_str = margins_to_ratio_string(left, right)
    tb_str = margins_to_ratio_string_tb(top, bottom)
    dev = _centering_deviation(left, right, top, bottom)

    psa = approximate_psa_from_centering(dev)
    bgs = approximate_bgs_from_centering(dev)
    cgc = approximate_cgc_from_centering(dev)

    hh, ww = warped_bgr.shape[:2]
    return AnalyzeCardResponse(
        centering=CenteringMetrics(left_right=lr_str, top_bottom=tb_str),
        estimated_grades=EstimatedGrades(PSA=psa, BGS=bgs, CGC=cgc),
        warp_width=ww,
        warp_height=hh,
        detection_confidence=detection_confidence,
    )
