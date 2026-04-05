"""
Edge-projection centering: outer warp boundary → inner frame edge.

Measures distance from the **physical edges of the warped image** (the card quad) to
the strongest **in-band** gradient peaks, using symmetric thresholds. Search is limited
to outer 25% of width/height so interior artwork edges are ignored.
"""

from __future__ import annotations

import cv2
import numpy as np

# Outer fraction of each axis where we look for the inner border (avoid art).
_SIDE_FRAC = 0.25
# Wide vertical band for LR (corners excluded enough by side search cap).
_CORE_Y0_FRAC = 0.12
_CORE_Y1_FRAC = 0.88
_CORE_X0_FRAC = 0.12
_CORE_X1_FRAC = 0.88


def _smooth_1d(sig: np.ndarray, window: int) -> np.ndarray:
    w = max(3, window | 1)
    if sig.size < w:
        return sig.astype(np.float32, copy=False)
    x = sig.astype(np.float32, copy=False).reshape(1, -1)
    return cv2.blur(x, (w, 1)).flatten()


def _first_cross_from_left(sig: np.ndarray, x0: int, x1: int, thresh: float) -> float | None:
    """First index in [x0, x1) where sig >= thresh, sub-pixel linear interp."""
    x1 = min(x1, sig.size)
    x0 = max(0, x0)
    for x in range(x0, x1):
        if sig[x] >= thresh:
            if x == 0:
                return 0.0
            a, b = float(sig[x - 1]), float(sig[x])
            if b <= a:
                return float(x)
            t = (thresh - a) / (b - a + 1e-9)
            return float(x - 1) + float(np.clip(t, 0.0, 1.0))
    return None


def _first_cross_from_right(sig: np.ndarray, x0: int, x1: int, thresh: float) -> float | None:
    """Sub-pixel distance from index (len-1) to crossing; search x1-1 down to x0."""
    x1 = min(x1, sig.size)
    x0 = max(0, x0)
    w = sig.size
    for x in range(x1 - 1, x0 - 1, -1):
        if sig[x] >= thresh:
            if x >= w - 1:
                return 0.0
            outer = float(sig[x + 1])
            inner = float(sig[x])
            if inner <= outer + 1e-9:
                return float(w - 1 - x)
            t = (thresh - outer) / (inner - outer + 1e-9)
            t = float(np.clip(t, 0.0, 1.0))
            crossing_from_left = (x + 1) + t * (float(x) - float(x + 1))
            return float(w - 1) - crossing_from_left
    return None


def _first_cross_from_top(sig: np.ndarray, y0: int, y1: int, thresh: float) -> float | None:
    y1 = min(y1, sig.size)
    y0 = max(0, y0)
    for y in range(y0, y1):
        if sig[y] >= thresh:
            if y == 0:
                return 0.0
            a, b = float(sig[y - 1]), float(sig[y])
            if b <= a:
                return float(y)
            t = (thresh - a) / (b - a + 1e-9)
            return float(y - 1) + float(np.clip(t, 0.0, 1.0))
    return None


def _first_cross_from_bottom(sig: np.ndarray, y0: int, y1: int, thresh: float) -> float | None:
    y1 = min(y1, sig.size)
    y0 = max(0, y0)
    h = sig.size
    for y in range(y1 - 1, y0 - 1, -1):
        if sig[y] >= thresh:
            if y >= h - 1:
                return 0.0
            outer = float(sig[y + 1])
            inner = float(sig[y])
            if inner <= outer + 1e-9:
                return float(h - 1 - y)
            t = (thresh - outer) / (inner - outer + 1e-9)
            t = float(np.clip(t, 0.0, 1.0))
            crossing_from_top = (y + 1) + t * (float(y) - float(y + 1))
            return float(h - 1) - crossing_from_top
    return None


def measure_margins_edge_projection(
    warped_bgr: np.ndarray,
    threshold_ratio_lr: float = 0.38,
    threshold_ratio_tb: float = 0.38,
) -> tuple[float, float, float, float, dict]:
    """
    Return ``(left, right, top, bottom, debug_dict)`` in pixels from warp edges.

    ``debug_dict`` holds 1D signals and thresholds for visualization.
    """
    h, w = warped_bgr.shape[:2]
    if h < 32 or w < 32:
        fb = (float(w) / 4.0, float(w) / 4.0, float(h) / 4.0, float(h) / 4.0)
        return (
            *fb,
            {
                "rejected": True,
                "reason": "image_too_small",
                "col_sig": np.zeros(w, dtype=np.float32),
                "row_sig": np.zeros(h, dtype=np.float32),
            },
        )

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)

    y_a, y_b = int(_CORE_Y0_FRAC * h), int(_CORE_Y1_FRAC * h)
    y_a, y_b = max(0, y_a), min(h, max(y_a + 1, y_b))
    x_a, x_b = int(_CORE_X0_FRAC * w), int(_CORE_X1_FRAC * w)
    x_a, x_b = max(0, x_a), min(w, max(x_a + 1, x_b))

    # LR: mean |Gx| over core rows (stable on uniform yellow border).
    col_sig = np.mean(np.abs(gx[y_a:y_b, :]), axis=0).astype(np.float32)
    sw = max(5, min(w // 32, 21) | 1)
    col_sig = _smooth_1d(col_sig, sw)
    cmax = float(np.max(col_sig)) or 1.0
    col_n = col_sig / cmax

    w_side = max(8, int(_SIDE_FRAC * w))
    h_side = max(8, int(_SIDE_FRAC * h))
    # Ignore a few pixels at the warp boundary (ringing / replicate border) so we do
    # not threshold immediately at x=0 and report tiny equal margins → false PSA 10.
    rim_x = max(2, min(int(0.008 * w), w_side // 4))
    rim_y = max(2, min(int(0.008 * h), h_side // 4))

    left_slab = col_n[rim_x:w_side]
    right_slab = col_n[w - w_side : w - rim_x] if rim_x > 0 else col_n[w - w_side :]
    peak_lr = max(float(np.max(left_slab)), float(np.max(right_slab)), 1e-6)
    thresh_lr = threshold_ratio_lr * peak_lr

    # Strongest vertical edge in each border band (yellow→art), not first threshold
    # crossing (crossings latch onto foil / text strokes too easily).
    li = int(np.argmax(left_slab))
    left_peak = float(rim_x + li)
    ridx = int(np.argmax(right_slab))
    r_col = (w - w_side) + ridx
    right_peak = float((w - 1) - r_col)

    left_cross = _first_cross_from_left(col_n, rim_x, w_side, thresh_lr)
    right_cross = _first_cross_from_right(col_n, w - w_side, w - rim_x, thresh_lr)
    agree = 0.18 * float(w_side)
    if left_cross is not None and abs(left_cross - left_peak) <= agree:
        left = float(np.clip(left_cross, 0.0, w / 2.0 - 1.0))
    else:
        left = float(np.clip(left_peak, 0.0, w / 2.0 - 1.0))
    if right_cross is not None and abs(right_cross - right_peak) <= agree:
        right = float(np.clip(right_cross, 0.0, w / 2.0 - 1.0))
    else:
        right = float(np.clip(right_peak, 0.0, w / 2.0 - 1.0))

    # TB: mean |Gy| over core columns.
    row_sig = np.mean(np.abs(gy[:, x_a:x_b]), axis=1).astype(np.float32)
    sh = max(5, min(h // 28, 25) | 1)
    row_sig = _smooth_1d(row_sig, sh)
    rmax = float(np.max(row_sig)) or 1.0
    row_n = row_sig / rmax

    top_slab = row_n[rim_y:h_side]
    bot_slab = row_n[h - h_side : h - rim_y] if rim_y > 0 else row_n[h - h_side :]
    peak_tb = max(float(np.max(top_slab)), float(np.max(bot_slab)), 1e-6)
    thresh_tb = threshold_ratio_tb * peak_tb

    ti = int(np.argmax(top_slab))
    top_peak = float(rim_y + ti)
    bidx = int(np.argmax(bot_slab))
    r_row = (h - h_side) + bidx
    bot_peak = float((h - 1) - r_row)

    top_cross = _first_cross_from_top(row_n, rim_y, h_side, thresh_tb)
    bot_cross = _first_cross_from_bottom(row_n, h - h_side, h - rim_y, thresh_tb)
    agree_tb = 0.18 * float(h_side)
    if top_cross is not None and abs(top_cross - top_peak) <= agree_tb:
        top = float(np.clip(top_cross, 0.0, h / 2.0 - 1.0))
    else:
        top = float(np.clip(top_peak, 0.0, h / 2.0 - 1.0))
    if bot_cross is not None and abs(bot_cross - bot_peak) <= agree_tb:
        bottom = float(np.clip(bot_cross, 0.0, h / 2.0 - 1.0))
    else:
        bottom = float(np.clip(bot_peak, 0.0, h / 2.0 - 1.0))

    meta: dict = {
        "rejected": False,
        "col_sig": col_sig,
        "row_sig": row_sig,
        "col_n": col_n,
        "row_n": row_n,
        "thresh_lr": thresh_lr,
        "thresh_tb": thresh_tb,
        "w_side": w_side,
        "h_side": h_side,
        "peak_lr": peak_lr,
        "peak_tb": peak_tb,
    }

    # Sanity: total measurable frame must be meaningful vs card size.
    min_lr = max(16.0, 0.05 * float(w))
    min_tb = max(16.0, 0.05 * float(h))
    if left + right < min_lr or top + bottom < min_tb:
        meta["rejected"] = True
        meta["reason"] = "margin_sum_too_small"
        # Keep raw measurements; grading applies a floor when ``rejected`` (see centering.py).
    elif left + right >= w or top + bottom >= h:
        meta["rejected"] = True
        meta["reason"] = "margins_overflow"
        left = right = float(w) / 4.0
        top = bottom = float(h) / 4.0

    return left, right, top, bottom, meta
