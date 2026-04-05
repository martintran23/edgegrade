"""
Edge-projection centering: outer warp boundary → inner frame edge.

Measures distance from the **physical edges of the warped image** (the card quad) to
the inner frame using mean ``|G|`` in a core band, then **parabolic sub-pixel** peaks on
that 1D profile (tight window around the coarse seam). A small calibrated inward bias
removes residual blur/Scharr shift on sharp step edges. Search stays in the outer 25%.
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

# Gaussian (5×5) + Scharr + smoothed projection peaks sit slightly **inside** the true
# print edge on sharp step borders. Parabolic subpixel refinement on |G| reduces that
# offset, so the scalar bias can be a bit smaller while keeping synth ratios correct.
_INWARD_BIAS_PER_SMOOTH_PX = 0.5
_INWARD_BIAS_INTERCEPT_PX = 0.92


def _parabolic_peak_subpx(sig: np.ndarray, i_center: int) -> float:
    """
    Sub-pixel location of a local maximum near ``i_center`` on 1D signal ``sig``.

    Fits a parabola through ``(i-1, i, i+1)`` and returns the vertex in index units.
    """
    n = int(sig.shape[0])
    i = int(np.clip(i_center, 1, n - 2))
    y0, y1, y2 = float(sig[i - 1]), float(sig[i]), float(sig[i + 1])
    if y1 < y0 and y1 < y2:
        return float(i_center)
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(i_center)
    dx = 0.5 * (y0 - y2) / denom
    return float(i) + float(np.clip(dx, -0.95, 0.95))


def _refine_margin_to_gradient_peak(
    sig: np.ndarray,
    coord_guess: float,
    band_lo: int,
    band_hi: int,
    *,
    window: int = 5,
) -> float:
    """
    Snap a margin coordinate (distance from the same physical edge as ``band_lo``)
    to the parabolic peak of ``sig`` in a small window around ``coord_guess``.

    For LR top-of-image / left edge: coordinate is column or row index from that edge.
    """
    band_hi = min(band_hi, sig.shape[0] - 1)
    band_lo = max(0, band_lo)
    if band_hi <= band_lo + 2:
        return float(coord_guess)
    ic = int(np.clip(round(coord_guess), band_lo + 1, band_hi - 2))
    half = window // 2
    lo = max(band_lo, ic - half)
    hi = min(band_hi + 1, ic + half + 1)
    if hi - lo < 3:
        lo = max(band_lo, band_hi - 3)
        hi = band_hi + 1
    seg = sig[lo:hi]
    if seg.size < 3:
        return float(coord_guess)
    k = int(np.argmax(seg))
    return _parabolic_peak_subpx(sig, lo + k)


def _edge_projection_inward_bias_px(smooth_window: int) -> float:
    wn = int(max(3, smooth_window | 1))
    return _INWARD_BIAS_PER_SMOOTH_PX * float(wn) + _INWARD_BIAS_INTERCEPT_PX


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

    # Parabolic peak on raw |G| projection (pre-normalize) for pixel-accurate seam position.
    left = float(
        np.clip(
            _refine_margin_to_gradient_peak(col_sig, left, rim_x, w_side - 1),
            0.0,
            w / 2.0 - 1.0,
        )
    )
    x_inner = (w - 1.0) - right
    x_inner = float(
        np.clip(
            _refine_margin_to_gradient_peak(col_sig, x_inner, w - w_side, w - 1 - rim_x),
            float(w - w_side),
            float(w - 1 - rim_x),
        )
    )
    right = float(np.clip((w - 1.0) - x_inner, 0.0, w / 2.0 - 1.0))
    top = float(
        np.clip(
            _refine_margin_to_gradient_peak(row_sig, top, rim_y, h_side - 1),
            0.0,
            h / 2.0 - 1.0,
        )
    )
    y_inner = (h - 1.0) - bottom
    y_inner = float(
        np.clip(
            _refine_margin_to_gradient_peak(row_sig, y_inner, h - h_side, h - 1 - rim_y),
            float(h - h_side),
            float(h - 1 - rim_y),
        )
    )
    bottom = float(np.clip((h - 1.0) - y_inner, 0.0, h / 2.0 - 1.0))

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

    bias_lr = _edge_projection_inward_bias_px(sw)
    bias_tb = _edge_projection_inward_bias_px(sh)
    left = float(np.clip(left + bias_lr, 0.0, w / 2.0 - 1.0))
    right = float(np.clip(right + bias_lr, 0.0, w / 2.0 - 1.0))
    top = float(np.clip(top + bias_tb, 0.0, h / 2.0 - 1.0))
    bottom = float(np.clip(bottom + bias_tb, 0.0, h / 2.0 - 1.0))
    meta["bias_lr_px"] = bias_lr
    meta["bias_tb_px"] = bias_tb

    # Sanity: total measurable frame must be meaningful vs card size (after bias correction).
    min_lr = max(16.0, 0.05 * float(w))
    min_tb = max(16.0, 0.05 * float(h))
    if left + right < min_lr or top + bottom < min_tb:
        meta["rejected"] = True
        meta["reason"] = "margin_sum_too_small"
        # Keep measurements; grading applies a floor when ``rejected`` (see centering.py).
    elif left + right >= w or top + bottom >= h:
        meta["rejected"] = True
        meta["reason"] = "margins_overflow"
        left = right = float(w) / 4.0
        top = bottom = float(h) / 4.0
        meta.pop("bias_lr_px", None)
        meta.pop("bias_tb_px", None)

    return left, right, top, bottom, meta
