"""
Color-aware border → art transitions for TCG-style fronts.

Silver-border / blue-face (e.g. Scarlet & Violet) margins use an HSV **blue panel** cue
so left/right track the silver→blue seam instead of interior holo or text edges.

Yellow frames use HSV yellow segmentation; otherwise we fall back to gradient peaks.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.services.centering_projection import _smooth_1d, measure_margins_edge_projection


def _first_ge_cross_subpx(sig: np.ndarray, lo: int, hi: int, thresh: float) -> float | None:
    """
    Sub-pixel position of the first transition to ``sig[i] >= thresh`` scanning ``i`` upward
    from ``lo + 1`` to ``hi - 1``. Returns distance from index 0 of ``sig`` (same units as
    integer column/row indices). ``None`` if no crossing in range.
    """
    hi = min(hi, len(sig))
    lo = max(0, lo)
    for i in range(max(lo + 1, 1), hi):
        v1 = float(sig[i])
        if v1 < thresh:
            continue
        v0 = float(sig[i - 1])
        if v1 <= v0 + 1e-6:
            return float(i)
        t = (thresh - v0) / (v1 - v0)
        return float(i - 1) + max(0.0, min(1.0, t))
    return None


def _symmetry_nudge(a: float, b: float, *, rel: float, floor_px: float) -> tuple[float, float]:
    """When imbalance is within expected seam noise, use the mean (physical cards are often symmetric)."""
    s = a + b
    if s <= 1e-6:
        return a, b
    if abs(a - b) <= max(floor_px, rel * s):
        m = 0.5 * (a + b)
        return m, m
    return a, b


def _try_blue_panel_margins(bgr: np.ndarray) -> tuple[float, float, float, float, dict] | None:
    """
    Silver/gray outer border with a **blue/cyan inner frame** (e.g. Scarlet & Violet).

    Uses HSV blue-like pixels aggregated in mid-rows / mid-columns so margins track the
    silver→blue seam instead of interior holo/text edges.
    """
    h, w = bgr.shape[:2]
    if h < 64 or w < 64:
        return None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    inner = ((H >= 88) & (H <= 135) & (S >= 30) & (V >= 55)).astype(np.float32)

    cy0, cy1 = int(0.34 * h), int(0.66 * h)
    cx0, cx1 = int(0.34 * w), int(0.66 * w)
    core_blue = float(np.mean(inner[cy0:cy1, cx0:cx1]))
    if core_blue < 0.07 or core_blue > 0.92:
        return None

    ry0, ry1 = int(0.18 * h), int(0.82 * h)
    rx0, rx1 = int(0.18 * w), int(0.82 * w)
    ry0, ry1 = max(0, ry0), min(h, max(ry0 + 1, ry1))
    rx0, rx1 = max(0, rx0), min(w, max(rx0 + 1, rx1))

    col_b = np.mean(inner[ry0:ry1, :], axis=0).astype(np.float32)
    row_b = np.mean(inner[:, rx0:rx1], axis=1).astype(np.float32)
    sw = max(7, min(w // 25, 29) | 1)
    sh = max(7, min(h // 25, 29) | 1)
    col_b = _smooth_1d(col_b, sw)
    row_b = _smooth_1d(row_b, sh)

    rim_x = max(2, min(int(0.007 * w), w // 25))
    rim_y = max(2, min(int(0.007 * h), h // 25))
    w_band = max(12, int(0.26 * w))
    h_band = max(12, int(0.26 * h))

    # Only the outermost ~2% of width — not the first 11%, which already includes the blue face
    # and falsely reads as "high blue" on centered SV cards (rejects valid silver-border frames).
    outer_w = max(6, min(int(0.022 * w), w // 45))
    l_lo, l_hi = rim_x, min(rim_x + outer_w, w // 2 - 1)
    r_lo = max(w // 2 + 1, w - rim_x - outer_w)
    r_hi = w - rim_x
    if l_hi <= l_lo or r_hi <= r_lo:
        return None
    l_strip = float(np.mean(col_b[l_lo:l_hi]))
    r_strip = float(np.mean(col_b[r_lo:r_hi]))
    if r_strip > 0.24 or l_strip > 0.24:
        return None

    ch, cw = max(4, int(0.024 * h)), max(4, int(0.024 * w))
    corner_samples = np.concatenate(
        [
            inner[:ch, :cw].ravel(),
            inner[:ch, -cw:].ravel(),
            inner[-ch:, :cw].ravel(),
            inner[-ch:, -cw:].ravel(),
        ]
    )
    corner_base = float(np.mean(corner_samples))

    mid_lo, mid_hi = int(0.36 * w), int(0.64 * w)
    peak_c = float(np.percentile(col_b[mid_lo:mid_hi], 93)) if mid_hi > mid_lo + 2 else 0.0
    mid_r0, mid_r1 = int(0.38 * h), int(0.62 * h)
    peak_r = float(np.percentile(row_b[mid_r0:mid_r1], 93)) if mid_r1 > mid_r0 + 2 else 0.0
    if peak_c < 0.18 or peak_r < 0.18:
        return None

    # Strip mean (silver) as column baseline; corners as row baseline (top/bottom strips mix
    # rounded corners / anti-alias and skew row-only baselines).
    base_c = 0.5 * (l_strip + r_strip)
    frac = 0.46
    thresh_c = base_c + frac * (peak_c - base_c)
    thresh_r = corner_base + frac * (peak_r - corner_base)

    left = _first_ge_cross_subpx(col_b, rim_x, w_band, thresh_c)
    right = _first_ge_cross_subpx(col_b[::-1], rim_x, w_band, thresh_c)
    top = _first_ge_cross_subpx(row_b, rim_y, h_band, thresh_r)
    bottom = _first_ge_cross_subpx(row_b[::-1], rim_y, h_band, thresh_r)

    if None in (left, right, top, bottom):
        return None

    left, right = _symmetry_nudge(left, right, rel=0.055, floor_px=2.5)
    top, bottom = _symmetry_nudge(top, bottom, rel=0.085, floor_px=3.5)

    if not (0.012 * w <= left <= 0.22 * w and 0.012 * w <= right <= 0.22 * w):
        return None
    if not (0.012 * h <= top <= 0.26 * h and 0.012 * h <= bottom <= 0.26 * h):
        return None
    if left + right > 0.42 * w or top + bottom > 0.50 * h:
        return None

    meta = {
        "method": "blue_panel_hsv",
        "rejected": False,
        "col_sig": col_b,
        "row_sig": row_b,
    }
    return left, right, top, bottom, meta


def _yellow_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Broad yellow / tan (washed scans, warm lighting)
    m = cv2.inRange(hsv, np.array([8, 38, 50], dtype=np.uint8), np.array([52, 255, 255], dtype=np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m


def _try_yellow_frame_margins(bgr: np.ndarray) -> tuple[float, float, float, float, dict] | None:
    h, w = bgr.shape[:2]
    if h < 64 or w < 64:
        return None

    mask = _yellow_mask(bgr)
    # Reject if the picture area is full of yellow-like pixels (not a classic frame).
    cy0, cy1 = int(0.34 * h), int(0.66 * h)
    cx0, cx1 = int(0.34 * w), int(0.66 * w)
    core_frac = float(np.mean(mask[cy0:cy1, cx0:cx1] > 0))
    if core_frac > 0.20:
        return None

    ry0, ry1 = int(0.18 * h), int(0.82 * h)
    rx0, rx1 = int(0.18 * w), int(0.82 * w)
    ry0, ry1 = max(0, ry0), min(h, max(ry0 + 1, ry1))
    rx0, rx1 = max(0, rx0), min(w, max(rx0 + 1, rx1))

    col_y = np.mean(mask[ry0:ry1, :] > 0, axis=0).astype(np.float32)
    row_y = np.mean(mask[:, rx0:rx1] > 0, axis=1).astype(np.float32)
    sw = max(7, min(w // 25, 31) | 1)
    sh = max(7, min(h // 25, 31) | 1)
    col_y = _smooth_1d(col_y, sw)
    row_y = _smooth_1d(row_y, sh)

    rim_x = max(2, min(int(0.007 * w), w // 25))
    rim_y = max(2, min(int(0.007 * h), h // 25))
    w_band = max(12, int(0.22 * w))
    h_band = max(12, int(0.22 * h))

    # Transition: high yellow coverage in border columns → low in art.
    hi_l = float(np.percentile(col_y[rim_x : min(w_band, w // 2)], 88)) if w_band > rim_x + 2 else 0.0
    hi_t = float(np.percentile(row_y[rim_y : min(h_band, h // 2)], 88)) if h_band > rim_y + 2 else 0.0
    if hi_l < 0.34 or hi_t < 0.34:
        return None

    thresh_l = 0.52 * hi_l
    thresh_t = 0.52 * hi_t

    left = None
    for x in range(rim_x, w_band):
        if col_y[x] < thresh_l:
            left = float(x)
            break
    right = None
    for x in range(w - 1 - rim_x, w - w_band - 1, -1):
        if col_y[x] < thresh_l:
            right = float((w - 1) - x)
            break
    top = None
    for y in range(rim_y, h_band):
        if row_y[y] < thresh_t:
            top = float(y)
            break
    bottom = None
    for y in range(h - 1 - rim_y, h - h_band - 1, -1):
        if row_y[y] < thresh_t:
            bottom = float((h - 1) - y)
            break

    if None in (left, right, top, bottom):
        return None

    # Physical yellow border on modern Pokémon is usually ~4–14% of side each; allow scan noise.
    if not (0.018 * w <= left <= 0.26 * w and 0.018 * w <= right <= 0.26 * w):
        return None
    if not (0.018 * h <= top <= 0.30 * h and 0.018 * h <= bottom <= 0.30 * h):
        return None
    if left + right > 0.48 * w or top + bottom > 0.55 * h:
        return None

    meta = {
        "method": "yellow_hsv",
        "rejected": False,
        "col_sig": col_y,
        "row_sig": row_y,
    }
    return left, right, top, bottom, meta


def measure_margins_combined(warped_bgr: np.ndarray) -> tuple[float, float, float, float, dict]:
    """
    Order: **blue panel** (silver-border era) → gradient; optional **yellow nudge** when
    it tightly agrees with gradient. Blue path fixes centered SV cards where edges alone
    track the wrong vertical features.
    """
    blue = _try_blue_panel_margins(warped_bgr)
    yellow = _try_yellow_frame_margins(warped_bgr)
    proj_l, proj_r, proj_t, proj_b, pmeta = measure_margins_edge_projection(warped_bgr)
    pmeta = dict(pmeta)
    pmeta["method"] = "edge_projection"

    if blue is not None:
        bl, br, bt, bb, bmeta = blue
        return bl, br, bt, bb, dict(bmeta)

    if yellow is None:
        return proj_l, proj_r, proj_t, proj_b, pmeta

    yl, yr, yt, yb, ymeta = yellow
    ymeta = dict(ymeta)

    def lr_share(l: float, r: float) -> float:
        t = l + r
        return 50.0 if t <= 1e-6 else 100.0 * l / t

    def tb_share(t: float, b: float) -> float:
        s = t + b
        return 50.0 if s <= 1e-6 else 100.0 * t / s

    # Geometry failed on gradient path — fall back to yellow if we have it.
    if pmeta.get("rejected"):
        ymeta.setdefault("col_sig", np.zeros(warped_bgr.shape[1], dtype=np.float32))
        ymeta.setdefault("row_sig", np.zeros(warped_bgr.shape[0], dtype=np.float32))
        return yl, yr, yt, yb, ymeta

    dlr = abs(lr_share(yl, yr) - lr_share(proj_l, proj_r))
    dtb = abs(tb_share(yt, yb) - tb_share(proj_t, proj_b))
    if dlr <= 8.0 and dtb <= 8.0:
        meta = {
            "method": "edge_projection+yellow_nudge",
            "rejected": False,
            "col_sig": pmeta.get("col_sig"),
            "row_sig": pmeta.get("row_sig"),
        }
        return (
            0.88 * proj_l + 0.12 * yl,
            0.88 * proj_r + 0.12 * yr,
            0.88 * proj_t + 0.12 * yt,
            0.88 * proj_b + 0.12 * yb,
            meta,
        )

    return proj_l, proj_r, proj_t, proj_b, pmeta
