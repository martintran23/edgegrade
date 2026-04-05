"""
Color-aware border → art transitions for TCG-style fronts.

Silver-border / blue-face (e.g. Scarlet & Violet) margins use an HSV **blue panel** cue
so left/right track the silver→blue seam instead of interior holo or text edges.

Yellow frames use HSV yellow segmentation; otherwise we fall back to gradient peaks.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.services.centering_projection import (
    _edge_projection_inward_bias_px,
    _smooth_1d,
    measure_margins_edge_projection,
)


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

    col_raw = np.mean(inner[ry0:ry1, :], axis=0).astype(np.float32)
    row_raw = np.mean(inner[:, rx0:rx1], axis=1).astype(np.float32)
    # Wide smoothing for baselines / peaks (stable); narrower for seam crossing (less LR bias).
    sw_stat = max(7, min(w // 25, 29) | 1)
    sh_stat = max(7, min(h // 25, 29) | 1)
    sw_cross = max(5, min(w // 32, 9) | 1)
    sh_cross = max(5, min(h // 32, 9) | 1)
    col_stat = _smooth_1d(col_raw, sw_stat)
    row_stat = _smooth_1d(row_raw, sh_stat)
    col_cross = _smooth_1d(col_raw, sw_cross)
    row_cross = _smooth_1d(row_raw, sh_cross)

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
    l_strip = float(np.mean(col_stat[l_lo:l_hi]))
    r_strip = float(np.mean(col_stat[r_lo:r_hi]))
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
    peak_c = float(np.percentile(col_stat[mid_lo:mid_hi], 93)) if mid_hi > mid_lo + 2 else 0.0
    mid_r0, mid_r1 = int(0.38 * h), int(0.62 * h)
    peak_r = float(np.percentile(row_stat[mid_r0:mid_r1], 93)) if mid_r1 > mid_r0 + 2 else 0.0
    if peak_c < 0.18 or peak_r < 0.18:
        return None

    # Strip mean (silver) as column baseline; corners as row baseline (top/bottom strips mix
    # rounded corners / anti-alias and skew row-only baselines).
    base_c = 0.5 * (l_strip + r_strip)
    frac = 0.46
    thresh_c = base_c + frac * (peak_c - base_c)
    thresh_r = corner_base + frac * (peak_r - corner_base)

    left = _first_ge_cross_subpx(col_cross, rim_x, w_band, thresh_c)
    right = _first_ge_cross_subpx(col_cross[::-1], rim_x, w_band, thresh_c)
    top = _first_ge_cross_subpx(row_cross, rim_y, h_band, thresh_r)
    bottom = _first_ge_cross_subpx(row_cross[::-1], rim_y, h_band, thresh_r)

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
        "col_sig": col_cross,
        "row_sig": row_cross,
    }
    return left, right, top, bottom, meta


def _yellow_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Broad yellow / tan — low S/V floors catch pale printer ink and flatbed shadows.
    m = cv2.inRange(hsv, np.array([6, 22, 40], dtype=np.uint8), np.array([54, 255, 255], dtype=np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m


def _yellow_rim_mean_score(bgr: np.ndarray) -> float:
    """Mean yellow-mask hit rate on thin outer strips (0–1)."""
    h, w = bgr.shape[:2]
    if h < 32 or w < 32:
        return 0.0
    m = _yellow_mask(bgr)
    mt = max(2, int(0.09 * h))
    mw = max(2, int(0.065 * w))
    top = float(np.mean(m[:mt, :] > 0))
    bot = float(np.mean(m[h - mt :, :] > 0))
    left = float(np.mean(m[:, :mw] > 0))
    right = float(np.mean(m[:, w - mw :] > 0))
    return 0.25 * (top + bot + left + right)


def _prefer_yellow_over_blue_panel(
    warped_bgr: np.ndarray,
    yellow: tuple[float, float, float, float, dict],
    blue: tuple[float, float, float, float, dict],
) -> bool:
    """
    Classic yellow-border fronts with blue (e.g. Water) art match **both** blue-panel
    (blue core) and yellow HSV. Prefer yellow seams. Printer scans may desaturate the rim,
    so we also fall back when blue TB is extreme but yellow is saner and some rim yellow remains.
    """
    yl, yr, yt, yb, _ = yellow
    _bl, _br, bt, bb, _ = blue
    rim = _yellow_rim_mean_score(warped_bgr)
    if rim >= 0.18:
        return True
    s_b = bt + bb
    s_y = yt + yb
    if s_b <= 1e-6 or s_y <= 1e-6:
        return False
    small_b = min(bt, bb) / s_b
    small_y = min(yt, yb) / s_y
    if rim >= 0.14 and small_b < 0.36 and small_y > small_b + 0.04:
        return True
    # Desaturated rim but blue TB clearly worse than yellow (common on printer scans).
    if rim >= 0.10:
        pct_b = 100.0 * bt / s_b
        pct_y = 100.0 * yt / s_y
        skew_b = abs(pct_b - 50.0)
        skew_y = abs(pct_y - 50.0)
        if skew_b >= skew_y + 6.0:
            return True
    return False


def _yellow_row_profile_tb(mask: np.ndarray) -> np.ndarray:
    """
    Per-row yellow fraction for **top/bottom** seams.

    Full-width means dip when the **center** of a row is non-yellow (name bar, holo, print
    voids) while the true outer yellow frame is still present — that fires the seam too early
    and shrinks ``top`` (bad splits like 25/75). Side **lobes** exclude the central ~24%–76%
    band and average left/right art-adjacent windows instead.
    """
    h, w = mask.shape[:2]
    trim_x = max(1, int(0.02 * w))
    x_mid0, x_mid1 = int(0.38 * w), int(0.62 * w)
    if x_mid1 <= x_mid0 + 8:
        return np.mean(mask[:, trim_x : w - trim_x] > 0, axis=1).astype(np.float32)
    left_lo, left_hi = trim_x, x_mid0
    right_lo, right_hi = x_mid1, w - trim_x
    if left_hi <= left_lo + 6 or right_hi <= right_lo + 6:
        return np.mean(mask[:, trim_x : w - trim_x] > 0, axis=1).astype(np.float32)
    rl = np.mean(mask[:, left_lo:left_hi] > 0, axis=1).astype(np.float32)
    rr = np.mean(mask[:, right_lo:right_hi] > 0, axis=1).astype(np.float32)
    return 0.5 * (rl + rr)


def _first_below_thresh_subpx_top(sig: np.ndarray, y0: int, y1: int, thresh: float) -> float | None:
    """First row (from top) where sig drops below thresh; linear sub-pixel between y-1 and y."""
    y1 = min(y1, len(sig))
    y0 = max(0, y0)
    for y in range(y0, y1):
        if float(sig[y]) >= thresh:
            continue
        if y == y0:
            return float(y)
        v0, v1 = float(sig[y - 1]), float(sig[y])
        if v0 < thresh:
            return float(y)
        if abs(v1 - v0) < 1e-9:
            return float(y)
        t = (thresh - v0) / (v1 - v0)
        return float(y - 1) + max(0.0, min(1.0, t))
    return None


def _first_below_thresh_subpx_bottom(
    sig: np.ndarray, y_lo: int, y_hi: int, thresh: float
) -> float | None:
    """
    Distance from image **bottom** to inner seam. Scan ``y`` from ``y_hi`` (near bottom)
    down to ``y_lo`` (exclusive). Row ``y+1`` is toward the bottom (yellow); ``y`` is inner.
    """
    h = len(sig)
    y_hi = min(h - 1, max(0, y_hi))
    y_lo = max(0, min(h - 1, y_lo))
    for y in range(y_hi, y_lo, -1):
        if float(sig[y]) >= thresh:
            continue
        if y + 1 >= h:
            return float((h - 1) - y)
        v_lo, v_hi = float(sig[y]), float(sig[y + 1])
        if v_hi < thresh:
            continue
        if abs(v_lo - v_hi) < 1e-9:
            y_seam = float(y)
        else:
            t = (thresh - v_hi) / (v_lo - v_hi)
            t = max(0.0, min(1.0, t))
            y_seam = float(y + 1) + t * (float(y) - float(y + 1))
        return float(h - 1) - y_seam
    return None


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
    ry0, ry1 = max(0, ry0), min(h, max(ry0 + 1, ry1))

    col_y = np.mean(mask[ry0:ry1, :] > 0, axis=0).astype(np.float32)
    row_raw = _yellow_row_profile_tb(mask)
    sw = max(7, min(w // 25, 31) | 1)
    sh_stat = max(7, min(h // 25, 31) | 1)
    sh_cross = max(5, min(h // 32, 13) | 1)
    col_y = _smooth_1d(col_y, sw)
    row_stat = _smooth_1d(row_raw, sh_stat)
    row_cross = _smooth_1d(row_raw, sh_cross)

    rim_x = max(2, min(int(0.007 * w), w // 25))
    rim_y = max(2, min(int(0.007 * h), h // 25))
    w_band = max(12, int(0.22 * w))
    h_band = max(12, int(0.22 * h))

    # Transition: high yellow coverage in border columns → low in art.
    hi_l = float(np.percentile(col_y[rim_x : min(w_band, w // 2)], 88)) if w_band > rim_x + 2 else 0.0
    hi_t = float(np.percentile(row_stat[rim_y : min(h_band, h // 2)], 88)) if h_band > rim_y + 2 else 0.0
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
    top = _first_below_thresh_subpx_top(row_cross, rim_y, h_band, thresh_t)
    bottom = _first_below_thresh_subpx_bottom(row_cross, h - h_band - 1, h - 1 - rim_y, thresh_t)

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
        "row_sig": row_cross,
    }
    return left, right, top, bottom, meta


def measure_margins_combined(warped_bgr: np.ndarray) -> tuple[float, float, float, float, dict]:
    """
    Order: **blue panel** (SV-style silver + blue face) **or** **yellow HSV** when both
    match (yellow-border + blue art) — yellow wins if the outer rim is yellow and/or blue
    TB is extreme vs yellow. Then gradient; optional **yellow nudge** when it agrees.
    """
    blue = _try_blue_panel_margins(warped_bgr)
    yellow = _try_yellow_frame_margins(warped_bgr)
    proj_l, proj_r, proj_t, proj_b, pmeta = measure_margins_edge_projection(warped_bgr)
    pmeta = dict(pmeta)
    pmeta["method"] = "edge_projection"

    if blue is not None:
        # Blue core matches SV silver-border **and** yellow-border + blue art; prefer yellow
        # HSV seams when the layout is clearly classic yellow (or blue TB is implausible).
        if yellow is not None and _prefer_yellow_over_blue_panel(warped_bgr, yellow, blue):
            yl, yr, yt, yb, ymeta = yellow
            ymeta = dict(ymeta)
            ymeta["method"] = "yellow_hsv"
            ymeta["skipped_blue_panel"] = True
            ymeta["yellow_rim_score"] = float(_yellow_rim_mean_score(warped_bgr))
            return yl, yr, yt, yb, ymeta
        bl, br, bt, bb, bmeta = blue
        return bl, br, bt, bb, dict(bmeta)

    if yellow is None:
        return proj_l, proj_r, proj_t, proj_b, pmeta

    yl, yr, yt, yb, ymeta = yellow
    ymeta = dict(ymeta)
    rim_yellow = float(_yellow_rim_mean_score(warped_bgr))

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
        hh, ww = warped_bgr.shape[:2]
        sw = max(5, min(ww // 32, 21) | 1)
        sh = max(5, min(hh // 28, 25) | 1)
        bias_lr = _edge_projection_inward_bias_px(sw)
        bias_tb = _edge_projection_inward_bias_px(sh)
        yl = float(np.clip(yl + bias_lr, 0.0, ww / 2.0 - 1.0))
        yr = float(np.clip(yr + bias_lr, 0.0, ww / 2.0 - 1.0))
        yt = float(np.clip(yt + bias_tb, 0.0, hh / 2.0 - 1.0))
        yb = float(np.clip(yb + bias_tb, 0.0, hh / 2.0 - 1.0))
        ymeta = dict(ymeta)
        ymeta["method"] = "yellow_hsv+inward_bias"
        ymeta["yellow_rim_score"] = rim_yellow
        return yl, yr, yt, yb, ymeta

    dlr = abs(lr_share(yl, yr) - lr_share(proj_l, proj_r))
    dtb = abs(tb_share(yt, yb) - tb_share(proj_t, proj_b))

    # Printer scans / shadows: |Gy| projection often disagrees strongly with yellow TB on
    # real yellow borders. Never drop yellow for pure projection when the rim is yellow or
    # when TB splits diverge by more than a few points (previous bug: 25/75 projection won).
    if rim_yellow >= 0.10:
        ymeta["method"] = "yellow_hsv"
        ymeta["yellow_rim_score"] = rim_yellow
        return yl, yr, yt, yb, ymeta
    if dtb > 8.0:
        ymeta["method"] = "yellow_hsv_preferred_tb_vs_projection"
        ymeta["yellow_rim_score"] = rim_yellow
        ymeta["proj_tb_disagree_pp"] = dtb
        return yl, yr, yt, yb, ymeta

    # Heavy projection weight when both paths agree (non–yellow-border lookalikes).
    _PROJ_BLEND = 0.94
    _YL_BLEND = 1.0 - _PROJ_BLEND
    if dlr <= 8.0 and dtb <= 8.0:
        meta = {
            "method": "edge_projection+yellow_nudge",
            "rejected": False,
            "col_sig": pmeta.get("col_sig"),
            "row_sig": pmeta.get("row_sig"),
            "yellow_rim_score": rim_yellow,
        }
        return (
            _PROJ_BLEND * proj_l + _YL_BLEND * yl,
            _PROJ_BLEND * proj_r + _YL_BLEND * yr,
            _PROJ_BLEND * proj_t + _YL_BLEND * yt,
            _PROJ_BLEND * proj_b + _YL_BLEND * yb,
            meta,
        )

    return proj_l, proj_r, proj_t, proj_b, pmeta
