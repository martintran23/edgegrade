"""
Card face isolation and inner-frame detection for centering.

Centering is defined relative to the **physical card** in the perspective-corrected crop:
outer bounds = tight card rectangle after removing uniform **background** margins; inner
bounds = the **picture / play area** frame (border around the artwork) when it can be
found from print structure. Falls back to edge-profile heuristics when inner geometry
is ambiguous (foil, dark borders, low contrast).
"""

from __future__ import annotations

import cv2
import numpy as np


def _outer_edge_px_lab(bgr: np.ndarray, edge_px: int) -> np.ndarray:
    """Flattened LAB rows for pixels on the outer ``edge_px`` frame (all four sides)."""
    h, w = bgr.shape[:2]
    e = max(1, min(edge_px, h // 4, w // 4))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    parts = [
        lab[0:e, :].reshape(-1, 3),
        lab[h - e : h, :].reshape(-1, 3),
        lab[:, 0:e].reshape(-1, 3),
        lab[:, w - e : w].reshape(-1, 3),
    ]
    return np.concatenate(parts, axis=0)


def _perimeter_suggests_card_border_not_desk(bgr: np.ndarray) -> bool:
    """
    If true, skip LAB margin stripping.

    Corners sample the physical card; a uniform yellow/red border matches those corners
    in LAB, so ``_isolate_lab_margin`` would walk inward and drop the real outer edge —
    destroying centering. Neutral (low-chroma) outer strips are usually desk / sleeve /
    scanner bed and are safe to strip.
    """
    edge_px = max(4, min(bgr.shape[0] // 10, bgr.shape[1] // 10, 12))
    pix = _outer_edge_px_lab(bgr, edge_px)
    l = pix[:, 0].astype(np.float32)
    a = pix[:, 1].astype(np.float32) - 128.0
    b = pix[:, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)
    med_c = float(np.median(chroma))
    med_l = float(np.median(l))
    # Colored TCG frame (yellow, red, blue, …); threshold is conservative for dim photos
    if med_c > 8.0:
        return True
    # Dark / black frame reads as low chroma but is still printed border, not desk
    if med_l < 52.0:
        return True
    return False


def _isolate_lab_margin(bgr: np.ndarray, lab_tol: float) -> np.ndarray | None:
    """
    Crop uniform margins via LAB distance from corners. Returns None if no safe shrink.
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    edge = max(4, min(h // 10, w // 10, 12))

    def corner_patch(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        return lab[y0:y1, x0:x1].reshape(-1, 3)

    patches = np.concatenate(
        [
            corner_patch(0, edge, 0, edge),
            corner_patch(0, edge, w - edge, w),
            corner_patch(h - edge, h, 0, edge),
            corner_patch(h - edge, h, w - edge, w),
        ],
        axis=0,
    )
    ref = np.median(patches, axis=0)
    dist = np.sqrt(np.sum((lab - ref) ** 2, axis=2))

    y0, y1, x0, x1 = 0, h - 1, 0, w - 1
    for y in range(h):
        if float(np.percentile(dist[y, :], 75)) > lab_tol:
            y0 = y
            break
    for y in range(h - 1, -1, -1):
        if float(np.percentile(dist[y, :], 75)) > lab_tol:
            y1 = y
            break
    for x in range(w):
        if float(np.percentile(dist[:, x], 75)) > lab_tol:
            x0 = x
            break
    for x in range(w - 1, -1, -1):
        if float(np.percentile(dist[:, x], 75)) > lab_tol:
            x1 = x
            break

    if y1 <= y0 or x1 <= x0:
        return None

    # When L/R (or T/B) insets almost match, nudge to exact symmetry so tiny scan
    # differences do not shift the outer reference. Skip when margins are clearly
    # asymmetric (e.g. more desk on one side) — full symmetrize would clip real context.
    li, ri = x0, (w - 1 - x1)
    ti, bi = y0, (h - 1 - y1)
    tot_lr = float(li + ri) + 1e-6
    if abs(li - ri) / tot_lr < 0.22:
        m_lr = min(li, ri)
        x0, x1 = m_lr, w - 1 - m_lr
    tot_tb = float(ti + bi) + 1e-6
    if abs(ti - bi) / tot_tb < 0.22:
        m_tb = min(ti, bi)
        y0, y1 = m_tb, h - 1 - m_tb
    if y1 <= y0 or x1 <= x0:
        return None

    crop_h, crop_w = y1 - y0 + 1, x1 - x0 + 1
    if crop_h < int(0.55 * h) or crop_w < int(0.55 * w):
        return None

    out = bgr[y0 : y1 + 1, x0 : x1 + 1]
    if out.size < 3000:
        return None
    return out


def isolate_card_face(bgr: np.ndarray, lab_tol: float = 13.5) -> np.ndarray:
    """
    Crop the warped image to the **printed card** region:

    1. LAB distance from corner patches (removes desk/UI that matches corners).
    2. Retries with slightly looser LAB tolerance if corners are noisy (foil / yellow).

    When the perimeter already looks like a **printed border** (saturated color or very
    dark frame), LAB isolation is skipped so the physical outer edge stays in frame for
    centering.
    """
    h, w = bgr.shape[:2]
    if h < 16 or w < 16:
        return bgr

    if _perimeter_suggests_card_border_not_desk(bgr):
        return bgr

    for tol in (lab_tol, lab_tol * 1.35, lab_tol * 1.7):
        cropped = _isolate_lab_margin(bgr, tol)
        if cropped is not None:
            ch, cw = cropped.shape[:2]
            if ch <= int(0.97 * h) or cw <= int(0.97 * w):
                return cropped

    return bgr


def _best_inner_from_adaptive(
    blur: np.ndarray, h: int, w: int, bsz: int, c_const: int
) -> tuple[tuple[int, int, int, int] | None, float]:
    """Return (rect or None, score) for one adaptive threshold setting."""
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bsz, c_const
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = float(h * w)
    min_inset_px = max(3, int(0.009 * min(w, h)))

    best: tuple[int, int, int, int] | None = None
    best_score = -1.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.09 * total or area > 0.93 * total:
            continue
        x, yy, rw, rh = cv2.boundingRect(c)
        if rw < int(0.30 * w) or rh < int(0.30 * h):
            continue
        ar = rw / max(rh, 1)
        if ar < 0.52 or ar > 0.90:
            continue

        inset_l, inset_r = x, w - x - rw
        inset_t, inset_b = yy, h - yy - rh
        min_inset = min(inset_l, inset_r, inset_t, inset_b)
        if min_inset < min_inset_px:
            continue

        hull = cv2.convexHull(c)
        hull_a = cv2.contourArea(hull)
        solidity = area / hull_a if hull_a > 1e-6 else 0.0
        if solidity < 0.66:
            continue

        score = area + 1.8 * min_inset * min(w, h) + 45.0 * solidity
        if score > best_score:
            best_score = score
            best = (x, yy, x + rw, yy + rh)

    return best, best_score


def find_inner_picture_box(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Find axis-aligned inner box ``(x0, y0, x1, y1)`` with **x1,y1 exclusive**,
    approximating the artwork / inner print frame inside the colored border.

    Tries a few adaptive **C** constants on the same block size; keeps the strongest
    valid candidate. Canny fallback if none qualify.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    bsz = int(0.045 * min(h, w)) | 1
    bsz = max(15, min(55, bsz))
    if bsz % 2 == 0:
        bsz += 1

    global_best: tuple[int, int, int, int] | None = None
    global_score = -1.0
    for c_const in (10, 7, 13):
        cand, sc = _best_inner_from_adaptive(blur, h, w, bsz, c_const)
        if cand is not None and sc > global_score:
            global_score = sc
            global_best = cand

    if global_best is None:
        return _find_inner_picture_box_canny(bgr)

    x0, y0, x1, y1 = global_best
    total = float(h * w)
    iw, ih = x1 - x0, y1 - y0
    if iw < 1 or ih < 1 or iw * ih < 0.12 * total:
        return _find_inner_picture_box_canny(bgr)
    return global_best


def _find_inner_picture_box_canny(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Fallback: closed Canny edges for inner-like rectangles."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    e = cv2.Canny(gray, 40, 120)
    e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = float(h * w)
    best = None
    best_score = -1.0
    min_inset_px = max(3, int(0.009 * min(w, h)))
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.08 * total or area > 0.90 * total:
            continue
        x, yy, rw, rh = cv2.boundingRect(c)
        if rw < int(0.28 * w) or rh < int(0.28 * h):
            continue
        ar = rw / max(rh, 1)
        if ar < 0.52 or ar > 0.90:
            continue
        min_inset = min(x, yy, w - x - rw, h - yy - rh)
        if min_inset < min_inset_px:
            continue
        score = area + 1.5 * min_inset * min(w, h)
        if score > best_score:
            best_score = score
            best = (x, yy, x + rw, yy + rh)
    return best
