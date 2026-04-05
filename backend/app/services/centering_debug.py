"""
Optional disk + log output for centering QA.

Enable with ``CARDGRADING_DEBUG_CENTERING=1`` (and optionally ``CARDGRADING_DEBUG_OUTPUTS_DIR``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from app.services.centering_grades import compute_centering_ratios

logger = logging.getLogger(__name__)


def resolve_debug_dir(configured: str) -> Path:
    if configured and configured.strip():
        return Path(configured).expanduser().resolve()
    # backend/debug_outputs
    here = Path(__file__).resolve().parent  # services
    backend = here.parent.parent
    return (backend / "debug_outputs").resolve()


def save_centering_debug_bundle(
    debug_dir: Path,
    warped_bgr: np.ndarray,
    left: float,
    right: float,
    top: float,
    bottom: float,
    meta: dict,
) -> str:
    """
    Write JPEGs + log JSON margins. Returns session id (folder suffix).
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    sid = str(time.time_ns())
    sub = debug_dir / sid
    sub.mkdir(parents=True, exist_ok=False)

    cv2.imwrite(str(sub / "warped_bgr.jpg"), warped_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # LAB isolation disabled in production path; keep file for diff-friendly workflows.
    cv2.imwrite(str(sub / "isolated.jpg"), warped_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    mag_n = mag / (float(np.percentile(mag, 99.5)) or 1.0)
    mag_u8 = np.clip(mag_n * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(sub / "edges.jpg"), mag_u8)

    h, w = warped_bgr.shape[:2]
    inner_overlay = warped_bgr.copy()
    method = str(meta.get("method", "unknown"))
    cv2.putText(
        inner_overlay,
        f"centering: {method}",
        (max(8, w // 40), max(24, h // 45)),
        cv2.FONT_HERSHEY_SIMPLEX,
        min(w, h) / 900.0,
        (0, 220, 255),
        max(1, min(w, h) // 400),
        cv2.LINE_AA,
    )
    cv2.imwrite(str(sub / "inner_box_overlay.jpg"), inner_overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    prof = warped_bgr.copy()
    xl = int(round(np.clip(left, 0, w - 1)))
    xr = int(round(np.clip(w - 1 - right, 0, w - 1)))
    yt = int(round(np.clip(top, 0, h - 1)))
    yb = int(round(np.clip(h - 1 - bottom, 0, h - 1)))
    cv2.line(prof, (xl, 0), (xl, h - 1), (0, 255, 0), max(1, min(w, h) // 400))
    cv2.line(prof, (xr, 0), (xr, h - 1), (0, 200, 0), max(1, min(w, h) // 400))
    cv2.line(prof, (0, yt), (w - 1, yt), (255, 0, 0), max(1, min(w, h) // 400))
    cv2.line(prof, (0, yb), (w - 1, yb), (200, 0, 0), max(1, min(w, h) // 400))
    cv2.imwrite(str(sub / "profile_overlay.jpg"), prof, [cv2.IMWRITE_JPEG_QUALITY, 92])

    ratios = compute_centering_ratios(left, right, top, bottom)
    payload = {
        "left": round(left, 2),
        "right": round(right, 2),
        "top": round(top, 2),
        "bottom": round(bottom, 2),
        "lr_small_pct": round(float(ratios["lr_small"]), 3),
        "tb_small_pct": round(float(ratios["tb_small"]), 3),
        "lr_display": ratios["lr_display"],
        "tb_display": ratios["tb_display"],
        "rejected": bool(meta.get("rejected")),
        "reason": meta.get("reason"),
        "method": meta.get("method"),
        "session": sid,
        "path": str(sub),
    }
    (sub / "margins.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("centering_margins_px %s", json.dumps(payload, separators=(",", ":")))

    return sid
