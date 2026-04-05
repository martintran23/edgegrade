"""
Centering from the **warped card rectangle** using edge projection (symmetric thresholds).

Definition: distance from each **physical edge of the warp** to the inner print frame.
Uses **HSV yellow-frame** detection when the layout matches (typical Pokémon fronts), else
**gradient peaks** in outer bands (see ``centering_borders.py``, ``centering_projection.py``).

Grading uses **threshold-based** PSA-style tiers from raw margins (``centering_grades``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.models.schemas import AnalyzeCardResponse, CenteringMetrics, EstimatedGrades, MarginsPx
from app.services.centering_borders import measure_margins_combined
from app.services.centering_debug import resolve_debug_dir, save_centering_debug_bundle
from app.services.centering_grades import (
    companion_estimated_grades,
    compute_centering_ratios,
    compute_psa_grade,
)

logger = logging.getLogger(__name__)


def compute_centering_margins(
    warped_bgr: np.ndarray,
) -> tuple[float, float, float, float, int, int, dict]:
    """
    Margins in pixels from warp edges. Also returns ``meta`` for debug overlays.

    Returns ``(left, right, top, bottom, width, height, meta)``.
    """
    hh, ww = warped_bgr.shape[:2]
    left, right, top, bottom, meta = measure_margins_combined(warped_bgr)
    return left, right, top, bottom, ww, hh, meta


def build_analyze_response(
    warped_bgr: np.ndarray,
    detection_confidence: str,
    *,
    debug_enabled: bool | None = None,
    debug_dir: Path | str | None = None,
) -> AnalyzeCardResponse:
    """Run centering + PSA threshold grade on the normalized card image (full warp)."""
    left, right, top, bottom, ww, hh, meta = compute_centering_margins(warped_bgr)

    do_debug = settings.debug_centering if debug_enabled is None else debug_enabled
    if do_debug:
        dpath = resolve_debug_dir(
            settings.debug_outputs_dir if debug_dir is None else str(debug_dir)
        )
        try:
            save_centering_debug_bundle(dpath, warped_bgr, left, right, top, bottom, meta)
        except OSError as e:
            logger.warning("centering debug save failed: %s", e)

    ratios = compute_centering_ratios(left, right, top, bottom)
    lr_str = str(ratios["lr_display"])
    tb_str = str(ratios["tb_display"])
    lr_small = float(ratios["lr_small"])
    tb_small = float(ratios["tb_small"])

    psa_tier = compute_psa_grade(left, right, top, bottom)
    if meta.get("rejected"):
        psa_tier = 5
    psa_f, bgs_f, cgc_f = companion_estimated_grades(psa_tier)

    return AnalyzeCardResponse(
        centering=CenteringMetrics(
            left_right=lr_str,
            top_bottom=tb_str,
            lr_small_pct=lr_small,
            tb_small_pct=tb_small,
            margins_px=MarginsPx(left=left, right=right, top=top, bottom=bottom),
        ),
        estimated_grades=EstimatedGrades(PSA=psa_f, BGS=bgs_f, CGC=cgc_f),
        warp_width=ww,
        warp_height=hh,
        detection_confidence=detection_confidence,
    )
