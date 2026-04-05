"""
Legacy centering heuristics (LAB crop + inner rectangle + blended Scharr profiles).

**Not used** by the production path in ``centering.py`` as of the edge-projection
refactor. Kept so experiments can ``import`` ``isolate_card_face`` /
``find_inner_picture_box`` from ``card_frame`` alongside archived logic if needed.
"""

from __future__ import annotations

# Re-export for optional A/B tests without pulling deleted code back into centering.py
from app.services.card_frame import find_inner_picture_box, isolate_card_face

__all__ = ["find_inner_picture_box", "isolate_card_face"]
