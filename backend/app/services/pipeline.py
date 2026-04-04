"""
End-to-end analysis pipeline: decode → detect/warp → centering + grades.

Keeps orchestration thin so new stages (defects, corners) can be inserted as services.
"""

from __future__ import annotations

from app.core.config import settings
from app.models.schemas import AnalyzeCardResponse
from app.services.card_detection import decode_image_from_bytes, extract_normalized_card
from app.services.centering import build_analyze_response


def analyze_card_image(file_bytes: bytes) -> AnalyzeCardResponse:
    """
    Run Phase-1 analysis on raw upload bytes.

    Steps:
        1. Decode image.
        2. Detect card quadrilateral; perspective-warp to canonical height.
        3. Heuristic centering + approximate grades (centering-only).
    """
    image = decode_image_from_bytes(file_bytes)
    warped, det_conf = extract_normalized_card(image, target_height=settings.warp_height)
    return build_analyze_response(warped, detection_confidence=det_conf)
