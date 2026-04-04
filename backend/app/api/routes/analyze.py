"""REST routes for card analysis."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import AnalyzeCardResponse
from app.services.pipeline import analyze_card_image

router = APIRouter(tags=["analyze"])


@router.post("/analyze-card", response_model=AnalyzeCardResponse)
async def analyze_card(file: UploadFile = File(..., description="Trading card photo (JPEG/PNG/WebP)")) -> AnalyzeCardResponse:
    """
    Accept a single image upload, detect the card, warp to top-down view, and return
    centering metrics plus **centering-only** grade approximations.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        return analyze_card_image(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
