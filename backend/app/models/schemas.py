"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field


class CenteringMetrics(BaseModel):
    """Left/right and top/bottom centering as percentage split strings (e.g. '55/45')."""

    left_right: str = Field(..., description="Left vs right margin balance, e.g. '55/45'")
    top_bottom: str = Field(..., description="Top vs bottom margin balance, e.g. '52/48'")


class EstimatedGrades(BaseModel):
    """Rough grade estimates from centering only; not official grading."""

    PSA: float = Field(..., ge=1, le=10, description="Approximate PSA-style whole number 1–10")
    BGS: float = Field(..., ge=1, le=10, description="Approximate BGS-style including half points")
    CGC: float = Field(..., ge=1, le=10, description="Approximate CGC-style 1–10")


class AnalyzeCardResponse(BaseModel):
    """Result of POST /analyze-card."""

    centering: CenteringMetrics
    estimated_grades: EstimatedGrades
    # Optional debug fields for UI / future persistence (kept lightweight for MVP)
    warp_width: int | None = Field(None, description="Pixel width of perspective-corrected card crop")
    warp_height: int | None = Field(None, description="Pixel height of perspective-corrected card crop")
    detection_confidence: str = Field(
        "medium",
        description="Heuristic confidence in contour-based card detection: low|medium|high",
    )
