"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field


class MarginsPx(BaseModel):
    """Raw margin depths from the warp edges (pixels)."""

    left: float = Field(..., description="Left margin depth in px")
    right: float = Field(..., description="Right margin depth in px")
    top: float = Field(..., description="Top margin depth in px")
    bottom: float = Field(..., description="Bottom margin depth in px")


class CenteringMetrics(BaseModel):
    """Centering display strings plus values used for tier logic."""

    left_right: str = Field(..., description="Left vs right split, e.g. '55/45'")
    top_bottom: str = Field(..., description="Top vs bottom split, e.g. '52/48'")
    lr_small_pct: float = Field(
        ...,
        description="Smaller LR margin as %% of (left+right); PSA tiers use this with safe_div smoothing",
    )
    tb_small_pct: float = Field(
        ...,
        description="Smaller TB margin as %% of (top+bottom); tiers use this with safe_div smoothing",
    )
    margins_px: MarginsPx = Field(..., description="Measured margins in pixels")


class EstimatedGrades(BaseModel):
    """Demo centering-only estimates (not official PSA / BGS / CGC grades)."""

    PSA: float = Field(..., ge=5, le=10, description="PSA-style centering tier (whole steps)")
    BGS: float = Field(..., ge=5, le=10, description="BGS-style with half-point mapping from PSA tier")
    CGC: float = Field(..., ge=5, le=10, description="CGC-style whole-number mapping from PSA tier")


class AnalyzeCardResponse(BaseModel):
    """Result of POST /analyze-card."""

    centering: CenteringMetrics
    estimated_grades: EstimatedGrades
    warp_width: int | None = Field(None, description="Pixel width of perspective-corrected card crop")
    warp_height: int | None = Field(None, description="Pixel height of perspective-corrected card crop")
    detection_confidence: str = Field(
        "medium",
        description="Heuristic confidence in contour-based card detection: low|medium|high",
    )
