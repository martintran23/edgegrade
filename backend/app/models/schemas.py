"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field


class MarginsPx(BaseModel):
    """Margin depths from the warp edges (pixels, sub-pixel precision)."""

    left: float = Field(..., description="Left margin depth in px (typically 3 decimal places)")
    right: float = Field(..., description="Right margin depth in px (typically 3 decimal places)")
    top: float = Field(..., description="Top margin depth in px (typically 3 decimal places)")
    bottom: float = Field(..., description="Bottom margin depth in px (typically 3 decimal places)")


class CenteringMetrics(BaseModel):
    """Centering display strings plus values used for tier logic."""

    left_right: str = Field(..., description="Left vs right split, e.g. '55/45'")
    top_bottom: str = Field(..., description="Top vs bottom split, e.g. '52/48'")
    lr_small_pct: float = Field(
        ...,
        description="Smaller LR margin as %% of (left+right); PSA tiers use this value",
    )
    tb_small_pct: float = Field(
        ...,
        description="Smaller TB margin as %% of (top+bottom); PSA tiers use this value",
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
    centering_method: str | None = Field(
        None,
        description="Which seam detector produced margins (e.g. yellow_hsv, blue_panel_hsv, edge_projection)",
    )
    centering_build: str | None = Field(
        None,
        description="Pipeline build id — if this does not match your checkout, the API is stale",
    )
