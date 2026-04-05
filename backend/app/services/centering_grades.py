"""
PSA-style centering **classification** from raw margins (threshold-based, not regression).

``lr_small`` / ``tb_small`` are the percentage that the **smaller** margin represents of
the total border on that axis — the usual way centering splits are read for subgrades.
"""

from __future__ import annotations


def safe_div(a: float, b: float) -> float:
    """Stabilize ratios against tiny pixel noise and divide-by-zero."""
    return (a + 1.0) / (b + 1.0)


def compute_centering_ratios(
    left: float, right: float, top: float, bottom: float
) -> dict[str, float | str]:
    s_lr = left + right
    s_tb = top + bottom

    if s_lr <= 0:
        lr_small = 50.0
        lr_display_left = 50
        lr_display_right = 50
    else:
        lr_small = 100.0 * safe_div(min(left, right), s_lr)
        lr_display_left = round(100.0 * left / s_lr)
        lr_display_right = 100 - lr_display_left  # complementary so split always sums to 100

    if s_tb <= 0:
        tb_small = 50.0
        tb_display_top = 50
        tb_display_bottom = 50
    else:
        tb_small = 100.0 * safe_div(min(top, bottom), s_tb)
        tb_display_top = round(100.0 * top / s_tb)
        tb_display_bottom = 100 - tb_display_top

    return {
        "lr_small": lr_small,
        "tb_small": tb_small,
        "lr_display": f"{lr_display_left}/{lr_display_right}",
        "tb_display": f"{tb_display_top}/{tb_display_bottom}",
    }


def compute_psa_grade(left: float, right: float, top: float, bottom: float) -> int:
    """
    Threshold-based PSA centering subgrade (both axes must clear each tier).

    This is an **educational mapping** inspired by common centering tables; not official PSA.
    """
    s_lr = left + right
    s_tb = top + bottom
    if s_lr <= 0 or s_tb <= 0:
        return 5

    lr_small = 100.0 * safe_div(min(left, right), s_lr)
    tb_small = 100.0 * safe_div(min(top, bottom), s_tb)

    # Top/bottom margins on real scans are often noisier than left/right; TB floors are
    # slightly looser than LR so a true 50/50 gem is not over-penalized on one axis.
    if lr_small >= 45 and tb_small >= 35:
        return 10
    if lr_small >= 40 and tb_small >= 30:
        return 9
    if lr_small >= 35 and tb_small >= 25:
        return 8
    if lr_small >= 30 and tb_small >= 22:
        return 7
    if lr_small >= 25 and tb_small >= 18:
        return 6
    return 5


def companion_estimated_grades(psa_tier: int) -> tuple[float, float, float]:
    """BGS half-point bump below 10; CGC mirrors PSA whole-number tier (demo mapping)."""
    psa_f = float(psa_tier)
    if psa_tier >= 10:
        return 10.0, 10.0, 10.0
    bgs = min(9.5, psa_f + 0.5)
    return psa_f, bgs, psa_f
