"""
PSA-style centering **classification** from raw margins (threshold-based, not regression).

``lr_small`` / ``tb_small`` are the percentage that the **smaller** margin represents of
the total border on that axis — the usual way centering splits are read for subgrades.
"""

from __future__ import annotations


def _nearest_pct_display(p: float) -> tuple[int, int]:
    """Integer ``a/b`` with ``a + b == 100``; use banker's rounding (Python ``round``) so
    values like 40.50000002 do not become 41/59 from ``int(p + 0.5)``."""
    p = max(0.0, min(100.0, p))
    a = max(0, min(100, round(p)))
    return a, 100 - a


def pct_smaller_on_axis(a: float, b: float) -> float:
    """Smaller margin as a fraction of total border on that axis (standard centering read)."""
    s = a + b
    if s <= 1e-12:
        return 0.5
    return min(a, b) / s


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
        lr_small = 100.0 * pct_smaller_on_axis(left, right)
        lr_pct_left = 100.0 * left / s_lr
        lr_display_left, lr_display_right = _nearest_pct_display(lr_pct_left)

    if s_tb <= 0:
        tb_small = 50.0
        tb_display_top = 50
        tb_display_bottom = 50
    else:
        tb_small = 100.0 * pct_smaller_on_axis(top, bottom)
        tb_pct_top = 100.0 * top / s_tb
        tb_display_top, tb_display_bottom = _nearest_pct_display(tb_pct_top)

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

    lr_small = 100.0 * pct_smaller_on_axis(left, right)
    tb_small = 100.0 * pct_smaller_on_axis(top, bottom)

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
