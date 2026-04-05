"""
Synthetic centering validation (edge projection).

Run from ``backend``::

    python -m unittest tests.test_centering_validation -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.services.card_detection import decode_image_from_bytes, extract_normalized_card
from app.services.centering_borders import measure_margins_combined, _try_yellow_frame_margins
from app.services.centering_grades import compute_centering_ratios
from app.services.centering_projection import measure_margins_edge_projection


def _synth_card(
    h: int,
    w: int,
    bl: int,
    br: int,
    bt: int,
    bb: int,
    outer_bgr: tuple[int, int, int] = (0, 220, 255),
    inner_bgr: tuple[int, int, int] = (120, 120, 120),
) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = outer_bgr
    y0, y1 = bt, h - bb
    x0, x1 = bl, w - br
    img[y0:y1, x0:x1] = inner_bgr
    return img


def _parse_split(s: str) -> int:
    a, _ = s.split("/")
    return int(a)


class TestCenteringProjectionSynthetic(unittest.TestCase):
    def test_near_50_50(self) -> None:
        h, w = 1120, 800
        m = 56
        im = _synth_card(h, w, m, m, m, m)
        l, r, t, b, meta = measure_margins_edge_projection(im)
        self.assertFalse(meta.get("rejected"), msg=str(meta))
        rd = compute_centering_ratios(l, r, t, b)
        lr = _parse_split(str(rd["lr_display"]))
        tb = _parse_split(str(rd["tb_display"]))
        self.assertLess(abs(lr - 50), 1.1, msg=f"LR got {lr} margins={l:.1f},{r:.1f}")
        self.assertLess(abs(tb - 50), 1.1, msg=f"TB got {tb} margins={t:.1f},{b:.1f}")

    def test_60_40_left_right(self) -> None:
        h, w = 1120, 800
        bl, br = 72, 48
        m = 56
        im = _synth_card(h, w, bl, br, m, m)
        l, r, t, b, meta = measure_margins_edge_projection(im)
        self.assertFalse(meta.get("rejected"), msg=str(meta))
        rd = compute_centering_ratios(l, r, t, b)
        lr = _parse_split(str(rd["lr_display"]))
        self.assertLess(abs(lr - 60), 1.1, msg=f"LR got {lr} expected ~60 margins={l:.1f},{r:.1f}")
        self.assertEqual(str(rd["lr_display"]), "60/40", msg=f"display {rd['lr_display']} margins={l:.3f},{r:.3f}")

    def test_40_60_left_right(self) -> None:
        h, w = 1120, 800
        bl, br = 48, 72
        m = 56
        im = _synth_card(h, w, bl, br, m, m)
        l, r, t, b, meta = measure_margins_edge_projection(im)
        self.assertFalse(meta.get("rejected"), msg=str(meta))
        rd = compute_centering_ratios(l, r, t, b)
        self.assertEqual(str(rd["lr_display"]), "40/60", msg=f"display {rd['lr_display']} margins={l:.3f},{r:.3f}")

    def test_60_40_combined_with_yellow_nudge(self) -> None:
        h, w = 1120, 800
        im = _synth_card(h, w, 72, 48, 56, 56)
        l, r, t, b, meta = measure_margins_combined(im)
        self.assertFalse(meta.get("rejected"), msg=str(meta))
        rd = compute_centering_ratios(l, r, t, b)
        self.assertEqual(str(rd["lr_display"]), "60/40", msg=f"{meta.get('method')} {rd['lr_display']} {l:.3f},{r:.3f}")
        self.assertIn(
            meta.get("method"),
            ("edge_projection+yellow_nudge", "yellow_hsv", "yellow_hsv_preferred_tb_vs_projection"),
            msg=str(meta),
        )

    def test_55_45_top_bottom(self) -> None:
        h, w = 1120, 800
        bt, bb = 62, 50
        m = 56
        im = _synth_card(h, w, m, m, bt, bb)
        l, r, t, b, meta = measure_margins_edge_projection(im)
        self.assertFalse(meta.get("rejected"), msg=str(meta))
        rd = compute_centering_ratios(l, r, t, b)
        tb = _parse_split(str(rd["tb_display"]))
        exp = 100.0 * bt / (bt + bb)
        self.assertLess(abs(tb - exp), 1.1, msg=f"TB got {tb} expected ~{exp:.0f} t,b={t:.1f},{b:.1f}")


class TestYellowTbRobustness(unittest.TestCase):
    """TB seams must not trigger on full-width dips from center-only print voids / text."""

    def test_side_lobe_row_profile_ignores_center_hole_in_top_bar(self) -> None:
        h, w = 1120, 800
        b = 56
        im = _synth_card(h, w, b, b, b, b)
        im[b : b + 24, int(0.38 * w) : int(0.62 * w), :] = (70, 70, 70)
        got = _try_yellow_frame_margins(im)
        self.assertIsNotNone(got, msg="yellow frame should still be detected")
        _l, _r, t, bb, _meta = got
        self.assertLess(abs(float(t) - b), 6.0, msg=f"top seam {t} expected ~{b} (not center-void early)")
        self.assertLess(abs(float(bb) - b), 6.0, msg=f"bottom seam {bb} expected ~{b}")


class TestYellowBorderBlueCorePrefersYellow(unittest.TestCase):
    """Yellow-border + blue art matches blue-panel HSV; combined must still use yellow seams."""

    def test_skips_blue_panel_when_rim_is_yellow(self) -> None:
        h, w = 1120, 800
        b = 52
        im = np.zeros((h, w, 3), dtype=np.uint8)
        im[:, :] = (0, 230, 255)
        roi_h, roi_w = h - 2 * b, w - 2 * b
        hsv_fill = np.full((roi_h, roi_w, 3), (105, 200, 220), dtype=np.uint8)
        inner = cv2.cvtColor(hsv_fill, cv2.COLOR_HSV2BGR)
        inner[::3, :, :] = 60
        im[b : h - b, b : w - b] = inner
        _l, _r, _t, _b, meta = measure_margins_combined(im)
        self.assertTrue(meta.get("skipped_blue_panel"), msg=str(meta))
        self.assertEqual(meta.get("method"), "yellow_hsv")


class TestReferenceSilverBlueCentered(unittest.TestCase):
    """Regression: SV-style silver border + blue face must not use wrong vertical edges for LR."""

    _fixture = Path(__file__).resolve().parent / "fixtures" / "reference_silver_blue_centered.png"

    def test_fixture_uses_blue_panel_and_balanced_lr(self) -> None:
        if not self._fixture.is_file():
            self.skipTest("reference fixture missing")
        raw = decode_image_from_bytes(self._fixture.read_bytes())
        warped, _ = extract_normalized_card(raw, target_height=settings.warp_height)
        l, r, t, b, meta = measure_margins_combined(warped)
        self.assertEqual(meta.get("method"), "blue_panel_hsv", msg=str(meta))
        rd = compute_centering_ratios(l, r, t, b)
        lr = _parse_split(str(rd["lr_display"]))
        tb = _parse_split(str(rd["tb_display"]))
        self.assertLess(abs(lr - 50), 4.0, msg=f"LR {lr} margins L,R={l:.1f},{r:.1f}")
        self.assertLess(abs(tb - 50), 4.0, msg=f"TB {tb} margins T,B={t:.1f},{b:.1f}")
        self.assertLess(abs(l - r), max(6.0, 0.04 * (l + r)), msg=f"asymmetric LR {l:.1f},{r:.1f}")
        self.assertLess(abs(t - b), max(6.0, 0.06 * (t + b)), msg=f"asymmetric TB {t:.1f},{b:.1f}")


if __name__ == "__main__":
    unittest.main()
