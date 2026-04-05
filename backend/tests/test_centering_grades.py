"""Unit tests for threshold-based PSA centering grades."""

from __future__ import annotations

import unittest

from app.services.centering_grades import compute_centering_ratios, compute_psa_grade


class TestComputePsaGrade(unittest.TestCase):
    def test_perfect_centering(self) -> None:
        self.assertEqual(compute_psa_grade(100, 100, 100, 100), 10)

    def test_psa_10_boundary(self) -> None:
        self.assertEqual(compute_psa_grade(90, 110, 80, 120), 10)

    def test_psa_9(self) -> None:
        self.assertEqual(compute_psa_grade(80, 120, 70, 130), 9)

    def test_psa_8(self) -> None:
        self.assertEqual(compute_psa_grade(70, 130, 60, 140), 8)

    def test_ratios_keys(self) -> None:
        r = compute_centering_ratios(90, 110, 80, 120)
        self.assertIn("lr_small", r)
        self.assertIn("tb_small", r)
        self.assertEqual(r["lr_display"], "45/55")
        self.assertEqual(r["tb_display"], "40/60")


if __name__ == "__main__":
    unittest.main()
