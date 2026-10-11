"""Tests for the pure end-to-end pipeline efficiency report."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from efficiency import pipeline_report


class TestPipelineReport(unittest.TestCase):
    def test_totals_and_percent(self):
        report = pipeline_report(10_000, 1_000, 40, 20)
        self.assertEqual(report["baseline_tokens"], 10_040)
        self.assertEqual(report["sent_tokens"], 1_020)
        self.assertEqual(report["tokens_saved"], 9_020)
        self.assertEqual(report["reduction_percent"], 89.8)
        self.assertEqual(report["stages"]["question"]["saved"], 20)
        self.assertIsNone(report["cost"])

    def test_cost_uses_caller_price(self):
        report = pipeline_report(1_000_000, 100_000, 0, 0, price_per_million=3.0)
        self.assertEqual(report["cost"]["baseline_usd"], 3.0)
        self.assertEqual(report["cost"]["optimized_usd"], 0.3)
        self.assertEqual(report["cost"]["saved_usd"], 2.7)
        self.assertEqual(report["cost"]["saved_usd_per_1000_queries"], 2700.0)

    def test_never_negative_and_no_divide_by_zero(self):
        report = pipeline_report(0, 50, 0, 0, price_per_million=1.0)
        self.assertEqual(report["tokens_saved"], 0)
        self.assertEqual(report["reduction_percent"], 0.0)
        self.assertEqual(report["cost"]["saved_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
