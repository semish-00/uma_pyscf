"""Versioned candidate-portfolio report schema."""

from __future__ import annotations

import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.ids import canonical_json_fingerprint
from uma_pyscf.schemas.portfolio import PortfolioReport, PortfolioSourceSummary


class PortfolioReportTests(unittest.TestCase):
    def report(self) -> PortfolioReport:
        config = {"portfolio_id": "calibration_v1", "seed": 7}
        return PortfolioReport(
            portfolio_id="calibration_v1",
            config_sha256=canonical_json_fingerprint(config),
            config=config,
            counts={"source_manifests": 1, "available": 3, "selected": 2},
            sources=(
                PortfolioSourceSummary(
                    category="local",
                    source_id="local_pool",
                    source_sha256="a" * 64,
                    quota=2,
                    available_count=3,
                    selected_record_ids=("local_1", "local_2"),
                    skipped_counts={
                        "duplicate_geometry_state": 0,
                        "parent_limit": 0,
                        "trajectory_limit": 0,
                        "quota_reached": 1,
                    },
                ),
            ),
        )

    def test_round_trip_is_lossless(self) -> None:
        report = self.report()
        self.assertEqual(PortfolioReport.from_dict(report.to_dict()), report)
        self.assertEqual(report.selected_record_ids, ("local_1", "local_2"))

    def test_config_digest_and_selected_count_fail_closed(self) -> None:
        report = self.report().to_dict()
        report["config_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValidationError, "fingerprints"):
            PortfolioReport.from_dict(report)

        report = self.report().to_dict()
        report["counts"]["selected"] = 1  # type: ignore[index]
        with self.assertRaisesRegex(ValidationError, "selected"):
            PortfolioReport.from_dict(report)


if __name__ == "__main__":
    unittest.main()
