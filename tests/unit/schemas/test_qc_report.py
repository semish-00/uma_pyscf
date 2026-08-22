"""The QC report schema: round trips, and the three things it refuses to believe.

A report states a config digest, a verdict per record, and a set of
distributions. All three are derived from content the report itself carries, so
all three are recomputed on the way in: a report whose digest, whose verdict, or
whose counts have been edited is refused rather than read.
"""

from __future__ import annotations

import json
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.ids import canonical_json_fingerprint
from uma_pyscf.schemas.qc_report import QC_REPORT_SCHEMA, QcReport

CONFIG = {
    "schema_version": 1,
    "qc_id": "unit_qc_v1",
    "electronic": {"require_converged": True, "gradient_max_abs_hartree_per_bohr": 1.0},
    "geometry": {"covalent_factor": 0.65},
}
CONFIG_SHA256 = canonical_json_fingerprint(CONFIG)

PASSING_CHECK = {
    "name": "converged",
    "passed": True,
    "observed": True,
    "threshold": True,
}
FAILING_CHECK = {
    "name": "gradient_max_component",
    "passed": False,
    "observed": 3.0,
    "threshold": 1.0,
}


def entry(
    record_id: str = "h2_a",
    *,
    checks: list[dict[str, object]] | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    composition: str = "H2",
    atom_count: int = 2,
    energy: float = -1.17,
    gradient_max: float = 0.01,
) -> dict[str, object]:
    """Return one report entry whose status follows from its checks."""
    resolved = checks if checks is not None else [dict(PASSING_CHECK)]
    failed = [check["name"] for check in resolved if not check["passed"]]
    return {
        "record_id": record_id,
        "status": "rejected" if failed else "accepted",
        "checks": resolved,
        "failed_checks": failed,
        "composition": composition,
        "charge": charge,
        "multiplicity": multiplicity,
        "atom_count": atom_count,
        "energy_hartree": energy,
        "gradient_max_abs_hartree_per_bohr": gradient_max,
    }


def report(*entries: dict[str, object]) -> QcReport:
    """Return a report over `entries` with the reference config."""
    return QcReport(
        qc_id="unit_qc_v1",
        config_sha256=CONFIG_SHA256,
        config=CONFIG,
        entries=entries or (entry(),),
    )


def round_trip(data: dict[str, object]) -> QcReport:
    """Return `data` read back through JSON, the way a written report is read."""
    return QcReport.from_dict(json.loads(json.dumps(data)))


class RoundTripTests(unittest.TestCase):
    def test_a_report_survives_a_json_round_trip_unchanged(self) -> None:
        original = report(entry("h2_a"), entry("h2_b", checks=[dict(FAILING_CHECK)]))
        self.assertEqual(round_trip(original.to_dict()).to_dict(), original.to_dict())

    def test_the_schema_string_is_written_and_required(self) -> None:
        self.assertEqual(report().to_dict()["schema"], QC_REPORT_SCHEMA)
        data = report().to_dict()
        data["schema"] = "uma-pyscf-qc-report-v0"
        with self.assertRaises(ValidationError) as caught:
            QcReport.from_dict(data)
        self.assertIn(QC_REPORT_SCHEMA, str(caught.exception))

    def test_entries_are_sorted_by_record_id(self) -> None:
        built = report(entry("h2_z"), entry("h2_a"))
        self.assertEqual([item["record_id"] for item in built.entries], ["h2_a", "h2_z"])

    def test_an_unknown_top_level_key_is_refused(self) -> None:
        data = report().to_dict()
        data["generated_utc"] = "2026-08-22T00:00:00+00:00"
        with self.assertRaises(ValidationError) as caught:
            QcReport.from_dict(data)
        self.assertIn("generated_utc", str(caught.exception))

    def test_an_unknown_entry_key_is_refused(self) -> None:
        data = report().to_dict()
        data["entries"][0]["reason"] = "because"  # type: ignore[index]
        with self.assertRaises(ValidationError) as caught:
            QcReport.from_dict(data)
        self.assertIn("reason", str(caught.exception))

    def test_an_unknown_check_key_is_refused(self) -> None:
        data = report().to_dict()
        data["entries"][0]["checks"][0]["skipped"] = True  # type: ignore[index]
        with self.assertRaises(ValidationError) as caught:
            QcReport.from_dict(data)
        self.assertIn("skipped", str(caught.exception))

    def test_the_report_carries_no_timestamp(self) -> None:
        # The report's own structure states no clock. An embedded config may of
        # course carry a `created` header of its own; that is the config's
        # provenance, not the report's, and it is fixed by the config file.
        text = json.dumps(report().to_dict())
        for token in ("utc", "timestamp", "generated_at"):
            with self.subTest(token=token):
                self.assertNotIn(token, text)


class ConfigDigestTests(unittest.TestCase):
    def test_a_digest_that_does_not_match_the_config_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            QcReport(qc_id="unit_qc_v1", config_sha256="a" * 64, config=CONFIG, entries=(entry(),))
        self.assertIn("fingerprints to", str(caught.exception))

    def test_an_edited_config_no_longer_matches_its_digest(self) -> None:
        data = report().to_dict()
        data["config"]["geometry"]["covalent_factor"] = 0.9  # type: ignore[index]
        with self.assertRaises(ValidationError):
            QcReport.from_dict(data)

    def test_a_digest_that_is_not_a_sha256_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            QcReport(qc_id="unit_qc_v1", config_sha256="abc", config=CONFIG, entries=(entry(),))
        self.assertIn("64 hexadecimal", str(caught.exception))


class EntryConsistencyTests(unittest.TestCase):
    def test_a_status_that_contradicts_the_failed_list_is_refused(self) -> None:
        data = report(entry("h2_a", checks=[dict(FAILING_CHECK)])).to_dict()
        data["entries"][0]["status"] = "accepted"  # type: ignore[index]
        with self.assertRaises(ValidationError) as caught:
            QcReport.from_dict(data)
        self.assertIn("if and only if", str(caught.exception))

    def test_a_failed_list_that_contradicts_the_checks_is_refused(self) -> None:
        data = report().to_dict()
        data["entries"][0]["failed_checks"] = ["converged"]  # type: ignore[index]
        data["entries"][0]["status"] = "rejected"  # type: ignore[index]
        with self.assertRaises(ValidationError) as caught:
            QcReport.from_dict(data)
        self.assertIn("failed_checks", str(caught.exception))

    def test_an_entry_with_no_checks_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            report(entry("h2_a", checks=[]))
        self.assertIn("at least one check", str(caught.exception))

    def test_an_unknown_status_is_refused(self) -> None:
        data = report().to_dict()
        data["entries"][0]["status"] = "pending"  # type: ignore[index]
        with self.assertRaises(ValidationError) as caught:
            QcReport.from_dict(data)
        self.assertIn("'accepted'", str(caught.exception))

    def test_a_repeated_record_id_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            report(entry("h2_a"), entry("h2_a"))
        self.assertIn("exactly once", str(caught.exception))

    def test_a_missing_entry_field_is_refused_by_name(self) -> None:
        data = report().to_dict()
        del data["entries"][0]["composition"]  # type: ignore[attr-defined]
        with self.assertRaises(ValidationError) as caught:
            QcReport.from_dict(data)
        self.assertIn("composition", str(caught.exception))

    def test_a_negative_gradient_magnitude_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            report(entry("h2_a", gradient_max=-0.5))
        self.assertIn("must not be negative", str(caught.exception))

    def test_a_check_whose_observation_is_not_json_is_refused(self) -> None:
        broken = {"name": "duplicate", "passed": True, "observed": {1, 2}, "threshold": 3}
        with self.assertRaises(ValidationError):
            report(entry("h2_a", checks=[broken]))


class DistributionTests(unittest.TestCase):
    def built(self) -> QcReport:
        """Return a report with a mix of verdicts, charges, and compositions."""
        return report(
            entry("h2_a", energy=-1.0, gradient_max=0.10),
            entry("h2_b", energy=-2.0, gradient_max=0.20, charge=1, multiplicity=2),
            entry(
                "sih4_c",
                checks=[dict(FAILING_CHECK)],
                composition="H4Si",
                atom_count=5,
                energy=-291.0,
                gradient_max=3.0,
            ),
        )

    def test_the_counts_summarize_the_verdicts(self) -> None:
        self.assertEqual(
            self.built().distributions["counts"],
            {"accepted": 2, "rejected": 1, "total": 3},
        )

    def test_each_breakdown_reports_both_verdicts_per_group(self) -> None:
        distributions = self.built().distributions
        self.assertEqual(
            distributions["by_composition"],
            {"H2": {"accepted": 2, "rejected": 0}, "H4Si": {"accepted": 0, "rejected": 1}},
        )
        self.assertEqual(
            distributions["by_charge"],
            {"0": {"accepted": 1, "rejected": 1}, "1": {"accepted": 1, "rejected": 0}},
        )
        self.assertEqual(
            distributions["by_multiplicity"],
            {"1": {"accepted": 1, "rejected": 1}, "2": {"accepted": 1, "rejected": 0}},
        )
        self.assertEqual(
            distributions["by_atom_count"],
            {"2": {"accepted": 2, "rejected": 0}, "5": {"accepted": 0, "rejected": 1}},
        )

    def test_the_ranges_ignore_rejected_records(self) -> None:
        distributions = self.built().distributions
        self.assertEqual(distributions["energy_hartree"], {"min": -2.0, "max": -1.0, "mean": -1.5})
        self.assertEqual(
            distributions["gradient_max_abs_hartree_per_bohr"],
            {"min": 0.10, "max": 0.20, "mean": 0.15000000000000002},
        )

    def test_a_report_with_nothing_accepted_has_no_ranges(self) -> None:
        built = report(entry("h2_a", checks=[dict(FAILING_CHECK)]))
        self.assertIsNone(built.distributions["energy_hartree"])
        self.assertIsNone(built.distributions["gradient_max_abs_hartree_per_bohr"])

    def test_an_empty_report_counts_nothing(self) -> None:
        empty = QcReport(
            qc_id="unit_qc_v1", config_sha256=CONFIG_SHA256, config=CONFIG, entries=()
        )
        self.assertEqual(empty.counts, {"accepted": 0, "rejected": 0, "total": 0})
        self.assertEqual(empty.distributions["by_charge"], {})

    def test_a_tampered_count_is_refused(self) -> None:
        data = self.built().to_dict()
        data["distributions"]["counts"]["accepted"] = 3  # type: ignore[index]
        with self.assertRaises(ValidationError) as caught:
            QcReport.from_dict(data)
        self.assertIn("never stated apart", str(caught.exception))

    def test_a_tampered_breakdown_is_refused(self) -> None:
        data = self.built().to_dict()
        data["distributions"]["by_charge"]["9"] = {  # type: ignore[index]
            "accepted": 1,
            "rejected": 0,
        }
        with self.assertRaises(ValidationError):
            QcReport.from_dict(data)

    def test_a_tampered_energy_range_is_refused(self) -> None:
        data = self.built().to_dict()
        data["distributions"]["energy_hartree"]["mean"] = 0.0  # type: ignore[index]
        with self.assertRaises(ValidationError):
            QcReport.from_dict(data)

    def test_the_distributions_survive_a_round_trip(self) -> None:
        original = self.built()
        self.assertEqual(round_trip(original.to_dict()).distributions, original.distributions)


if __name__ == "__main__":
    unittest.main()
