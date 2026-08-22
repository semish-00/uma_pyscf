"""Applying a QC config to a batch: the verdicts, the histories, and the refusals.

The interesting properties are the ones a later reader of a record depends on:
that an existing history is preserved and exactly one entry is appended, that
the caller's timestamp is written verbatim, that nothing outside the `qc` block
changes, and that a batch which cannot be judged honestly stops instead of
producing a report that looks complete.
"""

from __future__ import annotations

from dataclasses import replace
import json
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.ids import canonical_json_fingerprint
from uma_pyscf.qc.run import apply_qc, composition_formula
from uma_pyscf.schemas.label_record import (
    ElectronicState,
    Engine,
    LabelRecord,
    Method,
    QcState,
    RawArtifact,
    Results,
    Structure,
)

UTC = "2026-08-22T04:05:06+00:00"

CONFIG = {
    "schema_version": 1,
    "qc_id": "unit_qc_v1",
    "electronic": {
        "require_converged": True,
        "s2_max_abs_deviation": 0.05,
        "require_s2_for_open_shell": True,
        "gradient_max_abs_hartree_per_bohr": 1.0,
        "gradient_norm_max_hartree_per_bohr": 2.0,
    },
    "geometry": {
        "covalent_factor": 0.65,
        "bond_factor": 1.3,
        "allow_fragments": False,
        "duplicate_decimals": 3,
    },
}
CONFIG_SHA256 = canonical_json_fingerprint(CONFIG)

Vectors = tuple[tuple[float, float, float], ...]


def stretched(distance: float) -> Vectors:
    """Return an H2 geometry with the two atoms `distance` angstrom apart."""
    return ((0.0, 0.0, 0.0), (0.0, 0.0, distance))


H2: Vectors = stretched(0.74144)
COLLIDED: Vectors = stretched(0.2)
SMALL_GRADIENT: Vectors = ((0.0, 0.0, -0.01), (0.0, 0.0, 0.01))
HUGE_GRADIENT: Vectors = ((0.0, 0.0, -3.0), (0.0, 0.0, 3.0))


def make_record(
    *,
    record_id: str = "h2_a",
    positions: Vectors = H2,
    gradient: Vectors = SMALL_GRADIENT,
    charge: int = 0,
    multiplicity: int = 1,
    converged: bool = True,
    energy: float = -1.1730407,
    status: str = "pending",
    history: tuple[dict[str, object], ...] = (),
) -> LabelRecord:
    """Return an H2 label record with the facts a test cares about."""
    return LabelRecord(
        record_id=record_id,
        structure=Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=positions,
            parent_structure_id="h2_seed",
            sampling_method="bond_scan",
        ),
        state=ElectronicState(charge=charge, multiplicity=multiplicity, spin_2s=multiplicity - 1),
        method=Method(
            functional="wb97m-v",
            basis="def2-tzvpd",
            ecp=None,
            aux_basis=None,
            grid_level=3,
            nlc_grid_level=1,
            grid_response=True,
            density_fit=False,
            scf_conv_tol=1e-10,
            scf_max_cycle=200,
        ),
        engine=Engine(name="gpu4pyscf", versions={"pyscf": "2.6.2"}),
        results=Results(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient,
            converged=converged,
            s2=None if multiplicity == 1 else 0.7501,
            s2_target=None if multiplicity == 1 else 0.75,
            s2_deviation=None if multiplicity == 1 else 0.0001,
        ),
        raw=RawArtifact(),
        qc=QcState(status=status, history=history),
    )


class CompositionFormulaTests(unittest.TestCase):
    def test_symbols_are_alphabetical_and_a_count_of_one_is_omitted(self) -> None:
        self.assertEqual(composition_formula((14, 1, 1, 1, 1)), "H4Si")
        self.assertEqual(composition_formula((1, 17)), "ClH")

    def test_a_single_atom_is_its_symbol(self) -> None:
        self.assertEqual(composition_formula((32,)), "Ge")


class AcceptanceTests(unittest.TestCase):
    def test_a_clean_record_is_accepted_with_no_failed_checks(self) -> None:
        records, report = apply_qc([make_record()], CONFIG, utc=UTC)
        self.assertEqual(records[0].qc.status, "accepted")
        entry = report.entries[0]
        self.assertEqual(entry["status"], "accepted")
        self.assertEqual(entry["failed_checks"], [])

    def test_an_accepted_record_still_reports_every_check_that_ran(self) -> None:
        _, report = apply_qc([make_record()], CONFIG, utc=UTC)
        names = [check["name"] for check in report.entries[0]["checks"]]
        self.assertEqual(
            names,
            [
                "converged",
                "s2_deviation",
                "gradient_max_component",
                "gradient_norm",
                "minimum_distance",
                "fragments",
                "duplicate",
            ],
        )

    def test_every_check_dict_is_json_safe(self) -> None:
        _, report = apply_qc([make_record()], CONFIG, utc=UTC)
        for check in report.entries[0]["checks"]:
            with self.subTest(check=check["name"]):
                self.assertEqual(sorted(check), ["name", "observed", "passed", "threshold"])
                self.assertEqual(json.loads(json.dumps(check)), check)


class HistoryTests(unittest.TestCase):
    def test_the_appended_entry_states_the_run_the_config_and_the_result(self) -> None:
        records, _ = apply_qc([make_record()], CONFIG, utc=UTC)
        self.assertEqual(
            records[0].qc.history[-1],
            {
                "utc": UTC,
                "event": "qc_evaluated",
                "qc_id": "unit_qc_v1",
                "config_sha256": CONFIG_SHA256,
                "result": "accepted",
                "failed_checks": [],
            },
        )

    def test_a_rejected_record_lists_its_failed_checks_in_the_history(self) -> None:
        records, _ = apply_qc([make_record(gradient=HUGE_GRADIENT)], CONFIG, utc=UTC)
        entry = records[0].qc.history[-1]
        self.assertEqual(entry["result"], "rejected")
        self.assertEqual(entry["failed_checks"], ["gradient_max_component", "gradient_norm"])

    def test_an_existing_history_is_preserved_and_exactly_one_entry_is_added(self) -> None:
        earlier = {"utc": "2026-08-13T00:00:00+00:00", "event": "imported", "note": "part i"}
        records, _ = apply_qc([make_record(history=(earlier,))], CONFIG, utc=UTC)
        history = records[0].qc.history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], earlier)
        self.assertEqual(history[1]["event"], "qc_evaluated")

    def test_the_caller_timestamp_is_written_verbatim(self) -> None:
        stamp = "not-really-a-timestamp-but-a-string"
        records, _ = apply_qc([make_record()], CONFIG, utc=stamp)
        self.assertEqual(records[0].qc.history[-1]["utc"], stamp)

    def test_an_empty_timestamp_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            apply_qc([make_record()], CONFIG, utc="")


class RecordPreservationTests(unittest.TestCase):
    def test_nothing_outside_the_qc_block_changes(self) -> None:
        record = make_record()
        judged, _ = apply_qc([record], CONFIG, utc=UTC)
        before = record.to_dict()
        after = judged[0].to_dict()
        del before["qc"], after["qc"]
        self.assertEqual(before, after)

    def test_the_input_record_is_left_pending(self) -> None:
        record = make_record()
        apply_qc([record], CONFIG, utc=UTC)
        self.assertEqual(record.qc.status, "pending")
        self.assertEqual(record.qc.history, ())


class MixedBatchTests(unittest.TestCase):
    def batch(self) -> list[LabelRecord]:
        """Return one clean record and three records that each fail differently.

        Every geometry differs, so no record here is a duplicate of another and
        each rejection has exactly the one reason it was built to have.
        """
        return [
            make_record(record_id="h2_clean"),
            make_record(record_id="h2_unconverged", positions=stretched(0.75), converged=False),
            make_record(record_id="h2_collided", positions=COLLIDED),
            make_record(
                record_id="h2_forceful", positions=stretched(0.76), gradient=HUGE_GRADIENT
            ),
        ]

    def test_each_record_gets_the_status_its_checks_imply(self) -> None:
        records, _ = apply_qc(self.batch(), CONFIG, utc=UTC)
        self.assertEqual(
            [(record.record_id, record.qc.status) for record in records],
            [
                ("h2_clean", "accepted"),
                ("h2_unconverged", "rejected"),
                ("h2_collided", "rejected"),
                ("h2_forceful", "rejected"),
            ],
        )

    def test_the_report_names_the_reason_for_each_rejection(self) -> None:
        _, report = apply_qc(self.batch(), CONFIG, utc=UTC)
        failed = {entry["record_id"]: entry["failed_checks"] for entry in report.entries}
        self.assertEqual(
            failed,
            {
                "h2_clean": [],
                "h2_unconverged": ["converged"],
                "h2_collided": ["minimum_distance"],
                "h2_forceful": ["gradient_max_component", "gradient_norm"],
            },
        )

    def test_the_returned_records_keep_the_input_order(self) -> None:
        records, _ = apply_qc(self.batch()[::-1], CONFIG, utc=UTC)
        self.assertEqual(
            [record.record_id for record in records],
            ["h2_forceful", "h2_collided", "h2_unconverged", "h2_clean"],
        )

    def test_the_report_entries_are_sorted_by_record_id(self) -> None:
        _, report = apply_qc(self.batch()[::-1], CONFIG, utc=UTC)
        self.assertEqual(
            [entry["record_id"] for entry in report.entries],
            ["h2_clean", "h2_collided", "h2_forceful", "h2_unconverged"],
        )

    def test_the_distributions_count_only_accepted_records_in_the_ranges(self) -> None:
        _, report = apply_qc(self.batch(), CONFIG, utc=UTC)
        distributions = report.distributions
        self.assertEqual(distributions["counts"], {"accepted": 1, "rejected": 3, "total": 4})
        self.assertEqual(
            distributions["energy_hartree"],
            {"min": -1.1730407, "max": -1.1730407, "mean": -1.1730407},
        )
        self.assertEqual(distributions["by_composition"], {"H2": {"accepted": 1, "rejected": 3}})

    def test_a_batch_with_nothing_accepted_reports_no_range(self) -> None:
        _, report = apply_qc(self.batch()[1:], CONFIG, utc=UTC)
        self.assertIsNone(report.distributions["energy_hartree"])
        self.assertIsNone(report.distributions["gradient_max_abs_hartree_per_bohr"])


class DuplicateBatchTests(unittest.TestCase):
    def test_the_later_duplicate_is_rejected_and_names_the_kept_record(self) -> None:
        records, report = apply_qc(
            [make_record(record_id="h2_first"), make_record(record_id="h2_second")],
            CONFIG,
            utc=UTC,
        )
        self.assertEqual([record.qc.status for record in records], ["accepted", "rejected"])
        duplicate = next(
            check
            for entry in report.entries
            if entry["record_id"] == "h2_second"
            for check in entry["checks"]
            if check["name"] == "duplicate"
        )
        self.assertEqual(duplicate["observed"], "h2_first")

    def test_charge_siblings_are_both_accepted(self) -> None:
        records, _ = apply_qc(
            [
                make_record(record_id="h2_neutral"),
                make_record(record_id="h2_cation", charge=1, multiplicity=2),
            ],
            CONFIG,
            utc=UTC,
        )
        self.assertEqual([record.qc.status for record in records], ["accepted", "accepted"])


class RefusalTests(unittest.TestCase):
    def test_a_record_that_already_has_a_verdict_stops_the_run_by_name(self) -> None:
        judged = replace(make_record(record_id="h2_done"), qc=QcState(status="accepted"))
        with self.assertRaises(ValidationError) as caught:
            apply_qc([make_record(), judged], CONFIG, utc=UTC)
        message = str(caught.exception)
        self.assertIn("h2_done", message)
        self.assertIn("pending", message)

    def test_a_repeated_record_id_stops_the_run(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            apply_qc([make_record(), make_record()], CONFIG, utc=UTC)
        self.assertIn("h2_a", str(caught.exception))

    def test_something_that_is_not_a_record_stops_the_run(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            apply_qc([make_record().to_dict()], CONFIG, utc=UTC)  # type: ignore[list-item]
        self.assertIn("records[0]", str(caught.exception))

    def test_an_unknown_config_key_stops_the_run(self) -> None:
        broken = {**CONFIG, "retry": True}
        with self.assertRaises(ValidationError) as caught:
            apply_qc([make_record()], broken, utc=UTC)
        self.assertIn("'retry'", str(caught.exception))

    def test_a_missing_threshold_stops_the_run(self) -> None:
        electronic = {
            key: value
            for key, value in CONFIG["electronic"].items()  # type: ignore[union-attr]
            if key != "gradient_norm_max_hartree_per_bohr"
        }
        with self.assertRaises(ValidationError) as caught:
            apply_qc([make_record()], {**CONFIG, "electronic": electronic}, utc=UTC)
        self.assertIn("gradient_norm_max_hartree_per_bohr", str(caught.exception))

    def test_an_empty_batch_produces_an_empty_report(self) -> None:
        records, report = apply_qc([], CONFIG, utc=UTC)
        self.assertEqual(records, ())
        self.assertEqual(report.counts, {"accepted": 0, "rejected": 0, "total": 0})


class ReportProvenanceTests(unittest.TestCase):
    def test_the_report_embeds_the_config_and_fingerprints_it(self) -> None:
        _, report = apply_qc([make_record()], CONFIG, utc=UTC)
        self.assertEqual(report.qc_id, "unit_qc_v1")
        self.assertEqual(report.config, CONFIG)
        self.assertEqual(report.config_sha256, CONFIG_SHA256)

    def test_the_record_history_and_the_report_name_the_same_config(self) -> None:
        records, report = apply_qc([make_record()], CONFIG, utc=UTC)
        self.assertEqual(records[0].qc.history[-1]["config_sha256"], report.config_sha256)

    def test_the_report_is_identical_for_the_same_records_and_config(self) -> None:
        first = apply_qc(self_records(), CONFIG, utc=UTC)[1].to_dict()
        second = apply_qc(self_records(), CONFIG, utc="a-different-instant")[1].to_dict()
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))


def self_records() -> list[LabelRecord]:
    """Return a small mixed batch, rebuilt each call so no state is shared."""
    return [
        make_record(record_id="h2_clean"),
        make_record(record_id="h2_forceful", positions=stretched(0.76), gradient=HUGE_GRADIENT),
    ]


if __name__ == "__main__":
    unittest.main()
