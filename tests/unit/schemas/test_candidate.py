"""The candidate manifest and geometry QC report schemas.

Both records are built from one small H2 candidate, and every rejection test
breaks exactly one thing about it, so what each check is responsible for stays
visible.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.ids import canonical_json_fingerprint
from uma_pyscf.core.io import read_json, write_json_atomic
from uma_pyscf.schemas.candidate import (
    CANDIDATE_MANIFEST_SCHEMA,
    GEOMETRY_QC_SCHEMA,
    CandidateManifest,
    CandidateRecord,
    GeometryQcReport,
)
from uma_pyscf.schemas.label_record import ElectronicState, Structure

CONFIG: dict[str, object] = {
    "schema_version": 1,
    "sampling_id": "h2_scan_v1",
    "structures": [{"id": "h2_seed", "xyz_path": "structures/h2.xyz"}],
    "operations": [
        {
            "kind": "bond_scan",
            "structure": "h2_seed",
            "charge": 0,
            "multiplicity": 1,
            "anchor_index": 0,
            "moved_index": 1,
            "factors": [0.9, 1.1],
        }
    ],
}
CONFIG_SHA256 = canonical_json_fingerprint(CONFIG)


def h2_candidate(record_id: str = "h2_scan_v1_h2_seed_bond01_x0p9") -> CandidateRecord:
    """Return a valid H2 candidate built through the dataclasses."""
    return CandidateRecord(
        record_id=record_id,
        structure=Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.667296)),
            parent_structure_id="h2_seed",
            sampling_method="bond_scan",
        ),
        state=ElectronicState(charge=0, multiplicity=1, spin_2s=0),
        generation_parameters={
            "operation": {"kind": "bond_scan", "factors": [0.9, 1.1]},
            "factor": 0.9,
        },
    )


def manifest(*records: CandidateRecord) -> CandidateManifest:
    """Return a manifest carrying ``records`` and the shared example config."""
    return CandidateManifest(
        sampling_id="h2_scan_v1",
        config_sha256=CONFIG_SHA256,
        config=dict(CONFIG),
        records=records or (h2_candidate(),),
    )


def report(*entries: dict[str, object]) -> GeometryQcReport:
    """Return a QC report carrying ``entries``."""
    return GeometryQcReport(sampling_id="h2_scan_v1", config_sha256=CONFIG_SHA256, entries=entries)


def accepted_entry(record_id: str = "h2_scan_v1_h2_seed_bond01_x0p9") -> dict[str, object]:
    """Return one accepted QC entry."""
    return {
        "record_id": record_id,
        "status": "accepted",
        "checks": {"finite_coordinates": True, "fragments": {"count": 1}},
        "reason": None,
    }


def rejected_entry(record_id: str = "h2_scan_v1_h2_seed_bond01_x1p1") -> dict[str, object]:
    """Return one rejected QC entry."""
    return {
        "record_id": record_id,
        "status": "rejected",
        "checks": {"finite_coordinates": True, "fragments": {"count": 2}},
        "reason": "the geometry separates into 2 fragments.",
    }


class CandidateRecordTests(unittest.TestCase):
    def test_a_valid_candidate_keeps_its_provenance(self) -> None:
        candidate = h2_candidate()
        self.assertEqual(candidate.structure.parent_structure_id, "h2_seed")
        self.assertEqual(candidate.structure.sampling_method, "bond_scan")
        self.assertEqual(candidate.electron_count, 2)

    def test_round_trip_through_dicts(self) -> None:
        candidate = h2_candidate()
        self.assertEqual(CandidateRecord.from_dict(candidate.to_dict()), candidate)

    def test_generation_parameters_stay_free_form(self) -> None:
        candidate = CandidateRecord(
            record_id="h2_free_form",
            structure=Structure(
                atomic_numbers=(1, 1), positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74))
            ),
            state=ElectronicState(charge=0, multiplicity=1, spin_2s=0),
            generation_parameters={"anything": {"nested": [1, 2.5, None, "text", True]}},
        )
        self.assertEqual(
            CandidateRecord.from_dict(candidate.to_dict()).generation_parameters,
            candidate.generation_parameters,
        )

    def test_generation_parameters_must_be_json(self) -> None:
        with self.assertRaises(ValidationError):
            CandidateRecord(
                record_id="h2_bad_parameters",
                structure=Structure(
                    atomic_numbers=(1, 1), positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74))
                ),
                state=ElectronicState(charge=0, multiplicity=1, spin_2s=0),
                generation_parameters={"factor": {1.0, 2.0}},  # type: ignore[dict-item]
            )

    def test_the_state_has_to_be_reachable_from_the_atoms(self) -> None:
        with self.assertRaises(ValidationError):
            CandidateRecord(
                record_id="h2_bad_parity",
                structure=Structure(
                    atomic_numbers=(1, 1), positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74))
                ),
                state=ElectronicState(charge=0, multiplicity=2, spin_2s=1),
            )

    def test_an_unknown_key_is_rejected(self) -> None:
        data = h2_candidate().to_dict() | {"energy_hartree": -1.17}
        with self.assertRaises(ValidationError) as caught:
            CandidateRecord.from_dict(data)
        self.assertIn("energy_hartree", str(caught.exception))

    def test_a_malformed_record_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            h2_candidate("H2 Scan")


class CandidateManifestTests(unittest.TestCase):
    def test_round_trip_through_a_written_file(self) -> None:
        original = manifest(h2_candidate(), h2_candidate("h2_scan_v1_h2_seed_bond01_x1p1"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            write_json_atomic(path, original.to_dict())
            restored = CandidateManifest.from_dict(read_json(path))
        self.assertEqual(restored, original)
        self.assertEqual(restored.schema, CANDIDATE_MANIFEST_SCHEMA)
        self.assertEqual(
            [record.record_id for record in restored.records],
            [record.record_id for record in original.records],
        )

    def test_the_config_is_embedded_verbatim(self) -> None:
        self.assertEqual(manifest().to_dict()["config"], CONFIG)

    def test_the_digest_is_derived_from_the_embedded_config(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            CandidateManifest(
                sampling_id="h2_scan_v1",
                config_sha256="0" * 64,
                config=dict(CONFIG),
                records=(h2_candidate(),),
            )
        self.assertIn("config_sha256", str(caught.exception))

    def test_an_edited_config_no_longer_matches_its_digest(self) -> None:
        data = manifest().to_dict()
        config = dict(data["config"])  # type: ignore[arg-type]
        config["sampling_id"] = "something_else"
        data["config"] = config
        with self.assertRaises(ValidationError):
            CandidateManifest.from_dict(data)

    def test_duplicate_record_ids_are_rejected(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            manifest(h2_candidate(), h2_candidate())
        self.assertIn("h2_scan_v1_h2_seed_bond01_x0p9", str(caught.exception))

    def test_an_empty_manifest_is_allowed(self) -> None:
        empty = CandidateManifest(
            sampling_id="h2_scan_v1", config_sha256=CONFIG_SHA256, config=dict(CONFIG)
        )
        self.assertEqual(empty.records, ())
        self.assertEqual(CandidateManifest.from_dict(empty.to_dict()), empty)

    def test_a_foreign_schema_string_is_refused_by_name(self) -> None:
        data = manifest().to_dict() | {"schema": "uma-pyscf-candidate-manifest-v0"}
        with self.assertRaises(ValidationError) as caught:
            CandidateManifest.from_dict(data)
        self.assertIn(CANDIDATE_MANIFEST_SCHEMA, str(caught.exception))

    def test_an_unknown_key_is_rejected(self) -> None:
        data = manifest().to_dict() | {"created": "2026-08-22"}
        with self.assertRaises(ValidationError) as caught:
            CandidateManifest.from_dict(data)
        self.assertIn("created", str(caught.exception))

    def test_a_malformed_digest_is_rejected(self) -> None:
        for digest in ("abc", "z" * 64, 1234):
            with self.subTest(digest=digest):
                with self.assertRaises(ValidationError):
                    CandidateManifest(
                        sampling_id="h2_scan_v1",
                        config_sha256=digest,  # type: ignore[arg-type]
                        config=dict(CONFIG),
                    )

    def test_a_malformed_sampling_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CandidateManifest(
                sampling_id="H2 Scan", config_sha256=CONFIG_SHA256, config=dict(CONFIG)
            )


class GeometryQcReportTests(unittest.TestCase):
    def test_round_trip_through_a_written_file(self) -> None:
        original = report(accepted_entry(), rejected_entry())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry_qc.json"
            write_json_atomic(path, original.to_dict())
            restored = GeometryQcReport.from_dict(read_json(path))
        self.assertEqual(restored, original)
        self.assertEqual(restored.schema, GEOMETRY_QC_SCHEMA)

    def test_the_counts_are_derived_from_the_entries(self) -> None:
        data = report(accepted_entry(), rejected_entry()).to_dict()
        self.assertEqual(data["counts"], {"total": 2, "accepted": 1, "rejected": 1})

    def test_counts_that_disagree_with_the_entries_are_rejected(self) -> None:
        data = report(accepted_entry(), rejected_entry()).to_dict()
        data["counts"] = {"total": 2, "accepted": 2, "rejected": 0}
        with self.assertRaises(ValidationError) as caught:
            GeometryQcReport.from_dict(data)
        self.assertIn("counts", str(caught.exception))

    def test_a_rejection_must_carry_a_reason(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            report(rejected_entry() | {"reason": None})
        self.assertIn("reason", str(caught.exception))

    def test_an_acceptance_must_not_carry_a_reason(self) -> None:
        with self.assertRaises(ValidationError):
            report(accepted_entry() | {"reason": "looked fine"})

    def test_an_unknown_status_is_rejected(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            report(accepted_entry() | {"status": "pending"})
        self.assertIn("status", str(caught.exception))

    def test_duplicate_entry_record_ids_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            report(accepted_entry(), accepted_entry())

    def test_an_unknown_entry_key_is_rejected(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            report(accepted_entry() | {"note": "extra"})
        self.assertIn("note", str(caught.exception))

    def test_checks_must_be_a_json_object(self) -> None:
        with self.assertRaises(ValidationError):
            report(accepted_entry() | {"checks": "everything passed"})

    def test_a_foreign_schema_string_is_refused_by_name(self) -> None:
        data = report(accepted_entry()).to_dict() | {"schema": "uma-pyscf-geometry-qc-v0"}
        with self.assertRaises(ValidationError) as caught:
            GeometryQcReport.from_dict(data)
        self.assertIn(GEOMETRY_QC_SCHEMA, str(caught.exception))

    def test_count_reports_one_status_at_a_time(self) -> None:
        built = report(accepted_entry(), rejected_entry())
        self.assertEqual(built.count("accepted"), 1)
        self.assertEqual(built.count("rejected"), 1)


if __name__ == "__main__":
    unittest.main()
