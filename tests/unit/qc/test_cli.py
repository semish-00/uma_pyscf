"""The pieces the `qc` subcommand is built from: input resolution and output writing.

The end-to-end run of the subcommand lives in tests/unit/cli/test_main.py next
to the other subcommands. What is tested here is the part that decides *which*
files a run reads and *where* it writes, including the refusal that keeps a run
from overwriting its own inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.io import write_json_atomic
from uma_pyscf.qc.cli import load_records, resolve_record_paths, write_qc_outputs
from uma_pyscf.qc.config import load_qc_config
from uma_pyscf.qc.run import apply_qc
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
from uma_pyscf.schemas.qc_report import QcReport

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_QC_CONFIG = REPO_ROOT / "configs" / "datasets" / "example_qc_v1.yaml"
UTC = "2026-08-22T04:05:06+00:00"


def make_record(record_id: str = "h2_a", *, distance: float = 0.74144) -> LabelRecord:
    """Return a clean H2 label record awaiting QC."""
    return LabelRecord(
        record_id=record_id,
        structure=Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, distance)),
        ),
        state=ElectronicState(charge=0, multiplicity=1, spin_2s=0),
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
            energy_hartree=-1.1730407,
            gradient_hartree_per_bohr=((0.0, 0.0, -0.01), (0.0, 0.0, 0.01)),
            converged=True,
        ),
        raw=RawArtifact(),
        qc=QcState(status="pending"),
    )


class ResolveRecordPathsTests(unittest.TestCase):
    def test_a_file_resolves_to_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            write_json_atomic(path, make_record().to_dict())
            self.assertEqual(resolve_record_paths([str(path)]), (path,))

    def test_a_directory_resolves_to_its_json_files_in_name_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("c.json", "a.json", "b.json"):
                write_json_atomic(root / name, make_record().to_dict())
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            resolved = resolve_record_paths([str(root)])
        self.assertEqual([path.name for path in resolved], ["a.json", "b.json", "c.json"])

    def test_several_paths_are_concatenated_in_the_order_given(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            nested = root / "nested"
            write_json_atomic(first, make_record().to_dict())
            write_json_atomic(nested / "second.json", make_record().to_dict())
            resolved = resolve_record_paths([str(first), str(nested)])
        self.assertEqual([path.name for path in resolved], ["first.json", "second.json"])

    def test_a_directory_with_no_records_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValidationError) as caught:
                resolve_record_paths([directory])
        self.assertIn("no *.json record files", str(caught.exception))

    def test_no_paths_at_all_stops_the_run(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            resolve_record_paths([])
        self.assertIn("nothing to quality control", str(caught.exception))


class LoadRecordsTests(unittest.TestCase):
    def test_valid_records_are_returned_in_path_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_atomic(root / "a.json", make_record("h2_a").to_dict())
            write_json_atomic(root / "b.json", make_record("h2_b").to_dict())
            records = load_records(resolve_record_paths([str(root)]))
        self.assertEqual([record.record_id for record in records], ["h2_a", "h2_b"])

    def test_a_file_that_is_not_a_record_is_reported_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text('{"schema": "uma-pyscf-candidate-manifest-v1"}', encoding="utf-8")
            with self.assertRaises(ValidationError) as caught:
                load_records((path,))
        message = str(caught.exception)
        self.assertIn("broken.json", message)
        self.assertIn("uma-pyscf-label-record-v1", message)

    def test_a_missing_file_is_reported_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.json"
            with self.assertRaises(ValidationError) as caught:
                load_records((missing,))
        self.assertIn("absent.json", str(caught.exception))


class WriteOutputsTests(unittest.TestCase):
    def judge(self, records: list[LabelRecord]) -> tuple[tuple[LabelRecord, ...], QcReport]:
        """Run the committed example config over `records`."""
        return apply_qc(records, load_qc_config(EXAMPLE_QC_CONFIG), utc=UTC)

    def test_records_are_written_under_records_and_the_report_beside_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            judged, report = self.judge([make_record("h2_a"), make_record("h2_b", distance=0.8)])
            written, report_path = write_qc_outputs(judged, report, root)
            self.assertEqual(
                [path.relative_to(root).as_posix() for path in written],
                ["records/h2_a.json", "records/h2_b.json"],
            )
            self.assertEqual(report_path, root / "example_qc_v1_report.json")
            self.assertTrue(all(path.is_file() for path in written))
            written_record = json.loads(written[0].read_text(encoding="utf-8"))
        self.assertEqual(written_record["qc"]["status"], "accepted")
        self.assertEqual(written_record["qc"]["history"][0]["utc"], UTC)

    def test_writing_over_an_input_record_is_refused_before_anything_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "records" / "h2_a.json"
            write_json_atomic(source, make_record("h2_a").to_dict())
            before = source.read_bytes()
            judged, report = self.judge([make_record("h2_a")])
            with self.assertRaises(ValidationError) as caught:
                write_qc_outputs(judged, report, root, inputs=(source,))
            self.assertIn("would overwrite an input record", str(caught.exception))
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse((root / "example_qc_v1_report.json").exists())

    def test_the_report_is_byte_identical_for_the_same_records_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_judged, first_report = self.judge([make_record("h2_a")])
            _, first_path = write_qc_outputs(first_judged, first_report, root / "one")
            second_judged, second_report = apply_qc(
                [make_record("h2_a")], load_qc_config(EXAMPLE_QC_CONFIG), utc="another-instant"
            )
            _, second_path = write_qc_outputs(second_judged, second_report, root / "two")
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
