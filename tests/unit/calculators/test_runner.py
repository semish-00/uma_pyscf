"""Attempt ledger, direct fallback, atomic publication, and resume."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import tempfile
from typing import Any
import unittest

from uma_pyscf.calculators.config import load_dft_config
from uma_pyscf.calculators.model import CalculationFailure, CalculationOutput
from uma_pyscf.calculators.runner import LABEL_EVENT, build_label_plan, run_label_batch
from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.ids import canonical_json_fingerprint, sha256_of_file
from uma_pyscf.core.io import read_json
from uma_pyscf.schemas.candidate import CandidateManifest, CandidateRecord
from uma_pyscf.schemas.label_record import (
    ElectronicState,
    LabelRecord,
    Method,
    Results,
    Structure,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DFT_CONFIG = REPO_ROOT / "configs" / "dft" / "omol_wb97mv_tzvpd_v1.yaml"


def h2_candidate(record_id: str = "h2_case") -> CandidateRecord:
    return CandidateRecord(
        record_id=record_id,
        structure=Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)),
            parent_structure_id="h2_seed",
            sampling_method="bond_scan",
        ),
        state=ElectronicState(
            charge=0,
            multiplicity=1,
            spin_2s=0,
            state_provenance="ground_state_default",
        ),
    )


def manifest(*records: CandidateRecord) -> CandidateManifest:
    source = {"fixture": "p2.3"}
    return CandidateManifest(
        sampling_id="label_runner_fixture_v1",
        config_sha256=canonical_json_fingerprint(source),
        config=source,
        records=records,
    )


class FakeAdapter:
    """Return a canonical H2 result, optionally failing selected calls."""

    def __init__(self, config: Mapping[str, Any], failures: tuple[str | None, ...] = ()) -> None:
        self.config = config
        self.failures = failures
        self.calls: list[tuple[str, bool, str]] = []

    def calculate(
        self,
        candidate: CandidateRecord,
        method: Method,
        config: Mapping[str, Any],
        *,
        attempt_id: str,
        resource: Mapping[str, Any],
    ) -> CalculationOutput:
        del config, resource
        call_index = len(self.calls)
        self.calls.append((candidate.record_id, method.density_fit, attempt_id))
        if call_index < len(self.failures) and self.failures[call_index] is not None:
            category = self.failures[call_index]
            assert category is not None
            raise CalculationFailure(category, f"fixture failure: {category}")
        engine = self.config["engine"]
        versions = dict(engine["required_versions"])
        versions.update(
            {
                "cuda_runtime_version": "12080",
                "cuda_driver_version": "12020",
                "cuda_device_name": engine["required_gpu_name"],
                "container_image_sha256": "a" * 64,
                "python_lock_sha256": "b" * 64,
            }
        )
        return CalculationOutput(
            engine_name="gpu4pyscf",
            engine_versions=versions,
            results=Results(
                energy_hartree=-1.1,
                gradient_hartree_per_bohr=((0.0, 0.0, -0.01), (0.0, 0.0, 0.01)),
                converged=True,
                s2=0.0,
                s2_target=0.0,
                s2_deviation=0.0,
                wall_time_seconds=1.25,
            ),
            raw_payload={"fixture": True},
        )


def clock() -> Any:
    """Return a deterministic, monotonically labelled timestamp callable."""
    state = {"value": 0}

    def now() -> str:
        state["value"] += 1
        return f"2026-08-31T00:00:{state['value']:02d}+00:00"

    return now


class PlanTests(unittest.TestCase):
    def test_dry_run_contains_scope_resource_and_both_attempts(self) -> None:
        config = load_dft_config(DFT_CONFIG)
        plan = build_label_plan(manifest(h2_candidate()), config)
        self.assertEqual(plan["counts"], {"total": 1, "ready": 1, "blocked": 0})
        self.assertEqual(plan["records"][0]["resource"]["max_memory_mb"], 24000)
        self.assertEqual(
            [attempt["id"] for attempt in plan["records"][0]["attempts"]],
            ["primary_density_fit", "direct_fallback"],
        )


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_dft_config(DFT_CONFIG)
        self.manifest = manifest(h2_candidate())

    def test_primary_success_publishes_raw_record_ledger_and_summary(self) -> None:
        adapter = FakeAdapter(self.config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_label_batch(
                self.manifest, self.config, root, adapter, now=clock()
            )
            record_path = root / "records" / "h2_case.json"
            record = LabelRecord.from_dict(read_json(record_path))
            ledger = read_json(root / "attempt_ledger.json")
            raw_path = root / record.raw.logical_location
            self.assertTrue(raw_path.is_file())
            self.assertEqual(record.raw.checksum_sha256, sha256_of_file(raw_path))
            self.assertEqual(ledger["records"]["h2_case"]["status"], "completed")
            self.assertEqual(summary["counts"]["completed"], 1)
            self.assertFalse(summary["release_allowed"])
        self.assertEqual(adapter.calls, [("h2_case", True, "primary_density_fit")])
        self.assertEqual(record.state.initial_guess, "minao")
        self.assertEqual(record.qc.status, "pending")
        self.assertEqual(record.qc.history[0]["event"], LABEL_EVENT)
        self.assertTrue(record.method.density_fit)

    def test_resume_validates_and_skips_a_completed_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_label_batch(
                self.manifest, self.config, directory, FakeAdapter(self.config), now=clock()
            )
            second = FakeAdapter(self.config)
            summary = run_label_batch(
                self.manifest, self.config, directory, second, now=clock()
            )
        self.assertEqual(second.calls, [])
        self.assertEqual(summary["counts"]["skipped"], 1)

    def test_scf_failure_uses_the_direct_fallback_and_keeps_both_attempts(self) -> None:
        adapter = FakeAdapter(self.config, failures=("scf_not_converged", None))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_label_batch(
                self.manifest, self.config, root, adapter, now=clock()
            )
            record = LabelRecord.from_dict(read_json(root / "records" / "h2_case.json"))
            ledger = read_json(root / "attempt_ledger.json")
        self.assertEqual(summary["counts"]["completed"], 1)
        self.assertFalse(record.method.density_fit)
        self.assertEqual(
            [attempt["status"] for attempt in ledger["records"]["h2_case"]["attempts"]],
            ["failed", "completed"],
        )
        self.assertEqual(
            [call[2] for call in adapter.calls], ["primary_density_fit", "direct_fallback"]
        )

    def test_unapproved_state_is_blocked_without_calling_the_adapter(self) -> None:
        blocked = h2_candidate("h2_charged")
        blocked = CandidateRecord(
            record_id=blocked.record_id,
            structure=blocked.structure,
            state=ElectronicState(
                charge=1,
                multiplicity=2,
                spin_2s=1,
                state_provenance="pending_scientific_review",
            ),
        )
        adapter = FakeAdapter(self.config)
        with tempfile.TemporaryDirectory() as directory:
            summary = run_label_batch(
                manifest(blocked), self.config, directory, adapter, now=clock()
            )
            ledger = read_json(Path(directory) / "attempt_ledger.json")
        self.assertEqual(adapter.calls, [])
        self.assertEqual(summary["counts"]["blocked"], 1)
        self.assertEqual(ledger["records"]["h2_charged"]["status"], "blocked")

    def test_non_retryable_failure_is_terminal_and_resume_does_not_repeat_it(self) -> None:
        first = FakeAdapter(self.config, failures=("runtime_environment",))
        with tempfile.TemporaryDirectory() as directory:
            summary = run_label_batch(
                self.manifest, self.config, directory, first, now=clock()
            )
            second = FakeAdapter(self.config)
            resumed = run_label_batch(
                self.manifest, self.config, directory, second, now=clock()
            )
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(second.calls, [])
        self.assertEqual(resumed["counts"]["failed"], 1)

    def test_changed_protocol_cannot_reuse_an_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_label_batch(
                self.manifest, self.config, directory, FakeAdapter(self.config), now=clock()
            )
            changed = dict(self.config)
            changed["description"] = "different fingerprint"
            with self.assertRaises(ValidationError) as caught:
                run_label_batch(
                    self.manifest, changed, directory, FakeAdapter(changed), now=clock()
                )
        self.assertIn("protocol_sha256", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
