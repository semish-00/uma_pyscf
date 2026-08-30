"""Production QC mirrors every machine-checkable Gate 1 condition."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from uma_pyscf.calculators.config import load_dft_config, method_from_config
from uma_pyscf.core.ids import canonical_json_fingerprint
from uma_pyscf.qc.config import load_qc_config
from uma_pyscf.qc.protocol import PROTOCOL_CHECK_NAMES, protocol_checks
from uma_pyscf.qc.run import apply_qc
from uma_pyscf.schemas.label_record import (
    ElectronicState,
    Engine,
    LabelRecord,
    QcState,
    RawArtifact,
    Results,
    Structure,
)
from uma_pyscf.schemas.state_registry import StateRegistry, StateRegistryEntry
from uma_pyscf.states.registry import registry_identity

REPO_ROOT = Path(__file__).resolve().parents[3]
DFT_CONFIG = REPO_ROOT / "configs" / "dft" / "omol_wb97mv_tzvpd_v1.yaml"
QC_CONFIG = (
    REPO_ROOT / "configs" / "datasets" / "omol_wb97mv_tzvpd_conditional_qc_v1.yaml"
)


def production_h2() -> LabelRecord:
    dft = load_dft_config(DFT_CONFIG)
    protocol_sha256 = canonical_json_fingerprint(dft)
    versions = dict(dft["engine"]["required_versions"])
    versions.update(
        {
            "protocol_id": dft["protocol_id"],
            "protocol_sha256": protocol_sha256,
            "initial_density": "minao",
            "initial_density_generated_on": "cpu_before_device_conversion",
            "cuda_runtime_version": "12080",
            "cuda_driver_version": "12020",
            "cuda_device_name": "NVIDIA A100-SXM4-80GB",
            "container_image_sha256": "b" * 64,
            "python_lock_sha256": "c" * 64,
        }
    )
    return LabelRecord(
        record_id="production_h2",
        structure=Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)),
        ),
        state=ElectronicState(
            charge=0,
            multiplicity=1,
            spin_2s=0,
            initial_guess="minao",
            state_provenance="ground_state_default",
        ),
        method=method_from_config(dft),
        engine=Engine(name="gpu4pyscf", versions=versions),
        results=Results(
            energy_hartree=-1.1,
            gradient_hartree_per_bohr=((0.0, 0.0, -0.01), (0.0, 0.0, 0.01)),
            converged=True,
            s2=0.0,
            s2_target=0.0,
            s2_deviation=0.0,
        ),
        raw=RawArtifact(logical_location="raw/production_h2.json", checksum_sha256="a" * 64),
        qc=QcState(status="pending"),
    )


class ConfigPairTests(unittest.TestCase):
    def test_qc_config_pins_the_exact_dft_config_fingerprint(self) -> None:
        dft = load_dft_config(DFT_CONFIG)
        qc = load_qc_config(QC_CONFIG)
        self.assertEqual(qc["protocol"]["protocol_id"], dft["protocol_id"])
        self.assertEqual(
            qc["protocol"]["protocol_sha256"], canonical_json_fingerprint(dft)
        )
        self.assertEqual(qc["release_status"], "engineering_only_pending_scientific_freeze")


class ProtocolCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_qc_config(QC_CONFIG)
        self.section = self.config["protocol"]

    def failed(self, record: LabelRecord) -> list[str]:
        return [
            check["name"]
            for check in protocol_checks(record, self.section)
            if not check["passed"]
        ]

    def test_production_record_passes_all_protocol_checks(self) -> None:
        checks = protocol_checks(production_h2(), self.section)
        self.assertEqual(tuple(check["name"] for check in checks), PROTOCOL_CHECK_NAMES)
        self.assertTrue(all(check["passed"] for check in checks))

    def test_wrong_protocol_fingerprint_is_rejected(self) -> None:
        record = production_h2()
        versions = dict(record.engine.versions)
        versions["protocol_sha256"] = "b" * 64
        changed = replace(record, engine=replace(record.engine, versions=versions))
        self.assertEqual(self.failed(changed), ["protocol_identity"])

    def test_changed_grid_is_rejected_as_a_method_change(self) -> None:
        record = production_h2()
        changed = replace(record, method=replace(record.method, grid_level=4))
        self.assertEqual(self.failed(changed), ["protocol_method"])

    def test_missing_raw_checksum_is_rejected(self) -> None:
        record = production_h2()
        changed = replace(record, raw=RawArtifact(logical_location="raw/missing.json"))
        self.assertEqual(self.failed(changed), ["raw_checksum_present"])

    def test_unregistered_non_default_state_is_rejected(self) -> None:
        record = production_h2()
        changed = replace(
            record,
            state=ElectronicState(
                charge=1,
                multiplicity=2,
                spin_2s=1,
                initial_guess="minao",
                state_provenance="pending_scientific_review",
            ),
            results=replace(
                record.results,
                s2=0.75,
                s2_target=0.75,
                s2_deviation=0.0,
            ),
        )
        self.assertEqual(self.failed(changed), ["state_registry"])

    def test_approved_non_default_state_requires_matching_registry_checksum(self) -> None:
        registry = StateRegistry(
            registry_id="unit_h2_states_v1",
            created="2026-08-31",
            description="Approved QC fixture.",
            entries=(
                StateRegistryEntry(
                    entry_id="h2_cation_doublet",
                    composition="H2",
                    charge=1,
                    multiplicity=2,
                    status="approved",
                    evidence=("unit-reference",),
                    reviewer="unit-reviewer",
                    decision="unit-decision",
                ),
            ),
        )
        record = production_h2()
        identity = registry_identity(registry)
        assert identity is not None
        versions = dict(record.engine.versions) | identity
        changed = replace(
            record,
            state=ElectronicState(
                charge=1,
                multiplicity=2,
                spin_2s=1,
                initial_guess="minao",
                state_provenance=(
                    "state_registry:unit_h2_states_v1:h2_cation_doublet"
                ),
            ),
            engine=replace(record.engine, versions=versions),
            results=replace(
                record.results,
                s2=0.75,
                s2_target=0.75,
                s2_deviation=0.0,
            ),
        )
        checks = protocol_checks(changed, self.section, state_registry=registry)
        registry_check = next(check for check in checks if check["name"] == "state_registry")
        self.assertTrue(registry_check["passed"])
        forged_versions = dict(versions)
        forged_versions["state_registry_sha256"] = "0" * 64
        forged = replace(changed, engine=replace(changed.engine, versions=forged_versions))
        forged_check = next(
            check
            for check in protocol_checks(forged, self.section, state_registry=registry)
            if check["name"] == "state_registry"
        )
        self.assertFalse(forged_check["passed"])

    def test_full_qc_accepts_engineering_record_without_changing_release_status(
        self,
    ) -> None:
        records, report = apply_qc(
            [production_h2()], self.config, utc="2026-08-31T00:00:00+00:00"
        )
        self.assertEqual(records[0].qc.status, "accepted")
        self.assertEqual(report.entries[0]["failed_checks"], [])
        self.assertEqual(
            report.config["release_status"], "engineering_only_pending_scientific_freeze"
        )


if __name__ == "__main__":
    unittest.main()
