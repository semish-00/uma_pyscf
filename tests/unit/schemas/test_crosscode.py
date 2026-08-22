"""Importing Part I `crosscode-result-v1` results as canonical label records.

The result dicts below are written out as literals that mirror what
`validation/orca_gpu4pyscf/run_pyscf.py` and `parse_orca.py` emit. They are
deliberately not imported from `validation/`: that experiment is frozen, the
package must not depend on it, and the format this importer promises to read is
pinned here instead.
"""

from __future__ import annotations

import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.schemas.crosscode import label_record_from_crosscode_result
from uma_pyscf.schemas.label_record import CANONICAL_UNITS, LabelRecord

FINGERPRINT = "9b4c" * 16
CREATED_UTC = "2026-08-13T02:31:44.512345+00:00"


def h2_case() -> dict[str, object]:
    """Return the `case` block `common.case_record()` writes for neutral H2."""
    return {
        "case_id": "h2_neutral_singlet",
        "input_fingerprint_sha256": FINGERPRINT,
        "structure_manifest_path": "structures/h2.xyz",
        "atoms": [
            {"element": "H", "xyz_angstrom": [0.0, 0.0, 0.0]},
            {"element": "H", "xyz_angstrom": [0.0, 0.0, 0.74144]},
        ],
        "charge": 0,
        "multiplicity": 1,
        "pyscf_spin_2s": 0,
        "electron_count": 2,
        "functional": "wb97m-v",
        "basis": "def2-tzvpd",
        "calculation": "energy_gradient",
    }


def pyscf_result() -> dict[str, object]:
    """Return a full `crosscode-result-v1` result as `run_pyscf.py` writes it."""
    return {
        "schema": "crosscode-result-v1",
        "created_utc": CREATED_UTC,
        "engine": "pyscf-cpu",
        "engine_runtime": {
            "python": "3.11.15",
            "pyscf": "2.6.2",
            "libxc": "6.2.2",
            "gpu4pyscf": None,
            "cupy": None,
            "cuda_device": None,
        },
        "case": h2_case(),
        "settings": {
            "scf": {"conv_tol": 1e-10, "max_cycle": 200},
            "verbose": 3,
            "grid_level": 3,
            "nlc_grid_level": 1,
            "grid_response": True,
            "density_fit": False,
            "max_memory_mb": 8000.0,
            "reference": "RKS",
            "nonlocal_correlation_active": True,
        },
        "converged": True,
        "energy_hartree": -1.1730407,
        "gradient_hartree_per_bohr": [
            [0.0, 0.0, -0.0123456],
            [0.0, 0.0, 0.0123456],
        ],
        "s2": 0.0,
        "s2_target": 0.0,
        "s2_deviation": 0.0,
        "multiplicity_from_spin_square": 1.0,
        "wall_time_seconds": 12.5,
        "scf_wall_time_seconds": 9.25,
        "gradient_wall_time_seconds": 3.25,
        "tolerances": {
            "energy_abs_hartree": 1e-06,
            "gradient_rms_hartree_per_bohr": 1e-05,
            "gradient_max_hartree_per_bohr": 2e-05,
        },
        "tolerance_status": None,
    }


def orca_result() -> dict[str, object]:
    """Return an ORCA result as `parse_orca.py` writes it: no PySCF settings."""
    return {
        "schema": "crosscode-result-v1",
        "created_utc": CREATED_UTC,
        "engine": "orca",
        "engine_runtime": {"orca": "6.0.1"},
        "case": h2_case(),
        "settings": {
            "version": "6.0.1",
            "nprocs": 8,
            "maxcore_mb_per_process": 3000,
            "keywords": ["wB97M-V", "def2-TZVPD", "EnGrad", "TightSCF"],
            "scf": {"conv_tol": 1e-10, "max_cycle": 200},
            "coordinates_verified_max_abs_angstrom": 0.0,
        },
        "converged": True,
        "energy_hartree": -1.1730405,
        "gradient_hartree_per_bohr": [
            [0.0, 0.0, -0.0123451],
            [0.0, 0.0, 0.0123451],
        ],
        "s2": None,
        "s2_target": 0.0,
        "s2_deviation": None,
        "tolerances": {
            "energy_abs_hartree": 1e-06,
            "gradient_rms_hartree_per_bohr": 1e-05,
            "gradient_max_hartree_per_bohr": 2e-05,
        },
        "tolerance_status": None,
        "source_engrad": "h2_neutral_singlet.engrad",
        "source_output": "h2_neutral_singlet.out",
    }


class FieldMappingTests(unittest.TestCase):
    def test_structure_is_taken_from_the_case_atoms(self) -> None:
        record = label_record_from_crosscode_result(pyscf_result())
        self.assertEqual(record.structure.atomic_numbers, (1, 1))
        self.assertEqual(
            record.structure.positions_angstrom,
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.74144)),
        )
        self.assertIsNone(record.structure.parent_structure_id)
        self.assertIsNone(record.structure.sampling_method)
        self.assertIsNone(record.structure.random_seed)

    def test_element_symbols_are_canonicalized(self) -> None:
        result = pyscf_result()
        case = dict(result["case"])  # type: ignore[arg-type]
        case["atoms"] = [
            {"element": "si", "xyz_angstrom": [0.0, 0.0, 0.0]},
            {"element": "CL ", "xyz_angstrom": [0.0, 0.0, 2.05]},
        ]
        case["charge"] = 0
        case["multiplicity"] = 2
        case["pyscf_spin_2s"] = 1
        result["case"] = case
        result["gradient_hartree_per_bohr"] = [[0.0, 0.0, -0.01], [0.0, 0.0, 0.01]]
        record = label_record_from_crosscode_result(result)
        self.assertEqual(record.structure.atomic_numbers, (14, 17))
        self.assertEqual(record.electron_count, 31)

    def test_electronic_state_comes_from_the_case(self) -> None:
        record = label_record_from_crosscode_result(pyscf_result())
        self.assertEqual(record.state.charge, 0)
        self.assertEqual(record.state.multiplicity, 1)
        self.assertEqual(record.state.spin_2s, 0)

    def test_method_comes_from_the_case_and_the_settings_block(self) -> None:
        record = label_record_from_crosscode_result(pyscf_result())
        self.assertEqual(record.method.functional, "wb97m-v")
        self.assertEqual(record.method.basis, "def2-tzvpd")
        self.assertEqual(record.method.grid_level, 3)
        self.assertEqual(record.method.nlc_grid_level, 1)
        self.assertTrue(record.method.grid_response)
        self.assertFalse(record.method.density_fit)
        self.assertEqual(record.method.scf_conv_tol, 1e-10)
        self.assertEqual(record.method.scf_max_cycle, 200)

    def test_no_ecp_or_aux_basis_is_invented(self) -> None:
        record = label_record_from_crosscode_result(pyscf_result())
        self.assertIsNone(record.method.ecp)
        self.assertIsNone(record.method.aux_basis)

    def test_engine_name_and_versions_are_carried_over(self) -> None:
        record = label_record_from_crosscode_result(pyscf_result())
        self.assertEqual(record.engine.name, "pyscf-cpu")
        self.assertEqual(
            record.engine.versions,
            {
                "python": "3.11.15",
                "pyscf": "2.6.2",
                "libxc": "6.2.2",
                "gpu4pyscf": None,
                "cupy": None,
                "cuda_device": None,
            },
        )

    def test_numeric_runtime_provenance_is_kept_as_text(self) -> None:
        result = pyscf_result()
        result["engine"] = "gpu4pyscf"
        result["engine_runtime"] = {
            "pyscf": "2.6.2",
            "gpu4pyscf": "1.0.2",
            "cuda_device": 0,
            "cuda_runtime_version": 12040,
            "cuda_device_total_memory_bytes": 25396576256,
        }
        record = label_record_from_crosscode_result(result)
        self.assertEqual(record.engine.name, "gpu4pyscf")
        self.assertEqual(record.engine.versions["cuda_device"], "0")
        self.assertEqual(record.engine.versions["cuda_runtime_version"], "12040")
        self.assertEqual(record.engine.versions["cuda_device_total_memory_bytes"], "25396576256")

    def test_missing_engine_runtime_yields_no_versions(self) -> None:
        result = pyscf_result()
        del result["engine_runtime"]
        self.assertEqual(label_record_from_crosscode_result(result).engine.versions, {})

    def test_structured_runtime_values_are_rejected(self) -> None:
        result = pyscf_result()
        result["engine_runtime"] = {"pyscf": {"version": "2.6.2"}}
        with self.assertRaises(ValidationError) as caught:
            label_record_from_crosscode_result(result)
        self.assertIn("engine_runtime.pyscf", str(caught.exception))

    def test_results_are_carried_over_in_native_units(self) -> None:
        record = label_record_from_crosscode_result(pyscf_result())
        self.assertEqual(record.results.energy_hartree, -1.1730407)
        self.assertEqual(
            record.results.gradient_hartree_per_bohr,
            ((0.0, 0.0, -0.0123456), (0.0, 0.0, 0.0123456)),
        )
        self.assertTrue(record.results.converged)
        self.assertEqual(record.results.s2, 0.0)
        self.assertEqual(record.results.s2_target, 0.0)
        self.assertEqual(record.results.s2_deviation, 0.0)
        self.assertEqual(record.results.wall_time_seconds, 12.5)
        self.assertEqual(record.results.scf_wall_time_seconds, 9.25)
        self.assertEqual(record.results.gradient_wall_time_seconds, 3.25)

    def test_units_are_the_canonical_mapping(self) -> None:
        record = label_record_from_crosscode_result(pyscf_result())
        self.assertEqual(record.units, dict(CANONICAL_UNITS))

    def test_record_id_defaults_to_the_case_id(self) -> None:
        self.assertEqual(
            label_record_from_crosscode_result(pyscf_result()).record_id,
            "h2_neutral_singlet",
        )

    def test_record_id_can_be_overridden(self) -> None:
        record = label_record_from_crosscode_result(
            pyscf_result(), record_id="ds_sigehcl_001_h2_0001"
        )
        self.assertEqual(record.record_id, "ds_sigehcl_001_h2_0001")

    def test_invalid_record_id_override_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            label_record_from_crosscode_result(pyscf_result(), record_id="H2 Record")

    def test_raw_location_is_recorded_without_a_checksum(self) -> None:
        record = label_record_from_crosscode_result(
            pyscf_result(), raw_location="runs/gate1/h2_neutral_singlet/pyscf_cpu.json"
        )
        self.assertEqual(
            record.raw.logical_location, "runs/gate1/h2_neutral_singlet/pyscf_cpu.json"
        )
        self.assertIsNone(record.raw.checksum_sha256)

    def test_raw_location_defaults_to_unknown(self) -> None:
        record = label_record_from_crosscode_result(pyscf_result())
        self.assertIsNone(record.raw.logical_location)
        self.assertIsNone(record.raw.checksum_sha256)

    def test_qc_starts_pending_with_the_import_event_and_fingerprint(self) -> None:
        record = label_record_from_crosscode_result(pyscf_result())
        self.assertEqual(record.qc.status, "pending")
        self.assertEqual(len(record.qc.history), 1)
        self.assertEqual(
            record.qc.history[0],
            {
                "utc": CREATED_UTC,
                "event": "imported_from_crosscode_result_v1",
                "input_fingerprint_sha256": FINGERPRINT,
            },
        )

    def test_history_timestamp_falls_back_to_unknown(self) -> None:
        result = pyscf_result()
        del result["created_utc"]
        record = label_record_from_crosscode_result(result)
        self.assertEqual(record.qc.history[0]["utc"], "unknown")

    def test_imported_record_round_trips(self) -> None:
        record = label_record_from_crosscode_result(pyscf_result())
        self.assertEqual(LabelRecord.from_dict(record.to_dict()), record)

    def test_extra_result_keys_are_ignored(self) -> None:
        result = pyscf_result()
        result["tolerance_status"] = {"energy_abs_hartree": "pass"}
        result["future_field"] = [1, 2, 3]
        self.assertEqual(
            label_record_from_crosscode_result(result).record_id, "h2_neutral_singlet"
        )


class RefusalTests(unittest.TestCase):
    def test_wrong_result_schema_is_rejected(self) -> None:
        result = pyscf_result()
        result["schema"] = "crosscode-result-v2"
        with self.assertRaises(ValidationError) as caught:
            label_record_from_crosscode_result(result)
        self.assertIn("crosscode-result-v1", str(caught.exception))

    def test_unconverged_result_is_rejected(self) -> None:
        result = pyscf_result()
        result["converged"] = False
        with self.assertRaises(ValidationError) as caught:
            label_record_from_crosscode_result(result)
        self.assertIn("converged", str(caught.exception))

    def test_missing_convergence_flag_is_rejected(self) -> None:
        result = pyscf_result()
        del result["converged"]
        with self.assertRaises(ValidationError) as caught:
            label_record_from_crosscode_result(result)
        self.assertIn("result.converged", str(caught.exception))

    def test_spin_that_disagrees_with_the_multiplicity_is_rejected(self) -> None:
        result = pyscf_result()
        case = dict(result["case"])  # type: ignore[arg-type]
        case["pyscf_spin_2s"] = 1
        result["case"] = case
        with self.assertRaises(ValidationError) as caught:
            label_record_from_crosscode_result(result)
        message = str(caught.exception)
        self.assertIn("pyscf_spin_2s", message)
        self.assertIn("source of truth", message)

    def test_impossible_charge_and_multiplicity_are_rejected(self) -> None:
        result = pyscf_result()
        case = dict(result["case"])  # type: ignore[arg-type]
        case["multiplicity"] = 2
        case["pyscf_spin_2s"] = 1
        result["case"] = case
        with self.assertRaises(ValidationError):
            label_record_from_crosscode_result(result)

    def test_orca_result_without_pyscf_settings_is_rejected(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            label_record_from_crosscode_result(orca_result())
        message = str(caught.exception)
        self.assertIn("method settings", message)
        self.assertIn("settings.grid_level", message)

    def test_result_without_any_settings_block_is_rejected(self) -> None:
        result = orca_result()
        del result["settings"]
        with self.assertRaises(ValidationError) as caught:
            label_record_from_crosscode_result(result)
        self.assertIn("method settings", str(caught.exception))

    def test_settings_without_an_scf_block_is_rejected(self) -> None:
        result = pyscf_result()
        settings = dict(result["settings"])  # type: ignore[arg-type]
        del settings["scf"]
        result["settings"] = settings
        with self.assertRaises(ValidationError) as caught:
            label_record_from_crosscode_result(result)
        self.assertIn("settings.scf", str(caught.exception))

    def test_orca_result_with_the_pyscf_settings_filled_in_is_accepted(self) -> None:
        result = orca_result()
        settings = dict(result["settings"])  # type: ignore[arg-type]
        settings.update(
            {
                "grid_level": 3,
                "nlc_grid_level": 1,
                "grid_response": True,
                "density_fit": False,
            }
        )
        result["settings"] = settings
        record = label_record_from_crosscode_result(result)
        self.assertEqual(record.engine.name, "orca")
        self.assertEqual(record.engine.versions, {"orca": "6.0.1"})
        self.assertIsNone(record.results.s2)

    def test_missing_case_keys_name_their_path(self) -> None:
        for key in (
            "atoms",
            "charge",
            "multiplicity",
            "functional",
            "basis",
            "input_fingerprint_sha256",
            "case_id",
        ):
            with self.subTest(key=key):
                result = pyscf_result()
                case = dict(result["case"])  # type: ignore[arg-type]
                del case[key]
                result["case"] = case
                with self.assertRaises(ValidationError) as caught:
                    label_record_from_crosscode_result(result)
                self.assertIn(f"case.{key}", str(caught.exception))

    def test_missing_case_block_is_rejected(self) -> None:
        result = pyscf_result()
        del result["case"]
        with self.assertRaises(ValidationError) as caught:
            label_record_from_crosscode_result(result)
        self.assertIn("result.case", str(caught.exception))

    def test_missing_energy_or_gradient_is_rejected(self) -> None:
        for key in ("energy_hartree", "gradient_hartree_per_bohr"):
            with self.subTest(key=key):
                result = pyscf_result()
                del result[key]
                with self.assertRaises(ValidationError) as caught:
                    label_record_from_crosscode_result(result)
                self.assertIn(key, str(caught.exception))

    def test_gradient_that_does_not_match_the_atom_count_is_rejected(self) -> None:
        result = pyscf_result()
        result["gradient_hartree_per_bohr"] = [[0.0, 0.0, -0.0123456]]
        with self.assertRaises(ValidationError):
            label_record_from_crosscode_result(result)

    def test_unknown_element_symbol_is_rejected(self) -> None:
        result = pyscf_result()
        case = dict(result["case"])  # type: ignore[arg-type]
        case["atoms"] = [
            {"element": "Xx", "xyz_angstrom": [0.0, 0.0, 0.0]},
            {"element": "H", "xyz_angstrom": [0.0, 0.0, 0.74144]},
        ]
        result["case"] = case
        with self.assertRaises(ValidationError) as caught:
            label_record_from_crosscode_result(result)
        self.assertIn("'Xx'", str(caught.exception))

    def test_non_object_result_is_rejected(self) -> None:
        for result in (None, [], "result"):
            with self.subTest(result=result):
                with self.assertRaises(ValidationError):
                    label_record_from_crosscode_result(result)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
