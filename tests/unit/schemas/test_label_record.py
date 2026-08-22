"""The canonical label record schema.

The record under test is a small H2 label with hard-coded numbers; every
rejection test starts from the same valid dict and breaks exactly one thing, so
what each check is responsible for stays visible.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.io import read_json, write_json_atomic
from uma_pyscf.schemas.label_record import (
    CANONICAL_UNITS,
    LABEL_RECORD_SCHEMA,
    ElectronicState,
    Engine,
    LabelRecord,
    Method,
    QcState,
    RawArtifact,
    Results,
    Structure,
)

CHECKSUM = "3a" * 32


def h2_record_dict() -> dict[str, object]:
    """Return a valid H2 label record as a freshly built JSON-shaped dict."""
    return {
        "schema": "uma-pyscf-label-record-v1",
        "record_id": "h2_neutral_singlet",
        "structure": {
            "atomic_numbers": [1, 1],
            "positions_angstrom": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74144]],
            "parent_structure_id": "h2_equilibrium",
            "sampling_method": "bond_scan_v1",
            "random_seed": 20260822,
        },
        "state": {
            "charge": 0,
            "multiplicity": 1,
            "spin_2s": 0,
            "initial_guess": "minao",
            "state_provenance": "ground_state_default",
        },
        "method": {
            "functional": "wb97m-v",
            "basis": "def2-tzvpd",
            "ecp": None,
            "aux_basis": None,
            "grid_level": 3,
            "nlc_grid_level": 1,
            "grid_response": True,
            "density_fit": False,
            "scf_conv_tol": 1e-10,
            "scf_max_cycle": 200,
        },
        "engine": {
            "name": "gpu4pyscf",
            "versions": {
                "python": "3.11.15",
                "pyscf": "2.6.2",
                "gpu4pyscf": "1.0.2",
                "cupy": "13.3.0",
                "libxc": "6.2.2",
                "cuda_device_name": "NVIDIA RTX A5000",
            },
        },
        "results": {
            "energy_hartree": -1.1730407,
            "gradient_hartree_per_bohr": [[0.0, 0.0, -0.0123456], [0.0, 0.0, 0.0123456]],
            "converged": True,
            "n_iterations": 12,
            "s2": 0.0,
            "s2_target": 0.0,
            "s2_deviation": 0.0,
            "wall_time_seconds": 12.5,
            "scf_wall_time_seconds": 9.25,
            "gradient_wall_time_seconds": 3.25,
        },
        "raw": {
            "logical_location": "runs/label/h2_neutral_singlet/result.json",
            "checksum_sha256": CHECKSUM,
        },
        "qc": {
            "status": "pending",
            "history": [
                {
                    "utc": "2026-08-22T00:00:00+00:00",
                    "event": "imported_from_crosscode_result_v1",
                    "input_fingerprint_sha256": CHECKSUM,
                }
            ],
        },
        "units": {
            "energy": "hartree",
            "gradient": "hartree/bohr",
            "positions": "angstrom",
        },
    }


def h2_record() -> LabelRecord:
    """Return the same H2 label record, built through the dataclasses."""
    return LabelRecord.from_dict(h2_record_dict())


class ConstructionTests(unittest.TestCase):
    def test_valid_record_exposes_its_blocks(self) -> None:
        record = h2_record()
        self.assertEqual(record.schema, LABEL_RECORD_SCHEMA)
        self.assertEqual(record.record_id, "h2_neutral_singlet")
        self.assertEqual(record.structure.atomic_numbers, (1, 1))
        self.assertEqual(record.structure.atom_count, 2)
        self.assertEqual(
            record.structure.positions_angstrom,
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.74144)),
        )
        self.assertEqual(record.state.multiplicity, 1)
        self.assertEqual(record.state.spin_2s, 0)
        self.assertEqual(record.method.basis, "def2-tzvpd")
        self.assertEqual(record.engine.name, "gpu4pyscf")
        self.assertEqual(record.results.n_iterations, 12)
        self.assertEqual(record.qc.status, "pending")
        self.assertEqual(record.electron_count, 2)

    def test_units_are_the_canonical_mapping(self) -> None:
        self.assertEqual(h2_record().units, dict(CANONICAL_UNITS))
        self.assertEqual(
            dict(CANONICAL_UNITS),
            {"energy": "hartree", "gradient": "hartree/bohr", "positions": "angstrom"},
        )

    def test_direct_construction_matches_from_dict(self) -> None:
        record = LabelRecord(
            record_id="h2_neutral_singlet",
            structure=Structure(
                atomic_numbers=(1, 1),
                positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74144)),
                parent_structure_id="h2_equilibrium",
                sampling_method="bond_scan_v1",
                random_seed=20260822,
            ),
            state=ElectronicState(
                charge=0,
                multiplicity=1,
                spin_2s=0,
                initial_guess="minao",
                state_provenance="ground_state_default",
            ),
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
            engine=Engine(
                name="gpu4pyscf",
                versions={
                    "python": "3.11.15",
                    "pyscf": "2.6.2",
                    "gpu4pyscf": "1.0.2",
                    "cupy": "13.3.0",
                    "libxc": "6.2.2",
                    "cuda_device_name": "NVIDIA RTX A5000",
                },
            ),
            results=Results(
                energy_hartree=-1.1730407,
                gradient_hartree_per_bohr=(
                    (0.0, 0.0, -0.0123456),
                    (0.0, 0.0, 0.0123456),
                ),
                converged=True,
                n_iterations=12,
                s2=0.0,
                s2_target=0.0,
                s2_deviation=0.0,
                wall_time_seconds=12.5,
                scf_wall_time_seconds=9.25,
                gradient_wall_time_seconds=3.25,
            ),
            raw=RawArtifact(
                logical_location="runs/label/h2_neutral_singlet/result.json",
                checksum_sha256=CHECKSUM,
            ),
            qc=QcState(
                status="pending",
                history=(
                    {
                        "utc": "2026-08-22T00:00:00+00:00",
                        "event": "imported_from_crosscode_result_v1",
                        "input_fingerprint_sha256": CHECKSUM,
                    },
                ),
            ),
        )
        self.assertEqual(record, h2_record())

    def test_sequences_are_normalized_to_tuples(self) -> None:
        structure = Structure.from_dict(
            {"atomic_numbers": [1, 1], "positions_angstrom": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]}
        )
        self.assertEqual(structure.atomic_numbers, (1, 1))
        self.assertEqual(structure.positions_angstrom, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.7)))

    def test_record_is_immutable(self) -> None:
        record = h2_record()
        with self.assertRaises(AttributeError):
            record.record_id = "other"  # type: ignore[misc]

    def test_engine_versions_are_copied_from_the_caller(self) -> None:
        versions = {"pyscf": "2.6.2"}
        engine = Engine(name="pyscf-cpu", versions=versions)
        versions["pyscf"] = "0.0.0"
        self.assertEqual(engine.versions, {"pyscf": "2.6.2"})

    def test_invalid_record_id_is_rejected(self) -> None:
        for record_id in ("H2 Record", "H2", "", "-leading", 7):
            with self.subTest(record_id=record_id):
                with self.assertRaises(ValidationError):
                    LabelRecord.from_dict(h2_record_dict() | {"record_id": record_id})


class RoundTripTests(unittest.TestCase):
    def test_to_dict_from_dict_round_trips_through_a_file(self) -> None:
        record = h2_record()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            write_json_atomic(destination, record.to_dict())
            restored = LabelRecord.from_dict(read_json(destination))
        self.assertEqual(restored, record)
        self.assertEqual(restored.to_dict(), record.to_dict())

    def test_to_dict_matches_the_source_dict(self) -> None:
        self.assertEqual(h2_record().to_dict(), h2_record_dict())

    def test_to_dict_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "record.json"
            write_json_atomic(destination, h2_record().to_dict())
            self.assertEqual(read_json(destination)["record_id"], "h2_neutral_singlet")

    def test_optional_fields_survive_absence(self) -> None:
        data = h2_record_dict()
        structure = dict(data["structure"])  # type: ignore[arg-type]
        for key in ("parent_structure_id", "sampling_method", "random_seed"):
            del structure[key]
        data["structure"] = structure
        record = LabelRecord.from_dict(data)
        self.assertIsNone(record.structure.parent_structure_id)
        self.assertIsNone(record.structure.sampling_method)
        self.assertIsNone(record.structure.random_seed)
        self.assertEqual(LabelRecord.from_dict(record.to_dict()), record)


class SpinAndChargeTests(unittest.TestCase):
    def test_seventeen_electron_structure_rejects_multiplicity_three(self) -> None:
        data = h2_record_dict()
        data["structure"] = {
            "atomic_numbers": [14, 1, 1, 1],
            "positions_angstrom": [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.48],
                [1.4, 0.0, -0.49],
                [-0.7, 1.21, -0.49],
            ],
            "parent_structure_id": None,
            "sampling_method": None,
            "random_seed": None,
        }
        data["state"] = {
            "charge": 0,
            "multiplicity": 3,
            "spin_2s": 2,
            "initial_guess": None,
            "state_provenance": None,
        }
        data["results"] = dict(data["results"]) | {  # type: ignore[operator]
            "gradient_hartree_per_bohr": [
                [0.0, 0.0, 0.001],
                [0.0, 0.0, -0.001],
                [0.001, 0.0, 0.0],
                [-0.001, 0.0, 0.0],
            ]
        }
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        self.assertIn("multiplicity 3", str(caught.exception))

    def test_same_structure_accepts_a_doublet(self) -> None:
        data = h2_record_dict()
        data["structure"] = {
            "atomic_numbers": [14, 1, 1, 1],
            "positions_angstrom": [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.48],
                [1.4, 0.0, -0.49],
                [-0.7, 1.21, -0.49],
            ],
            "parent_structure_id": None,
            "sampling_method": None,
            "random_seed": None,
        }
        data["state"] = {
            "charge": 0,
            "multiplicity": 2,
            "spin_2s": 1,
            "initial_guess": None,
            "state_provenance": None,
        }
        data["results"] = dict(data["results"]) | {  # type: ignore[operator]
            "gradient_hartree_per_bohr": [
                [0.0, 0.0, 0.001],
                [0.0, 0.0, -0.001],
                [0.001, 0.0, 0.0],
                [-0.001, 0.0, 0.0],
            ]
        }
        record = LabelRecord.from_dict(data)
        self.assertEqual(record.electron_count, 17)
        self.assertEqual(record.state.spin_2s, 1)

    def test_spin_2s_that_disagrees_with_multiplicity_is_rejected(self) -> None:
        data = h2_record_dict()
        data["state"] = dict(data["state"]) | {"spin_2s": 2}  # type: ignore[operator]
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        message = str(caught.exception)
        self.assertIn("spin_2s", message)
        self.assertIn("source of truth", message)

    def test_multiplicity_below_one_is_rejected(self) -> None:
        data = h2_record_dict()
        data["state"] = dict(data["state"]) | {"multiplicity": 0, "spin_2s": -1}  # type: ignore[operator]
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)

    def test_charge_that_empties_the_molecule_is_rejected(self) -> None:
        data = h2_record_dict()
        data["state"] = dict(data["state"]) | {"charge": 2}  # type: ignore[operator]
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)


class ShapeAndFinitenessTests(unittest.TestCase):
    def test_gradient_row_count_must_match_the_atom_count(self) -> None:
        data = h2_record_dict()
        data["results"] = dict(data["results"]) | {  # type: ignore[operator]
            "gradient_hartree_per_bohr": [[0.0, 0.0, -0.01]]
        }
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        message = str(caught.exception)
        self.assertIn("1 rows", message)
        self.assertIn("2 atoms", message)

    def test_position_row_count_must_match_the_atom_count(self) -> None:
        data = h2_record_dict()
        data["structure"] = dict(data["structure"]) | {  # type: ignore[operator]
            "positions_angstrom": [[0.0, 0.0, 0.0]]
        }
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)

    def test_rows_must_have_three_components(self) -> None:
        data = h2_record_dict()
        data["results"] = dict(data["results"]) | {  # type: ignore[operator]
            "gradient_hartree_per_bohr": [[0.0, 0.0], [0.0, 0.0]]
        }
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)

    def test_nan_energy_is_rejected(self) -> None:
        data = h2_record_dict()
        data["results"] = dict(data["results"]) | {"energy_hartree": float("nan")}  # type: ignore[operator]
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        self.assertIn("results.energy_hartree", str(caught.exception))

    def test_infinite_gradient_component_is_rejected(self) -> None:
        data = h2_record_dict()
        data["results"] = dict(data["results"]) | {  # type: ignore[operator]
            "gradient_hartree_per_bohr": [
                [0.0, 0.0, -0.0123456],
                [0.0, 0.0, float("inf")],
            ]
        }
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        self.assertIn("results.gradient_hartree_per_bohr[1][2]", str(caught.exception))

    def test_non_numeric_position_is_rejected(self) -> None:
        data = h2_record_dict()
        data["structure"] = dict(data["structure"]) | {  # type: ignore[operator]
            "positions_angstrom": [[0.0, 0.0, "0.0"], [0.0, 0.0, 0.74144]]
        }
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)

    def test_atomic_numbers_outside_the_table_are_rejected(self) -> None:
        for atomic_numbers in ([0, 1], [1, 119], [1, -1], [1, 1.0], [1, True]):
            with self.subTest(atomic_numbers=atomic_numbers):
                data = h2_record_dict()
                data["structure"] = dict(data["structure"]) | {  # type: ignore[operator]
                    "atomic_numbers": atomic_numbers
                }
                with self.assertRaises(ValidationError):
                    LabelRecord.from_dict(data)

    def test_empty_structure_is_rejected(self) -> None:
        data = h2_record_dict()
        data["structure"] = {"atomic_numbers": [], "positions_angstrom": []}
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)

    def test_boolean_is_not_an_integer_field(self) -> None:
        data = h2_record_dict()
        data["state"] = dict(data["state"]) | {"charge": True}  # type: ignore[operator]
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)


class FailClosedTests(unittest.TestCase):
    def test_unknown_top_level_key_is_rejected(self) -> None:
        data = h2_record_dict()
        data["extra_block"] = {"anything": 1}
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        self.assertIn("extra_block", str(caught.exception))

    def test_unknown_key_inside_results_is_rejected(self) -> None:
        data = h2_record_dict()
        data["results"] = dict(data["results"]) | {"dipole_debye": 0.0}  # type: ignore[operator]
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        self.assertIn("dipole_debye", str(caught.exception))

    def test_unknown_keys_in_every_nested_block_are_rejected(self) -> None:
        for block in ("structure", "state", "method", "engine", "raw", "qc"):
            with self.subTest(block=block):
                data = h2_record_dict()
                data[block] = dict(data[block]) | {"surprise": 1}  # type: ignore[operator,arg-type]
                with self.assertRaises(ValidationError) as caught:
                    LabelRecord.from_dict(data)
                self.assertIn("surprise", str(caught.exception))

    def test_engine_versions_accepts_unknown_keys(self) -> None:
        data = h2_record_dict()
        engine = dict(data["engine"])  # type: ignore[arg-type]
        engine["versions"] = dict(engine["versions"]) | {
            "some_future_library": "1.2.3",
            "cuda_device_total_memory_bytes": None,
        }
        data["engine"] = engine
        record = LabelRecord.from_dict(data)
        self.assertEqual(record.engine.versions["some_future_library"], "1.2.3")
        self.assertIsNone(record.engine.versions["cuda_device_total_memory_bytes"])

    def test_force_key_in_results_is_rejected_with_the_export_layer_rule(self) -> None:
        data = h2_record_dict()
        data["results"] = dict(data["results"]) | {  # type: ignore[operator]
            "forces_ev_per_angstrom": [[0.0, 0.0, 0.63], [0.0, 0.0, -0.63]]
        }
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        message = str(caught.exception)
        self.assertIn("forces_ev_per_angstrom", message)
        self.assertIn("export layer", message)
        self.assertIn("gradient", message)

    def test_force_key_at_the_top_level_is_rejected_with_the_export_layer_rule(self) -> None:
        data = h2_record_dict()
        data["forces_ev_per_angstrom"] = [[0.0, 0.0, 0.63], [0.0, 0.0, -0.63]]
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        self.assertIn("export layer", str(caught.exception))

    def test_any_spelling_of_force_is_refused(self) -> None:
        for key in ("force_hartree_per_bohr", "FORCES", "atomic_forces"):
            with self.subTest(key=key):
                data = h2_record_dict()
                data["results"] = dict(data["results"]) | {key: 1}  # type: ignore[operator]
                with self.assertRaises(ValidationError) as caught:
                    LabelRecord.from_dict(data)
                self.assertIn("export layer", str(caught.exception))

    def test_missing_required_key_names_its_path(self) -> None:
        data = h2_record_dict()
        method = dict(data["method"])  # type: ignore[arg-type]
        del method["grid_level"]
        data["method"] = method
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        self.assertIn("method.grid_level", str(caught.exception))

    def test_missing_top_level_block_names_its_path(self) -> None:
        data = h2_record_dict()
        del data["qc"]
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        self.assertIn("record.qc", str(caught.exception))

    def test_wrong_schema_string_is_rejected(self) -> None:
        for schema in ("uma-pyscf-label-record-v0", "crosscode-result-v1", "", None):
            with self.subTest(schema=schema):
                data = h2_record_dict()
                data["schema"] = schema
                with self.assertRaises(ValidationError) as caught:
                    LabelRecord.from_dict(data)
                self.assertIn(LABEL_RECORD_SCHEMA, str(caught.exception))

    def test_missing_schema_string_is_rejected(self) -> None:
        data = h2_record_dict()
        del data["schema"]
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)

    def test_wrong_units_mapping_is_rejected(self) -> None:
        deviations: tuple[dict[str, str], ...] = (
            {"energy": "ev", "gradient": "ev/angstrom", "positions": "angstrom"},
            {"energy": "hartree", "gradient": "hartree/bohr", "positions": "bohr"},
            {"energy": "hartree", "gradient": "hartree/bohr"},
            {
                "energy": "hartree",
                "gradient": "hartree/bohr",
                "positions": "angstrom",
                "time": "seconds",
            },
        )
        for units in deviations:
            with self.subTest(units=units):
                data = h2_record_dict()
                data["units"] = units
                with self.assertRaises(ValidationError) as caught:
                    LabelRecord.from_dict(data)
                self.assertIn("record.units", str(caught.exception))

    def test_missing_units_mapping_is_rejected(self) -> None:
        data = h2_record_dict()
        del data["units"]
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)

    def test_non_object_input_is_rejected(self) -> None:
        for data in (None, [], "record", 3):
            with self.subTest(data=data):
                with self.assertRaises(ValidationError):
                    LabelRecord.from_dict(data)


class MethodBlockTests(unittest.TestCase):
    def test_grid_levels_must_be_non_negative_integers(self) -> None:
        for key, value in (
            ("grid_level", -1),
            ("nlc_grid_level", -2),
            ("grid_level", 3.0),
            ("nlc_grid_level", "1"),
        ):
            with self.subTest(key=key, value=value):
                data = h2_record_dict()
                data["method"] = dict(data["method"]) | {key: value}  # type: ignore[operator]
                with self.assertRaises(ValidationError):
                    LabelRecord.from_dict(data)

    def test_scf_thresholds_are_checked(self) -> None:
        for key, value in (
            ("scf_conv_tol", 0.0),
            ("scf_conv_tol", -1e-10),
            ("scf_max_cycle", 0),
            ("scf_max_cycle", -5),
        ):
            with self.subTest(key=key, value=value):
                data = h2_record_dict()
                data["method"] = dict(data["method"]) | {key: value}  # type: ignore[operator]
                with self.assertRaises(ValidationError):
                    LabelRecord.from_dict(data)

    def test_grid_response_and_density_fit_must_be_booleans(self) -> None:
        for key in ("grid_response", "density_fit"):
            with self.subTest(key=key):
                data = h2_record_dict()
                data["method"] = dict(data["method"]) | {key: 1}  # type: ignore[operator]
                with self.assertRaises(ValidationError):
                    LabelRecord.from_dict(data)

    def test_functional_and_basis_must_be_non_empty(self) -> None:
        for key in ("functional", "basis"):
            with self.subTest(key=key):
                data = h2_record_dict()
                data["method"] = dict(data["method"]) | {key: "  "}  # type: ignore[operator]
                with self.assertRaises(ValidationError):
                    LabelRecord.from_dict(data)

    def test_ecp_and_aux_basis_may_be_named(self) -> None:
        data = h2_record_dict()
        data["method"] = dict(data["method"]) | {  # type: ignore[operator]
            "ecp": "def2-ecp",
            "aux_basis": "def2-universal-jkfit",
        }
        record = LabelRecord.from_dict(data)
        self.assertEqual(record.method.ecp, "def2-ecp")
        self.assertEqual(record.method.aux_basis, "def2-universal-jkfit")


class RawArtifactTests(unittest.TestCase):
    def test_checksum_must_be_a_sha256_digest(self) -> None:
        for checksum in ("abc", "z" * 64, CHECKSUM[:63]):
            with self.subTest(checksum=checksum):
                data = h2_record_dict()
                data["raw"] = dict(data["raw"]) | {"checksum_sha256": checksum}  # type: ignore[operator]
                with self.assertRaises(ValidationError):
                    LabelRecord.from_dict(data)

    def test_checksum_is_normalized_to_lowercase(self) -> None:
        data = h2_record_dict()
        data["raw"] = dict(data["raw"]) | {"checksum_sha256": CHECKSUM.upper()}  # type: ignore[operator]
        self.assertEqual(LabelRecord.from_dict(data).raw.checksum_sha256, CHECKSUM)

    def test_a_record_may_carry_no_raw_artifact_yet(self) -> None:
        data = h2_record_dict()
        data["raw"] = {"logical_location": None, "checksum_sha256": None}
        record = LabelRecord.from_dict(data)
        self.assertIsNone(record.raw.logical_location)
        self.assertIsNone(record.raw.checksum_sha256)


class QcStateTests(unittest.TestCase):
    def test_only_the_three_defined_statuses_are_accepted(self) -> None:
        for status in ("pending", "accepted", "rejected"):
            with self.subTest(status=status):
                data = h2_record_dict()
                data["qc"] = dict(data["qc"]) | {"status": status}  # type: ignore[operator]
                self.assertEqual(LabelRecord.from_dict(data).qc.status, status)

    def test_unknown_status_is_rejected(self) -> None:
        for status in ("maybe", "PENDING", "ok", ""):
            with self.subTest(status=status):
                data = h2_record_dict()
                data["qc"] = dict(data["qc"]) | {"status": status}  # type: ignore[operator]
                with self.assertRaises(ValidationError):
                    LabelRecord.from_dict(data)

    def test_history_entry_without_event_is_rejected(self) -> None:
        data = h2_record_dict()
        data["qc"] = {
            "status": "pending",
            "history": [{"utc": "2026-08-22T00:00:00+00:00", "note": "imported"}],
        }
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        self.assertIn("qc.history[0].event", str(caught.exception))

    def test_history_entry_without_utc_is_rejected(self) -> None:
        data = h2_record_dict()
        data["qc"] = {"status": "pending", "history": [{"event": "imported"}]}
        with self.assertRaises(ValidationError) as caught:
            LabelRecord.from_dict(data)
        self.assertIn("qc.history[0].utc", str(caught.exception))

    def test_history_entries_may_carry_extra_provenance(self) -> None:
        data = h2_record_dict()
        data["qc"] = {
            "status": "rejected",
            "history": [
                {"utc": "2026-08-22T00:00:00+00:00", "event": "imported"},
                {
                    "utc": "2026-08-22T01:00:00+00:00",
                    "event": "rejected_by_qc",
                    "rule": "s2_deviation",
                    "observed": 0.42,
                    "threshold": 0.1,
                    "detail": {"attempt": 2},
                },
            ],
        }
        record = LabelRecord.from_dict(data)
        self.assertEqual(len(record.qc.history), 2)
        self.assertEqual(record.qc.history[1]["rule"], "s2_deviation")
        self.assertEqual(LabelRecord.from_dict(record.to_dict()), record)

    def test_history_entry_with_a_non_finite_value_is_rejected(self) -> None:
        data = h2_record_dict()
        data["qc"] = {
            "status": "pending",
            "history": [
                {
                    "utc": "2026-08-22T00:00:00+00:00",
                    "event": "imported",
                    "observed": float("nan"),
                }
            ],
        }
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)

    def test_history_must_be_a_list_of_objects(self) -> None:
        data = h2_record_dict()
        data["qc"] = {"status": "pending", "history": ["imported"]}
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)

    def test_empty_history_is_allowed(self) -> None:
        data = h2_record_dict()
        data["qc"] = {"status": "pending", "history": []}
        self.assertEqual(LabelRecord.from_dict(data).qc.history, ())


class ResultsBlockTests(unittest.TestCase):
    def test_unconverged_results_are_representable(self) -> None:
        data = h2_record_dict()
        data["results"] = dict(data["results"]) | {"converged": False}  # type: ignore[operator]
        self.assertFalse(LabelRecord.from_dict(data).results.converged)

    def test_optional_result_fields_may_be_absent(self) -> None:
        data = h2_record_dict()
        data["results"] = {
            "energy_hartree": -1.1730407,
            "gradient_hartree_per_bohr": [[0.0, 0.0, -0.0123456], [0.0, 0.0, 0.0123456]],
            "converged": True,
        }
        record = LabelRecord.from_dict(data)
        self.assertIsNone(record.results.s2)
        self.assertIsNone(record.results.wall_time_seconds)
        self.assertIsNone(record.results.n_iterations)

    def test_negative_wall_time_is_rejected(self) -> None:
        data = h2_record_dict()
        data["results"] = dict(data["results"]) | {"wall_time_seconds": -1.0}  # type: ignore[operator]
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)

    def test_converged_must_be_a_boolean(self) -> None:
        data = h2_record_dict()
        data["results"] = dict(data["results"]) | {"converged": "yes"}  # type: ignore[operator]
        with self.assertRaises(ValidationError):
            LabelRecord.from_dict(data)


if __name__ == "__main__":
    unittest.main()
