"""End-to-end candidate generation from a sampling config.

The committed reference config is generated here rather than a fixture built in
the test, because reproducing *that* file is the milestone's completion
condition. Everything else is written into a temporary directory: configs that
are meant to fail, and configs that are meant to produce a rejection.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.ids import canonical_json_fingerprint
from uma_pyscf.core.io import read_json
from uma_pyscf.sampling.generate import (
    generate_candidates,
    load_sampling_config,
    read_xyz_structure,
    write_outputs,
)
from uma_pyscf.schemas.candidate import CandidateManifest, GeometryQcReport

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "sampling" / "example_bond_scan_v1.yaml"
EXAMPLE_XYZ = REPO_ROOT / "configs" / "sampling" / "structures" / "sih4_seed_example.xyz"
EXPECTED_IDS = (
    "example_bond_scan_v1_sih4_seed_bond01_x0p85",
    "example_bond_scan_v1_sih4_seed_bond01_x1p15",
    "example_bond_scan_v1_sih4_seed_bond01_x1p3",
    "example_bond_scan_v1_sih4_seed_disp0p04_s20260813",
    "example_bond_scan_v1_sih4_seed_disp0p04_s20260814",
    "example_bond_scan_v1_sih4_seed_q0m1",
    "example_bond_scan_v1_sih4_seed_q1m2",
)
SIH4_XYZ = (
    "\n".join(
        [
            "5",
            "SiH4 tetrahedral seed",
            *(
                f"{symbol:<2s} {x: .12f} {y: .12f} {z: .12f}"
                for symbol, x, y, z in [
                    ("Si", 0.0, 0.0, 0.0),
                    *(
                        (
                            "H",
                            1.480 / math.sqrt(3.0) * sx,
                            1.480 / math.sqrt(3.0) * sy,
                            1.480 / math.sqrt(3.0) * sz,
                        )
                        for sx, sy, sz in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))
                    ),
                ]
            ),
        ]
    )
    + "\n"
)


def write_case(directory: Path, config_text: str, xyz_text: str = SIH4_XYZ) -> Path:
    """Write a config and its seed structure into ``directory`` and return the config path."""
    (directory / "sih4.xyz").write_text(xyz_text, encoding="utf-8")
    config_path = directory / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def config_text(operations: str, filters: str = "", sampling_id: str = "tmp_run_v1") -> str:
    """Return a minimal config with the given operations block."""
    return (
        f"schema_version: 1\n"
        f"sampling_id: {sampling_id}\n"
        "structures:\n"
        "  - id: sih4_seed\n"
        "    xyz_path: sih4.xyz\n"
        f"operations:\n{operations}"
        f"{filters}"
    )


BOND_SCAN = (
    "  - kind: bond_scan\n"
    "    structure: sih4_seed\n"
    "    charge: 0\n"
    "    multiplicity: 1\n"
    "    anchor_index: 0\n"
    "    moved_index: 1\n"
    "    factors: [{factors}]\n"
)


class ReferenceExampleTests(unittest.TestCase):
    def test_the_committed_example_generates_the_expected_candidates(self) -> None:
        manifest, report = generate_candidates(EXAMPLE_CONFIG)
        self.assertEqual(tuple(entry["record_id"] for entry in report.entries), EXPECTED_IDS)
        self.assertEqual(report.counts, {"total": 7, "accepted": 7, "rejected": 0})
        self.assertEqual(tuple(record.record_id for record in manifest.records), EXPECTED_IDS)

    def test_every_rejection_would_carry_a_reason(self) -> None:
        _, report = generate_candidates(EXAMPLE_CONFIG)
        for entry in report.entries:
            with self.subTest(record_id=entry["record_id"]):
                if entry["status"] == "rejected":
                    self.assertTrue(entry["reason"])
                else:
                    self.assertIsNone(entry["reason"])

    def test_the_manifest_names_the_config_by_its_content(self) -> None:
        manifest, report = generate_candidates(EXAMPLE_CONFIG)
        config = load_sampling_config(EXAMPLE_CONFIG)
        self.assertEqual(manifest.config, config)
        self.assertEqual(manifest.config_sha256, canonical_json_fingerprint(config))
        self.assertEqual(report.config_sha256, manifest.config_sha256)

    def test_every_candidate_records_where_it_came_from(self) -> None:
        manifest, _ = generate_candidates(EXAMPLE_CONFIG)
        for record in manifest.records:
            with self.subTest(record_id=record.record_id):
                self.assertEqual(record.structure.parent_structure_id, "sih4_seed")
                self.assertIn(
                    record.structure.sampling_method,
                    ("bond_scan", "cartesian_displacement", "state_expansion"),
                )
                self.assertIn("operation", record.generation_parameters)

    def test_only_displacements_carry_a_random_seed(self) -> None:
        manifest, _ = generate_candidates(EXAMPLE_CONFIG)
        seeds = {record.record_id: record.structure.random_seed for record in manifest.records}
        self.assertEqual(seeds["example_bond_scan_v1_sih4_seed_disp0p04_s20260813"], 20260813)
        self.assertEqual(seeds["example_bond_scan_v1_sih4_seed_disp0p04_s20260814"], 20260814)
        self.assertIsNone(seeds["example_bond_scan_v1_sih4_seed_bond01_x0p85"])
        self.assertIsNone(seeds["example_bond_scan_v1_sih4_seed_q1m2"])

    def test_the_state_expansion_keeps_the_seed_geometry_in_both_states(self) -> None:
        manifest, _ = generate_candidates(EXAMPLE_CONFIG)
        seed = read_xyz_structure(EXAMPLE_XYZ)
        states = [
            record
            for record in manifest.records
            if record.structure.sampling_method == "state_expansion"
        ]
        self.assertEqual(
            [(record.state.charge, record.state.multiplicity) for record in states],
            [(0, 1), (1, 2)],
        )
        for record in states:
            with self.subTest(record_id=record.record_id):
                self.assertEqual(record.structure.positions_angstrom, seed.positions_angstrom)

    def test_the_scan_moves_only_the_scanned_ligand(self) -> None:
        manifest, _ = generate_candidates(EXAMPLE_CONFIG)
        seed = read_xyz_structure(EXAMPLE_XYZ)
        scanned = next(
            record for record in manifest.records if record.record_id.endswith("bond01_x0p85")
        )
        self.assertAlmostEqual(
            math.dist(
                scanned.structure.positions_angstrom[0], scanned.structure.positions_angstrom[1]
            ),
            1.480 * 0.85,
            places=9,
        )
        self.assertEqual(scanned.structure.positions_angstrom[2:], seed.positions_angstrom[2:])


class DeterminismTests(unittest.TestCase):
    def test_regeneration_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_outputs(*generate_candidates(EXAMPLE_CONFIG), root / "first")
            second = write_outputs(*generate_candidates(EXAMPLE_CONFIG), root / "second")
            for left, right in zip(first, second, strict=True):
                with self.subTest(file=left.name):
                    self.assertEqual(left.name, right.name)
                    self.assertEqual(left.read_bytes(), right.read_bytes())

    def test_neither_output_mentions_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_outputs(*generate_candidates(EXAMPLE_CONFIG), Path(directory))
            for path in paths:
                text = path.read_text(encoding="utf-8")
                with self.subTest(file=path.name):
                    for token in ("utc", "timestamp", "generated_at", "20260822T"):
                        self.assertNotIn(token, text.lower())

    def test_the_files_are_named_after_the_sampling_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, report_path = write_outputs(
                *generate_candidates(EXAMPLE_CONFIG), Path(directory)
            )
        self.assertEqual(manifest_path.name, "example_bond_scan_v1_candidates.json")
        self.assertEqual(report_path.name, "example_bond_scan_v1_geometry_qc.json")

    def test_the_written_pair_reads_back_through_the_schema(self) -> None:
        manifest, report = generate_candidates(EXAMPLE_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, report_path = write_outputs(manifest, report, Path(directory))
            restored_manifest = CandidateManifest.from_dict(read_json(manifest_path))
            restored_report = GeometryQcReport.from_dict(read_json(report_path))
        self.assertEqual(restored_manifest, manifest)
        self.assertEqual(restored_report, report)

    def test_a_mismatched_pair_is_not_written(self) -> None:
        manifest, report = generate_candidates(EXAMPLE_CONFIG)
        other = GeometryQcReport(
            sampling_id="another_run_v1",
            config_sha256=report.config_sha256,
            entries=report.entries,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValidationError):
                write_outputs(manifest, other, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])


class RejectionTests(unittest.TestCase):
    def test_a_scan_that_reproduces_its_seed_is_rejected_as_a_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(BOND_SCAN.format(factors="1.0, 1.2")))
            manifest, report = generate_candidates(path)
        rejected = [entry for entry in report.entries if entry["status"] == "rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["record_id"], "tmp_run_v1_sih4_seed_bond01_x1p0")
        self.assertIn("sih4_seed", rejected[0]["reason"])
        self.assertEqual(rejected[0]["checks"]["duplicate"]["duplicate_of"], "sih4_seed")
        self.assertEqual(
            [record.record_id for record in manifest.records],
            ["tmp_run_v1_sih4_seed_bond01_x1p2"],
        )

    def test_a_symmetry_equivalent_candidate_is_rejected_naming_the_kept_one(self) -> None:
        # Stretching Si-H1 and stretching Si-H2 of a tetrahedron give the same
        # geometry up to a rotation, so the second one is a duplicate.
        operations = BOND_SCAN.format(factors="1.2") + (
            "  - kind: bond_scan\n"
            "    structure: sih4_seed\n"
            "    charge: 0\n"
            "    multiplicity: 1\n"
            "    anchor_index: 0\n"
            "    moved_index: 2\n"
            "    factors: [1.2]\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(operations))
            manifest, report = generate_candidates(path)
        rejected = [entry for entry in report.entries if entry["status"] == "rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["record_id"], "tmp_run_v1_sih4_seed_bond02_x1p2")
        self.assertIn("tmp_run_v1_sih4_seed_bond01_x1p2", rejected[0]["reason"])
        self.assertEqual(len(manifest.records), 1)

    def test_the_same_geometry_in_two_states_is_not_a_duplicate(self) -> None:
        operations = (
            "  - kind: state_expansion\n"
            "    structure: sih4_seed\n"
            "    states:\n"
            "      - charge: 0\n"
            "        multiplicity: 1\n"
            "      - charge: 0\n"
            "        multiplicity: 3\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(operations))
            manifest, report = generate_candidates(path)
        self.assertEqual(report.counts, {"total": 2, "accepted": 2, "rejected": 0})
        self.assertEqual(len(manifest.records), 2)

    def test_a_collapsed_bond_is_rejected_for_distance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(BOND_SCAN.format(factors="0.5")))
            manifest, report = generate_candidates(path)
        entry = report.entries[0]
        self.assertEqual(entry["status"], "rejected")
        self.assertIn("minimum distance", entry["reason"])
        self.assertEqual(entry["checks"]["minimum_distance"]["violation"]["symbols"], ["Si", "H"])
        self.assertEqual(manifest.records, ())

    def test_a_dissociated_geometry_is_rejected_for_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(BOND_SCAN.format(factors="3.0")))
            _, report = generate_candidates(path)
        entry = report.entries[0]
        self.assertEqual(entry["status"], "rejected")
        self.assertIn("fragments", entry["reason"])
        self.assertEqual(entry["checks"]["fragments"]["count"], 2)

    def test_fragments_are_kept_when_the_config_allows_them(self) -> None:
        filters = "filters:\n  allow_fragments: true\n"
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(
                Path(directory), config_text(BOND_SCAN.format(factors="3.0"), filters)
            )
            manifest, report = generate_candidates(path)
        self.assertEqual(report.entries[0]["status"], "accepted")
        self.assertEqual(len(manifest.records), 1)

    def test_a_rejected_candidate_is_reported_and_left_out_of_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(BOND_SCAN.format(factors="0.5, 1.15")))
            manifest, report = generate_candidates(path)
        self.assertEqual(report.counts, {"total": 2, "accepted": 1, "rejected": 1})
        self.assertEqual(
            [record.record_id for record in manifest.records],
            ["tmp_run_v1_sih4_seed_bond01_x1p15"],
        )


class ConfigErrorTests(unittest.TestCase):
    def test_an_unknown_operation_kind_is_an_error(self) -> None:
        operations = (
            "  - kind: normal_mode_scan\n"
            "    structure: sih4_seed\n"
            "    charge: 0\n"
            "    multiplicity: 1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(operations))
            with self.assertRaises(ValidationError) as caught:
                generate_candidates(path)
        self.assertIn("normal_mode_scan", str(caught.exception))
        self.assertIn("config.operations[0].kind", str(caught.exception))

    def test_an_unknown_operation_key_is_an_error(self) -> None:
        operations = BOND_SCAN.format(factors="1.1") + "    sigma_angstrom: 0.04\n"
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(operations))
            with self.assertRaises(ValidationError) as caught:
                generate_candidates(path)
        self.assertIn("sigma_angstrom", str(caught.exception))

    def test_a_state_expansion_with_impossible_parity_is_an_error(self) -> None:
        operations = (
            "  - kind: state_expansion\n"
            "    structure: sih4_seed\n"
            "    states:\n"
            "      - charge: 0\n"
            "        multiplicity: 2\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(operations))
            with self.assertRaises(ValidationError) as caught:
                generate_candidates(path)
        self.assertIn("multiplicity 2", str(caught.exception))

    def test_a_bond_scan_with_impossible_parity_is_an_error(self) -> None:
        operations = BOND_SCAN.format(factors="1.1").replace("multiplicity: 1", "multiplicity: 2")
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(operations))
            with self.assertRaises(ValidationError) as caught:
                generate_candidates(path)
        self.assertIn("config.operations[0]", str(caught.exception))

    def test_an_unknown_top_level_key_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(
                Path(directory),
                config_text(BOND_SCAN.format(factors="1.1")) + "temperature_k: 300\n",
            )
            with self.assertRaises(ValidationError) as caught:
                load_sampling_config(path)
        self.assertIn("temperature_k", str(caught.exception))

    def test_an_unknown_filter_key_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(
                Path(directory),
                config_text(BOND_SCAN.format(factors="1.1"), "filters:\n  max_atoms: 30\n"),
            )
            with self.assertRaises(ValidationError) as caught:
                load_sampling_config(path)
        self.assertIn("max_atoms", str(caught.exception))

    def test_a_wrong_schema_version_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(
                Path(directory),
                config_text(BOND_SCAN.format(factors="1.1")).replace(
                    "schema_version: 1", "schema_version: 2"
                ),
            )
            with self.assertRaises(ValidationError) as caught:
                load_sampling_config(path)
        self.assertIn("schema_version", str(caught.exception))

    def test_an_unquoted_date_is_an_error_because_it_is_not_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(
                Path(directory),
                config_text(BOND_SCAN.format(factors="1.1")) + "created: 2026-08-22\n",
            )
            with self.assertRaises(ValidationError) as caught:
                load_sampling_config(path)
        self.assertIn("config.created", str(caught.exception))

    def test_a_quoted_date_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(
                Path(directory),
                config_text(BOND_SCAN.format(factors="1.1")) + 'created: "2026-08-22"\n',
            )
            self.assertEqual(load_sampling_config(path)["created"], "2026-08-22")

    def test_an_operation_on_an_unknown_structure_is_an_error(self) -> None:
        operations = BOND_SCAN.format(factors="1.1").replace(
            "structure: sih4_seed", "structure: geh4_seed"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(operations))
            with self.assertRaises(ValidationError) as caught:
                load_sampling_config(path)
        self.assertIn("geh4_seed", str(caught.exception))

    def test_a_repeated_scan_factor_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(BOND_SCAN.format(factors="1.1, 1.1")))
            with self.assertRaises(ValidationError) as caught:
                load_sampling_config(path)
        self.assertIn("factors[1]", str(caught.exception))

    def test_an_empty_operation_list_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text("  []\n"))
            with self.assertRaises(ValidationError):
                load_sampling_config(path)

    def test_a_missing_config_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValidationError):
                load_sampling_config(Path(directory) / "absent.yaml")

    def test_a_missing_structure_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(BOND_SCAN.format(factors="1.1")))
            (Path(directory) / "sih4.xyz").unlink()
            with self.assertRaises(ValidationError) as caught:
                generate_candidates(path)
        self.assertIn("sih4.xyz", str(caught.exception))

    def test_a_scan_index_outside_the_structure_is_an_error(self) -> None:
        operations = BOND_SCAN.format(factors="1.1").replace("moved_index: 1", "moved_index: 9")
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(operations))
            with self.assertRaises(ValidationError) as caught:
                generate_candidates(path)
        self.assertIn("moved_index", str(caught.exception))

    def test_two_operations_that_would_share_an_id_are_an_error(self) -> None:
        operations = BOND_SCAN.format(factors="1.1") + BOND_SCAN.format(factors="1.1")
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), config_text(operations))
            with self.assertRaises(ValidationError) as caught:
                generate_candidates(path)
        self.assertIn("tmp_run_v1_sih4_seed_bond01_x1p1", str(caught.exception))


class XyzReaderTests(unittest.TestCase):
    def test_the_committed_seed_reads_back_as_sih4(self) -> None:
        structure = read_xyz_structure(EXAMPLE_XYZ)
        self.assertEqual(structure.atomic_numbers, (14, 1, 1, 1, 1))
        self.assertEqual(structure.positions_angstrom[0], (0.0, 0.0, 0.0))
        for index in range(1, 5):
            with self.subTest(atom=index):
                self.assertAlmostEqual(
                    math.dist(
                        structure.positions_angstrom[0], structure.positions_angstrom[index]
                    ),
                    1.480,
                    places=9,
                )

    def test_a_count_that_disagrees_with_the_body_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.xyz"
            path.write_text("3\ncomment\nH 0.0 0.0 0.0\n", encoding="utf-8")
            with self.assertRaises(ValidationError) as caught:
                read_xyz_structure(path)
        self.assertIn("3 atoms", str(caught.exception))

    def test_content_after_the_declared_atoms_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.xyz"
            path.write_text("1\ncomment\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", encoding="utf-8")
            with self.assertRaises(ValidationError):
                read_xyz_structure(path)

    def test_a_trailing_blank_line_is_fine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.xyz"
            path.write_text("1\ncomment\nH 0.0 0.0 0.0\n\n\n", encoding="utf-8")
            self.assertEqual(read_xyz_structure(path).atomic_numbers, (1,))

    def test_an_unknown_element_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.xyz"
            path.write_text("1\ncomment\nXx 0.0 0.0 0.0\n", encoding="utf-8")
            with self.assertRaises(ValidationError) as caught:
                read_xyz_structure(path)
        self.assertIn("Xx", str(caught.exception))

    def test_a_non_numeric_coordinate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.xyz"
            path.write_text("1\ncomment\nH 0.0 zero 0.0\n", encoding="utf-8")
            with self.assertRaises(ValidationError):
                read_xyz_structure(path)

    def test_a_non_finite_coordinate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inf.xyz"
            path.write_text("1\ncomment\nH 0.0 inf 0.0\n", encoding="utf-8")
            with self.assertRaises(ValidationError):
                read_xyz_structure(path)

    def test_a_missing_count_line_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nocount.xyz"
            path.write_text("comment\nH 0.0 0.0 0.0\n", encoding="utf-8")
            with self.assertRaises(ValidationError):
                read_xyz_structure(path)

    def test_a_wrong_field_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fields.xyz"
            path.write_text("1\ncomment\nH 0.0 0.0\n", encoding="utf-8")
            with self.assertRaises(ValidationError):
                read_xyz_structure(path)


if __name__ == "__main__":
    unittest.main()
