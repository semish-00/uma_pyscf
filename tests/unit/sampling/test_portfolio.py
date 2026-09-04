"""Score-independent, quota-controlled candidate-portfolio assembly."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from uma_pyscf.cli.main import main
from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.ids import canonical_json_fingerprint
from uma_pyscf.core.io import write_json_atomic
from uma_pyscf.sampling.portfolio import assemble_portfolio, load_portfolio_config
from uma_pyscf.schemas.candidate import CandidateManifest, CandidateRecord
from uma_pyscf.schemas.label_record import ElectronicState, Structure

REPO_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_CONFIG = REPO_ROOT / "configs/sampling/calibration_portfolio_180_v1.yaml"


def candidate(
    record_id: str,
    parent_id: str,
    distance: float,
    *,
    multiplicity: int = 1,
    **parameters: object,
) -> CandidateRecord:
    return CandidateRecord(
        record_id=record_id,
        structure=Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (distance, 0.0, 0.0)),
            parent_structure_id=parent_id,
            sampling_method="test",
        ),
        state=ElectronicState(
            charge=0, multiplicity=multiplicity, spin_2s=multiplicity - 1
        ),
        generation_parameters=dict(parameters),
    )


def write_manifest(path: Path, sampling_id: str, records: tuple[CandidateRecord, ...]) -> None:
    config = {"sampling_id": sampling_id}
    manifest = CandidateManifest(
        sampling_id=sampling_id,
        config_sha256=canonical_json_fingerprint(config),
        config=config,
        records=records,
    )
    write_json_atomic(path, manifest.to_dict())


def write_config(
    path: Path, *, second_quota: int = 2, strategy: str = "parent_round_robin"
) -> None:
    path.write_text(
        f"""\
schema_version: 1
portfolio_id: calibration_test_v1
seed: 42
strategy: {strategy}
duplicate_decimals: 3
max_per_parent: 1
max_per_trajectory: 1
sources:
  - category: local
    manifest: local.json
    quota: 2
  - category: path
    manifest: path.json
    quota: {second_quota}
""",
        encoding="utf-8",
    )


class PortfolioAssemblyTests(unittest.TestCase):
    def test_committed_180_record_allocation_is_valid(self) -> None:
        config = load_portfolio_config(CALIBRATION_CONFIG)
        self.assertEqual(sum(int(source["quota"]) for source in config["sources"]), 180)
        self.assertEqual(len(config["sources"]), 6)

    def test_assembly_is_deterministic_auditable_and_state_aware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root / "portfolio.yaml")
            write_manifest(
                root / "local.json",
                "local_pool",
                (
                    candidate("local_p1_a", "p1", 0.70),
                    candidate("local_p1_b", "p1", 0.72),
                    candidate("local_p2", "p2", 0.74),
                ),
            )
            write_manifest(
                root / "path.json",
                "path_pool",
                (
                    candidate("path_p3_a", "p3", 0.76, trajectory_id="t1"),
                    candidate("path_p3_b", "p3", 0.78, trajectory_id="t1"),
                    candidate("path_p4", "p4", 0.80, trajectory_id="t2"),
                ),
            )

            first_manifest, first_report = assemble_portfolio(root / "portfolio.yaml", root)
            second_manifest, second_report = assemble_portfolio(root / "portfolio.yaml", root)

        self.assertEqual(first_manifest.to_dict(), second_manifest.to_dict())
        self.assertEqual(first_report, second_report)
        self.assertEqual(first_report["counts"]["selected"], 4)
        selected_ids = tuple(
            record_id
            for source in first_report["sources"]
            for record_id in source["selected_record_ids"]
        )
        self.assertEqual(
            tuple(record.record_id for record in first_manifest.records),
            selected_ids,
        )
        self.assertEqual(
            {record.structure.parent_structure_id for record in first_manifest.records},
            {"p1", "p2", "p3", "p4"},
        )
        for record in first_manifest.records:
            self.assertIn("portfolio_source_category", record.generation_parameters)
            self.assertEqual(
                len(record.generation_parameters["portfolio_source_manifest_sha256"]), 64
            )

    def test_duplicate_geometry_is_removed_across_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root / "portfolio.yaml", second_quota=2)
            write_manifest(
                root / "local.json",
                "local_pool",
                (
                    candidate("local_p1", "p1", 0.70),
                    candidate("local_p2", "p2", 0.74),
                ),
            )
            write_manifest(
                root / "path.json",
                "path_pool",
                (
                    candidate("path_duplicate", "p3", 0.70, trajectory_id="t1"),
                    candidate(
                        "path_triplet",
                        "p4",
                        0.70,
                        multiplicity=3,
                        trajectory_id="t2",
                    ),
                    candidate("path_unique", "p5", 0.80, trajectory_id="t3"),
                ),
            )
            manifest, report = assemble_portfolio(root / "portfolio.yaml", root)

        path_summary = next(
            source for source in report["sources"] if source["category"] == "path"
        )
        self.assertEqual(
            set(path_summary["selected_record_ids"]), {"path_triplet", "path_unique"}
        )
        self.assertEqual(path_summary["skipped_counts"]["duplicate_geometry_state"], 1)
        self.assertIn("path_triplet", tuple(record.record_id for record in manifest.records))

    def test_d_optimal_is_deterministic_and_selects_distance_extremes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root / "portfolio.yaml", second_quota=1, strategy="d_optimal")
            write_manifest(
                root / "local.json",
                "local_pool",
                tuple(
                    candidate(f"local_p{index}", f"p{index}", distance)
                    for index, distance in enumerate((0.60, 0.75, 0.90, 1.05, 1.20), start=1)
                ),
            )
            write_manifest(
                root / "path.json",
                "path_pool",
                (candidate("path_p6", "p6", 1.40, trajectory_id="t6"),),
            )
            first, _ = assemble_portfolio(root / "portfolio.yaml", root)
            second, _ = assemble_portfolio(root / "portfolio.yaml", root)

        self.assertEqual(first.to_dict(), second.to_dict())
        local_distances = {
            record.structure.positions_angstrom[1][0]
            for record in first.records
            if record.generation_parameters["portfolio_source_category"] == "local"
        }
        self.assertEqual(local_distances, {0.60, 1.20})

    def test_impossible_quota_and_missing_parent_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root / "portfolio.yaml")
            write_manifest(
                root / "local.json",
                "local_pool",
                (
                    candidate("local_p1_a", "p1", 0.70),
                    candidate("local_p1_b", "p1", 0.72),
                ),
            )
            write_manifest(
                root / "path.json",
                "path_pool",
                (
                    candidate("path_p2", "p2", 0.76, trajectory_id="t1"),
                    candidate("path_p3", "p3", 0.78, trajectory_id="t2"),
                ),
            )
            with self.assertRaisesRegex(ValidationError, "could select only"):
                assemble_portfolio(root / "portfolio.yaml", root)

            write_manifest(
                root / "local.json",
                "local_pool",
                (
                    candidate("local_p1", "p1", 0.70),
                    candidate("local_p4", "p4", 0.72),
                ),
            )
            raw = json.loads((root / "path.json").read_text(encoding="utf-8"))
            raw["records"][0]["structure"]["parent_structure_id"] = None
            write_json_atomic(root / "path.json", raw)
            with self.assertRaisesRegex(ValidationError, "parent_structure_id"):
                assemble_portfolio(root / "portfolio.yaml", root)

    def test_cli_writes_manifest_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "portfolio.yaml"
            config.write_text(
                """\
schema_version: 1
portfolio_id: cli_portfolio_v1
seed: 1
strategy: parent_round_robin
duplicate_decimals: 3
max_per_parent: 1
sources:
  - category: local
    manifest: local.json
    quota: 1
""",
                encoding="utf-8",
            )
            write_manifest(
                root / "local.json",
                "local_pool",
                (candidate("local_p1", "p1", 0.70),),
            )
            output_dir = root / "output"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main(
                    [
                        "assemble-portfolio",
                        str(config),
                        "--source-root",
                        str(root),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            manifest_path = output_dir / "cli_portfolio_v1_candidates.json"
            report_path = output_dir / "cli_portfolio_v1_portfolio_report.json"
            self.assertEqual(exit_code, 0)
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertIn("selected=1", stream.getvalue())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["sources"][0]["selected_record_ids"], ["local_p1"])


if __name__ == "__main__":
    unittest.main()
