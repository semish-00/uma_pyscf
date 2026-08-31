"""ASE trajectory import into the unlabeled candidate schema."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from uma_pyscf.sampling.trajectory_import import (
    import_trajectory_candidates,
    uniform_frame_indices,
)


class _Atoms:
    def __init__(self, distance: float) -> None:
        self.numbers = [1, 1]
        self.positions = [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]
        self.pbc = [False, False, False]


class TrajectoryImportTests(unittest.TestCase):
    def test_uniform_indices_include_both_endpoints(self) -> None:
        self.assertEqual(uniform_frame_indices(10, 4), (0, 3, 6, 9))
        self.assertEqual(uniform_frame_indices(5, 1), (2,))

    def test_import_records_source_hash_and_original_frame_index(self) -> None:
        config = """\
schema_version: 1
sampling_id: trajectory_test_v1
state: {charge: 0, multiplicity: 1}
trajectories:
  - trajectory_id: reaction_forward
    parent_id: reaction_family
    path: runs/reaction/forward.traj
    count: 3
filters:
  covalent_factor: 0.65
  bond_factor: 1.3
  allow_fragments: true
  duplicate_decimals: 3
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(config, encoding="utf-8")
            trajectory_path = root / "sources/runs/reaction/forward.traj"
            trajectory_path.parent.mkdir(parents=True)
            trajectory_path.write_bytes(b"trajectory fixture")
            frames = [_Atoms(0.70 + 0.02 * index) for index in range(5)]
            with patch(
                "uma_pyscf.sampling.trajectory_import._read_trajectory",
                return_value=frames,
            ):
                manifest, report = import_trajectory_candidates(config_path, root / "sources")

        self.assertEqual(report.counts, {"total": 3, "accepted": 3, "rejected": 0})
        self.assertEqual(
            [record.generation_parameters["frame_index"] for record in manifest.records],
            [0, 2, 4],
        )
        self.assertTrue(
            all(
                record.generation_parameters["trajectory_id"] == "reaction_forward"
                for record in manifest.records
            )
        )
        source = manifest.config["resolved_sources"][0]
        self.assertEqual(source["selected_frame_indices"], [0, 2, 4])
        self.assertEqual(len(source["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
