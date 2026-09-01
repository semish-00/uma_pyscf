from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "run_langevin_md.py"
SPEC = importlib.util.spec_from_file_location("matlantis_pfp_langevin_md", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
langevin_md = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(langevin_md)

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs/pfp_v9_r2scan_plus_d3_langevin_preflight_v1.json"
)


def candidate(record_id: str = "sih4_seed") -> dict:
    return {
        "record_id": record_id,
        "structure": {
            "atomic_numbers": [14, 1, 1, 1, 1],
            "positions_angstrom": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            "parent_structure_id": "sih4_parent",
        },
        "state": {"charge": 0, "multiplicity": 1},
    }


def manifest() -> dict:
    return {
        "schema": "uma-pyscf-candidate-manifest-v1",
        "sampling_id": "md_seed_pool_v1",
        "records": [candidate()],
    }


class PfpLangevinMdTests(unittest.TestCase):
    def test_protocol_grid_and_state_guards(self) -> None:
        protocol = langevin_md._load_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol["model_version"], "v9.0.0")
        self.assertEqual(protocol["timestep_fs"], 0.5)
        self.assertEqual(protocol["temperatures_K"], [300, 600, 900, 1200])
        first = langevin_md._run_identity(manifest(), protocol)
        second = langevin_md._run_identity(manifest(), protocol)
        self.assertEqual(first, second)
        self.assertEqual(len(first["runs"]), 8)
        self.assertEqual(first["runs"][0]["parent_id"], "sih4_parent")
        self.assertEqual(
            first["runs"][0]["path"],
            "trajectories/sih4_seed_t0300_s2026090101.traj",
        )

        charged = candidate()
        charged["state"]["charge"] = 1
        with self.assertRaisesRegex(ValueError, "neutral singlet"):
            langevin_md._validate_candidate(charged)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            invalid = dict(protocol)
            invalid["timestep_fs"] = 1.5
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "<= 1.0"):
                langevin_md._load_protocol(path)

    def test_dry_run_needs_no_matlantis_dependencies_and_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
            output_dir = root / "output"
            args = argparse.Namespace(
                config=PROTOCOL_PATH,
                manifest=manifest_path,
                output_dir=output_dir,
                dry_run=True,
                keep_going=False,
            )
            self.assertEqual(langevin_md.run(args), 0)
            first = (output_dir / "run_identity.json").read_bytes()
            self.assertEqual(langevin_md.run(args), 0)
            second = (output_dir / "run_identity.json").read_bytes()

        self.assertEqual(first, second)

if __name__ == "__main__":
    unittest.main()
