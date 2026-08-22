"""The split config format and the pieces the ``split`` subcommand is built from.

The config is checked here; the end-to-end run of the subcommand lives in
tests/unit/cli/test_main.py next to the other subcommands.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.datasets.cli import load_split_config, split_from_candidates, write_split
from uma_pyscf.datasets.splits import generate_split
from uma_pyscf.sampling.generate import generate_candidates, write_outputs

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_SAMPLING_CONFIG = REPO_ROOT / "configs" / "sampling" / "example_bond_scan_v1.yaml"
EXAMPLE_SPLIT_CONFIG = REPO_ROOT / "configs" / "datasets" / "example_parent_split_v1.yaml"

VALID_CONFIG = (
    "schema_version: 1\n"
    "split_id: unit_split_v1\n"
    "axis: multiplicity\n"
    "seed: 7\n"
    "partitions:\n"
    "  train: 0.5\n"
    "  holdout: 0.5\n"
)


def write_config(directory: Path, text: str, name: str = "split.yaml") -> Path:
    """Write `text` as a config file inside `directory` and return its path."""
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def example_candidates(directory: Path) -> Path:
    """Generate the committed P2.2 example candidates into `directory`."""
    manifest, report = generate_candidates(EXAMPLE_SAMPLING_CONFIG)
    manifest_path, _ = write_outputs(manifest, report, directory)
    return manifest_path


class ConfigLoadingTests(unittest.TestCase):
    def test_a_valid_config_is_returned_verbatim_and_in_declaration_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_split_config(write_config(Path(directory), VALID_CONFIG))
        self.assertEqual(config["split_id"], "unit_split_v1")
        self.assertEqual(config["axis"], "multiplicity")
        self.assertEqual(list(config["partitions"]), ["train", "holdout"])

    def test_the_committed_example_config_loads(self) -> None:
        config = load_split_config(EXAMPLE_SPLIT_CONFIG)
        self.assertEqual(config["split_id"], "example_parent_split_v1")
        self.assertEqual(config["axis"], "parent")
        self.assertEqual(config["seed"], 20260822)
        self.assertEqual(list(config["partitions"]), ["train", "val", "test"])

    def test_an_unknown_key_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(Path(directory), VALID_CONFIG + "shuffle: true\n")
            with self.assertRaises(ValidationError) as caught:
                load_split_config(path)
        self.assertIn("shuffle", str(caught.exception))

    def test_a_missing_required_key_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(Path(directory), VALID_CONFIG.replace("axis: multiplicity\n", ""))
            with self.assertRaises(ValidationError) as caught:
                load_split_config(path)
        self.assertIn("axis", str(caught.exception))

    def test_a_wrong_schema_version_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(
                Path(directory), VALID_CONFIG.replace("schema_version: 1", "schema_version: 2")
            )
            with self.assertRaises(ValidationError):
                load_split_config(path)

    def test_an_unquoted_yaml_date_is_refused_with_its_key_named(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(Path(directory), VALID_CONFIG + "created: 2026-08-22\n")
            with self.assertRaises(ValidationError) as caught:
                load_split_config(path)
        self.assertIn("created", str(caught.exception))

    def test_a_quoted_date_and_a_description_are_accepted_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(
                Path(directory),
                VALID_CONFIG + 'created: "2026-08-22"\ndescription: a unit split\n',
            )
            self.assertEqual(load_split_config(path)["created"], "2026-08-22")

    def test_an_unknown_axis_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(Path(directory), VALID_CONFIG.replace("multiplicity", "random"))
            with self.assertRaises(ValidationError) as caught:
                load_split_config(path)
        self.assertIn("random axis", str(caught.exception))

    def test_fractions_that_do_not_sum_to_one_stop_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(
                Path(directory), VALID_CONFIG.replace("holdout: 0.5", "holdout: 0.4")
            )
            with self.assertRaises(ValidationError):
                load_split_config(path)

    def test_a_missing_file_is_reported_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValidationError) as caught:
                load_split_config(Path(directory) / "absent.yaml")
        self.assertIn("absent.yaml", str(caught.exception))

    def test_malformed_yaml_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(Path(directory), "schema_version: 1\n  bad: [\n")
            with self.assertRaises(ValidationError):
                load_split_config(path)


class SplitFromCandidatesTests(unittest.TestCase):
    def test_the_source_names_the_sampling_run_and_fingerprints_its_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = example_candidates(root)
            config = load_split_config(write_config(root, VALID_CONFIG))
            split = split_from_candidates(config, candidates)
        self.assertEqual(split.source["id"], "example_bond_scan_v1")
        self.assertEqual(len(split.source["sha256"]), 64)
        self.assertEqual(split.record_count, 7)

    def test_the_multiplicity_axis_finds_the_singlets_and_the_one_doublet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = example_candidates(root)
            config = load_split_config(write_config(root, VALID_CONFIG))
            split = split_from_candidates(config, candidates)
        self.assertEqual(sorted(split.group_assignments), ["1", "2"])
        self.assertNotEqual(split.group_assignments["1"], split.group_assignments["2"])

    def test_a_candidate_file_that_is_not_a_manifest_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broken = root / "candidates.json"
            broken.write_text('{"schema": "uma-pyscf-label-record-v1"}', encoding="utf-8")
            config = load_split_config(write_config(root, VALID_CONFIG))
            with self.assertRaises(ValidationError) as caught:
                split_from_candidates(config, broken)
        self.assertIn("uma-pyscf-candidate-manifest-v1", str(caught.exception))


class WriteSplitTests(unittest.TestCase):
    def test_an_empty_item_set_has_no_split_to_write(self) -> None:
        with self.assertRaises(ValidationError):
            generate_split(
                (),
                split_id="named_split_v1",
                axis="parent",
                partitions={"train": 0.5, "test": 0.5},
                seed=1,
                source_id="src_v1",
                source_sha256="a" * 64,
            )

    def test_writing_creates_the_directory_and_names_the_file_after_the_split_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = example_candidates(root)
            config = load_split_config(write_config(root, VALID_CONFIG))
            split = split_from_candidates(config, candidates)
            path = write_split(split, root / "splits")
            self.assertEqual(path, root / "splits" / "unit_split_v1.json")
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
