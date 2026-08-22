"""The QC config format: what a threshold file has to say before it is trusted.

The committed reference config is loaded here as well as by the CLI tests, so a
change to `configs/datasets/example_qc_v1.yaml` that the loader would reject is
caught by the unit suite rather than by a run.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.qc.config import load_qc_config, validate_qc_config

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_QC_CONFIG = REPO_ROOT / "configs" / "datasets" / "example_qc_v1.yaml"

VALID_CONFIG = (
    "schema_version: 1\n"
    "qc_id: unit_qc_v1\n"
    "electronic:\n"
    "  require_converged: true\n"
    "  s2_max_abs_deviation: 0.05\n"
    "  require_s2_for_open_shell: true\n"
    "  gradient_max_abs_hartree_per_bohr: 1.0\n"
    "  gradient_norm_max_hartree_per_bohr: 2.0\n"
    "geometry:\n"
    "  covalent_factor: 0.65\n"
    "  bond_factor: 1.3\n"
    "  allow_fragments: false\n"
    "  duplicate_decimals: 3\n"
)


def write_config(directory: Path, text: str, name: str = "qc.yaml") -> Path:
    """Write `text` as a config file inside `directory` and return its path."""
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def load(text: str) -> dict[str, object]:
    """Load `text` as a QC config through a temporary file."""
    with tempfile.TemporaryDirectory() as directory:
        return load_qc_config(write_config(Path(directory), text))


class CommittedExampleTests(unittest.TestCase):
    def test_the_committed_example_config_loads(self) -> None:
        config = load_qc_config(EXAMPLE_QC_CONFIG)
        self.assertEqual(config["qc_id"], "example_qc_v1")
        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(config["created"], "2026-08-22")

    def test_the_committed_example_states_every_threshold(self) -> None:
        config = load_qc_config(EXAMPLE_QC_CONFIG)
        self.assertEqual(
            config["electronic"],
            {
                "require_converged": True,
                "s2_max_abs_deviation": 0.05,
                "require_s2_for_open_shell": True,
                "gradient_max_abs_hartree_per_bohr": 1.0,
                "gradient_norm_max_hartree_per_bohr": 2.0,
            },
        )
        self.assertEqual(
            config["geometry"],
            {
                "covalent_factor": 0.65,
                "bond_factor": 1.3,
                "allow_fragments": False,
                "duplicate_decimals": 3,
            },
        )


class ConfigLoadingTests(unittest.TestCase):
    def test_a_valid_config_is_returned_verbatim(self) -> None:
        config = load(VALID_CONFIG)
        self.assertEqual(config["qc_id"], "unit_qc_v1")
        self.assertEqual(sorted(config), ["electronic", "geometry", "qc_id", "schema_version"])

    def test_a_quoted_date_and_a_description_are_accepted_headers(self) -> None:
        config = load(VALID_CONFIG + 'created: "2026-08-22"\ndescription: a unit config\n')
        self.assertEqual(config["created"], "2026-08-22")
        self.assertEqual(config["description"], "a unit config")

    def test_an_unquoted_yaml_date_is_refused_with_its_key_named(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG + "created: 2026-08-22\n")
        self.assertIn("created", str(caught.exception))

    def test_a_missing_file_is_reported_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValidationError) as caught:
                load_qc_config(Path(directory) / "absent.yaml")
        self.assertIn("absent.yaml", str(caught.exception))

    def test_malformed_yaml_is_reported(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load("schema_version: 1\n  bad: [\n")
        self.assertIn("YAML", str(caught.exception))

    def test_a_wrong_schema_version_stops_the_run(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG.replace("schema_version: 1", "schema_version: 2"))
        self.assertIn("schema_version", str(caught.exception))


class UnknownKeyTests(unittest.TestCase):
    def test_an_unknown_top_level_key_names_its_path(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG + "retry: true\n")
        message = str(caught.exception)
        self.assertIn("config has unknown key", message)
        self.assertIn("'retry'", message)

    def test_an_unknown_electronic_key_names_its_section(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG.replace("  require_converged: true\n", "  s2_max: 0.1\n"))
        message = str(caught.exception)
        self.assertIn("config.electronic has unknown key", message)
        self.assertIn("'s2_max'", message)

    def test_an_unknown_geometry_key_names_its_section(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG + "  fragment_factor: 1.2\n")
        message = str(caught.exception)
        self.assertIn("config.geometry has unknown key", message)
        self.assertIn("'fragment_factor'", message)


class RequiredFieldTests(unittest.TestCase):
    def test_a_missing_section_stops_the_run(self) -> None:
        text = VALID_CONFIG.split("geometry:")[0]
        with self.assertRaises(ValidationError) as caught:
            load(text)
        self.assertIn("config.geometry", str(caught.exception))

    def test_a_missing_threshold_stops_the_run_with_its_path(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG.replace("  s2_max_abs_deviation: 0.05\n", ""))
        self.assertIn("config.electronic.s2_max_abs_deviation", str(caught.exception))

    def test_a_missing_geometry_threshold_stops_the_run_with_its_path(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG.replace("  bond_factor: 1.3\n", ""))
        self.assertIn("config.geometry.bond_factor", str(caught.exception))


class ThresholdValueTests(unittest.TestCase):
    def test_a_negative_threshold_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG.replace("s2_max_abs_deviation: 0.05", "s2_max_abs_deviation: -0.05"))
        message = str(caught.exception)
        self.assertIn("config.electronic.s2_max_abs_deviation", message)
        self.assertIn("positive", message)

    def test_a_zero_threshold_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            load(VALID_CONFIG.replace("covalent_factor: 0.65", "covalent_factor: 0.0"))

    def test_a_negative_duplicate_decimals_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG.replace("duplicate_decimals: 3", "duplicate_decimals: -1"))
        self.assertIn("config.geometry.duplicate_decimals", str(caught.exception))

    def test_a_non_numeric_threshold_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            load(VALID_CONFIG.replace("bond_factor: 1.3", 'bond_factor: "1.3"'))

    def test_a_flag_that_is_not_a_boolean_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG.replace("allow_fragments: false", "allow_fragments: 0"))
        self.assertIn("config.geometry.allow_fragments", str(caught.exception))

    def test_require_s2_for_open_shell_may_be_false(self) -> None:
        config = load(
            VALID_CONFIG.replace(
                "require_s2_for_open_shell: true", "require_s2_for_open_shell: false"
            )
        )
        electronic = config["electronic"]
        self.assertIs(electronic["require_s2_for_open_shell"], False)  # type: ignore[index]

    def test_require_converged_may_not_be_false_in_v1(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG.replace("require_converged: true", "require_converged: false"))
        message = str(caught.exception)
        self.assertIn("require_converged", message)
        self.assertIn("unconverged", message)


class QcIdTests(unittest.TestCase):
    def test_an_upper_case_qc_id_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            load(VALID_CONFIG.replace("qc_id: unit_qc_v1", "qc_id: Unit_QC_v1"))
        self.assertIn("Record id", str(caught.exception))

    def test_a_qc_id_with_a_slash_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            load(VALID_CONFIG.replace("qc_id: unit_qc_v1", "qc_id: unit/qc/v1"))

    def test_an_empty_qc_id_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            load(VALID_CONFIG.replace("qc_id: unit_qc_v1", 'qc_id: ""'))


class ValidateDirectlyTests(unittest.TestCase):
    def test_a_mapping_can_be_validated_without_a_file(self) -> None:
        config = validate_qc_config(load(VALID_CONFIG))
        self.assertEqual(config["qc_id"], "unit_qc_v1")

    def test_a_non_mapping_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            validate_qc_config([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
