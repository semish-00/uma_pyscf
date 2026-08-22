"""The split manifest schema.

One valid two-partition manifest is built once, and every rejection test breaks
exactly one thing about its on-disk form, so what each check is responsible for
stays visible. The tampering tests matter most: a split manifest is read by the
training stage, and a hand-edited one that moved a record between partitions has
to be refused rather than obeyed.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
from typing import Any
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.io import read_json, write_json_atomic
from uma_pyscf.schemas.split_manifest import (
    SPLIT_AXES,
    SPLIT_MANIFEST_SCHEMA,
    SplitManifest,
    validate_axis,
    validate_partitions,
)

SOURCE = {"id": "example_bond_scan_v1", "sha256": "a" * 64}


def manifest() -> SplitManifest:
    """Return a valid two-partition split of five records over three groups."""
    return SplitManifest(
        split_id="example_parent_split_v1",
        axis="parent",
        seed=20260822,
        partitions={"train": 0.6, "test": 0.4},
        source=dict(SOURCE),
        group_assignments={"sih4_seed": "train", "geh4_seed": "train", "sih3cl_seed": "test"},
        record_assignments={
            "train": ("sih4_a", "sih4_b", "geh4_a"),
            "test": ("sih3cl_a", "sih3cl_b"),
        },
    )


def tampered(**changes: Any) -> dict[str, Any]:
    """Return the manifest's dict form with top-level keys replaced."""
    data = manifest().to_dict()
    data.update(changes)
    return data


class RoundTripTests(unittest.TestCase):
    def test_a_manifest_survives_to_dict_and_from_dict(self) -> None:
        original = manifest()
        rebuilt = SplitManifest.from_dict(original.to_dict())
        self.assertEqual(rebuilt.to_dict(), original.to_dict())
        self.assertEqual(rebuilt.axis, "parent")
        self.assertEqual(rebuilt.seed, 20260822)

    def test_a_manifest_survives_a_write_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            write_json_atomic(path, manifest().to_dict())
            self.assertEqual(
                SplitManifest.from_dict(read_json(path)).to_dict(), manifest().to_dict()
            )

    def test_records_are_sorted_within_each_partition(self) -> None:
        built = SplitManifest(
            split_id="sorted_v1",
            axis="parent",
            seed=1,
            partitions={"train": 0.5, "test": 0.5},
            source=dict(SOURCE),
            group_assignments={"p0": "train", "p1": "test"},
            record_assignments={"train": ("r9", "r1", "r5"), "test": ("r2",)},
        )
        self.assertEqual(built.record_assignments["train"], ("r1", "r5", "r9"))

    def test_counts_summarize_both_assignment_maps(self) -> None:
        self.assertEqual(
            manifest().to_dict()["counts"],
            {
                "records": 5,
                "groups": 3,
                "records_by_partition": {"train": 3, "test": 2},
                "groups_by_partition": {"train": 2, "test": 1},
            },
        )

    def test_an_empty_partition_is_listed_rather_than_omitted(self) -> None:
        built = SplitManifest(
            split_id="empty_v1",
            axis="charge",
            seed=1,
            partitions={"train": 0.9, "holdout": 0.1},
            source=dict(SOURCE),
            group_assignments={"0": "train", "1": "train"},
            record_assignments={"train": ("r0", "r1"), "holdout": ()},
        )
        self.assertEqual(built.to_dict()["record_assignments"]["holdout"], [])
        self.assertEqual(built.counts["records_by_partition"]["holdout"], 0)


class TamperingTests(unittest.TestCase):
    def test_a_record_moved_between_partitions_is_caught_by_the_counts(self) -> None:
        data = manifest().to_dict()
        records = deepcopy(data["record_assignments"])
        records["train"].remove("geh4_a")
        records["test"].append("geh4_a")
        data["record_assignments"] = records
        with self.assertRaises(ValidationError) as caught:
            SplitManifest.from_dict(data)
        self.assertIn("counts", str(caught.exception))

    def test_a_record_copied_into_two_partitions_is_caught_by_disjointness(self) -> None:
        data = manifest().to_dict()
        records = deepcopy(data["record_assignments"])
        records["test"].append("sih4_a")
        data["record_assignments"] = records
        with self.assertRaises(ValidationError) as caught:
            SplitManifest.from_dict(data)
        message = str(caught.exception)
        self.assertIn("sih4_a", message)
        self.assertIn("disjoint", message)

    def test_a_group_moved_to_another_partition_is_caught_by_the_counts(self) -> None:
        data = manifest().to_dict()
        groups = deepcopy(data["group_assignments"])
        groups["geh4_seed"] = "test"
        data["group_assignments"] = groups
        with self.assertRaises(ValidationError) as caught:
            SplitManifest.from_dict(data)
        self.assertIn("counts", str(caught.exception))

    def test_a_removed_record_is_caught_by_the_counts(self) -> None:
        data = manifest().to_dict()
        records = deepcopy(data["record_assignments"])
        records["test"].remove("sih3cl_b")
        data["record_assignments"] = records
        with self.assertRaises(ValidationError):
            SplitManifest.from_dict(data)

    def test_counts_may_be_absent_and_are_then_derived(self) -> None:
        data = manifest().to_dict()
        del data["counts"]
        self.assertEqual(SplitManifest.from_dict(data).counts, manifest().counts)


class RejectionTests(unittest.TestCase):
    def test_an_unknown_key_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            SplitManifest.from_dict(tampered(comment="added by hand"))
        self.assertIn("comment", str(caught.exception))

    def test_a_foreign_schema_string_is_refused_by_name(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            SplitManifest.from_dict(tampered(schema="uma-pyscf-split-manifest-v0"))
        self.assertIn(SPLIT_MANIFEST_SCHEMA, str(caught.exception))

    def test_a_missing_key_is_refused(self) -> None:
        data = manifest().to_dict()
        del data["seed"]
        with self.assertRaises(ValidationError) as caught:
            SplitManifest.from_dict(data)
        self.assertIn("seed", str(caught.exception))

    def test_a_split_id_that_is_not_an_identifier_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            SplitManifest.from_dict(tampered(split_id="Parent Split"))

    def test_an_unknown_axis_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            SplitManifest.from_dict(tampered(axis="random"))
        self.assertIn("random axis", str(caught.exception))

    def test_a_group_naming_an_undeclared_partition_is_refused(self) -> None:
        data = manifest().to_dict()
        data["group_assignments"] = dict(data["group_assignments"]) | {"sih3cl_seed": "val"}
        with self.assertRaises(ValidationError) as caught:
            SplitManifest.from_dict(data)
        self.assertIn("'val'", str(caught.exception))

    def test_a_record_partition_that_is_not_declared_is_refused(self) -> None:
        data = manifest().to_dict()
        records = deepcopy(data["record_assignments"])
        records["val"] = []
        data["record_assignments"] = records
        with self.assertRaises(ValidationError) as caught:
            SplitManifest.from_dict(data)
        self.assertIn("'val'", str(caught.exception))

    def test_a_declared_partition_missing_from_the_records_is_refused(self) -> None:
        data = manifest().to_dict()
        records = deepcopy(data["record_assignments"])
        del records["test"]
        data["record_assignments"] = records
        with self.assertRaises(ValidationError) as caught:
            SplitManifest.from_dict(data)
        self.assertIn("missing the declared partition", str(caught.exception))

    def test_an_empty_group_map_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            SplitManifest.from_dict(tampered(group_assignments={}))

    def test_a_source_digest_that_is_not_a_sha256_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            SplitManifest.from_dict(
                tampered(source={"id": "example_bond_scan_v1", "sha256": "abc"})
            )
        self.assertIn("64 hexadecimal", str(caught.exception))

    def test_an_unknown_source_key_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            SplitManifest.from_dict(
                tampered(source={"id": "example_bond_scan_v1", "sha256": "a" * 64, "path": "x"})
            )

    def test_a_record_id_that_is_not_an_identifier_is_refused(self) -> None:
        data = manifest().to_dict()
        records = deepcopy(data["record_assignments"])
        records["test"].append("Not An Id")
        data["record_assignments"] = records
        with self.assertRaises(ValidationError):
            SplitManifest.from_dict(data)


class PartitionValidatorTests(unittest.TestCase):
    def test_the_declared_fractions_are_returned_in_declaration_order(self) -> None:
        self.assertEqual(
            list(validate_partitions({"train": 0.6, "val": 0.2, "test": 0.2}, "p")),
            ["train", "val", "test"],
        )

    def test_binary_representation_noise_is_tolerated(self) -> None:
        self.assertEqual(len(validate_partitions({"a": 0.1, "b": 0.2, "c": 0.7}, "p")), 3)

    def test_a_sum_outside_the_tolerance_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            validate_partitions({"a": 0.5, "b": 0.5 + 1e-6}, "p")

    def test_one_partition_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            validate_partitions({"all": 1.0}, "p")

    def test_a_non_numeric_fraction_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            validate_partitions({"train": "0.5", "test": 0.5}, "p")


class AxisValidatorTests(unittest.TestCase):
    def test_every_documented_axis_validates(self) -> None:
        for axis in SPLIT_AXES:
            with self.subTest(axis=axis):
                self.assertEqual(validate_axis(axis, "axis"), axis)

    def test_the_axis_list_is_the_one_the_plan_names(self) -> None:
        self.assertEqual(SPLIT_AXES, ("parent", "composition", "charge", "multiplicity"))


if __name__ == "__main__":
    unittest.main()
