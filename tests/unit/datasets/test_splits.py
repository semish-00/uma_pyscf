"""Group-wise split assignment, and the leakage invariants it exists to keep.

Most of these tests do not check a specific assignment. They check a property
that has to hold for *every* assignment: a group is never divided, related
records travel together on the ordinary axes, and the generalization axes pull
charge/spin siblings apart on purpose. The few tests that do pin a specific
outcome are the determinism ones, because reproducing a split byte for byte is
the milestone's completion condition.
"""

from __future__ import annotations

import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.ids import canonical_json_fingerprint
from uma_pyscf.datasets.splits import (
    AXES,
    SplitItem,
    assign_groups,
    composition_formula,
    generate_split,
    group_key_for,
    split_item_from_label_record,
    split_items_from_candidate_manifest,
)
from uma_pyscf.schemas.candidate import CandidateManifest, CandidateRecord
from uma_pyscf.schemas.label_record import (
    ElectronicState,
    Engine,
    LabelRecord,
    Method,
    QcState,
    RawArtifact,
    Results,
    Structure,
)

HALVES: dict[str, float] = {"train": 0.5, "test": 0.5}


def item(
    record_id: str,
    *,
    parent: str | None = "p0",
    composition: str = "H4Si",
    charge: int = 0,
    multiplicity: int = 1,
) -> SplitItem:
    """Return a split item, defaulting every axis to one shared value."""
    return SplitItem(
        record_id=record_id,
        parent_structure_id=parent,
        composition=composition,
        charge=charge,
        multiplicity=multiplicity,
    )


def one_item_per_parent(count: int) -> tuple[SplitItem, ...]:
    """Return `count` items that are each alone in their own parent group."""
    return tuple(item(f"r{index:03d}", parent=f"p{index:03d}") for index in range(count))


def partition_of(manifest_records: dict[str, tuple[str, ...]], record_id: str) -> str:
    """Return the name of the partition holding `record_id`."""
    for name, ids in manifest_records.items():
        if record_id in ids:
            return name
    raise AssertionError(f"{record_id} was not assigned to any partition.")


def h2_structure(parent: str | None = "h2_seed") -> Structure:
    """Return a small H2 geometry carrying `parent` as its provenance."""
    return Structure(
        atomic_numbers=(1, 1),
        positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74144)),
        parent_structure_id=parent,
    )


def h2_label_record(record_id: str = "h2_neutral_singlet") -> LabelRecord:
    """Return a valid minimal H2 label record."""
    return LabelRecord(
        record_id=record_id,
        structure=h2_structure(),
        state=ElectronicState(charge=0, multiplicity=1, spin_2s=0),
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
        engine=Engine(name="gpu4pyscf", versions={"pyscf": "2.6.2"}),
        results=Results(
            energy_hartree=-1.1730407,
            gradient_hartree_per_bohr=((0.0, 0.0, -0.0123456), (0.0, 0.0, 0.0123456)),
            converged=True,
        ),
        raw=RawArtifact(),
        qc=QcState(status="pending"),
    )


class CompositionFormulaTests(unittest.TestCase):
    def test_symbols_are_alphabetical_and_a_single_atom_drops_its_count(self) -> None:
        self.assertEqual(composition_formula((14, 1, 1, 1, 1)), "H4Si")

    def test_the_same_atoms_in_any_order_give_the_same_formula(self) -> None:
        self.assertEqual(
            composition_formula((1, 14, 1, 32, 1)), composition_formula((32, 1, 1, 1, 14))
        )

    def test_a_removed_hydrogen_changes_the_formula(self) -> None:
        self.assertNotEqual(
            composition_formula((14, 1, 1, 1, 1)), composition_formula((14, 1, 1, 1))
        )

    def test_an_impossible_atomic_number_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            composition_formula((14, 0))

    def test_an_empty_structure_has_no_composition(self) -> None:
        with self.assertRaises(ValidationError):
            composition_formula(())


class SplitItemTests(unittest.TestCase):
    def test_a_record_id_that_is_not_an_identifier_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            item("Not An Id")

    def test_a_parent_that_is_not_an_identifier_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            item("r0", parent="Not An Id")

    def test_a_multiplicity_below_one_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            item("r0", multiplicity=0)

    def test_parity_is_not_re_checked_because_the_record_already_passed_it(self) -> None:
        # An H2 item declared as a triplet would fail the schema layer's parity
        # check, but a split item carries what a validated record said and does
        # not re-litigate it.
        self.assertEqual(item("r0", charge=0, multiplicity=3).multiplicity, 3)


class GroupKeyTests(unittest.TestCase):
    def test_the_axes_are_the_four_the_plan_names_and_there_is_no_random_one(self) -> None:
        self.assertEqual(AXES, ("parent", "composition", "charge", "multiplicity"))
        self.assertNotIn("random", AXES)

    def test_every_axis_produces_a_key(self) -> None:
        subject = item("r0", parent="sih4_seed", composition="H4Si", charge=1, multiplicity=2)
        self.assertEqual(
            [group_key_for(subject, axis) for axis in AXES], ["sih4_seed", "H4Si", "1", "2"]
        )

    def test_a_record_without_a_parent_is_its_own_parent_group(self) -> None:
        self.assertEqual(group_key_for(item("orphan_r0", parent=None), "parent"), "orphan_r0")

    def test_an_unknown_axis_is_refused_and_names_the_known_ones(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            group_key_for(item("r0"), "random")
        self.assertIn("random axis", str(caught.exception))


class ExtractorTests(unittest.TestCase):
    def test_candidate_manifest_items_keep_manifest_order_and_provenance(self) -> None:
        config = {"schema_version": 1}
        records = tuple(
            CandidateRecord(
                record_id=record_id,
                structure=h2_structure(),
                state=ElectronicState(charge=charge, multiplicity=multiplicity, spin_2s=spin),
            )
            for record_id, charge, multiplicity, spin in (
                ("h2_q0m1", 0, 1, 0),
                ("h2_q1m2", 1, 2, 1),
            )
        )
        manifest = CandidateManifest(
            sampling_id="h2_v1",
            config_sha256=canonical_json_fingerprint(config),
            config=config,
            records=records,
        )
        items = split_items_from_candidate_manifest(manifest)
        self.assertEqual([entry.record_id for entry in items], ["h2_q0m1", "h2_q1m2"])
        self.assertEqual([entry.charge for entry in items], [0, 1])
        self.assertEqual({entry.composition for entry in items}, {"H2"})
        self.assertEqual({entry.parent_structure_id for entry in items}, {"h2_seed"})

    def test_a_label_record_becomes_one_item(self) -> None:
        extracted = split_item_from_label_record(h2_label_record())
        self.assertEqual(extracted.record_id, "h2_neutral_singlet")
        self.assertEqual(extracted.composition, "H2")
        self.assertEqual(extracted.parent_structure_id, "h2_seed")
        self.assertEqual((extracted.charge, extracted.multiplicity), (0, 1))

    def test_the_extractors_refuse_the_wrong_record_type(self) -> None:
        with self.assertRaises(ValidationError):
            split_item_from_label_record("not a record")  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            split_items_from_candidate_manifest("not a manifest")  # type: ignore[arg-type]


class LeakageInvariantTests(unittest.TestCase):
    """The properties every assignment has to satisfy, on every axis."""

    def mixed_items(self) -> tuple[SplitItem, ...]:
        """Return items spanning three parents, two compositions, and four states."""
        items: list[SplitItem] = []
        for parent, composition in (
            ("sih4_seed", "H4Si"),
            ("geh4_seed", "GeH4"),
            ("sih3cl_seed", "ClH3Si"),
        ):
            for index in range(4):
                items.append(
                    item(
                        f"{parent}_scan{index}",
                        parent=parent,
                        composition=composition,
                        charge=0,
                        multiplicity=1,
                    )
                )
            # The charge/spin siblings of the same geometry.
            items.append(
                item(
                    f"{parent}_q1m2",
                    parent=parent,
                    composition=composition,
                    charge=1,
                    multiplicity=2,
                )
            )
            items.append(
                item(
                    f"{parent}_qm1m2",
                    parent=parent,
                    composition=composition,
                    charge=-1,
                    multiplicity=2,
                )
            )
        return tuple(items)

    def test_no_group_is_ever_divided_between_partitions_on_any_axis(self) -> None:
        items = self.mixed_items()
        for axis in AXES:
            with self.subTest(axis=axis):
                split = generate_split(
                    items,
                    split_id="leak_v1",
                    axis=axis,
                    partitions=HALVES,
                    seed=11,
                    source_id="src_v1",
                    source_sha256="a" * 64,
                )
                landed: dict[str, set[str]] = {}
                for entry in items:
                    key = group_key_for(entry, axis)
                    where = partition_of(split.record_assignments, entry.record_id)
                    landed.setdefault(key, set()).add(where)
                for key, partitions in landed.items():
                    self.assertEqual(
                        len(partitions), 1, f"group {key!r} on axis {axis!r} spans {partitions}"
                    )

    def test_every_record_is_assigned_exactly_once(self) -> None:
        items = self.mixed_items()
        split = generate_split(
            items,
            split_id="leak_v1",
            axis="parent",
            partitions=HALVES,
            seed=11,
            source_id="src_v1",
            source_sha256="a" * 64,
        )
        assigned = [rid for ids in split.record_assignments.values() for rid in ids]
        self.assertEqual(sorted(assigned), sorted(entry.record_id for entry in items))
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_the_parent_axis_keeps_a_scan_and_its_siblings_together(self) -> None:
        items = self.mixed_items()
        split = generate_split(
            items,
            split_id="leak_v1",
            axis="parent",
            partitions=HALVES,
            seed=11,
            source_id="src_v1",
            source_sha256="a" * 64,
        )
        for parent in ("sih4_seed", "geh4_seed", "sih3cl_seed"):
            with self.subTest(parent=parent):
                related = [
                    partition_of(split.record_assignments, entry.record_id)
                    for entry in items
                    if entry.parent_structure_id == parent
                ]
                self.assertEqual(len(set(related)), 1)

    def test_the_composition_axis_keeps_charge_and_spin_siblings_together(self) -> None:
        # The deliberate contrast with the charge and multiplicity axes below:
        # siblings share a geometry, so an ordinary split must not separate them.
        items = self.mixed_items()
        split = generate_split(
            items,
            split_id="leak_v1",
            axis="composition",
            partitions=HALVES,
            seed=3,
            source_id="src_v1",
            source_sha256="a" * 64,
        )
        for parent in ("sih4_seed", "geh4_seed", "sih3cl_seed"):
            siblings = [f"{parent}_scan0", f"{parent}_q1m2", f"{parent}_qm1m2"]
            with self.subTest(parent=parent):
                where = {
                    partition_of(split.record_assignments, record_id) for record_id in siblings
                }
                self.assertEqual(len(where), 1)

    def test_the_charge_axis_keeps_one_charge_together(self) -> None:
        items = self.mixed_items()
        split = generate_split(
            items,
            split_id="leak_v1",
            axis="charge",
            partitions=HALVES,
            seed=5,
            source_id="src_v1",
            source_sha256="a" * 64,
        )
        for charge in (-1, 0, 1):
            with self.subTest(charge=charge):
                where = {
                    partition_of(split.record_assignments, entry.record_id)
                    for entry in items
                    if entry.charge == charge
                }
                self.assertEqual(len(where), 1)

    def test_the_charge_axis_does_separate_two_different_charges(self) -> None:
        # Two equally sized charge groups and two equal partitions: the first
        # group fills one partition's share exactly, so the second has to go to
        # the other. Siblings are pulled apart, which is what a charge holdout is.
        items = (
            item("neutral_a", charge=0, multiplicity=1),
            item("neutral_b", charge=0, multiplicity=1),
            item("cation_a", charge=1, multiplicity=2),
            item("cation_b", charge=1, multiplicity=2),
        )
        split = generate_split(
            items,
            split_id="charge_holdout_v1",
            axis="charge",
            partitions=HALVES,
            seed=1,
            source_id="src_v1",
            source_sha256="a" * 64,
        )
        self.assertNotEqual(
            partition_of(split.record_assignments, "neutral_a"),
            partition_of(split.record_assignments, "cation_a"),
        )

    def test_the_multiplicity_axis_separates_a_singlet_from_a_doublet(self) -> None:
        items = (
            item("singlet_a", charge=0, multiplicity=1),
            item("singlet_b", charge=0, multiplicity=1),
            item("doublet_a", charge=1, multiplicity=2),
            item("doublet_b", charge=1, multiplicity=2),
        )
        split = generate_split(
            items,
            split_id="spin_holdout_v1",
            axis="multiplicity",
            partitions=HALVES,
            seed=1,
            source_id="src_v1",
            source_sha256="a" * 64,
        )
        self.assertNotEqual(
            partition_of(split.record_assignments, "singlet_a"),
            partition_of(split.record_assignments, "doublet_a"),
        )


class DeterminismTests(unittest.TestCase):
    def test_the_same_inputs_produce_an_identical_manifest(self) -> None:
        items = one_item_per_parent(40)
        first, second = (
            generate_split(
                items,
                split_id="det_v1",
                axis="parent",
                partitions={"train": 0.7, "test": 0.3},
                seed=20260822,
                source_id="src_v1",
                source_sha256="b" * 64,
            ).to_dict()
            for _ in range(2)
        )
        self.assertEqual(first, second)

    def test_the_item_order_does_not_change_the_assignment(self) -> None:
        items = one_item_per_parent(40)
        forward = assign_groups(items, "parent", HALVES, 4, "det_v1")
        backward = assign_groups(tuple(reversed(items)), "parent", HALVES, 4, "det_v1")
        self.assertEqual(forward, backward)

    def test_a_different_seed_reshuffles_the_assignment(self) -> None:
        # Forty groups over two partitions: two seeds agreeing on every single
        # group would be a one-in-2**39 coincidence, so inequality of the whole
        # mapping is the honest assertion rather than inequality of one group.
        items = one_item_per_parent(40)
        first = assign_groups(items, "parent", HALVES, 1, "det_v1")
        second = assign_groups(items, "parent", HALVES, 2, "det_v1")
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first, second)

    def test_a_different_split_id_reshuffles_the_assignment(self) -> None:
        items = one_item_per_parent(40)
        first = assign_groups(items, "parent", HALVES, 1, "det_a_v1")
        second = assign_groups(items, "parent", HALVES, 1, "det_b_v1")
        self.assertNotEqual(first, second)


class FractionTests(unittest.TestCase):
    def test_uniform_groups_honour_the_fractions_exactly(self) -> None:
        items = one_item_per_parent(100)
        split = generate_split(
            items,
            split_id="frac_v1",
            axis="parent",
            partitions={"train": 0.8, "test": 0.2},
            seed=99,
            source_id="src_v1",
            source_sha256="c" * 64,
        )
        self.assertEqual(len(split.record_assignments["train"]), 80)
        self.assertEqual(len(split.record_assignments["test"]), 20)

    def test_three_uniform_partitions_honour_their_fractions_exactly(self) -> None:
        items = one_item_per_parent(100)
        split = generate_split(
            items,
            split_id="frac_v1",
            axis="parent",
            partitions={"train": 0.6, "val": 0.2, "test": 0.2},
            seed=99,
            source_id="src_v1",
            source_sha256="c" * 64,
        )
        self.assertEqual(
            {name: len(ids) for name, ids in split.record_assignments.items()},
            {"train": 60, "val": 20, "test": 20},
        )

    def test_uneven_groups_are_still_never_divided(self) -> None:
        # One group of ten records and ten groups of one. No fraction can be hit
        # exactly here, and that is the trade the machinery makes: whole groups
        # first, fractions second.
        items = [item(f"big{index}", parent="big_parent") for index in range(10)]
        items.extend(item(f"small{index}", parent=f"small_p{index}") for index in range(10))
        split = generate_split(
            tuple(items),
            split_id="uneven_v1",
            axis="parent",
            partitions=HALVES,
            seed=8,
            source_id="src_v1",
            source_sha256="d" * 64,
        )
        big = {partition_of(split.record_assignments, f"big{index}") for index in range(10)}
        self.assertEqual(len(big), 1)
        self.assertEqual(split.record_count, 20)


class RefusalTests(unittest.TestCase):
    def test_fractions_that_do_not_sum_to_one_are_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            assign_groups(one_item_per_parent(4), "parent", {"train": 0.7, "test": 0.2}, 1, "s_v1")
        self.assertIn("sum to", str(caught.exception))

    def test_a_zero_fraction_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            assign_groups(one_item_per_parent(4), "parent", {"train": 1.0, "test": 0.0}, 1, "s_v1")
        self.assertIn("positive fraction", str(caught.exception))

    def test_a_negative_fraction_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            assign_groups(
                one_item_per_parent(4), "parent", {"train": 1.2, "test": -0.2}, 1, "s_v1"
            )

    def test_a_single_partition_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            assign_groups(one_item_per_parent(4), "parent", {"train": 1.0}, 1, "s_v1")
        self.assertIn("at least two partitions", str(caught.exception))

    def test_fewer_groups_than_partitions_is_refused_with_the_counts(self) -> None:
        items = tuple(item(f"r{index}", parent="only_parent") for index in range(7))
        with self.assertRaises(ValidationError) as caught:
            assign_groups(items, "parent", {"train": 0.6, "val": 0.2, "test": 0.2}, 1, "single_v1")
        message = str(caught.exception)
        self.assertIn("1 distinct group", message)
        self.assertIn("3 partitions", message)
        self.assertIn("more distinct groups", message)

    def test_an_unknown_axis_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            assign_groups(one_item_per_parent(4), "scaffold", HALVES, 1, "s_v1")
        self.assertIn("'parent'", str(caught.exception))

    def test_a_duplicate_record_id_is_refused(self) -> None:
        items = (item("r0", parent="p0"), item("r0", parent="p1"))
        with self.assertRaises(ValidationError) as caught:
            assign_groups(items, "parent", HALVES, 1, "s_v1")
        self.assertIn("appears twice", str(caught.exception))

    def test_a_partition_name_that_is_not_an_identifier_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            assign_groups(one_item_per_parent(4), "parent", {"Train": 0.5, "test": 0.5}, 1, "s_v1")
        self.assertIn("partition name", str(caught.exception))

    def test_a_split_id_that_is_not_an_identifier_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            assign_groups(one_item_per_parent(4), "parent", HALVES, 1, "Split V1")

    def test_something_that_is_not_a_split_item_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            assign_groups(("r0",), "parent", HALVES, 1, "s_v1")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
