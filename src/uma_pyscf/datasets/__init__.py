"""Dataset assembly: how validated records become a teaching set.

``datasets`` sits above ``schemas`` in the one-way dependency chain
``core -> schemas -> (sampling | calculators | qc | datasets | ...) -> cli`` and
takes records from the stages before it as *files*, not as imports -- it reads a
candidate manifest or a set of label records and writes a split manifest.

The package owns both leakage-safe split assignment and ASE-LMDB export. The
export is the sole place where canonical gradients become forces and where
Hartree/Bohr becomes fairchem's eV/Angstrom.
"""

from __future__ import annotations

from .ase_lmdb import (
    export_ase_lmdb_dataset,
    label_record_to_atoms,
    verify_ase_lmdb_dataset,
)
from .ase_lmdb_cli import (
    ASE_DATASET_CONFIG_SCHEMA_VERSION,
    configure_dataset,
    configure_verify_dataset,
    load_dataset_config,
    run_dataset,
    run_verify_dataset,
)
from .baseline import (
    atomic_counts,
    fit_atomic_composition_baseline,
    predict_baseline_energy,
)
from .baseline_cli import (
    BASELINE_CONFIG_SCHEMA_VERSION,
    configure_fit_baseline,
    load_baseline_config,
    run_fit_baseline,
)
from .cli import (
    SPLIT_CONFIG_SCHEMA_VERSION,
    configure_split,
    load_split_config,
    run_split,
    split_from_candidates,
    write_split,
)
from .splits import (
    AXES,
    SplitItem,
    assign_groups,
    composition_formula,
    generate_split,
    group_key_for,
    split_item_from_label_record,
    split_items_from_candidate_manifest,
)

__all__ = [
    "AXES",
    "ASE_DATASET_CONFIG_SCHEMA_VERSION",
    "BASELINE_CONFIG_SCHEMA_VERSION",
    "SPLIT_CONFIG_SCHEMA_VERSION",
    "SplitItem",
    "atomic_counts",
    "assign_groups",
    "composition_formula",
    "configure_dataset",
    "configure_verify_dataset",
    "configure_fit_baseline",
    "configure_split",
    "export_ase_lmdb_dataset",
    "generate_split",
    "group_key_for",
    "load_split_config",
    "load_dataset_config",
    "load_baseline_config",
    "fit_atomic_composition_baseline",
    "predict_baseline_energy",
    "label_record_to_atoms",
    "run_fit_baseline",
    "run_dataset",
    "run_verify_dataset",
    "run_split",
    "split_from_candidates",
    "split_item_from_label_record",
    "split_items_from_candidate_manifest",
    "write_split",
    "verify_ase_lmdb_dataset",
]
