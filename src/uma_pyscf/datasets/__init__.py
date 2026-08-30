"""Dataset assembly: how validated records become a teaching set.

``datasets`` sits above ``schemas`` in the one-way dependency chain
``core -> schemas -> (sampling | calculators | qc | datasets | ...) -> cli`` and
takes records from the stages before it as *files*, not as imports -- it reads a
candidate manifest or a set of label records and writes a split manifest.

What lives here first is the split machinery, which the plan places in P2.6 and
decision 0002 sanctioned early because it is pure data processing over records
that already exist: nothing about grouping a dataset by parent structure or by
composition depends on the Gate 1 outcome. The rest of the milestone -- the
LMDB/ASE export, the fairchem unit conversion, the one place where a gradient
becomes a force -- arrives with the stages that need it and is not anticipated
here.
"""

from __future__ import annotations

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
    "BASELINE_CONFIG_SCHEMA_VERSION",
    "SPLIT_CONFIG_SCHEMA_VERSION",
    "SplitItem",
    "atomic_counts",
    "assign_groups",
    "composition_formula",
    "configure_fit_baseline",
    "configure_split",
    "generate_split",
    "group_key_for",
    "load_split_config",
    "load_baseline_config",
    "fit_atomic_composition_baseline",
    "predict_baseline_energy",
    "run_fit_baseline",
    "run_split",
    "split_from_candidates",
    "split_item_from_label_record",
    "split_items_from_candidate_manifest",
    "write_split",
]
