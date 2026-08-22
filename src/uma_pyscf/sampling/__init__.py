"""Deterministic structure candidate generation and geometry QC.

``sampling`` is the first stage of the teaching-data pipeline: it turns a
versioned config into candidate structures and decides, before any DFT time is
spent, which of them are worth computing. It sits above ``schemas`` in the
one-way dependency chain and hands its result on as records, not as imports --
the label stage reads the candidate manifest file that this module writes.

The milestone's own completion condition is reproducibility: the same config
regenerates the same manifest and the same QC report, byte for byte. That rules
out timestamps in either file and any randomness that is not seeded from the
config, and it is why every operation here is written as a pure function of its
inputs.

The Part I validation ladder is the ancestor of these operations -- the bond
scans, the seeded Cartesian displacements, the covalent-radius collision filter
-- generalized and given tests. Nothing is imported from ``validation/``, which
stays frozen.
"""

from __future__ import annotations

from .cli import configure_sample, run_sample
from .filters import (
    fragment_count,
    is_duplicate,
    minimum_distance_violation,
    pair_distance_fingerprint,
)
from .generate import (
    DEFAULT_FILTERS,
    OPERATION_KINDS,
    SAMPLING_CONFIG_SCHEMA_VERSION,
    FilterSettings,
    generate_candidates,
    load_sampling_config,
    read_xyz_structure,
    write_outputs,
)
from .geometry import gaussian_displacement, scale_bond
from .siblings import expand_states

__all__ = [
    "DEFAULT_FILTERS",
    "OPERATION_KINDS",
    "SAMPLING_CONFIG_SCHEMA_VERSION",
    "FilterSettings",
    "configure_sample",
    "expand_states",
    "fragment_count",
    "gaussian_displacement",
    "generate_candidates",
    "is_duplicate",
    "load_sampling_config",
    "minimum_distance_violation",
    "pair_distance_fingerprint",
    "read_xyz_structure",
    "run_sample",
    "scale_bond",
    "write_outputs",
]
