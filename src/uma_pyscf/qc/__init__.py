"""Quality control: the gate between a computed label and a dataset.

``qc`` answers one question about every canonical label record -- accepted or
rejected -- and answers it with machine-checkable reasons. Nothing reaches a
dataset without passing through here, which is the fail-closed rule of the
project plan applied to teaching data: a label whose convergence, spin state, or
geometry nobody verified is not a label anybody should train on.

Three properties are structural rather than conventional:

* **Thresholds are not in this package.** Every number a verdict depends on
  comes from a versioned config under ``configs/datasets/`` (structure design
  section 4: "閾値の値はconfigs/datasets/へ"). This package owns how a record is
  checked; the config owns what it is checked against, and the report embeds
  that config verbatim so a verdict can always be traced to the conditions that
  produced it.
* **A verdict is appended, never substituted.** Records arrive with
  ``qc.status`` ``pending`` and leave ``accepted`` or ``rejected`` with one new
  history entry naming the run, the config fingerprint, the result, and the
  checks that failed. Everything else in the record is copied through untouched.
* **A check that cannot be evaluated is an error.** A missing threshold, an
  unknown config key, a record that already carries a verdict: these stop the
  run. Only a check that ran and said no produces a rejection.

What is deliberately *not* here yet: the retry protocol and the attempt ledger
(they belong with the calculator stage that produces the retries), the
outlier-detection checks that need a distribution over real labels to calibrate,
and the parent/trajectory leakage check, which the split machinery in
``datasets`` already enforces structurally by assigning whole groups.
"""

from __future__ import annotations

from .cli import (
    configure_qc,
    load_records,
    resolve_record_paths,
    run_qc,
    write_qc_outputs,
)
from .config import (
    ELECTRONIC_KEYS,
    GEOMETRY_KEYS,
    QC_CONFIG_SCHEMA_VERSION,
    load_qc_config,
    validate_qc_config,
)
from .electronic import (
    ELECTRONIC_CHECK_NAMES,
    check_converged,
    check_gradient_max_component,
    check_gradient_norm,
    check_s2_deviation,
    electronic_checks,
    gradient_max_abs,
    gradient_norm,
)
from .geometry import (
    GEOMETRY_CHECK_NAMES,
    check_duplicate,
    check_fragments,
    check_minimum_distance,
    duplicate_map,
    geometry_checks,
    state_qualified_key,
)
from .run import QC_EVENT, apply_qc, composition_formula

__all__ = [
    "ELECTRONIC_CHECK_NAMES",
    "ELECTRONIC_KEYS",
    "GEOMETRY_CHECK_NAMES",
    "GEOMETRY_KEYS",
    "QC_CONFIG_SCHEMA_VERSION",
    "QC_EVENT",
    "apply_qc",
    "check_converged",
    "check_duplicate",
    "check_fragments",
    "check_gradient_max_component",
    "check_gradient_norm",
    "check_minimum_distance",
    "check_s2_deviation",
    "composition_formula",
    "configure_qc",
    "duplicate_map",
    "electronic_checks",
    "geometry_checks",
    "gradient_max_abs",
    "gradient_norm",
    "load_qc_config",
    "load_records",
    "resolve_record_paths",
    "run_qc",
    "state_qualified_key",
    "validate_qc_config",
    "write_qc_outputs",
]
