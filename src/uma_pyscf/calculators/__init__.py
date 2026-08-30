"""Production DFT labeling: protocol, scope, adapter, retry, and resume."""

from __future__ import annotations

from .cli import configure_label, run_label
from .config import (
    DFT_CONFIG_SCHEMA_VERSION,
    load_dft_config,
    method_from_config,
    resource_for_candidate,
    scope_violations,
    validate_dft_config,
)
from .model import CalculationFailure, CalculationOutput, CalculatorAdapter
from .runner import (
    LABEL_EVENT,
    LABEL_LEDGER_SCHEMA,
    build_label_plan,
    run_label_batch,
)

__all__ = [
    "DFT_CONFIG_SCHEMA_VERSION",
    "LABEL_EVENT",
    "LABEL_LEDGER_SCHEMA",
    "CalculationFailure",
    "CalculationOutput",
    "CalculatorAdapter",
    "build_label_plan",
    "configure_label",
    "load_dft_config",
    "method_from_config",
    "resource_for_candidate",
    "run_label",
    "run_label_batch",
    "scope_violations",
    "validate_dft_config",
]
