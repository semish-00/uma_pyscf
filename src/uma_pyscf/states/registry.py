"""Load and enforce the non-default electronic-state registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..core.elements import PERIODIC_SYMBOLS
from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint
from ..schemas.candidate import CandidateRecord
from ..schemas.label_record import LabelRecord
from ..schemas.state_registry import StateRegistry

__all__ = [
    "candidate_composition",
    "load_state_registry",
    "registry_identity",
    "state_registry_violations",
]


def _composition(atomic_numbers: tuple[int, ...]) -> str:
    counts: dict[str, int] = {}
    for number in atomic_numbers:
        symbol = PERIODIC_SYMBOLS[number]
        counts[symbol] = counts.get(symbol, 0) + 1
    return "".join(
        symbol + (str(count) if count > 1 else "")
        for symbol, count in sorted(counts.items())
    )


def candidate_composition(candidate: CandidateRecord | LabelRecord) -> str:
    """Return the canonical composition used as a registry key."""
    if not isinstance(candidate, CandidateRecord | LabelRecord):
        raise ValidationError(
            f"A CandidateRecord or LabelRecord is required; got {type(candidate).__name__}."
        )
    return _composition(candidate.structure.atomic_numbers)


def load_state_registry(path: str | Path) -> StateRegistry:
    source = Path(path)
    try:
        loaded: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"State registry {source} cannot be read: {exc}.") from exc
    except yaml.YAMLError as exc:
        raise ValidationError(f"State registry {source} is not valid YAML: {exc}.") from exc
    return StateRegistry.from_dict(loaded)


def registry_identity(registry: StateRegistry | None) -> dict[str, str] | None:
    if registry is None:
        return None
    return {
        "state_registry_id": registry.registry_id,
        "state_registry_sha256": canonical_json_fingerprint(registry.to_dict()),
    }


def state_registry_violations(
    record: CandidateRecord | LabelRecord, registry: StateRegistry | None
) -> tuple[str, ...]:
    """Return violations for a non-default record against an actual registry."""
    non_default = record.state.charge != 0 or record.state.multiplicity != 1
    if not non_default:
        return ()
    if registry is None:
        return ("non_default_state_registry_not_supplied",)
    composition = candidate_composition(record)
    entry = registry.entry_for(composition, record.state.charge, record.state.multiplicity)
    if entry is None:
        return (
            "non_default_state_not_listed_in_registry:"
            f"{composition}:q{record.state.charge}:m{record.state.multiplicity}",
        )
    violations: list[str] = []
    if entry.status != "approved":
        violations.append(f"non_default_state_registry_status:{entry.status}")
    expected = entry.provenance(registry.registry_id)
    if record.state.state_provenance != expected:
        violations.append(f"state_provenance_registry_mismatch:expected={expected}")
    return tuple(violations)
