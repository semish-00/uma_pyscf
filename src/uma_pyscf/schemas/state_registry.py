"""Versioned scientific approval registry for non-default electronic states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import validate_record_id
from ._fields import (
    optional_str,
    reject_unknown_keys,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
)

__all__ = [
    "STATE_REGISTRY_SCHEMA",
    "STATE_REGISTRY_STATUSES",
    "StateRegistry",
    "StateRegistryEntry",
]

STATE_REGISTRY_SCHEMA = "uma-pyscf-state-registry-v1"
STATE_REGISTRY_STATUSES = ("pending_scientific_review", "approved", "rejected")

_REGISTRY_KEYS = ("schema", "registry_id", "created", "description", "entries")
_ENTRY_KEYS = (
    "entry_id",
    "composition",
    "charge",
    "multiplicity",
    "status",
    "evidence",
    "reviewer",
    "decision",
)


@dataclass(frozen=True, kw_only=True)
class StateRegistryEntry:
    """One composition/charge/multiplicity state and its review status."""

    entry_id: str
    composition: str
    charge: int
    multiplicity: int
    status: str
    evidence: tuple[str, ...] = ()
    reviewer: str | None = None
    decision: str | None = None

    def __post_init__(self) -> None:
        multiplicity = require_int(self.multiplicity, "state_entry.multiplicity")
        if multiplicity < 1:
            raise ValidationError(
                f"state_entry.multiplicity must be at least 1; got {multiplicity}."
            )
        status = require_str(self.status, "state_entry.status")
        if status not in STATE_REGISTRY_STATUSES:
            raise ValidationError(
                f"state_entry.status must be one of {STATE_REGISTRY_STATUSES!r}; "
                f"got {status!r}."
            )
        evidence = tuple(
            require_str(value, f"state_entry.evidence[{index}]")
            for index, value in enumerate(
                require_sequence(self.evidence, "state_entry.evidence")
            )
        )
        reviewer = optional_str(self.reviewer, "state_entry.reviewer")
        decision = optional_str(self.decision, "state_entry.decision")
        if status == "approved" and (not evidence or reviewer is None or decision is None):
            raise ValidationError(
                "An approved state registry entry requires evidence, reviewer, and decision."
            )
        object.__setattr__(self, "entry_id", validate_record_id(self.entry_id))
        object.__setattr__(
            self,
            "composition",
            require_str(self.composition, "state_entry.composition"),
        )
        object.__setattr__(self, "charge", require_int(self.charge, "state_entry.charge"))
        object.__setattr__(self, "multiplicity", multiplicity)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "reviewer", reviewer)
        object.__setattr__(self, "decision", decision)

    @property
    def state_key(self) -> tuple[str, int, int]:
        return (self.composition, self.charge, self.multiplicity)

    def provenance(self, registry_id: str) -> str:
        return f"state_registry:{registry_id}:{self.entry_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "composition": self.composition,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "status": self.status,
            "evidence": list(self.evidence),
            "reviewer": self.reviewer,
            "decision": self.decision,
        }

    @classmethod
    def from_dict(cls, data: Any) -> StateRegistryEntry:
        mapping = require_mapping(data, "state_entry")
        reject_unknown_keys(mapping, _ENTRY_KEYS, "state_entry")
        for key in ("entry_id", "composition", "charge", "multiplicity", "status"):
            require_key(mapping, key, "state_entry")
        return cls(
            entry_id=mapping["entry_id"],
            composition=mapping["composition"],
            charge=mapping["charge"],
            multiplicity=mapping["multiplicity"],
            status=mapping["status"],
            evidence=tuple(require_sequence(mapping.get("evidence", ()), "state_entry.evidence")),
            reviewer=mapping.get("reviewer"),
            decision=mapping.get("decision"),
        )


@dataclass(frozen=True, kw_only=True)
class StateRegistry:
    """A versioned set of reviewed state choices."""

    schema: str = STATE_REGISTRY_SCHEMA
    registry_id: str
    created: str
    description: str
    entries: tuple[StateRegistryEntry, ...]

    def __post_init__(self) -> None:
        if self.schema != STATE_REGISTRY_SCHEMA:
            raise ValidationError(
                f"state_registry.schema must be {STATE_REGISTRY_SCHEMA!r}; got {self.schema!r}."
            )
        entries: list[StateRegistryEntry] = []
        ids: set[str] = set()
        keys: set[tuple[str, int, int]] = set()
        for index, value in enumerate(require_sequence(self.entries, "state_registry.entries")):
            if not isinstance(value, StateRegistryEntry):
                raise ValidationError(
                    f"state_registry.entries[{index}] must be StateRegistryEntry; "
                    f"got {type(value).__name__}."
                )
            if value.entry_id in ids:
                raise ValidationError(
                    f"state_registry entry_id {value.entry_id!r} appears more than once."
                )
            if value.state_key in keys:
                raise ValidationError(
                    f"state_registry state {value.state_key!r} appears more than once."
                )
            ids.add(value.entry_id)
            keys.add(value.state_key)
            entries.append(value)
        if not entries:
            raise ValidationError("state_registry.entries must not be empty.")
        object.__setattr__(self, "registry_id", validate_record_id(self.registry_id))
        object.__setattr__(self, "created", require_str(self.created, "state_registry.created"))
        object.__setattr__(
            self, "description", require_str(self.description, "state_registry.description")
        )
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda item: item.entry_id)))

    def entry_for(
        self, composition: str, charge: int, multiplicity: int
    ) -> StateRegistryEntry | None:
        key = (composition, charge, multiplicity)
        return next((entry for entry in self.entries if entry.state_key == key), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "registry_id": self.registry_id,
            "created": self.created,
            "description": self.description,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Any) -> StateRegistry:
        mapping = require_mapping(data, "state_registry")
        schema = mapping.get("schema")
        if schema != STATE_REGISTRY_SCHEMA:
            raise ValidationError(
                f"state_registry.schema must be {STATE_REGISTRY_SCHEMA!r}; got {schema!r}."
            )
        reject_unknown_keys(mapping, _REGISTRY_KEYS, "state_registry")
        for key in _REGISTRY_KEYS:
            require_key(mapping, key, "state_registry")
        return cls(
            schema=schema,
            registry_id=mapping["registry_id"],
            created=mapping["created"],
            description=mapping["description"],
            entries=tuple(
                StateRegistryEntry.from_dict(value)
                for value in require_sequence(mapping["entries"], "state_registry.entries")
            ),
        )
