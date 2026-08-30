"""Small typed boundary shared by label runners and calculator adapters."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.errors import UmaPyscfError, ValidationError
from ..schemas._fields import require_mapping, require_str, validated_json_object
from ..schemas.candidate import CandidateRecord
from ..schemas.label_record import Method, Results

__all__ = [
    "CalculationFailure",
    "CalculationOutput",
    "CalculatorAdapter",
]


class CalculationFailure(UmaPyscfError):
    """A categorized calculation failure that the retry policy may act on."""

    def __init__(self, category: str, message: str) -> None:
        self.category = require_str(category, "calculation_failure.category")
        super().__init__(require_str(message, "calculation_failure.message"))


@dataclass(frozen=True, kw_only=True)
class CalculationOutput:
    """Engine-neutral result returned to the batch runner by one attempt."""

    engine_name: str
    engine_versions: dict[str, str]
    results: Results
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        versions = require_mapping(self.engine_versions, "calculation.engine_versions")
        checked: dict[str, str] = {}
        for key, value in versions.items():
            checked[require_str(key, "calculation.engine_versions key")] = require_str(
                value, f"calculation.engine_versions.{key}"
            )
        if not isinstance(self.results, Results):
            raise ValidationError(
                f"calculation.results must be Results; got {type(self.results).__name__}."
            )
        object.__setattr__(
            self, "engine_name", require_str(self.engine_name, "calculation.engine_name")
        )
        object.__setattr__(self, "engine_versions", checked)
        object.__setattr__(
            self,
            "raw_payload",
            validated_json_object(self.raw_payload, "calculation.raw_payload"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe worker envelope."""
        return {
            "engine_name": self.engine_name,
            "engine_versions": dict(self.engine_versions),
            "results": self.results.to_dict(),
            "raw_payload": deepcopy(self.raw_payload),
        }

    @classmethod
    def from_dict(cls, data: Any) -> CalculationOutput:
        """Restore a worker envelope and revalidate every block."""
        mapping = require_mapping(data, "calculation")
        allowed = {"engine_name", "engine_versions", "results", "raw_payload"}
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ValidationError(f"calculation has unknown keys {unknown!r}.")
        return cls(
            engine_name=require_str(mapping.get("engine_name"), "calculation.engine_name"),
            engine_versions=require_mapping(
                mapping.get("engine_versions"), "calculation.engine_versions"
            ),
            results=Results.from_dict(mapping.get("results")),
            raw_payload=validated_json_object(
                mapping.get("raw_payload") or {}, "calculation.raw_payload"
            ),
        )


class CalculatorAdapter(Protocol):
    """The injectable scientific boundary used by the batch runner."""

    def calculate(
        self,
        candidate: CandidateRecord,
        method: Method,
        config: Mapping[str, Any],
        *,
        attempt_id: str,
        resource: Mapping[str, Any],
    ) -> CalculationOutput:
        """Calculate one candidate or raise :class:`CalculationFailure`."""
