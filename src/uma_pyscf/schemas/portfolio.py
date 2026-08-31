"""Versioned receipt for score-independent candidate-portfolio assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint, validate_record_id
from ._fields import (
    reject_unknown_keys,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
    validated_json_object,
)

__all__ = [
    "PORTFOLIO_REPORT_SCHEMA",
    "PORTFOLIO_SKIP_REASONS",
    "PortfolioReport",
    "PortfolioSourceSummary",
]

PORTFOLIO_REPORT_SCHEMA = "uma-pyscf-candidate-portfolio-report-v1"
PORTFOLIO_SKIP_REASONS = (
    "duplicate_geometry_state",
    "parent_limit",
    "trajectory_limit",
    "quota_reached",
)

_HEX_DIGITS = frozenset("0123456789abcdef")
_SOURCE_KEYS = (
    "category",
    "source_id",
    "source_sha256",
    "quota",
    "available_count",
    "selected_record_ids",
    "skipped_counts",
)
_REPORT_KEYS = (
    "schema",
    "portfolio_id",
    "config_sha256",
    "config",
    "counts",
    "sources",
)


def _require_sha256(value: object, path: str) -> str:
    digest = require_str(value, path).lower()
    if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
        raise ValidationError(f"{path} must be 64 hexadecimal characters; got {value!r}.")
    return digest


def _require_counts(value: object, path: str, allowed: tuple[str, ...]) -> dict[str, int]:
    mapping = require_mapping(value, path)
    reject_unknown_keys(mapping, allowed, path)
    counts: dict[str, int] = {}
    for key in allowed:
        count = require_int(require_key(mapping, key, path), f"{path}.{key}")
        if count < 0:
            raise ValidationError(f"{path}.{key} must not be negative; got {count}.")
        counts[key] = count
    return counts


@dataclass(frozen=True, kw_only=True)
class PortfolioSourceSummary:
    """Selection outcome for one immutable source candidate manifest."""

    category: str
    source_id: str
    source_sha256: str
    quota: int
    available_count: int
    selected_record_ids: tuple[str, ...]
    skipped_counts: dict[str, int]

    def __post_init__(self) -> None:
        quota = require_int(self.quota, "portfolio_source.quota")
        available = require_int(self.available_count, "portfolio_source.available_count")
        if quota < 1:
            raise ValidationError("portfolio_source.quota must be positive.")
        if available < quota:
            raise ValidationError(
                "portfolio_source.available_count must be at least its quota; "
                f"got {available} < {quota}."
            )
        selected = tuple(
            validate_record_id(require_str(item, "portfolio_source.selected_record_ids"))
            for item in require_sequence(
                self.selected_record_ids, "portfolio_source.selected_record_ids"
            )
        )
        if len(selected) != quota:
            raise ValidationError(
                "portfolio_source.selected_record_ids must exactly satisfy quota; "
                f"got {len(selected)} != {quota}."
            )
        if len(selected) != len(set(selected)):
            raise ValidationError("portfolio_source.selected_record_ids must be unique.")
        skipped = _require_counts(
            self.skipped_counts, "portfolio_source.skipped_counts", PORTFOLIO_SKIP_REASONS
        )
        accounted = len(selected) + sum(skipped.values())
        if accounted != available:
            raise ValidationError(
                "portfolio_source selected and skipped counts must account for every "
                f"available record; got {accounted} != {available}."
            )
        object.__setattr__(self, "category", validate_record_id(self.category))
        object.__setattr__(self, "source_id", validate_record_id(self.source_id))
        object.__setattr__(
            self,
            "source_sha256",
            _require_sha256(self.source_sha256, "portfolio_source.source_sha256"),
        )
        object.__setattr__(self, "quota", quota)
        object.__setattr__(self, "available_count", available)
        object.__setattr__(self, "selected_record_ids", selected)
        object.__setattr__(self, "skipped_counts", skipped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "quota": self.quota,
            "available_count": self.available_count,
            "selected_record_ids": list(self.selected_record_ids),
            "skipped_counts": dict(self.skipped_counts),
        }

    @classmethod
    def from_dict(cls, data: Any) -> PortfolioSourceSummary:
        mapping = require_mapping(data, "portfolio_source")
        reject_unknown_keys(mapping, _SOURCE_KEYS, "portfolio_source")
        return cls(
            category=require_key(mapping, "category", "portfolio_source"),
            source_id=require_key(mapping, "source_id", "portfolio_source"),
            source_sha256=require_key(mapping, "source_sha256", "portfolio_source"),
            quota=require_key(mapping, "quota", "portfolio_source"),
            available_count=require_key(mapping, "available_count", "portfolio_source"),
            selected_record_ids=tuple(
                require_sequence(
                    require_key(mapping, "selected_record_ids", "portfolio_source"),
                    "portfolio_source.selected_record_ids",
                )
            ),
            skipped_counts=require_key(mapping, "skipped_counts", "portfolio_source"),
        )


@dataclass(frozen=True, kw_only=True)
class PortfolioReport:
    """Auditable receipt for a blind, quota-controlled portfolio selection."""

    schema: str = PORTFOLIO_REPORT_SCHEMA
    portfolio_id: str
    config_sha256: str
    config: dict[str, Any]
    counts: dict[str, int]
    sources: tuple[PortfolioSourceSummary, ...]

    def __post_init__(self) -> None:
        if self.schema != PORTFOLIO_REPORT_SCHEMA:
            raise ValidationError(
                f"portfolio.schema must be {PORTFOLIO_REPORT_SCHEMA!r}; got {self.schema!r}."
            )
        config = validated_json_object(self.config, "portfolio.config")
        digest = _require_sha256(self.config_sha256, "portfolio.config_sha256")
        expected = canonical_json_fingerprint(config)
        if digest != expected:
            raise ValidationError(
                f"portfolio.config_sha256 is {digest} but config fingerprints to {expected}."
            )
        counts = _require_counts(
            self.counts, "portfolio.counts", ("source_manifests", "available", "selected")
        )
        sources: list[PortfolioSourceSummary] = []
        categories: set[str] = set()
        selected_ids: set[str] = set()
        for index, item in enumerate(require_sequence(self.sources, "portfolio.sources")):
            if not isinstance(item, PortfolioSourceSummary):
                raise ValidationError(
                    f"portfolio.sources[{index}] must be a PortfolioSourceSummary."
                )
            if item.category in categories:
                raise ValidationError(
                    f"portfolio.sources[{index}] repeats category {item.category!r}."
                )
            overlap = selected_ids.intersection(item.selected_record_ids)
            if overlap:
                raise ValidationError(
                    f"portfolio.sources[{index}] repeats selected record {min(overlap)!r}."
                )
            categories.add(item.category)
            selected_ids.update(item.selected_record_ids)
            sources.append(item)
        if counts["source_manifests"] != len(sources):
            raise ValidationError("portfolio.counts.source_manifests does not match sources.")
        if counts["available"] != sum(item.available_count for item in sources):
            raise ValidationError("portfolio.counts.available does not match sources.")
        if counts["selected"] != len(selected_ids):
            raise ValidationError("portfolio.counts.selected does not match selected records.")
        object.__setattr__(self, "portfolio_id", validate_record_id(self.portfolio_id))
        object.__setattr__(self, "config_sha256", digest)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "sources", tuple(sources))

    @property
    def selected_record_ids(self) -> tuple[str, ...]:
        return tuple(
            record_id for source in self.sources for record_id in source.selected_record_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "portfolio_id": self.portfolio_id,
            "config_sha256": self.config_sha256,
            "config": dict(self.config),
            "counts": dict(self.counts),
            "sources": [source.to_dict() for source in self.sources],
        }

    @classmethod
    def from_dict(cls, data: Any) -> PortfolioReport:
        mapping = require_mapping(data, "portfolio")
        reject_unknown_keys(mapping, _REPORT_KEYS, "portfolio")
        return cls(
            schema=require_key(mapping, "schema", "portfolio"),
            portfolio_id=require_key(mapping, "portfolio_id", "portfolio"),
            config_sha256=require_key(mapping, "config_sha256", "portfolio"),
            config=require_key(mapping, "config", "portfolio"),
            counts=require_key(mapping, "counts", "portfolio"),
            sources=tuple(
                PortfolioSourceSummary.from_dict(item)
                for item in require_sequence(
                    require_key(mapping, "sources", "portfolio"), "portfolio.sources"
                )
            ),
        )
