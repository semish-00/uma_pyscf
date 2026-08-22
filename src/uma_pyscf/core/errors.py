"""Exception hierarchy used by the fail-closed checks throughout the package.

Every deliberate refusal raises one of these, so a caller can distinguish an
input this project rejected from an unexpected interpreter or OS failure.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "ProvenanceError",
    "UmaPyscfError",
    "ValidationError",
]


class UmaPyscfError(Exception):
    """Base class for every error this package raises on purpose."""


class ValidationError(UmaPyscfError):
    """An input violates a structural or scientific invariant.

    Charge/spin parity, unit expectations, identifier syntax, and required
    fields all fail closed through this error.
    """


class ConfigError(UmaPyscfError):
    """A configuration file is missing, unreadable, or internally inconsistent."""


class ProvenanceError(UmaPyscfError):
    """Provenance for a record is incomplete, contradictory, or unverifiable."""
