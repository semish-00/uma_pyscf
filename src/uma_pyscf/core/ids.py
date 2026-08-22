"""Record identifiers and deterministic content fingerprints.

Identifiers are lowercase and filesystem safe so a record id can be a directory
name. Fingerprints are sha256 over canonical JSON, optionally extended with raw
file bytes, which is the composition the Part I validation experiment uses for
its input fingerprint (canonical scientific input, then the structure file).
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from .errors import ValidationError

__all__ = [
    "CASE_ID_PATTERN",
    "canonical_json_fingerprint",
    "sha256_of_file",
    "validate_record_id",
]

CASE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_FILE_CHUNK_BYTES = 1 << 20


def validate_record_id(value: str) -> str:
    """Return ``value`` unchanged if it is a valid record id, else raise."""
    if not isinstance(value, str) or not CASE_ID_PATTERN.fullmatch(value):
        raise ValidationError(
            f"Record id {value!r} must start with a lowercase letter or digit and contain "
            "only lowercase letters, digits, underscores, and hyphens."
        )
    return value


def canonical_json_fingerprint(data: Any, extra_bytes: bytes | None = None) -> str:
    """Return the sha256 hex digest of ``data`` in canonical JSON form.

    ``data`` is serialized with sorted keys and no insignificant whitespace, so
    the digest depends on content only and not on key order or formatting. When
    ``extra_bytes`` is given it is hashed after the JSON, which is how a
    structure file is bound to the manifest that references it.
    """
    try:
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise ValidationError(f"Fingerprint input is not JSON serializable: {exc}.") from exc
    digest = sha256()
    digest.update(canonical.encode("utf-8"))
    if extra_bytes is not None:
        digest.update(extra_bytes)
    return digest.hexdigest()


def sha256_of_file(path: str | Path) -> str:
    """Return the sha256 hex digest of a file, read in chunks."""
    source = Path(path)
    digest = sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(_FILE_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
