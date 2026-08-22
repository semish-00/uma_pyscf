"""Atomic file publication and JSON reading.

Everything this package writes is serialized in full before anything is
published, then moved into place with a single rename. An interrupted or
failing run therefore leaves either the previous file or no file at all, never
a truncated one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import ValidationError

__all__ = ["read_json", "write_json_atomic", "write_text_atomic"]


def write_text_atomic(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` through a scratch file in the same directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        scratch.write_text(text, encoding="utf-8")
        os.replace(scratch, destination)
    finally:
        scratch.unlink(missing_ok=True)


def write_json_atomic(path: str | Path, data: Any) -> None:
    """Serialize ``data`` first, then publish it atomically at ``path``.

    Serialization happens before the destination directory is touched, so data
    that cannot be represented as JSON fails without creating or damaging any
    file. Keys are sorted and the output is indented, which keeps written
    records diffable and their byte content independent of insertion order.
    """
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    write_text_atomic(path, text)


def read_json(path: str | Path) -> Any:
    """Read JSON from ``path``, naming the file if it cannot be parsed."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{source} is not valid JSON: {exc}.") from exc
