"""Reproducible UMA fine-tuning on GPU4PySCF labels.

The package is deliberately dependency free at its core: ``uma_pyscf.core``
holds the cross-cutting units, spin, identifier, and atomic I/O primitives that
every later module builds on, and heavier scientific dependencies enter only in
the modules that need them.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("uma-pyscf")
except PackageNotFoundError:  # Source tree without an installed distribution.
    __version__ = "0.0.0"

__all__ = ["__version__"]
