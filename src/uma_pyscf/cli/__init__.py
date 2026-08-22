"""Command line entry points.

Argument parsing and configuration loading only. Scientific logic lives in the
module a subcommand delegates to. The entry point itself is
``uma_pyscf.cli.main:main``; this package intentionally imports nothing at
import time so ``python -m uma_pyscf.cli.main`` behaves like the installed
``uma-pyscf`` console script.
"""

from __future__ import annotations
