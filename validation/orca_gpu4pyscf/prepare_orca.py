#!/usr/bin/env python3
"""Generate a deterministic ORCA input from a cross-code case manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from common import Case, load_case


def orca_functional_name(functional: str) -> str:
    normalized = functional.strip().lower().replace("_", "-")
    supported = {"wb97m-v": "WB97M-V"}
    try:
        return supported[normalized]
    except KeyError:
        raise ValueError(
            f"No reviewed ORCA keyword mapping for functional {functional!r}."
        ) from None


def orca_basis_name(basis: str) -> str:
    normalized = basis.strip().lower()
    supported = {"def2-tzvpd": "def2-TZVPD"}
    try:
        return supported[normalized]
    except KeyError:
        raise ValueError(f"No reviewed ORCA keyword mapping for basis {basis!r}.") from None


def render_orca_input(case: Case) -> str:
    settings = case.raw["orca"]
    keywords = [str(keyword) for keyword in settings["keywords"]]
    simple_keywords = [
        orca_functional_name(case.functional),
        orca_basis_name(case.basis),
        "EnGrad",
        *keywords,
    ]
    lines = [
        f"# Generated from {case.config_path.name}; case_id={case.case_id}",
        f"! {' '.join(simple_keywords)}",
        "",
        f"%maxcore {int(settings['maxcore_mb_per_process'])}",
        "%pal",
        f"  nprocs {int(settings['nprocs'])}",
        "end",
        "",
        f"%scf",
        f"  MaxIter {int(case.raw['scf']['max_cycle'])}",
        "end",
        "",
        f"* xyz {case.charge} {case.multiplicity}",
    ]
    for atom in case.atoms:
        lines.append(f"  {atom.symbol:<2s} {atom.x: .12f} {atom.y: .12f} {atom.z: .12f}")
    lines.extend(("*", ""))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Cross-code case manifest JSON.")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", help="Destination .inp file.")
    output.add_argument("--stdout", action="store_true", help="Print input without writing.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    text = render_orca_input(load_case(args.config))
    if args.stdout:
        sys.stdout.write(text)
        return 0
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

