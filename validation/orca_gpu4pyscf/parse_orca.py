#!/usr/bin/env python3
"""Parse ORCA .engrad into the normalized cross-code result schema."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from common import (
    BOHR_TO_ANGSTROM,
    RESULT_SCHEMA,
    case_record,
    load_case,
    target_s2,
    write_json,
)

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")


def _numbers_between(
    lines: list[str],
    start_marker: str,
    end_marker: str | None,
) -> list[float]:
    start = next((i for i, line in enumerate(lines) if start_marker in line), None)
    if start is None:
        raise ValueError(f"ORCA engrad is missing marker {start_marker!r}.")
    if end_marker is None:
        end = len(lines)
    else:
        end = next(
            (i for i in range(start + 1, len(lines)) if end_marker in lines[i]),
            None,
        )
        if end is None:
            raise ValueError(f"ORCA engrad is missing marker {end_marker!r}.")
    values: list[float] = []
    for line in lines[start + 1 : end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in FLOAT_RE.findall(stripped):
            values.append(float(token.replace("D", "E").replace("d", "e")))
    return values


def parse_engrad(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    natom_values = _numbers_between(
        lines,
        "Number of atoms",
        "The current total energy",
    )
    if len(natom_values) != 1:
        raise ValueError("Expected exactly one atom count in ORCA engrad.")
    atom_count = int(natom_values[0])

    energies = _numbers_between(
        lines,
        "The current total energy",
        "The current gradient",
    )
    if len(energies) != 1:
        raise ValueError("Expected exactly one total energy in ORCA engrad.")
    gradient_flat = _numbers_between(
        lines,
        "The current gradient",
        "The atomic numbers and current coordinates",
    )
    if len(gradient_flat) != atom_count * 3:
        raise ValueError(
            f"Expected {atom_count * 3} gradient components, found {len(gradient_flat)}."
        )
    coordinates_flat = _numbers_between(
        lines,
        "The atomic numbers and current coordinates",
        None,
    )
    if len(coordinates_flat) != atom_count * 4:
        raise ValueError(
            f"Expected {atom_count * 4} atomic-number/coordinate values, "
            f"found {len(coordinates_flat)}."
        )

    gradient = [gradient_flat[index : index + 3] for index in range(0, len(gradient_flat), 3)]
    coordinates: list[dict[str, Any]] = []
    for index in range(0, len(coordinates_flat), 4):
        atomic_number, x, y, z = coordinates_flat[index : index + 4]
        coordinates.append(
            {
                "atomic_number": int(atomic_number),
                "xyz_bohr": [x, y, z],
                "xyz_angstrom": [
                    x * BOHR_TO_ANGSTROM,
                    y * BOHR_TO_ANGSTROM,
                    z * BOHR_TO_ANGSTROM,
                ],
            }
        )
    return {
        "atom_count": atom_count,
        "energy_hartree": energies[0],
        "gradient_hartree_per_bohr": gradient,
        "coordinates": coordinates,
    }


def parse_orca_output(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "normal_termination": None,
            "orca_version": None,
            "s2": None,
            "raw_output_supplied": False,
        }
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    version_match = re.search(r"Program Version\s+([0-9][^\s]*)", text)
    s2_matches = re.findall(
        r"(?:Expectation value of\s+)?<S\*\*2>\s*:\s*(%s)" % FLOAT_RE.pattern,
        text,
    )
    return {
        "normal_termination": "ORCA TERMINATED NORMALLY" in text,
        "orca_version": version_match.group(1) if version_match else None,
        "s2": float(s2_matches[-1].replace("D", "E")) if s2_matches else None,
        "raw_output_supplied": True,
    }


def normalized_result(config_path: str | Path, engrad_path: str | Path, output_path: str | Path | None) -> dict[str, Any]:
    case = load_case(config_path)
    parsed = parse_engrad(engrad_path)
    output_info = parse_orca_output(output_path)
    if parsed["atom_count"] != len(case.atoms):
        raise ValueError(
            f"ORCA engrad contains {parsed['atom_count']} atoms, manifest has {len(case.atoms)}."
        )

    max_coordinate_error = 0.0
    for expected, actual in zip(case.atoms, parsed["coordinates"]):
        expected_xyz = (expected.x, expected.y, expected.z)
        max_coordinate_error = max(
            max_coordinate_error,
            *(abs(left - right) for left, right in zip(expected_xyz, actual["xyz_angstrom"])),
        )
    if max_coordinate_error > 1e-7:
        raise ValueError(
            "ORCA engrad coordinates do not match the manifest geometry: "
            f"maximum error is {max_coordinate_error:.3e} angstrom."
        )
    if output_info["raw_output_supplied"] and not output_info["normal_termination"]:
        raise ValueError("The supplied ORCA output does not report normal termination.")
    if output_info["raw_output_supplied"]:
        expected_version = str(case.raw["orca"]["version"])
        actual_version = output_info["orca_version"]
        if actual_version is None:
            raise ValueError("The supplied ORCA output does not report its program version.")
        if actual_version != expected_version:
            raise ValueError(
                f"ORCA version mismatch: manifest requires {expected_version}, "
                f"output reports {actual_version}."
            )

    s2 = output_info["s2"]
    return {
        "schema": RESULT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "orca",
        "engine_runtime": {"orca": output_info["orca_version"]},
        "case": case_record(case),
        "settings": {
            **case.raw["orca"],
            "scf": case.raw["scf"],
            "coordinates_verified_max_abs_angstrom": max_coordinate_error,
        },
        "converged": output_info["normal_termination"],
        "energy_hartree": parsed["energy_hartree"],
        "gradient_hartree_per_bohr": parsed["gradient_hartree_per_bohr"],
        "s2": s2,
        "s2_target": target_s2(case.spin_2s),
        "s2_deviation": None if s2 is None else s2 - target_s2(case.spin_2s),
        "tolerances": case.tolerances,
        "tolerance_status": case.raw.get("tolerance_status"),
        "source_engrad": str(Path(engrad_path).name),
        "source_output": None if output_path is None else str(Path(output_path).name),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Cross-code case manifest JSON.")
    parser.add_argument("engrad", help="ORCA .engrad file.")
    parser.add_argument("--orca-output", help="ORCA stdout file for termination/version/S2.")
    parser.add_argument("--output", required=True, help="Normalized result JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = normalized_result(args.config, args.engrad, args.orca_output)
    write_json(args.output, result)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
