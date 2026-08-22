"""Shared manifest and result helpers for the cross-code validation experiment."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

SCHEMA_VERSION = 1
RESULT_SCHEMA = "crosscode-result-v1"
BOHR_TO_ANGSTROM = 0.529177210903

PERIODIC_SYMBOLS = (
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
ATOMIC_NUMBERS = {symbol: number for number, symbol in enumerate(PERIODIC_SYMBOLS) if symbol}


@dataclass(frozen=True)
class Atom:
    symbol: str
    x: float
    y: float
    z: float

    def as_record(self) -> dict[str, Any]:
        return {"element": self.symbol, "xyz_angstrom": [self.x, self.y, self.z]}


@dataclass(frozen=True)
class Case:
    config_path: Path
    raw: dict[str, Any]
    structure_path: Path
    atoms: tuple[Atom, ...]

    @property
    def case_id(self) -> str:
        return str(self.raw["case_id"])

    @property
    def charge(self) -> int:
        return int(self.raw["charge"])

    @property
    def multiplicity(self) -> int:
        return int(self.raw["multiplicity"])

    @property
    def spin_2s(self) -> int:
        return multiplicity_to_pyscf_spin(self.multiplicity)

    @property
    def functional(self) -> str:
        return str(self.raw["method"]["functional"])

    @property
    def basis(self) -> str:
        return str(self.raw["method"]["basis"])

    @property
    def tolerances(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.raw["tolerances"].items()}


def multiplicity_to_pyscf_spin(multiplicity: int) -> int:
    """Convert 2S+1 multiplicity to PySCF's n_alpha-n_beta = 2S."""
    if not isinstance(multiplicity, int) or isinstance(multiplicity, bool):
        raise ValueError("Multiplicity must be an integer.")
    if multiplicity < 1:
        raise ValueError("Multiplicity must be at least 1.")
    return multiplicity - 1


def target_s2(spin_2s: int) -> float:
    total_spin = spin_2s / 2.0
    return total_spin * (total_spin + 1.0)


def canonical_symbol(raw: str) -> str:
    symbol = raw.strip().capitalize()
    if symbol not in ATOMIC_NUMBERS:
        raise ValueError(f"Unknown element symbol {raw!r}.")
    return symbol


def read_xyz(path: str | Path) -> tuple[Atom, ...]:
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file {source} is incomplete.")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"XYZ file {source} has an invalid atom count.") from exc
    coordinate_lines = [line for line in lines[2:] if line.strip()]
    if len(coordinate_lines) != atom_count:
        raise ValueError(
            f"XYZ file {source} declares {atom_count} atoms but contains "
            f"{len(coordinate_lines)} coordinate rows."
        )
    atoms: list[Atom] = []
    for index, line in enumerate(coordinate_lines, start=1):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"XYZ atom row {index} in {source} must have four fields.")
        try:
            atoms.append(
                Atom(canonical_symbol(fields[0]), *(float(value) for value in fields[1:]))
            )
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ atom row {index} in {source}: {line!r}.") from exc
    return tuple(atoms)


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {key!r} must be an object.")
    return value


def _require_positive_number(mapping: dict[str, Any], key: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Manifest field {key!r} must be a positive number.") from exc
    if value <= 0:
        raise ValueError(f"Manifest field {key!r} must be positive.")
    return value


def load_case(path: str | Path) -> Case:
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest {config_path} is not valid JSON: {exc}.") from exc
    if not isinstance(raw, dict):
        raise ValueError("Manifest root must be an object.")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Manifest schema_version must be {SCHEMA_VERSION}.")

    case_id = raw.get("case_id")
    if not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", case_id):
        raise ValueError("case_id must use lowercase letters, digits, underscores, or hyphens.")
    if raw.get("calculation") != "energy_gradient":
        raise ValueError("Only calculation='energy_gradient' is supported.")

    try:
        charge = int(raw["charge"])
        multiplicity = int(raw["multiplicity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("charge and multiplicity must be integers.") from exc
    if isinstance(raw.get("charge"), bool) or isinstance(raw.get("multiplicity"), bool):
        raise ValueError("charge and multiplicity must be integers, not booleans.")
    spin_2s = multiplicity_to_pyscf_spin(multiplicity)

    structure_value = raw.get("structure")
    if not isinstance(structure_value, str) or not structure_value.strip():
        raise ValueError("Manifest structure must be a non-empty relative path.")
    structure_rel = Path(structure_value)
    if structure_rel.is_absolute():
        raise ValueError("Manifest structure must be relative to the manifest directory.")
    structure_path = (config_path.parent / structure_rel).resolve()
    atoms = read_xyz(structure_path)

    electron_count = sum(ATOMIC_NUMBERS[atom.symbol] for atom in atoms) - charge
    if electron_count <= 0:
        raise ValueError("The configured molecule must contain at least one electron.")
    if spin_2s > electron_count or (electron_count - spin_2s) % 2:
        raise ValueError(
            f"Electron count {electron_count}, charge {charge}, and multiplicity "
            f"{multiplicity} are inconsistent. PySCF spin would be {spin_2s}."
        )

    method = _require_mapping(raw, "method")
    for key in ("functional", "basis"):
        if not isinstance(method.get(key), str) or not method[key].strip():
            raise ValueError(f"method.{key} must be a non-empty string.")

    scf = _require_mapping(raw, "scf")
    _require_positive_number(scf, "conv_tol")
    if int(scf.get("max_cycle", 0)) < 1:
        raise ValueError("scf.max_cycle must be a positive integer.")

    pyscf = _require_mapping(raw, "pyscf")
    verbose = int(pyscf.get("verbose", 3))
    if not 0 <= verbose <= 9:
        raise ValueError("pyscf.verbose must be an integer from 0 through 9.")
    for key in ("grid_level", "nlc_grid_level"):
        if int(pyscf.get(key, -1)) < 0:
            raise ValueError(f"pyscf.{key} must be a non-negative integer.")
    if not isinstance(pyscf.get("grid_response"), bool):
        raise ValueError("pyscf.grid_response must be true or false.")
    if not isinstance(pyscf.get("density_fit"), bool):
        raise ValueError("pyscf.density_fit must be true or false.")
    _require_positive_number(pyscf, "max_memory_mb")

    orca = _require_mapping(raw, "orca")
    version = orca.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("orca.version must use a numeric major.minor.patch form.")
    if int(orca.get("nprocs", 0)) < 1:
        raise ValueError("orca.nprocs must be a positive integer.")
    if int(orca.get("maxcore_mb_per_process", 0)) < 1:
        raise ValueError("orca.maxcore_mb_per_process must be a positive integer.")
    keywords = orca.get("keywords")
    if not isinstance(keywords, list) or not keywords or not all(
        isinstance(keyword, str) and keyword.strip() for keyword in keywords
    ):
        raise ValueError("orca.keywords must be a non-empty list of strings.")

    tolerances = _require_mapping(raw, "tolerances")
    for key in (
        "energy_abs_hartree",
        "gradient_rms_hartree_per_bohr",
        "gradient_max_hartree_per_bohr",
    ):
        _require_positive_number(tolerances, key)

    return Case(config_path=config_path, raw=raw, structure_path=structure_path, atoms=atoms)


def input_fingerprint(case: Case) -> str:
    scientific_input = {
        key: value
        for key, value in case.raw.items()
        if key not in {"tolerances", "tolerance_status"}
    }
    digest = sha256()
    digest.update(json.dumps(scientific_input, sort_keys=True, separators=(",", ":")).encode())
    digest.update(case.structure_path.read_bytes())
    return digest.hexdigest()


def case_record(case: Case) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "input_fingerprint_sha256": input_fingerprint(case),
        "structure_manifest_path": str(case.raw["structure"]),
        "atoms": [atom.as_record() for atom in case.atoms],
        "charge": case.charge,
        "multiplicity": case.multiplicity,
        "pyscf_spin_2s": case.spin_2s,
        "electron_count": sum(ATOMIC_NUMBERS[atom.symbol] for atom in case.atoms)
        - case.charge,
        "functional": case.functional,
        "basis": case.basis,
        "calculation": "energy_gradient",
    }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_result(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"{source} is not a {RESULT_SCHEMA} result.")
    return data
