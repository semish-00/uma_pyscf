"""Import Part I ``crosscode-result-v1`` results as canonical label records.

The cross-code validation experiment (ORCA vs PySCF/GPU4PySCF) normalized every
calculation into a ``crosscode-result-v1`` JSON object. Those results are the
first real input the canonical schema has to read, so this module maps them
across. It reads the foreign format as data only -- nothing here imports from
``validation/``, which stays frozen -- and it invents nothing: a result that
does not state a method setting is refused rather than filled in with a
plausible default, because a label whose grid is guessed cannot be reproduced.

An ORCA result is therefore importable only if it carries the PySCF-side method
settings, which in practice it does not: ORCA results are Part I comparison
material, and the teaching engine is GPU4PySCF.

Unlike :meth:`LabelRecord.from_dict`, this reader does not reject keys it does
not know. The comparison tolerances, the tolerance status, and the ORCA source
file names belong to the other project's format, not to this schema, and the
record they produce is validated in full anyway.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.elements import atomic_number
from ..core.errors import ValidationError
from ..core.spin import multiplicity_to_spin_2s
from ._fields import (
    optional_finite_float,
    require_bool,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
)
from .label_record import (
    LABEL_RECORD_SCHEMA,
    ElectronicState,
    Engine,
    LabelRecord,
    Method,
    QcState,
    RawArtifact,
    Results,
    Structure,
)

__all__ = ["CROSSCODE_RESULT_SCHEMA", "IMPORT_EVENT", "label_record_from_crosscode_result"]

CROSSCODE_RESULT_SCHEMA = "crosscode-result-v1"
IMPORT_EVENT = "imported_from_crosscode_result_v1"


def _missing_method_settings(path: str) -> ValidationError:
    """Return the error for a result that cannot fill the canonical method block."""
    return ValidationError(
        f"The result lacks the method settings block: {path} is missing. A "
        f"{CROSSCODE_RESULT_SCHEMA} result written by an ORCA run carries the ORCA "
        "keywords instead of the PySCF grid, density-fitting, and SCF settings, so it "
        f"cannot become a {LABEL_RECORD_SCHEMA} record without inventing method "
        "settings that were never used."
    )


def _require_setting(settings: Mapping[str, Any], key: str, path: str) -> Any:
    """Return ``settings[key]``, or raise the missing-method-settings error."""
    if key not in settings:
        raise _missing_method_settings(f"{path}.{key}")
    return settings[key]


def _structure_from_case(case: Mapping[str, Any]) -> Structure:
    """Build the structure block from the case's element/coordinate list."""
    atoms = require_sequence(require_key(case, "atoms", "case"), "case.atoms")
    if not atoms:
        raise ValidationError("case.atoms must list at least one atom.")
    numbers: list[int] = []
    positions: list[Any] = []
    for index, atom in enumerate(atoms):
        path = f"case.atoms[{index}]"
        mapping = require_mapping(atom, path)
        symbol = require_str(require_key(mapping, "element", path), f"{path}.element")
        numbers.append(atomic_number(symbol))
        positions.append(require_key(mapping, "xyz_angstrom", path))
    return Structure(atomic_numbers=tuple(numbers), positions_angstrom=tuple(positions))


def _state_from_case(case: Mapping[str, Any]) -> ElectronicState:
    """Build the electronic state, checking the result's derived spin against it."""
    charge = require_int(require_key(case, "charge", "case"), "case.charge")
    multiplicity = require_int(require_key(case, "multiplicity", "case"), "case.multiplicity")
    spin_2s = multiplicity_to_spin_2s(multiplicity)
    stored = case.get("pyscf_spin_2s")
    if stored is not None:
        declared = require_int(stored, "case.pyscf_spin_2s")
        if declared != spin_2s:
            raise ValidationError(
                f"case.pyscf_spin_2s is {declared} but multiplicity {multiplicity} "
                f"derives {spin_2s}. The multiplicity is the source of truth, so a "
                "result that disagrees with it is not imported."
            )
    return ElectronicState(charge=charge, multiplicity=multiplicity, spin_2s=spin_2s)


def _method_from_result(result: Mapping[str, Any], case: Mapping[str, Any]) -> Method:
    """Build the method block, refusing to guess any setting the result omits."""
    functional = require_str(require_key(case, "functional", "case"), "case.functional")
    basis = require_str(require_key(case, "basis", "case"), "case.basis")
    settings_value = result.get("settings")
    if not isinstance(settings_value, Mapping):
        raise _missing_method_settings("settings")
    settings = require_mapping(settings_value, "settings")
    scf_value = _require_setting(settings, "scf", "settings")
    if not isinstance(scf_value, Mapping):
        raise _missing_method_settings("settings.scf")
    scf = require_mapping(scf_value, "settings.scf")
    return Method(
        functional=functional,
        basis=basis,
        ecp=None,
        aux_basis=None,
        grid_level=_require_setting(settings, "grid_level", "settings"),
        nlc_grid_level=_require_setting(settings, "nlc_grid_level", "settings"),
        grid_response=_require_setting(settings, "grid_response", "settings"),
        density_fit=_require_setting(settings, "density_fit", "settings"),
        scf_conv_tol=_require_setting(scf, "conv_tol", "settings.scf"),
        scf_max_cycle=_require_setting(scf, "max_cycle", "settings.scf"),
    )


def _engine_from_result(result: Mapping[str, Any]) -> Engine:
    """Build the engine block, flattening runtime provenance values to strings."""
    name = require_str(require_key(result, "engine", "result"), "result.engine")
    runtime = require_mapping(result.get("engine_runtime") or {}, "result.engine_runtime")
    versions: dict[str, str | None] = {}
    for key, value in runtime.items():
        if value is None or isinstance(value, str):
            versions[key] = value
        elif isinstance(value, bool | int | float):
            versions[key] = str(value)
        else:
            raise ValidationError(
                f"result.engine_runtime.{key} must be a string, a number, or null; got "
                f"{type(value).__name__}."
            )
    return Engine(name=name, versions=versions)


def _results_from_result(result: Mapping[str, Any]) -> Results:
    """Build the results block, refusing anything that did not converge."""
    converged = require_bool(require_key(result, "converged", "result"), "result.converged")
    if not converged:
        raise ValidationError(
            "result.converged is false. An unconverged calculation is not a label, so "
            "it is not imported; rerun it or record the failure in the QC ledger."
        )
    return Results(
        energy_hartree=require_key(result, "energy_hartree", "result"),
        gradient_hartree_per_bohr=require_key(result, "gradient_hartree_per_bohr", "result"),
        converged=True,
        n_iterations=result.get("n_iterations"),
        s2=optional_finite_float(result.get("s2"), "result.s2"),
        s2_target=optional_finite_float(result.get("s2_target"), "result.s2_target"),
        s2_deviation=optional_finite_float(result.get("s2_deviation"), "result.s2_deviation"),
        wall_time_seconds=result.get("wall_time_seconds"),
        scf_wall_time_seconds=result.get("scf_wall_time_seconds"),
        gradient_wall_time_seconds=result.get("gradient_wall_time_seconds"),
    )


def label_record_from_crosscode_result(
    result: dict[str, Any],
    *,
    record_id: str | None = None,
    raw_location: str | None = None,
) -> LabelRecord:
    """Convert one ``crosscode-result-v1`` object into a canonical label record.

    ``record_id`` defaults to the result's ``case.case_id``. ``raw_location`` is
    the logical location of the raw output this result was normalized from; no
    checksum is set, because the importer sees the normalized JSON and not the
    raw file it came from. The record starts QC life as ``pending`` with a
    single history entry that carries the case input fingerprint, which is what
    ties the record back to the exact scientific input Part I hashed.
    """
    mapping = require_mapping(result, "result")
    schema = mapping.get("schema")
    if schema != CROSSCODE_RESULT_SCHEMA:
        raise ValidationError(
            f"result.schema must be {CROSSCODE_RESULT_SCHEMA!r}; got {schema!r}."
        )
    case = require_mapping(require_key(mapping, "case", "result"), "case")
    fingerprint = require_str(
        require_key(case, "input_fingerprint_sha256", "case"),
        "case.input_fingerprint_sha256",
    )
    identifier = record_id if record_id is not None else require_key(case, "case_id", "case")
    created = mapping.get("created_utc")
    return LabelRecord(
        record_id=identifier,
        structure=_structure_from_case(case),
        state=_state_from_case(case),
        method=_method_from_result(mapping, case),
        engine=_engine_from_result(mapping),
        results=_results_from_result(mapping),
        raw=RawArtifact(logical_location=raw_location, checksum_sha256=None),
        qc=QcState(
            status="pending",
            history=(
                {
                    "utc": created if isinstance(created, str) and created.strip() else "unknown",
                    "event": IMPORT_EVENT,
                    "input_fingerprint_sha256": fingerprint,
                },
            ),
        ),
    )
