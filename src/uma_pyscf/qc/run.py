"""Apply a QC config to a batch of label records and write down what happened.

This is the gate the implementation plan puts between a computed label and a
dataset: nothing enters a dataset without a verdict here, and a verdict is only
as good as its reasons, so every check that was run -- passing and failing alike
-- is recorded with what it observed and what it was measured against.

Three rules govern what :func:`apply_qc` will and will not do.

* **It judges pending records only.** A record whose ``qc.status`` is already
  ``accepted`` or ``rejected`` stops the run by name. Re-running QC over a
  finished batch is a real need, but silently overwriting an earlier verdict is
  not the way to serve it: the thresholds may have changed, and the two verdicts
  would be indistinguishable in the record. A v1 run therefore refuses, and
  requalification will arrive as an explicit, reviewed mode.
* **It edits nothing but the verdict.** The updated record is the input record
  with a new ``qc`` block -- the status, and the history extended by exactly one
  entry. Energies, gradients, method, provenance are copied through untouched;
  QC is not a place where a label is corrected.
* **It fails closed.** A duplicate record id, a record that is not a record, a
  missing threshold, an unknown config key: each raises rather than producing a
  report that looks complete. A rejection, by contrast, is a normal outcome and
  never raises.

The history entry names the config by both its ``qc_id`` and the fingerprint of
its content, so "which thresholds rejected this record" is answerable from the
record alone, and answerable even if the file is later edited -- the digest
would no longer match, which is the point.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

from ..core.elements import PERIODIC_SYMBOLS
from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint
from ..schemas._fields import require_str
from ..schemas.label_record import LabelRecord, QcState
from ..schemas.qc_report import QcReport
from .config import validate_qc_config
from .electronic import electronic_checks, gradient_max_abs
from .geometry import duplicate_map, geometry_checks
from .protocol import protocol_checks

__all__ = ["PENDING_STATUS", "QC_EVENT", "apply_qc", "composition_formula"]

#: The status a record must carry to be judged, and the event a verdict appends.
PENDING_STATUS = "pending"
QC_EVENT = "qc_evaluated"


def composition_formula(atomic_numbers: Sequence[int]) -> str:
    """Return the canonical formula of a set of atoms, e.g. ``H4Si``.

    Element symbols are sorted alphabetically and a count of one is omitted.

    This repeats the convention of
    :func:`uma_pyscf.datasets.splits.composition_formula` rather than importing
    it, deliberately. ``datasets`` is the stage *after* this one in the pipeline
    (sampling -> calculators -> qc -> datasets), and a QC run that imported the
    dataset builder would make the gate depend on the thing it gates. The
    duplication is ten lines of string formatting, and the two are checked
    against each other where it matters: a composition string is only ever a
    grouping key in a report, so a divergence would rename a distribution
    bucket, never change a verdict.
    """
    counts: dict[str, int] = {}
    for number in atomic_numbers:
        symbol = PERIODIC_SYMBOLS[number]
        counts[symbol] = counts.get(symbol, 0) + 1
    return "".join(
        symbol + (str(count) if count > 1 else "") for symbol, count in sorted(counts.items())
    )


def _validated_batch(records: Sequence[LabelRecord]) -> tuple[LabelRecord, ...]:
    """Return the batch after checking it is a set of distinct pending records."""
    batch = tuple(records)
    seen: set[str] = set()
    for index, record in enumerate(batch):
        if not isinstance(record, LabelRecord):
            raise ValidationError(
                f"records[{index}] must be a LabelRecord; got {type(record).__name__}."
            )
        if record.record_id in seen:
            raise ValidationError(
                f"records[{index}].record_id {record.record_id!r} is already used by an "
                "earlier record in this batch; a record id names one calculation, and QC "
                "would otherwise write two verdicts to one file."
            )
        seen.add(record.record_id)
        if record.qc.status != PENDING_STATUS:
            raise ValidationError(
                f"Record {record.record_id!r} has qc.status {record.qc.status!r}; QC judges "
                f"records that are still {PENDING_STATUS!r}. A record that already carries a "
                "verdict is not requalified silently: the thresholds may differ from the "
                "ones that produced it, and the two verdicts would be indistinguishable."
            )
    return batch


def apply_qc(
    records: Sequence[LabelRecord], config: dict[str, Any], *, utc: str
) -> tuple[tuple[LabelRecord, ...], QcReport]:
    """Judge every record against ``config`` and return the verdicts.

    Returns the updated records in the order they were given, and the report
    covering all of them (its entries sorted by record id). ``utc`` is written
    verbatim into every history entry: the caller owns the clock, which is what
    lets a test state the timestamp and lets one batch share one instant.

    A record is accepted when every check passed. The checks that do not apply
    -- the spin-contamination check on a closed-shell record -- are reported as
    passing with nothing observed, so "accepted" always means "no check said
    no", never "no check ran".
    """
    checked = validate_qc_config(config)
    stamp = require_str(utc, "utc")
    batch = _validated_batch(records)
    qc_id = str(checked["qc_id"])
    digest = canonical_json_fingerprint(checked)
    electronic = checked["electronic"]
    geometry = checked["geometry"]
    protocol = checked.get("protocol")
    duplicates = duplicate_map(batch, geometry)

    updated: list[LabelRecord] = []
    entries: list[dict[str, Any]] = []
    for record in batch:
        checks = [
            *(protocol_checks(record, protocol) if protocol is not None else ()),
            *electronic_checks(record, electronic),
            *geometry_checks(record, geometry, duplicates.get(record.record_id)),
        ]
        failed = [check["name"] for check in checks if not check["passed"]]
        status = "rejected" if failed else "accepted"
        history_entry = {
            "utc": stamp,
            "event": QC_EVENT,
            "qc_id": qc_id,
            "config_sha256": digest,
            "result": status,
            "failed_checks": list(failed),
        }
        updated.append(
            replace(
                record,
                qc=QcState(status=status, history=(*record.qc.history, history_entry)),
            )
        )
        entries.append(
            {
                "record_id": record.record_id,
                "status": status,
                "checks": [deepcopy(check) for check in checks],
                "failed_checks": list(failed),
                "composition": composition_formula(record.structure.atomic_numbers),
                "charge": record.state.charge,
                "multiplicity": record.state.multiplicity,
                "atom_count": record.structure.atom_count,
                "energy_hartree": record.results.energy_hartree,
                "gradient_max_abs_hartree_per_bohr": gradient_max_abs(record),
            }
        )
    report = QcReport(qc_id=qc_id, config_sha256=digest, config=checked, entries=tuple(entries))
    return tuple(updated), report
