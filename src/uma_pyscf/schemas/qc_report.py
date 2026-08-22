"""The QC report: every record's verdict, with the evidence and the summary.

Schema ``uma-pyscf-qc-report-v1``. One report covers one QC run over one batch
of label records, and it is written to be *self-contained*: the thresholds it
judged by are embedded verbatim, and every entry carries not only its verdict
but the handful of per-record facts the release-gate distributions are computed
from. A reader holding the report alone can therefore recompute everything it
claims, which is what makes the claims checkable rather than trusted.

Three rules follow from that and are enforced here rather than assumed.

* **The digest is derived from the config, never stated beside it.**
  ``config_sha256`` is
  :func:`~uma_pyscf.core.ids.canonical_json_fingerprint` of the embedded
  ``config``; :meth:`QcReport.from_dict` recomputes it and refuses a mismatch,
  so a report cannot name thresholds it did not use.
* **The verdict is derived from the checks.** An entry's ``failed_checks`` must
  be exactly the names of its checks that did not pass, in order, and its
  ``status`` is ``rejected`` if and only if that list is non-empty. A report
  that says "accepted" over a failing check is refused, not read.
* **The distributions are derived from the entries.** :attr:`QcReport.counts`
  and :attr:`QcReport.distributions` are computed properties, written into
  :meth:`QcReport.to_dict` for the reader's convenience and recomputed on the
  way back in. Editing a count in the file makes the file unreadable rather
  than making it lie -- the same discipline
  :class:`~uma_pyscf.schemas.candidate.GeometryQcReport` applies to its counts.

There is deliberately no timestamp in the report. When a record was evaluated
belongs to that record's ``qc.history``, which is where an append-only audit
trail can live; a report that also carried the clock could not be regenerated
byte for byte from the same records and the same config, and the release gate's
determinism check is worth more than a redundant date.

The house rules of :mod:`uma_pyscf.schemas.label_record` apply unchanged: frozen
kw-only dataclasses, explicit ``to_dict``/``from_dict``, validation in the
constructor, and unknown keys refused -- except inside ``config`` and inside a
check's ``observed``/``threshold``, which are free-form JSON by design and are
validated as JSON values instead.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import fsum
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint, validate_record_id
from ._fields import (
    reject_unknown_keys,
    require_bool,
    require_finite_float,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
    validated_json_object,
)

__all__ = [
    "QC_REPORT_SCHEMA",
    "QC_REPORT_STATUSES",
    "QcReport",
]

QC_REPORT_SCHEMA = "uma-pyscf-qc-report-v1"

#: The verdicts a QC entry may carry. ``pending`` is a *record* state, not a
#: report state: a record that was not evaluated has no entry here at all.
QC_REPORT_STATUSES: tuple[str, ...] = ("accepted", "rejected")

_REPORT_KEYS = ("schema", "qc_id", "config_sha256", "config", "entries", "distributions")
_CHECK_KEYS = ("name", "passed", "observed", "threshold")
_ENTRY_KEYS = (
    "record_id",
    "status",
    "checks",
    "failed_checks",
    "composition",
    "charge",
    "multiplicity",
    "atom_count",
    "energy_hartree",
    "gradient_max_abs_hartree_per_bohr",
)

#: The per-record numbers the accepted-record distributions are computed over.
_DISTRIBUTED_VALUES = ("energy_hartree", "gradient_max_abs_hartree_per_bohr")

#: The entry fields the accepted/rejected breakdowns are grouped by.
_GROUPED_FIELDS = ("composition", "charge", "multiplicity", "atom_count")

_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_sha256(value: object, path: str) -> str:
    """Return ``value`` as a lowercase 64-character hex digest."""
    digest = require_str(value, path).lower()
    if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
        raise ValidationError(f"{path} must be 64 hexadecimal characters; got {value!r}.")
    return digest


def _validated_check(value: object, path: str) -> dict[str, Any]:
    """Return one check result, normalized and fully validated.

    A check states four things: which check it was (``name``), whether the
    record satisfied it (``passed``), what was measured (``observed``, ``null``
    when the check had nothing to report or did not apply), and what it was
    measured against (``threshold``, the config value or values that decided
    it). The last two are free-form JSON so a check can report a violating atom
    pair as readily as a number, and they are validated as JSON values: a check
    result that cannot survive a write-read round trip is not evidence.
    """
    mapping = require_mapping(value, path)
    reject_unknown_keys(mapping, _CHECK_KEYS, path)
    for key in _CHECK_KEYS:
        require_key(mapping, key, path)
    require_str(mapping["name"], f"{path}.name")
    require_bool(mapping["passed"], f"{path}.passed")
    return validated_json_object(mapping, path)


def _validated_entry(value: object, path: str) -> dict[str, Any]:
    """Return one report entry, normalized and fully validated.

    Besides the verdict and its evidence, an entry carries ``composition``,
    ``charge``, ``multiplicity``, ``atom_count``, ``energy_hartree``, and
    ``gradient_max_abs_hartree_per_bohr``. Those six exist so the release-gate
    distributions can be recomputed from the report alone; they are copied from
    the record and are not a second opinion about it.
    """
    mapping = require_mapping(value, path)
    reject_unknown_keys(mapping, _ENTRY_KEYS, path)
    for key in _ENTRY_KEYS:
        require_key(mapping, key, path)

    record_id = validate_record_id(require_str(mapping["record_id"], f"{path}.record_id"))
    status = require_str(mapping["status"], f"{path}.status")
    if status not in QC_REPORT_STATUSES:
        raise ValidationError(
            f"{path}.status must be one of "
            f"{', '.join(repr(name) for name in QC_REPORT_STATUSES)}; got {status!r}."
        )

    raw_checks = require_sequence(mapping["checks"], f"{path}.checks")
    if not raw_checks:
        raise ValidationError(
            f"{path}.checks must list at least one check; an entry with no checks records "
            "a verdict nothing was evaluated for."
        )
    checks = [
        _validated_check(check, f"{path}.checks[{index}]")
        for index, check in enumerate(raw_checks)
    ]
    failed = [
        require_str(name, f"{path}.failed_checks[{index}]")
        for index, name in enumerate(
            require_sequence(mapping["failed_checks"], f"{path}.failed_checks")
        )
    ]
    derived = [check["name"] for check in checks if not check["passed"]]
    if failed != derived:
        raise ValidationError(
            f"{path}.failed_checks is {failed!r} but the checks that did not pass are "
            f"{derived!r}. The failed list names the failing checks and is never stated "
            "apart from them."
        )
    if (status == "rejected") != bool(failed):
        raise ValidationError(
            f"{path}.status is {status!r} with {len(failed)} failed check(s); a record is "
            "rejected if and only if at least one check failed."
        )

    multiplicity = require_int(mapping["multiplicity"], f"{path}.multiplicity")
    if multiplicity < 1:
        raise ValidationError(f"{path}.multiplicity must be at least 1; got {multiplicity}.")
    atom_count = require_int(mapping["atom_count"], f"{path}.atom_count")
    if atom_count < 1:
        raise ValidationError(f"{path}.atom_count must be at least 1; got {atom_count}.")
    gradient_max = require_finite_float(
        mapping["gradient_max_abs_hartree_per_bohr"],
        f"{path}.gradient_max_abs_hartree_per_bohr",
    )
    if gradient_max < 0.0:
        raise ValidationError(
            f"{path}.gradient_max_abs_hartree_per_bohr is a magnitude and must not be "
            f"negative; got {gradient_max}."
        )
    return {
        "record_id": record_id,
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "composition": require_str(mapping["composition"], f"{path}.composition"),
        "charge": require_int(mapping["charge"], f"{path}.charge"),
        "multiplicity": multiplicity,
        "atom_count": atom_count,
        "energy_hartree": require_finite_float(
            mapping["energy_hartree"], f"{path}.energy_hartree"
        ),
        "gradient_max_abs_hartree_per_bohr": gradient_max,
    }


@dataclass(frozen=True, kw_only=True)
class QcReport:
    """The verdicts of one QC run, with the config that produced them.

    ``config`` is the loaded QC config verbatim and ``config_sha256`` is
    :func:`~uma_pyscf.core.ids.canonical_json_fingerprint` of it. Hashing the
    content rather than the file bytes is deliberate: the digest then depends on
    the thresholds the run applied and not on the YAML's comments or key order,
    and the same digest appears in every record's QC history entry, which is
    what ties a record's verdict to this report.

    Entries are sorted by ``record_id`` and each record appears once, so two
    runs over the same records in a different order produce the same report.
    """

    schema: str = QC_REPORT_SCHEMA
    qc_id: str
    config_sha256: str
    config: dict[str, Any]
    entries: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.schema != QC_REPORT_SCHEMA:
            raise ValidationError(
                f"report.schema must be {QC_REPORT_SCHEMA!r}; got {self.schema!r}."
            )
        config = validated_json_object(self.config, "report.config")
        digest = _require_sha256(self.config_sha256, "report.config_sha256")
        expected = canonical_json_fingerprint(config)
        if digest != expected:
            raise ValidationError(
                f"report.config_sha256 is {digest} but the embedded config fingerprints to "
                f"{expected}. The digest is derived from the config, never stated "
                "independently of it."
            )
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, entry in enumerate(require_sequence(self.entries, "report.entries")):
            checked = _validated_entry(entry, f"report.entries[{index}]")
            if checked["record_id"] in seen:
                raise ValidationError(
                    f"report.entries[{index}].record_id {checked['record_id']!r} is already "
                    "used by an earlier entry; every record is judged exactly once."
                )
            seen.add(checked["record_id"])
            entries.append(checked)
        object.__setattr__(self, "qc_id", validate_record_id(self.qc_id))
        object.__setattr__(self, "config_sha256", digest)
        object.__setattr__(self, "config", config)
        object.__setattr__(
            self, "entries", tuple(sorted(entries, key=lambda entry: entry["record_id"]))
        )

    def count(self, status: str) -> int:
        """Return how many entries carry ``status``."""
        return sum(1 for entry in self.entries if entry["status"] == status)

    @property
    def counts(self) -> dict[str, int]:
        """Return the derived verdict counts."""
        counts = {status: self.count(status) for status in QC_REPORT_STATUSES}
        counts["total"] = len(self.entries)
        return counts

    def breakdown(self, field: str) -> dict[str, dict[str, int]]:
        """Return accepted/rejected counts grouped by one entry field.

        Group keys are strings even when the field is a number, because they
        become JSON object keys: charge ``-1`` groups under ``"-1"``. Every
        group lists both verdicts, a zero included, so a reader never has to
        distinguish "none rejected" from "not reported".
        """
        if field not in _GROUPED_FIELDS:
            raise ValidationError(
                f"{field!r} is not a grouped field; this report groups by "
                f"{', '.join(repr(name) for name in _GROUPED_FIELDS)}."
            )
        grouped: dict[str, dict[str, int]] = {}
        for entry in self.entries:
            bucket = grouped.setdefault(
                str(entry[field]), {status: 0 for status in QC_REPORT_STATUSES}
            )
            bucket[entry["status"]] += 1
        return dict(sorted(grouped.items()))

    def accepted_range(self, field: str) -> dict[str, float] | None:
        """Return min/max/mean of one field over accepted records, or ``None``.

        Rejected records are excluded on purpose: the dataset card describes the
        data that is being released, and a rejected label's energy is not part
        of it. ``None`` means nothing was accepted, which is a different
        statement from a range of zero width.
        """
        if field not in _DISTRIBUTED_VALUES:
            raise ValidationError(
                f"{field!r} is not a distributed value; this report distributes "
                f"{', '.join(repr(name) for name in _DISTRIBUTED_VALUES)}."
            )
        values = [float(entry[field]) for entry in self.entries if entry["status"] == "accepted"]
        if not values:
            return None
        return {"min": min(values), "max": max(values), "mean": fsum(values) / len(values)}

    @property
    def distributions(self) -> dict[str, Any]:
        """Return the whole derived summary the dataset release gate reads."""
        summary: dict[str, Any] = {"counts": self.counts}
        for field in _GROUPED_FIELDS:
            summary[f"by_{field}"] = self.breakdown(field)
        for field in _DISTRIBUTED_VALUES:
            summary[field] = self.accepted_range(field)
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of the whole report."""
        return {
            "schema": self.schema,
            "qc_id": self.qc_id,
            "config_sha256": self.config_sha256,
            "config": deepcopy(self.config),
            "entries": [deepcopy(entry) for entry in self.entries],
            "distributions": self.distributions,
        }

    @classmethod
    def from_dict(cls, data: Any) -> QcReport:
        """Build a :class:`QcReport`, re-deriving the distributions on the way.

        The schema string is checked first, so a foreign or older report is
        refused by name rather than by a confusing field error further in.
        """
        mapping = require_mapping(data, "report")
        schema = mapping.get("schema")
        if schema != QC_REPORT_SCHEMA:
            raise ValidationError(f"report.schema must be {QC_REPORT_SCHEMA!r}; got {schema!r}.")
        reject_unknown_keys(mapping, _REPORT_KEYS, "report")
        report = cls(
            schema=schema,
            qc_id=require_key(mapping, "qc_id", "report"),
            config_sha256=require_key(mapping, "config_sha256", "report"),
            config=require_key(mapping, "config", "report"),
            entries=tuple(
                require_sequence(require_key(mapping, "entries", "report"), "report.entries")
            ),
        )
        stored = mapping.get("distributions")
        if stored is not None:
            distributions = require_mapping(stored, "report.distributions")
            if distributions != report.distributions:
                raise ValidationError(
                    f"report.distributions is {distributions!r} but the entries derive "
                    f"{report.distributions!r}. The distributions summarize the entries "
                    "and are never stated apart from them."
                )
        return report
