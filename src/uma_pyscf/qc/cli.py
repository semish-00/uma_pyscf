"""The ``uma-pyscf qc`` subcommand.

The CLI package only registers this; deciding what QC *is* belongs to the module
that owns it, which is why the handler lives here next to
:mod:`uma_pyscf.qc.run`.

Rejections are not failures. A run that judged its records and rejected some of
them did its job and exits 0, with one line per rejection naming the checks that
failed so the reason is visible without opening the report. Exit code 1 means
the run could not be trusted: an unreadable or invalid config, a record file
that does not validate, a batch that repeats a record id, or an output path that
would overwrite an input.

That last refusal deserves a note. Judged records are written to
``<output-dir>/records/<record_id>.json``, and the input records are left
exactly as they were: QC produces a new generation of records rather than
editing the old one in place, so a run can always be repeated from the same
inputs against a different config version. Pointing the output at the inputs
would destroy that, so it is refused rather than performed.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

from ..core.errors import UmaPyscfError, ValidationError
from ..core.io import read_json, write_json_atomic
from ..schemas.label_record import LabelRecord
from ..schemas.qc_report import QcReport
from .config import load_qc_config
from .run import apply_qc

__all__ = [
    "configure_qc",
    "load_records",
    "resolve_record_paths",
    "run_qc",
    "write_qc_outputs",
]


def resolve_record_paths(raw_paths: list[str]) -> tuple[Path, ...]:
    """Expand the ``--records`` arguments into the record files to read.

    A path may be a single label record JSON file or a directory, in which case
    every ``*.json`` file directly inside it is read in file-name order. The
    order matters and is therefore fixed here rather than left to the
    filesystem: it decides which of two duplicate records is the one kept.
    """
    resolved: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(path.glob("*.json"), key=lambda entry: entry.name)
            if not found:
                raise ValidationError(f"{path} is a directory with no *.json record files in it.")
            resolved.extend(found)
        else:
            resolved.append(path)
    if not resolved:
        raise ValidationError("--records named no files; there is nothing to quality control.")
    return tuple(resolved)


def load_records(paths: tuple[Path, ...]) -> tuple[LabelRecord, ...]:
    """Read and validate every record file, naming the file that fails."""
    records: list[LabelRecord] = []
    for path in paths:
        try:
            records.append(LabelRecord.from_dict(read_json(path)))
        except (UmaPyscfError, OSError, ValueError) as exc:
            raise ValidationError(f"{path}: {exc}") from exc
    return tuple(records)


def write_qc_outputs(
    records: tuple[LabelRecord, ...],
    report: QcReport,
    output_dir: str | Path,
    inputs: tuple[Path, ...] = (),
) -> tuple[tuple[Path, ...], Path]:
    """Write the judged records and the report atomically, returning their paths.

    Nothing is written until every destination has been checked against the
    input files, so a run that would clobber its own input leaves the output
    directory untouched rather than half rewritten.
    """
    directory = Path(output_dir)
    record_dir = directory / "records"
    destinations = tuple(record_dir / f"{record.record_id}.json" for record in records)
    report_path = directory / f"{report.qc_id}_report.json"
    protected = {path.resolve() for path in inputs}
    for destination in (*destinations, report_path):
        if destination.resolve() in protected:
            raise ValidationError(
                f"Writing {destination} would overwrite an input record. QC writes a new "
                "generation of records and never edits the ones it read; choose an output "
                "directory outside the inputs."
            )
    for record, destination in zip(records, destinations, strict=True):
        write_json_atomic(destination, record.to_dict())
    write_json_atomic(report_path, report.to_dict())
    return destinations, report_path


def configure_qc(parser: argparse.ArgumentParser) -> None:
    """Add the arguments of ``qc`` to its subparser."""
    parser.add_argument(
        "--config",
        required=True,
        metavar="<config>",
        help="QC config (YAML or JSON) naming the thresholds records are judged against.",
    )
    parser.add_argument(
        "--records",
        required=True,
        nargs="+",
        metavar="<path>",
        help="Label record JSON file, or a directory of them. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="<dir>",
        help="Directory to write the judged records and the QC report into.",
    )


def run_qc(args: argparse.Namespace) -> int:
    """Judge every named record against a QC config and write the results."""
    try:
        config = load_qc_config(Path(args.config))
        paths = resolve_record_paths(args.records)
        records = load_records(paths)
        judged, report = apply_qc(records, config, utc=datetime.now(UTC).isoformat())
        _, report_path = write_qc_outputs(judged, report, Path(args.output_dir), inputs=paths)
    except (UmaPyscfError, OSError, ValueError) as exc:
        print(f"{args.config}: ERROR {exc}", file=sys.stderr)
        return 1
    for entry in report.entries:
        if entry["status"] == "rejected":
            print(f"rejected {entry['record_id']}: {', '.join(entry['failed_checks'])}")
    print(
        f"accepted={report.count('accepted')} rejected={report.count('rejected')} "
        f"report={report_path}"
    )
    return 0
