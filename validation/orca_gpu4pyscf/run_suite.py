#!/usr/bin/env python3
"""Run a cross-code suite sequentially on this host with CPU PySCF or GPU4PySCF.

This is the non-PBS counterpart of submit_suite.py for hosts where cases are
executed directly, such as the GPU machine. Each case runs in its own child
process so a native crash or leaked GPU memory cannot poison later cases, and
every attempt is appended to a per-case ledger that is never overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from common import write_json
from run_pyscf import build_plan

ROOT = Path(__file__).resolve().parent
ENGINE_BY_DEVICE = {"cpu": "pyscf-cpu", "gpu": "gpu4pyscf"}
ATTEMPT_SCHEMA = "crosscode-suite-attempt-v1"
SESSION_SCHEMA = "crosscode-suite-session-v1"
TAIL_CHARACTERS = 4000


def load_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(suite, dict) or suite.get("schema") != "crosscode-suite-v1":
        raise ValueError(f"{path} is not a crosscode-suite-v1 manifest.")
    if not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError(f"{path} contains no cases.")
    return suite


def case_paths(root: Path, case: dict[str, Any], engine: str) -> tuple[Path, Path, Path]:
    config = root / str(case["config"])
    run_dir = root / "runs" / str(case["case_id"]) / engine
    return config, run_dir / "result.json", run_dir / "attempts.jsonl"


def existing_attempt_count(ledger: Path) -> int:
    if not ledger.is_file():
        return 0
    return sum(1 for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip())


def append_attempt(ledger: Path, entry: dict[str, Any]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")


def tail(text: str) -> str:
    return text[-TAIL_CHARACTERS:]


def execute_case(
    config: Path, device: str, output: Path, timeout_seconds: float | None
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "run_pyscf.py"),
        str(config),
        "--device",
        device,
        "--output",
        str(output),
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )


def dry_run(suite: dict[str, Any], root: Path, device: str) -> int:
    engine = ENGINE_BY_DEVICE[device]
    for case in suite["cases"]:
        config, result_path, _ = case_paths(root, case, engine)
        plan = build_plan(config, device)
        print(
            f"{case['case_id']}: {plan['reference']} charge={plan['case']['charge']} "
            f"multiplicity={plan['case']['multiplicity']} -> {result_path} "
            f"({'existing result' if result_path.is_file() else 'no result yet'})"
        )
    print(f"dry_run_ok suite={suite['suite_id']} cases={len(suite['cases'])} engine={engine}")
    return 0


def run_suite(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    suite = load_suite(args.suite)
    engine = ENGINE_BY_DEVICE[args.device]
    if args.dry_run:
        return dry_run(suite, root, args.device)

    timeout_seconds = args.case_timeout_minutes * 60 if args.case_timeout_minutes else None
    session: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "suite_id": suite["suite_id"],
        "engine": engine,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "keep_going": bool(args.keep_going),
        "overwrite": bool(args.overwrite),
        "cases": [],
    }
    stopped_early = False
    for case in suite["cases"]:
        case_id = str(case["case_id"])
        config, result_path, ledger_path = case_paths(root, case, engine)
        if stopped_early:
            session["cases"].append({"case_id": case_id, "status": "not_attempted"})
            continue
        if not config.is_file():
            raise FileNotFoundError(config)
        if result_path.is_file() and not args.overwrite:
            print(f"{case_id}: skipped, result exists at {result_path}")
            session["cases"].append({"case_id": case_id, "status": "skipped_existing_result"})
            continue

        # Fail before spawning if the manifest itself is invalid.
        build_plan(config, args.device)
        attempt_index = existing_attempt_count(ledger_path) + 1
        entry: dict[str, Any] = {
            "schema": ATTEMPT_SCHEMA,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "suite_id": suite["suite_id"],
            "case_id": case_id,
            "engine": engine,
            "attempt_index": attempt_index,
            "overwrote_existing": result_path.is_file(),
        }
        started = time.perf_counter()
        try:
            completed = execute_case(config, args.device, result_path, timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            entry.update(
                status="timeout",
                wall_time_seconds=time.perf_counter() - started,
                timeout_seconds=timeout_seconds,
                stdout_tail=tail(exc.stdout or "") if isinstance(exc.stdout, str) else None,
                stderr_tail=tail(exc.stderr or "") if isinstance(exc.stderr, str) else None,
            )
            failed = True
        else:
            succeeded = completed.returncode == 0 and result_path.is_file()
            entry.update(
                status="succeeded" if succeeded else "failed",
                returncode=completed.returncode,
                wall_time_seconds=time.perf_counter() - started,
                result_written=result_path.is_file(),
                stdout_tail=tail(completed.stdout),
                stderr_tail=tail(completed.stderr),
            )
            failed = not succeeded
        append_attempt(ledger_path, entry)
        session["cases"].append(
            {
                "case_id": case_id,
                "status": entry["status"],
                "attempt_index": attempt_index,
                "wall_time_seconds": entry["wall_time_seconds"],
            }
        )
        print(f"{case_id}: {entry['status']} ({entry['wall_time_seconds']:.1f}s, attempt {attempt_index})")
        if failed and not args.keep_going:
            stopped_early = True

    session["finished_utc"] = datetime.now(timezone.utc).isoformat()
    counts: dict[str, int] = {}
    for row in session["cases"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    session["status_counts"] = counts

    summary_path = args.summary_output
    if summary_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        summary_path = root / "runs" / "sessions" / f"{suite['suite_id']}_{engine}_{stamp}.json"
    write_json(summary_path, session)
    print(f"summary={summary_path} status_counts={json.dumps(counts, sort_keys=True)}")

    all_ok = all(
        row["status"] in ("succeeded", "skipped_existing_result") for row in session["cases"]
    )
    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="Suite manifest JSON.")
    parser.add_argument("--device", choices=tuple(ENGINE_BY_DEVICE), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every case manifest and print the plan without importing PySCF.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later cases after a failure. Default stops at the first "
        "failure, as the C1 smoke protocol requires.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run cases whose normalized result already exists.",
    )
    parser.add_argument(
        "--case-timeout-minutes",
        type=float,
        default=None,
        help="Optional per-case wall-clock limit; a timeout counts as a failure.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Session summary JSON path (default: runs/sessions/<suite>_<engine>_<stamp>.json).",
    )
    return parser


def main() -> int:
    return run_suite(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
