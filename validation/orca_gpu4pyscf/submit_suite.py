#!/usr/bin/env python3
"""Dry-run or submit a generated cross-code suite to Ujilab OpenPBS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def job_name(engine: str, case_id: str) -> str:
    prefix = "o" if engine == "orca" else "p"
    digest = hashlib.sha256(case_id.encode()).hexdigest()[:10]
    return f"{prefix}_{digest}"


def command(root: Path, case: dict[str, object], engine: str) -> list[str]:
    case_id = str(case["case_id"])
    config = f"validation/orca_gpu4pyscf/{case['config']}"
    allocation = dict(case["resources"])
    ncpus = int(allocation["ncpus"])
    memory_gb = int(allocation["memory_gb"])
    walltime = str(allocation["walltime"])
    common = ["/usr/openpbs/bin/qsub", "-N", job_name(engine, case_id)]
    if engine == "orca":
        variables = f"CONFIG={config},RUN_DIR=validation/orca_gpu4pyscf/runs/{case_id}/orca"
        select = f"select=1:ncpus={ncpus}:mpiprocs={ncpus}:mem={memory_gb}gb"
        script = "validation/orca_gpu4pyscf/jobs/run_orca_cpu_pbs.sh"
    else:
        variables = f"CONFIG={config},OUTPUT=validation/orca_gpu4pyscf/runs/{case_id}/pyscf-cpu/result.json,THREADS={ncpus}"
        select = f"select=1:ncpus={ncpus}:mpiprocs=1:mem={memory_gb}gb"
        script = "validation/orca_gpu4pyscf/jobs/run_pyscf_cpu_pbs.sh"
    return [*common, "-v", variables, "-l", select, "-l", f"walltime={walltime}", script]


def result_path(root: Path, case_id: str, engine: str) -> Path:
    return root / "validation/orca_gpu4pyscf/runs" / case_id / engine / "result.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--submit", action="store_true", help="Actually invoke qsub; default is dry-run.")
    parser.add_argument(
        "--skip-existing-results",
        action="store_true",
        help="Do not resubmit an engine/case pair whose normalized result already exists.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    receipt: dict[str, object] = {
        "schema": "pbs-submission-receipt-v1",
        "suite_id": suite["suite_id"],
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": [],
    }
    output = None
    if args.submit:
        destination = root / "validation/orca_gpu4pyscf/runs/submissions"
        destination.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = destination / f"{suite['suite_id']}_{stamp}.json"

    def save_receipt() -> None:
        if output is not None:
            output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    for case in suite["cases"]:
        config_path = root / "validation/orca_gpu4pyscf" / case["config"]
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        for engine in ("orca", "pyscf-cpu"):
            cmd = command(root, case, engine)
            existing = result_path(root, str(case["case_id"]), engine)
            if args.submit and args.skip_existing_results and existing.is_file():
                print(f"{case['case_id']} {engine}: skipped existing {existing}")
                receipt["jobs"].append({"case_id": case["case_id"], "engine": engine,
                                        "job_name": job_name(engine, str(case["case_id"])), "job_id": None,
                                        "status": "skipped_existing_result", "resources": case["resources"]})
                save_receipt()
                continue
            if args.submit:
                completed = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
                if completed.returncode != 0:
                    receipt["jobs"].append({"case_id": case["case_id"], "engine": engine,
                                            "job_name": job_name(engine, str(case["case_id"])), "job_id": None,
                                            "status": "submission_failed", "returncode": completed.returncode,
                                            "stderr": completed.stderr.strip(), "resources": case["resources"]})
                    save_receipt()
                    raise RuntimeError(f"qsub failed for {case['case_id']} {engine}: {completed.stderr.strip()}")
                job_id = completed.stdout.strip()
                print(f"{case['case_id']} {engine}: {job_id}")
            else:
                job_id = None
                print(" ".join(cmd))
            receipt["jobs"].append({"case_id": case["case_id"], "engine": engine,
                                    "job_name": job_name(engine, str(case["case_id"])), "job_id": job_id,
                                    "status": "submitted" if args.submit else "dry_run",
                                    "resources": case["resources"]})
            save_receipt()
    if args.submit:
        print(f"receipt={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
