"""Private subprocess entry point for one isolated GPU4PySCF attempt."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..core.errors import UmaPyscfError
from ..core.io import read_json, write_json_atomic
from ..schemas._fields import require_mapping, require_str
from ..schemas.candidate import CandidateRecord
from ..schemas.label_record import Method
from .config import validate_dft_config
from .model import CalculationFailure
from .pyscf_adapter import Gpu4PyscfAdapter


def run_worker(request_path: str | Path, response_path: str | Path) -> int:
    """Execute one request and always publish a success/failure envelope."""
    destination = Path(response_path)
    try:
        request = require_mapping(read_json(request_path), "worker_request")
        if request.get("schema") != "uma-pyscf-label-worker-request-v1":
            raise CalculationFailure(
                "worker_request", f"Unsupported worker request schema {request.get('schema')!r}."
            )
        candidate = CandidateRecord.from_dict(request.get("candidate"))
        method = Method.from_dict(request.get("method"))
        config = validate_dft_config(request.get("config"))
        attempt_id = require_str(request.get("attempt_id"), "worker_request.attempt_id")
        resource = require_mapping(request.get("resource"), "worker_request.resource")
        output = Gpu4PyscfAdapter().calculate(
            candidate,
            method,
            config,
            attempt_id=attempt_id,
            resource=resource,
        )
        envelope: dict[str, Any] = {"ok": True, "output": output.to_dict()}
        exit_code = 0
    except CalculationFailure as exc:
        envelope = {"ok": False, "category": exc.category, "message": str(exc)}
        exit_code = 1
    except (UmaPyscfError, OSError, ValueError, KeyError) as exc:
        envelope = {
            "ok": False,
            "category": "worker_error",
            "message": f"{type(exc).__name__}: {exc}",
        }
        exit_code = 1
    write_json_atomic(destination, envelope)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("response")
    args = parser.parse_args(argv)
    return run_worker(args.request, args.response)


if __name__ == "__main__":
    raise SystemExit(main())
