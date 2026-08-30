"""One-process-per-candidate adapter used by the production label CLI."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from ..core.io import read_json, write_json_atomic
from ..schemas._fields import require_bool, require_mapping, require_str
from ..schemas.candidate import CandidateRecord
from ..schemas.label_record import Method
from .model import CalculationFailure, CalculationOutput

__all__ = ["SubprocessGpu4PyscfAdapter"]


class SubprocessGpu4PyscfAdapter:
    """Invoke the real adapter in a fresh interpreter for each attempt."""

    def calculate(
        self,
        candidate: CandidateRecord,
        method: Method,
        config: Mapping[str, Any],
        *,
        attempt_id: str,
        resource: Mapping[str, Any],
    ) -> CalculationOutput:
        """Run ``uma_pyscf.calculators.worker`` and restore its typed envelope."""
        scratch_parent = os.environ.get("UMA_PYSCF_SCRATCH")
        with tempfile.TemporaryDirectory(
            prefix="uma-pyscf-label-", dir=scratch_parent
        ) as directory:
            root = Path(directory)
            request_path = root / "request.json"
            response_path = root / "response.json"
            write_json_atomic(
                request_path,
                {
                    "schema": "uma-pyscf-label-worker-request-v1",
                    "candidate": candidate.to_dict(),
                    "method": method.to_dict(),
                    "config": dict(config),
                    "attempt_id": attempt_id,
                    "resource": dict(resource),
                },
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "uma_pyscf.calculators.worker",
                    str(request_path),
                    str(response_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if not response_path.exists():
                message = (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or "no worker response"
                )
                raise CalculationFailure(
                    "subprocess_error",
                    f"GPU worker exited {completed.returncode}: {message}",
                )
            envelope = require_mapping(read_json(response_path), "worker_response")
            if not require_bool(envelope.get("ok"), "worker_response.ok"):
                category = require_str(envelope.get("category"), "worker_response.category")
                message = require_str(envelope.get("message"), "worker_response.message")
                raise CalculationFailure(category, message)
            output = CalculationOutput.from_dict(envelope.get("output"))
            payload = dict(output.raw_payload)
            if completed.stdout:
                payload["worker_stdout"] = completed.stdout
            if completed.stderr:
                payload["worker_stderr"] = completed.stderr
            payload["worker_returncode"] = completed.returncode
            return CalculationOutput(
                engine_name=output.engine_name,
                engine_versions=output.engine_versions,
                results=output.results,
                raw_payload=payload,
            )
