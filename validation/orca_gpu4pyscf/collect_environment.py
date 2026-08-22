#!/usr/bin/env python3
"""Collect the Workstream A1 hardware/software inventory of a GPU4PySCF host.

The script records what it can observe and stores explicit nulls plus a
collection-error map for everything it cannot, so an incomplete environment is
visible instead of silently omitted. Run it on the calculation host:

    python collect_environment.py            # writes configs/environments/
    python collect_environment.py --stdout   # print without writing
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
SCHEMA = "gpu4pyscf-environment-v1"

# The importable module name does not always match the installed distribution
# name, so probe the known aliases in order.
PACKAGE_DISTRIBUTIONS = {
    "pyscf": ("pyscf",),
    "gpu4pyscf": ("gpu4pyscf-cuda12x", "gpu4pyscf-cuda11x", "gpu4pyscf"),
    "cupy": ("cupy-cuda12x", "cupy-cuda11x", "cupy-cuda13x", "cupy"),
    "cutensor": ("cutensor-cu12", "cutensor-cu11", "cutensor-cu13", "cutensor"),
    "geometric": ("geometric",),
}


def first_distribution(names: tuple[str, ...]) -> dict[str, Any] | None:
    for name in names:
        try:
            return {"distribution": name, "version": metadata.version(name)}
        except metadata.PackageNotFoundError:
            continue
    return None


def read_os_release() -> str | None:
    path = Path("/etc/os-release")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def read_cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def read_ram_gb() -> float | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"MemTotal:\s+(\d+)\s+kB", line.strip())
        if match:
            return round(int(match.group(1)) / 1024 / 1024, 2)
    return None


def read_nvidia_driver() -> str | None:
    proc_version = Path("/proc/driver/nvidia/version")
    if proc_version.is_file():
        match = re.search(r"Kernel Module\s+(\S+)", proc_version.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    if shutil.which("nvidia-smi"):
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=60,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip().splitlines()[0].strip()
    return None


def read_gpus_via_nvidia_smi() -> list[dict[str, Any]] | None:
    if not shutil.which("nvidia-smi"):
        return None
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        timeout=60,
    )
    if completed.returncode != 0:
        return None
    gpus = []
    for line in completed.stdout.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                }
            )
    return gpus or None


def collect(errors: dict[str, str]) -> dict[str, Any]:
    def probe(name: str, callable_: Any) -> Any:
        try:
            return callable_()
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            return None

    inventory: dict[str, Any] = {
        "schema": SCHEMA,
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node() or None,
        "os": {
            "pretty_name": probe("os_release", read_os_release),
            "kernel": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": {
            "model": probe("cpu_model", read_cpu_model),
            "logical_cores": os.cpu_count(),
        },
        "ram_gb": probe("ram", read_ram_gb),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": {},
        "nvidia_driver": probe("nvidia_driver", read_nvidia_driver),
        "cuda": {"runtime_version": None},
        "gpus": None,
        "libxc": None,
        "environment_variables": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "CUPY_ACCELERATORS")
        },
    }
    for package, distributions in PACKAGE_DISTRIBUTIONS.items():
        inventory["packages"][package] = first_distribution(distributions)

    def cupy_probe() -> None:
        import cupy

        inventory["cuda"]["runtime_version"] = int(cupy.cuda.runtime.runtimeGetVersion())
        count = int(cupy.cuda.runtime.getDeviceCount())
        gpus = []
        for index in range(count):
            properties = cupy.cuda.runtime.getDeviceProperties(index)
            name = properties.get("name")
            gpus.append(
                {
                    "index": index,
                    "name": name.decode() if isinstance(name, bytes) else name,
                    "memory_total_mib": int(properties["totalGlobalMem"] // (1024 * 1024)),
                    "compute_capability": f"{properties['major']}.{properties['minor']}",
                }
            )
        inventory["gpus"] = gpus

    probe("cupy", cupy_probe)
    if inventory["gpus"] is None:
        inventory["gpus"] = probe("nvidia_smi_gpus", read_gpus_via_nvidia_smi)

    def libxc_probe() -> None:
        from pyscf import dft

        inventory["libxc"] = getattr(dft.libxc, "__version__", None)

    probe("libxc", libxc_probe)
    inventory["collection_errors"] = dict(sorted(errors.items())) or None
    return inventory


def emit_yaml(value: Any, indent: int = 0) -> str:
    """Serialize the inventory as YAML. Scalars use JSON syntax, which YAML 1.2
    accepts, so no third-party YAML dependency is needed."""
    prefix = "  " * indent
    if isinstance(value, dict):
        if not value:
            return f"{prefix}{{}}\n"
        lines = []
        for key, item in value.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
                raise ValueError(f"Refusing to emit non-identifier YAML key {key!r}.")
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{prefix}{key}:\n{emit_yaml(item, indent + 1)}")
            else:
                lines.append(f"{prefix}{key}: {json.dumps(item, ensure_ascii=False)}\n")
        return "".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                body = emit_yaml(item, indent + 1)
                first, _, rest = body.partition("\n")
                lines.append(f"{prefix}- {first.strip()}\n{rest}" if rest else f"{prefix}- {first.strip()}\n")
            else:
                lines.append(f"{prefix}- {json.dumps(item, ensure_ascii=False)}\n")
        return "".join(lines)
    return f"{prefix}{json.dumps(value, ensure_ascii=False)}\n"


def default_output(host: str | None) -> Path:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", host or "unknown-host").strip("-").lower()
    return REPO_ROOT / "configs" / "environments" / f"gpu4pyscf-{label or 'unknown-host'}.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Explicit output YAML path.")
    parser.add_argument("--stdout", action="store_true", help="Print instead of writing.")
    args = parser.parse_args()

    errors: dict[str, str] = {}
    inventory = collect(errors)
    text = emit_yaml(inventory)
    if args.stdout:
        print(text, end="")
        return 0
    output = args.output or default_output(inventory["host"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {output}")
    if inventory["collection_errors"]:
        print(f"collection_errors={json.dumps(inventory['collection_errors'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
