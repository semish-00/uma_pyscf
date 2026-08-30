#!/usr/bin/env python3
"""Plot C4 energy error, gradient parity, and speedup for Gate 1 review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from common import load_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    suite: dict[str, Any] = json.loads(args.suite.read_text(encoding="utf-8"))
    root = args.root.resolve()
    case_ids: list[str] = []
    energy_errors: list[float] = []
    cpu_gradients: list[float] = []
    gpu_gradients: list[float] = []
    speedups: list[float] = []
    for entry in suite["cases"]:
        candidate_id = str(entry["case_id"])
        base_id = str(entry["base_case_id"])
        gpu = load_result(root / "runs" / candidate_id / "gpu4pyscf" / "result.json")
        cpu = load_result(root / "runs" / base_id / "pyscf-cpu" / "result.json")
        case_ids.append(base_id)
        energy_errors.append(abs(float(gpu["energy_hartree"]) - float(cpu["energy_hartree"])))
        speedups.append(float(cpu["wall_time_seconds"]) / float(gpu["wall_time_seconds"]))
        for cpu_row, gpu_row in zip(
            cpu["gradient_hartree_per_bohr"], gpu["gradient_hartree_per_bohr"]
        ):
            cpu_gradients.extend(float(value) for value in cpu_row)
            gpu_gradients.extend(float(value) for value in gpu_row)

    width, height = 1800, 640
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = ImageFont.load_default(size=26)
    panel_font = ImageFont.load_default(size=20)
    label_font = ImageFont.load_default(size=15)
    draw.text(
        (width // 2, 18),
        "C4 GPU4PySCF density-fitting + explicit MINAO candidate",
        fill="black",
        font=title_font,
        anchor="ma",
    )

    panels = [(45, 90, 570, 560), (625, 90, 1150, 560), (1205, 90, 1730, 560)]

    def axes_box(panel: tuple[int, int, int, int], title: str) -> tuple[int, int, int, int]:
        left, top, right, bottom = panel
        plot = (left + 70, top + 42, right - 18, bottom - 55)
        draw.rectangle(plot, outline="#333333", width=2)
        draw.text(((left + right) // 2, top + 5), title, fill="black", font=panel_font, anchor="ma")
        return plot

    # Panel 1: absolute energy errors on a logarithmic axis.
    plot = axes_box(panels[0], "Absolute total-energy difference")
    x0, y0, x1, y1 = plot
    log_min, log_max = -8.0, -4.0
    for exponent in range(-8, -3):
        y = y1 - (exponent - log_min) / (log_max - log_min) * (y1 - y0)
        draw.line((x0, y, x1, y), fill="#dddddd", width=1)
        draw.text((x0 - 8, y), f"1e{exponent}", fill="#333333", font=label_font, anchor="rm")
    for index, value in enumerate(energy_errors):
        x = x0 + index / (len(energy_errors) - 1) * (x1 - x0)
        y = y1 - (max(log_min, min(log_max, math.log10(value))) - log_min) / (log_max - log_min) * (y1 - y0)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#31688e")
    tolerance_y = y1 - (math.log10(5e-5) - log_min) / (log_max - log_min) * (y1 - y0)
    draw.line((x0, tolerance_y, x1, tolerance_y), fill="#b2182b", width=2)
    draw.text((x1 - 4, tolerance_y - 6), "5e-5 Eh", fill="#b2182b", font=label_font, anchor="rb")
    draw.text(((x0 + x1) // 2, y1 + 34), "29-case ladder index", fill="black", font=label_font, anchor="ma")

    # Panel 2: all gradient components against the identity line.
    plot = axes_box(panels[1], "Gradient component parity")
    x0, y0, x1, y1 = plot
    extent = max(abs(value) for value in cpu_gradients + gpu_gradients) * 1.05
    draw.line((x0, y1, x1, y0), fill="#222222", width=2)
    for cpu_value, gpu_value in zip(cpu_gradients, gpu_gradients):
        x = x0 + (cpu_value + extent) / (2 * extent) * (x1 - x0)
        y = y1 - (gpu_value + extent) / (2 * extent) * (y1 - y0)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(53, 183, 121, 120))
    draw.text(((x0 + x1) // 2, y1 + 34), "CPU direct gradient (Eh/bohr)", fill="black", font=label_font, anchor="ma")
    draw.text((x0 + 8, y0 + 8), "GPU density-fit", fill="#247a56", font=label_font)

    # Panel 3: per-case speedup bars.
    plot = axes_box(panels[2], "Per-case speedup")
    x0, y0, x1, y1 = plot
    maximum = max(speedups) * 1.08
    bar_width = (x1 - x0) / len(speedups)
    for index, value in enumerate(speedups):
        left = x0 + index * bar_width + 1
        right = x0 + (index + 1) * bar_width - 1
        top = y1 - value / maximum * (y1 - y0)
        draw.rectangle((left, top, right, y1), fill="#e6cf28", outline="#6c6c3c")
    one_y = y1 - 1.0 / maximum * (y1 - y0)
    draw.line((x0, one_y, x1, one_y), fill="#222222", width=1)
    draw.text((x0 - 8, y0), f"{maximum:.0f}x", fill="#333333", font=label_font, anchor="ra")
    draw.text(((x0 + x1) // 2, y1 + 34), "29-case ladder index", fill="black", font=label_font, anchor="ma")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
