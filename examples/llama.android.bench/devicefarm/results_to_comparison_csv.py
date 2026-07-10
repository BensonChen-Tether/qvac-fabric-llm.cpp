#!/usr/bin/env python3
"""Pivot Device Farm results.tsv into comparison CSV: Model, Format, Device, tok/s CPU, tok/s GPU."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional


def format_quant_label(quant_type: str) -> str:
    if quant_type.upper().startswith("TQ2"):
        return "TQ2"
    if quant_type.upper().startswith("TQ1"):
        return "TQ1"
    return quant_type


def model_key_from_filename(model_name: str) -> str:
    stem = Path(model_name).stem
    return re.sub(
        r"-(BF16|F16|F32|Q[0-9]_[A-Z0-9_]+|TQ[0-9]_[0-9]+|IQ[0-9]_[A-Z0-9]+)$",
        "",
        stem,
        flags=re.IGNORECASE,
    )


def quant_from_filename(model_name: str) -> str:
    m = re.search(r"(TQ[0-9]_[0-9]+|Q[0-9]_[A-Z0-9_]+)", model_name, re.IGNORECASE)
    return format_quant_label(m.group(1)) if m else "UNKNOWN"


def parse_tsv(tsv_path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """Return model_key -> {format, cpu, gpu}."""
    rows: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("result") != "PASSED":
                continue
            model = row.get("model", "")
            key = model_key_from_filename(model)
            rows[key]["format"] = quant_from_filename(model)
            backend = row.get("backend", "")
            tg = row.get("tg_t_s", "").strip()
            if backend in ("cpu", "gpu") and tg:
                rows[key][backend] = float(tg)
    return rows


def write_comparison_csv(
    tsv_path: Path,
    output_path: Path,
    device_label: str,
) -> None:
    rows = parse_tsv(tsv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Format", "Device", "tok/s CPU", "tok/s GPU"])
        for key in sorted(rows):
            entry = rows[key]
            cpu = entry.get("cpu")
            gpu = entry.get("gpu")
            writer.writerow([
                key,
                entry.get("format", "UNKNOWN"),
                device_label,
                f"{cpu:.2f}" if cpu is not None else "",
                f"{gpu:.2f}" if gpu is not None else "",
            ])
    print(f"Comparison CSV: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path, help="results.tsv from s3 matrix run")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "-d",
        "--device",
        default="iPhone 17 Pro Max (Metal)",
        help="Device label for comparison CSV",
    )
    args = parser.parse_args()
    write_comparison_csv(args.tsv, args.output, args.device)


if __name__ == "__main__":
    main()
