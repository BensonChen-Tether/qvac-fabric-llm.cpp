#!/usr/bin/env python3
"""Collect Firebase Test Lab logcat output and generate an Excel benchmark report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    import pandas as pd
except ImportError:
    print("Missing dependency: pip3 install -r firebase/requirements.txt", file=sys.stderr)
    raise

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "testlab_results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reports"

RESULT_TAG = "LLAMA_BENCH_RESULT"
META_TAG = "LLAMA_BENCH_META"


def estimate_ttft_ms(bench_result: Union[Dict[str, Any], List[Any]]) -> Optional[float]:
    """Approximate TTFT as prefill latency plus mean single-token decode latency."""
    pp_avg_ns: Optional[int] = None
    tg_avg_ns: Optional[int] = None
    n_gen: Optional[int] = None

    entries: Iterable[Dict[str, Any]]
    if isinstance(bench_result, list):
        entries = (e for e in bench_result if isinstance(e, dict))
    elif isinstance(bench_result, dict):
        entries = [bench_result]
    else:
        return None

    for entry in entries:
        if entry.get("n_prompt", 0) > 0:
            pp_avg_ns = entry.get("avg_ns")
        elif entry.get("n_gen", 0) > 0:
            tg_avg_ns = entry.get("avg_ns")
            n_gen = entry.get("n_gen")

    if pp_avg_ns is None or tg_avg_ns is None or not n_gen:
        return None

    return pp_avg_ns / 1e6 + tg_avg_ns / n_gen / 1e6


def extract_metrics(bench_result: Union[Dict[str, Any], List[Any]]) -> Dict[str, Optional[Any]]:
    metrics: Dict[str, Optional[Any]] = {
        "pp_t_s": None,
        "tg_t_s": None,
        "ttft_ms": None,
        "model_size": None,
        "n_gpu_layers": None,
        "backend": None,
    }

    def extract_from_entry(entry: Dict[str, Any]) -> None:
        if metrics["n_gpu_layers"] is None:
            metrics["n_gpu_layers"] = entry.get("n_gpu_layers")

        if metrics["backend"] is None:
            if "backend" in entry:
                metrics["backend"] = entry["backend"]
            elif "gpu_info" in entry:
                gpu_info = entry["gpu_info"]
                if isinstance(gpu_info, str):
                    if "Metal" in gpu_info:
                        metrics["backend"] = "Metal"
                    elif "Vulkan" in gpu_info:
                        metrics["backend"] = "Vulkan"
                    elif "CUDA" in gpu_info or "NVIDIA" in gpu_info:
                        metrics["backend"] = "CUDA"
                    else:
                        metrics["backend"] = gpu_info
            elif metrics["n_gpu_layers"] == 0:
                metrics["backend"] = "CPU"

    entries: Iterable[Dict[str, Any]]
    if isinstance(bench_result, list):
        entries = bench_result
    elif isinstance(bench_result, dict):
        entries = [bench_result]
    else:
        return metrics

    for entry in entries:
        extract_from_entry(entry)
        if entry.get("n_prompt", 0) > 0:
            metrics["pp_t_s"] = entry.get("avg_ts")
            metrics["model_size"] = entry.get("model_size")
        elif entry.get("n_gen", 0) > 0:
            metrics["tg_t_s"] = entry.get("avg_ts")
            if metrics["model_size"] is None:
                metrics["model_size"] = entry.get("model_size")

    metrics["ttft_ms"] = estimate_ttft_ms(bench_result)
    return metrics


def parse_json_payload(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") or line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _extract_tag_payload(line: str, tag: str) -> Optional[str]:
    if tag not in line:
        return None
    payload = line.split(tag, 1)[-1].strip()
    if payload.startswith(":"):
        payload = payload[1:].strip()
    return payload or None


def _extract_json_array(text: str) -> Optional[List[Any]]:
    """Return the first top-level JSON array in text (llama-bench -o json output)."""
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    return None
    return None


def parse_logcat_text(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[Union[Dict[str, Any], List[Any]]]]:
    meta: Optional[Dict[str, Any]] = None
    result_parts: List[str] = []

    for line in text.splitlines():
        if META_TAG in line:
            payload = _extract_tag_payload(line, META_TAG)
            if payload:
                parsed = parse_json_payload(payload)
                if isinstance(parsed, dict):
                    meta = parsed

        if RESULT_TAG in line:
            payload = _extract_tag_payload(line, RESULT_TAG)
            if payload:
                result_parts.append(payload)
            continue

        # Fallback: single-line payloads where the tag was stripped by an exporter.
        if line.strip().startswith("{") and meta is None and "model_path" in line:
            parsed = parse_json_payload(line)
            if isinstance(parsed, dict):
                meta = parsed

    bench: Optional[Union[Dict[str, Any], List[Any]]] = None
    if result_parts:
        combined = "\n".join(result_parts)
        bench = _extract_json_array(combined)
        if bench is None:
            parsed = parse_json_payload(combined)
            if parsed is not None:
                bench = parsed

    return meta, bench


def infer_device_name(path: Path, meta: Optional[Dict[str, Any]]) -> str:
    if meta:
        manufacturer = meta.get("manufacturer")
        model = meta.get("device_model")
        if manufacturer and model:
            return f"{manufacturer} {model}"
        if model:
            return str(model)

    for part in reversed(path.parts):
        if re.search(r"-\d+-", part):
            return part
    return path.parent.name


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    candidate_files: List[Path] = []
    extracted_dir = results_dir / "extracted"

    for path in results_dir.rglob("*"):
        if not path.is_file():
            continue
        if _is_under(path, extracted_dir):
            continue
        name = path.name.lower()
        if name == "logcat" or name.endswith(".logcat") or "logcat" in name:
            candidate_files.append(path)
            continue
        if path.suffix.lower() in {".txt", ".log"}:
            candidate_files.append(path)

    meta_files = [
        path for path in results_dir.rglob("benchmark_meta.json")
        if not _is_under(path, extracted_dir)
    ]
    result_files = [
        path for path in results_dir.rglob("benchmark_result.json")
        if not _is_under(path, extracted_dir)
    ]

    seen_keys: set[str] = set()

    # Prefer shallow logcat paths so nested GCS download folders do not duplicate rows.
    for file_path in sorted(candidate_files, key=lambda p: (len(p.parts), str(p))):
        text = file_path.read_text(errors="ignore")
        if RESULT_TAG not in text and META_TAG not in text:
            continue

        meta, bench = parse_logcat_text(text)
        if bench is None:
            continue

        device = infer_device_name(file_path, meta)
        device_axis = file_path.parent.name
        model_path = meta.get("model_path") if meta else ""
        key = f"{device_axis}|{model_path}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        metrics = extract_metrics(bench)
        rows.append({
            "device": device,
            "manufacturer": meta.get("manufacturer") if meta else None,
            "device_model": meta.get("device_model") if meta else None,
            "android_release": meta.get("android_release") if meta else None,
            "model_path": meta.get("model_path") if meta else None,
            "model_file": meta.get("model_file") if meta else None,
            "repetitions": meta.get("repetitions") if meta else None,
            "prompt_tokens": meta.get("prompt_tokens") if meta else None,
            "gen_tokens": meta.get("gen_tokens") if meta else None,
            "n_gpu_layers": metrics.get("n_gpu_layers"),
            "backend": metrics.get("backend"),
            "pp_t_s": metrics.get("pp_t_s"),
            "tg_t_s": metrics.get("tg_t_s"),
            "ttft_ms": metrics.get("ttft_ms"),
            "model_size_bytes": metrics.get("model_size"),
            "source_file": str(file_path.relative_to(results_dir)),
        })

    for meta_path in meta_files:
        result_path = meta_path.parent / "benchmark_result.json"
        if not result_path.exists():
            continue

        meta = parse_json_payload(meta_path.read_text(errors="ignore"))
        bench = parse_json_payload(result_path.read_text(errors="ignore"))
        if not isinstance(meta, dict) or bench is None:
            continue

        device = infer_device_name(meta_path, meta)
        key = f"{meta_path.parent.name}|{meta.get('model_path')}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        metrics = extract_metrics(bench)
        rows.append({
            "device": device,
            "manufacturer": meta.get("manufacturer"),
            "device_model": meta.get("device_model"),
            "android_release": meta.get("android_release"),
            "model_path": meta.get("model_path"),
            "model_file": meta.get("model_file"),
            "repetitions": meta.get("repetitions"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "gen_tokens": meta.get("gen_tokens"),
            "n_gpu_layers": metrics.get("n_gpu_layers"),
            "backend": metrics.get("backend"),
            "pp_t_s": metrics.get("pp_t_s"),
            "tg_t_s": metrics.get("tg_t_s"),
            "ttft_ms": metrics.get("ttft_ms"),
            "model_size_bytes": metrics.get("model_size"),
            "source_file": str(meta_path.relative_to(results_dir)),
        })

    if not rows and result_files:
        for result_path in result_files:
            bench = parse_json_payload(result_path.read_text(errors="ignore"))
            if bench is None:
                continue
            metrics = extract_metrics(bench)
            rows.append({
                "device": result_path.parent.name,
                "manufacturer": None,
                "device_model": None,
                "android_release": None,
                "model_path": None,
                "model_file": result_path.name,
                "repetitions": None,
                "prompt_tokens": None,
                "gen_tokens": None,
                "n_gpu_layers": metrics.get("n_gpu_layers"),
                "backend": metrics.get("backend"),
                "pp_t_s": metrics.get("pp_t_s"),
                "tg_t_s": metrics.get("tg_t_s"),
                "model_size_bytes": metrics.get("model_size"),
                "source_file": str(result_path.relative_to(results_dir)),
            })

    return rows


def write_excel(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["model_path", "device"], na_position="last")
    df.to_excel(output_path, index=False)


def extract_artifacts(results_dir: Path, output_dir: Path) -> List[Path]:
    """Write parsed benchmark_meta.json / benchmark_result.json per device folder."""
    written: List[Path] = []
    seen_devices: set[str] = set()
    candidate_files: List[Path] = []
    skip_dir = results_dir / "extracted"

    for path in results_dir.rglob("*"):
        if not path.is_file() or _is_under(path, skip_dir):
            continue
        name = path.name.lower()
        if name == "logcat" or name.endswith(".logcat") or "logcat" in name:
            candidate_files.append(path)
        elif path.suffix.lower() in {".txt", ".log"} and (RESULT_TAG in path.read_text(errors="ignore")[:4096]):
            candidate_files.append(path)

    for logcat_path in sorted(candidate_files, key=lambda p: (len(p.parts), str(p))):
        text = logcat_path.read_text(errors="ignore")
        if RESULT_TAG not in text and META_TAG not in text:
            continue

        meta, bench = parse_logcat_text(text)
        if bench is None:
            continue

        device_key = logcat_path.parent.name
        if device_key in seen_devices:
            continue
        seen_devices.add(device_key)

        device_dir = output_dir / device_key
        device_dir.mkdir(parents=True, exist_ok=True)

        result_path = device_dir / "benchmark_result.json"
        result_path.write_text(json.dumps(bench, indent=2), encoding="utf-8")
        written.append(result_path)

        if meta is not None:
            meta_path = device_dir / "benchmark_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            written.append(meta_path)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Excel report from Firebase Test Lab results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing downloaded Test Lab artifacts/logcat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .xlsx path (default: <results-dir>/benchmark.xlsx)",
    )
    parser.add_argument(
        "--extract-json",
        action="store_true",
        help="Also write benchmark_meta.json / benchmark_result.json under <results-dir>/extracted/",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Directory for extracted JSON (default: <results-dir>/extracted)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        print("Download Test Lab output first, e.g.:")
        print("  ./firebase/fetch_firebase_results.sh gs://qvac-test-ftl-results/bench/MODEL/RUN_ID")
        return 1

    extracted: List[Path] = []
    json_dir = args.json_dir or (args.results_dir / "extracted")
    if args.extract_json:
        extracted = extract_artifacts(args.results_dir, json_dir)
        for path in extracted:
            print(f"Wrote {path}")

    rows = collect_rows(args.results_dir)
    if not rows:
        print(f"No benchmark rows found under {args.results_dir}")
        if extracted:
            return 0
        return 1

    output_path = args.output or (args.results_dir / "benchmark.xlsx")
    write_excel(rows, output_path)
    print(f"Wrote {len(rows)} row(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
