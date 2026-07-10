#!/usr/bin/env bash
# Run GPU+CPU Device Farm benchmarks for every model in s3-model-list.txt.
# Waits for each run, fetches artifacts, and appends metrics to a summary TSV.
set -euo pipefail

DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$DEVICEFARM_DIR/aws_creds.sh"

_AWS_DEVICE_ARN_OVERRIDE="${AWS_DEVICE_ARN:-}"
_AWS_PROJECT_ARN_OVERRIDE="${AWS_PROJECT_ARN:-}"
if [[ -f "$DEVICEFARM_DIR/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$DEVICEFARM_DIR/config.env"
fi
[[ -n "$_AWS_DEVICE_ARN_OVERRIDE" ]] && AWS_DEVICE_ARN="$_AWS_DEVICE_ARN_OVERRIDE"
[[ -n "$_AWS_PROJECT_ARN_OVERRIDE" ]] && AWS_PROJECT_ARN="$_AWS_PROJECT_ARN_OVERRIDE"

MODEL_FILE="${1:-$DEVICEFARM_DIR/s3-model-list.txt}"
RUN_CPU="${RUN_CPU:-1}"
RUN_GPU="${RUN_GPU:-1}"

export AWS_PROFILE="${AWS_PROFILE:-833707431398_Tether_DeviceFarm_FullAccess}"
export AWS_S3_PROFILE="${AWS_S3_PROFILE:-833707431398_Tether_S3_RW_tether-ai-dev}"

require_aws_profile "$AWS_PROFILE" "Device Farm" >/dev/null
require_aws_profile "$AWS_S3_PROFILE" "S3 presign" >/dev/null

SUMMARY_DIR="${SUMMARY_DIR:-$DEVICEFARM_DIR/devicefarm_results/s3_matrix_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SUMMARY_DIR"
SUMMARY_TSV="$SUMMARY_DIR/results.tsv"
echo -e "model\tbackend\trun_arn\tresult\tpp_t_s\ttg_t_s\tttft_ms\tresults_dir" > "$SUMMARY_TSV"

MODELS=()
while IFS= read -r line; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  MODELS+=("$line")
done < "$MODEL_FILE"

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No models in $MODEL_FILE" >&2
  exit 1
fi

extract_metrics() {
  local results_dir="$1"
  python3 - "$results_dir" <<'PY'
import json, re, sys
from pathlib import Path

results_dir = Path(sys.argv[1])
text = ""
for path in sorted(results_dir.rglob("*")):
    if not path.is_file():
        continue
    name = path.name.lower()
    if "syslog" not in name:
        continue
    chunk = path.read_text(errors="ignore")
    if "LLAMA_BENCH_RESULT" in chunk:
        text = chunk
        break

if not text:
    print("\t\t")
    raise SystemExit(0)

m = re.search(r"LLAMA_BENCH_RESULT (\[.*?\])", text)
if not m:
    print("\t\t")
    raise SystemExit(0)

bench = json.loads(m.group(1))
pp = next(e for e in bench if e.get("n_prompt", 0) > 0)
tg = next(e for e in bench if e.get("n_gen", 0) > 0)
ttft = pp["avg_ns"] / 1e6 + tg["avg_ns"] / tg["n_gen"] / 1e6
print(f"{pp['avg_ts']:.2f}\t{tg['avg_ts']:.2f}\t{ttft:.2f}")
PY
}

run_one() {
  local layers="$1"
  local model="$2"
  local backend="$3"

  echo ""
  echo "========================================"
  echo "[$backend] $model (n_gpu_layers=$layers)"
  echo "========================================"

  local output results_dir run_arn result metrics
  if ! output="$(N_GPU_LAYERS="$layers" "$DEVICEFARM_DIR/run_devicefarm_bench.sh" --wait "$model" 2>&1)"; then
    echo "$output" >&2
    echo -e "${model}\t${backend}\t\tFAILED\t\t\t" >> "$SUMMARY_TSV"
    return 1
  fi

  echo "$output"
  run_arn="$(echo "$output" | awk '/^Run ARN: / { print $3; exit }')"
  results_dir="$(echo "$output" | awk '/^Results: / { print $2; exit }')"
  result="$(echo "$output" | awk '/^Final result: / { print $3; exit }')"
  metrics="$(extract_metrics "$results_dir")"
  IFS=$'\t' read -r pp_t_s tg_t_s ttft_ms <<< "$metrics"

  echo -e "${model}\t${backend}\t${run_arn}\t${result}\t${pp_t_s}\t${tg_t_s}\t${ttft_ms}\t${results_dir}" >> "$SUMMARY_TSV"
  echo "Recorded: pp=$pp_t_s tg=$tg_t_s ttft=$ttft_ms ms"
}

total="${#MODELS[@]}"
idx=0
for model in "${MODELS[@]}"; do
  idx=$((idx + 1))
  echo ""
  echo ">>> Model $idx/$total: $model"

  if [[ "$RUN_GPU" -eq 1 ]]; then
    run_one 99 "$model" gpu || true
  fi
  if [[ "$RUN_CPU" -eq 1 ]]; then
    run_one 0 "$model" cpu || true
  fi
done

python3 "$DEVICEFARM_DIR/benchmark_devicefarm.py" \
  --results-dir "$DEVICEFARM_DIR/devicefarm_results" \
  --output "$SUMMARY_DIR/benchmark_matrix.xlsx" || true

COMPARISON_CSV="$SUMMARY_DIR/comparison.csv"
DEVICE_LABEL="${DEVICE_LABEL:-iPhone 17 Pro Max (Metal)}"
python3 "$DEVICEFARM_DIR/results_to_comparison_csv.py" \
  "$SUMMARY_TSV" -o "$COMPARISON_CSV" -d "$DEVICE_LABEL" || true

echo ""
echo "Summary TSV:  $SUMMARY_TSV"
echo "Summary XLSX: $SUMMARY_DIR/benchmark_matrix.xlsx"
echo "Comparison:   $COMPARISON_CSV"
cat "$COMPARISON_CSV" 2>/dev/null || cat "$SUMMARY_TSV"
