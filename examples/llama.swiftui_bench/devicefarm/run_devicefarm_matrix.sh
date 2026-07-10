#!/usr/bin/env bash
set -euo pipefail

DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$DEVICEFARM_DIR/aws_creds.sh"

PROFILE="${AWS_PROFILE:-devicefarm}"
WAIT_SEC=0
WAIT_FOR_RUN=0

usage() {
  cat <<'EOF'
Usage: run_devicefarm_matrix.sh [OPTIONS]

Schedule the full model × backend matrix on AWS Device Farm.
Returns immediately after scheduling unless --wait-for-runs is passed.

Options:
  --s3               Run all .gguf models from S3 (see config.env S3_MODEL_*).
                     Refreshes devicefarm/s3-model-list.txt, then runs GPU+CPU
                     iteratively with --wait-for-runs (presigned URL per run).
  --cpu-only         Run CPU (n_gpu_layers=0) only
  --gpu-only         Run GPU (n_gpu_layers=99) only
  --skip-build       Skip ./devicefarm/build_devicefarm.sh (ignored with --s3)
  --wait-for-runs    Wait for each run to finish and collect results
  --wait SECONDS     Poll for valid AWS credentials before starting (default: 0)
  -f FILE            Model list (default: model-matrix.txt; with --s3: s3-model-list.txt)
  -h, --help

Environment:
  AWS_PROFILE        Device Farm profile (default: 833707431398_Tether_DeviceFarm_FullAccess)
  AWS_S3_PROFILE     S3 presign profile (default: 833707431398_Tether_S3_RW_tether-ai-dev)
  S3_MODEL_BUCKET    Bucket in config.env (default: tether-ai-dev)
  S3_MODEL_PREFIX    Prefix in config.env (default: models/qwen3-checkpoints/gguf/)
EOF
}

check_creds() {
  aws sts get-caller-identity --profile "$PROFILE" >/dev/null 2>&1
}

require_aws_creds() {
  if check_creds; then
    require_aws_profile "$PROFILE" "Device Farm" >/dev/null
    return 0
  fi
  if [[ "$WAIT_SEC" -le 0 ]]; then
    fail_expired_creds "$PROFILE" "Device Farm" "NoCredentials: unable to locate credentials"
  fi
  echo "Waiting up to ${WAIT_SEC}s for AWS credentials (profile: $PROFILE)..."
  local deadline=$(( $(date +%s) + WAIT_SEC ))
  while ! check_creds; do
    if (( $(date +%s) >= deadline )); then
      fail_expired_creds "$PROFILE" "Device Farm" "Timed out waiting for credentials"
    fi
    sleep 3
  done
  require_aws_profile "$PROFILE" "Device Farm" >/dev/null
}

MATRIX_FILE="$DEVICEFARM_DIR/model-matrix.txt"
S3_MODELS=0
EXPLICIT_MODEL_FILE=0
SKIP_BUILD=0
RUN_CPU=1
RUN_GPU=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --s3) S3_MODELS=1; shift ;;
    --cpu-only) RUN_GPU=0; shift ;;
    --gpu-only) RUN_CPU=0; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --wait-for-runs) WAIT_FOR_RUN=1; shift ;;
    --wait) WAIT_SEC="$2"; shift 2 ;;
    -f) MATRIX_FILE="$2"; EXPLICIT_MODEL_FILE=1; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

export AWS_PROFILE="$PROFILE"
require_aws_creds

if [[ "$S3_MODELS" -eq 1 ]]; then
  if [[ "$EXPLICIT_MODEL_FILE" -eq 0 ]]; then
    MATRIX_FILE="$DEVICEFARM_DIR/s3-model-list.txt"
    echo "Refreshing S3 model list -> $MATRIX_FILE" >&2
    "$DEVICEFARM_DIR/sync_s3_model_list.sh" "$MATRIX_FILE"
  else
    echo "Using model list (no S3 refresh): $MATRIX_FILE" >&2
  fi
  export RUN_CPU RUN_GPU
  exec "$DEVICEFARM_DIR/run_s3_matrix_iterative.sh" "$MATRIX_FILE"
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  "$DEVICEFARM_DIR/build_devicefarm.sh"
fi

SUMMARY_DIR="$DEVICEFARM_DIR/devicefarm_results/matrix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SUMMARY_DIR"

MODELS=()
while IFS= read -r line; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  MODELS+=("$line")
done < "$MATRIX_FILE"

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No models in $MATRIX_FILE" >&2
  exit 1
fi

RUN_ARNS_FILE="$SUMMARY_DIR/scheduled_runs.txt"
: > "$RUN_ARNS_FILE"

bench_args=()
if [[ "$WAIT_FOR_RUN" -eq 1 ]]; then
  bench_args+=(--wait)
fi

schedule_bench() {
  local layers="$1"
  local model="$2"
  local label="$3"
  local output run_arn

  output="$(N_GPU_LAYERS="$layers" "$DEVICEFARM_DIR/run_devicefarm_bench.sh" "${bench_args[@]}" "$model" 2>&1)" || {
    echo "$output" >&2
    return 1
  }
  echo "$output"
  run_arn="$(echo "$output" | awk '/^Run ARN: / { print $3; exit }')"
  if [[ -n "$run_arn" ]]; then
    echo "$label $model $run_arn" >> "$RUN_ARNS_FILE"
  fi
}

for model in "${MODELS[@]}"; do
  if [[ "$RUN_GPU" -eq 1 ]]; then
    echo "=== GPU: $model ==="
    schedule_bench 99 "$model" gpu || exit 1
  fi
  if [[ "$RUN_CPU" -eq 1 ]]; then
    echo "=== CPU: $model ==="
    schedule_bench 0 "$model" cpu || exit 1
  fi
done

if [[ "$WAIT_FOR_RUN" -eq 0 ]]; then
  echo ""
  echo "All runs scheduled. Script exiting immediately."
  echo "Scheduled run ARNs: $RUN_ARNS_FILE"
  echo "Fetch a run when complete:"
  echo "  ./devicefarm/fetch_devicefarm_results.sh <RUN_ARN>"
  exit 0
fi

python3 "$DEVICEFARM_DIR/benchmark_devicefarm.py" \
  --results-dir "$DEVICEFARM_DIR/devicefarm_results" \
  --output "$SUMMARY_DIR/benchmark_matrix.xlsx"

echo "Matrix Excel: $SUMMARY_DIR/benchmark_matrix.xlsx"
