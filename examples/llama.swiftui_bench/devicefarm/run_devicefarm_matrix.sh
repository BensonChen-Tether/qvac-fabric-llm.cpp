#!/usr/bin/env bash
set -euo pipefail

DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${AWS_PROFILE:-devicefarm}"
WAIT_SEC=0

usage() {
  cat <<'EOF'
Usage: run_devicefarm_matrix.sh [OPTIONS]

Run the full model × backend matrix on AWS Device Farm.

Options:
  --cpu-only     Run CPU (n_gpu_layers=0) only
  --gpu-only     Run GPU (n_gpu_layers=99) only
  --skip-build   Skip ./devicefarm/build_devicefarm.sh
  --wait SECONDS Poll for valid AWS credentials before starting (default: 0 = fail fast)
  -f FILE        Model list (default: devicefarm/model-matrix.txt)
  -h, --help

Environment:
  AWS_PROFILE    AWS credentials profile (default: devicefarm)
  N_GPU_LAYERS   Override backend for single run_devicefarm_bench.sh calls
EOF
}

check_creds() {
  aws sts get-caller-identity --profile "$PROFILE" >/dev/null 2>&1
}

require_aws_creds() {
  if check_creds; then
    aws sts get-caller-identity --profile "$PROFILE"
    return 0
  fi
  if [[ "$WAIT_SEC" -le 0 ]]; then
    echo "AWS credentials invalid or expired for profile '$PROFILE'." >&2
    echo "Update ~/.aws/credentials with your access key (not SSO):" >&2
    echo "  [$PROFILE]" >&2
    echo "  aws_access_key_id = AKIA... or ASIA..." >&2
    echo "  aws_secret_access_key = ..." >&2
    echo "  aws_session_token = ...   # required when access key starts with ASIA" >&2
    echo "Verify, then rerun:" >&2
    echo "  aws sts get-caller-identity --profile $PROFILE" >&2
    echo "  AWS_PROFILE=$PROFILE $0" >&2
    exit 1
  fi
  echo "Waiting up to ${WAIT_SEC}s for AWS credentials (profile: $PROFILE)..."
  local deadline=$(( $(date +%s) + WAIT_SEC ))
  while ! check_creds; do
    if (( $(date +%s) >= deadline )); then
      echo "Timed out after ${WAIT_SEC}s. Refresh credentials and retry." >&2
      exit 1
    fi
    sleep 3
  done
  aws sts get-caller-identity --profile "$PROFILE"
}

MATRIX_FILE="$DEVICEFARM_DIR/model-matrix.txt"
SKIP_BUILD=0
RUN_CPU=1
RUN_GPU=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu-only) RUN_GPU=0; shift ;;
    --gpu-only) RUN_CPU=0; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --wait) WAIT_SEC="$2"; shift 2 ;;
    -f) MATRIX_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

export AWS_PROFILE="$PROFILE"
require_aws_creds

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

for model in "${MODELS[@]}"; do
  if [[ "$RUN_GPU" -eq 1 ]]; then
    echo "=== GPU: $model ==="
    if ! N_GPU_LAYERS=99 "$DEVICEFARM_DIR/run_devicefarm_bench.sh" "$model"; then
      echo "Matrix aborted after GPU failure for $model" >&2
      exit 1
    fi
  fi
  if [[ "$RUN_CPU" -eq 1 ]]; then
    echo "=== CPU: $model ==="
    if ! N_GPU_LAYERS=0 "$DEVICEFARM_DIR/run_devicefarm_bench.sh" "$model"; then
      echo "Matrix aborted after CPU failure for $model" >&2
      exit 1
    fi
  fi
done

python3 "$DEVICEFARM_DIR/benchmark_devicefarm.py" \
  --results-dir "$DEVICEFARM_DIR/devicefarm_results" \
  --output "$SUMMARY_DIR/benchmark_matrix.xlsx"

echo "Matrix Excel: $SUMMARY_DIR/benchmark_matrix.xlsx"
