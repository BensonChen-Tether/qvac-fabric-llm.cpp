#!/usr/bin/env bash
set -euo pipefail

DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$DEVICEFARM_DIR/aws_creds.sh"

PROFILE="${AWS_PROFILE:-833707431398_Tether_DeviceFarm_FullAccess}"
WAIT_SEC=0

usage() {
  cat <<'EOF'
Usage: run_devicefarm_matrix.sh [OPTIONS]

Schedule the full S3 model × backend matrix on AWS Device Farm (Android).

Options:
  --s3               Run all .gguf models from S3 (see config.env S3_MODEL_*).
  --cpu-only         Run CPU (n_gpu_layers=0) only
  --gpu-only         Run GPU (n_gpu_layers=999) only
  --wait SECONDS     Poll for valid AWS credentials before starting (default: 0)
  -f FILE            Model list (default with --s3: s3-model-list.txt)
  -h, --help

Environment:
  AWS_PROFILE        Device Farm profile
  AWS_S3_PROFILE     S3 presign profile
  AWS_DEVICE_ARN     Target device (override config.env)
  DEVICE_LABEL       Label for comparison.csv
  SUMMARY_DIR        Output directory (optional)
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

MATRIX_FILE="$DEVICEFARM_DIR/s3-model-list.txt"
S3_MODELS=0
EXPLICIT_MODEL_FILE=0
RUN_CPU=1
RUN_GPU=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --s3) S3_MODELS=1; shift ;;
    --cpu-only) RUN_GPU=0; shift ;;
    --gpu-only) RUN_CPU=0; shift ;;
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

echo "Use --s3 to run the S3 model matrix." >&2
exit 1
