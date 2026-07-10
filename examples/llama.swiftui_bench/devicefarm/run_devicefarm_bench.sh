#!/usr/bin/env bash
set -euo pipefail

DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$DEVICEFARM_DIR/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/devicefarm"

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

usage() {
  cat <<'EOF'
Usage: run_devicefarm_bench.sh [OPTIONS] [MODEL_FILENAME]

Schedule an iOS benchmark on AWS Device Farm. Returns immediately after
scheduling unless --wait is passed.

Options:
  --wait         Wait for run to finish, download artifacts, write Excel
  -h, --help     Show this help

Environment:
  AWS_PROFILE         Device Farm profile (default: devicefarm)
  AWS_S3_PROFILE      S3 presign profile (default: 833707431398_Tether_S3_RW_tether-ai-dev)
  WAIT_FOR_RUN=1      Same as --wait
  N_GPU_LAYERS        99=GPU, 0=CPU (default: 99)
EOF
}

WAIT_FOR_RUN="${WAIT_FOR_RUN:-0}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait) WAIT_FOR_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

export AWS_PROFILE="${AWS_PROFILE:-833707431398_Tether_DeviceFarm_FullAccess}"
REGION="${AWS_REGION:-us-west-2}"
PROJECT_ARN="${AWS_PROJECT_ARN:?Set AWS_PROJECT_ARN in devicefarm/config.env}"
DEVICE_ARN="${AWS_DEVICE_ARN:?Set AWS_DEVICE_ARN in devicefarm/config.env}"
REPETITIONS="${REPETITIONS:-${DEFAULT_REPETITIONS:-5}}"
TIMEOUT_MIN="${TEST_TIMEOUT_MINUTES:-60}"

MODEL_PATH="${ARGS[0]:-qwen3_1p7b-epoch01-TQ2_0.gguf}"
MODEL_FILENAME="$(basename "$MODEL_PATH")"
N_GPU_LAYERS="${N_GPU_LAYERS:-${DEFAULT_N_GPU_LAYERS:-99}}"

SAFE_MODEL="${MODEL_PATH//\//_}"
if [[ "$N_GPU_LAYERS" == "0" ]]; then
  SAFE_MODEL="${SAFE_MODEL}_cpu"
else
  SAFE_MODEL="${SAFE_MODEL}_gpu"
fi
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$DEVICEFARM_DIR/devicefarm_results/${SAFE_MODEL}_${RUN_ID}"
mkdir -p "$RESULTS_DIR"

S3_MODEL_BUCKET="${S3_MODEL_BUCKET:-tether-ai-dev}"
S3_MODEL_PREFIX="${S3_MODEL_PREFIX:-models/qwen3-checkpoints/gguf/}"
S3_REGION="${S3_REGION:-eu-central-1}"
AWS_S3_PROFILE="${AWS_S3_PROFILE:-833707431398_Tether_S3_RW_tether-ai-dev}"
PRESIGN_EXPIRES="${PRESIGN_EXPIRES:-7200}"

echo "Checking AWS credentials..." >&2
require_aws_profile "$AWS_PROFILE" "Device Farm" >/dev/null
require_aws_profile "$AWS_S3_PROFILE" "S3 presign" >/dev/null

presign_model_url() {
  local model_filename="$1"
  local s3_uri="s3://${S3_MODEL_BUCKET}/${S3_MODEL_PREFIX}${model_filename}"
  local presign_out

  if ! presign_out="$(AWS_PROFILE="$AWS_S3_PROFILE" aws s3 presign "$s3_uri" --expires-in "$PRESIGN_EXPIRES" --region "$S3_REGION" 2>&1)"; then
    echo "Cannot access model due to device farm error: failed to presign $s3_uri" >&2
    handle_aws_error "$AWS_S3_PROFILE" "S3 presign" "$presign_out" || true
    exit 1
  fi

  if [[ -z "$presign_out" || "$presign_out" != http* ]]; then
    echo "Cannot access model due to device farm error: invalid presign URL for $s3_uri" >&2
    exit 1
  fi

  echo "$presign_out"
}

echo "Presigning S3 model URL for $MODEL_FILENAME (profile=$AWS_S3_PROFILE)..." >&2
MODEL_DOWNLOAD_URL="$(presign_model_url "$MODEL_FILENAME")"
export MODEL_PATH MODEL_FILENAME MODEL_DOWNLOAD_URL N_GPU_LAYERS REPETITIONS

echo "Building app IPA with baked presigned URL (n_gpu_layers=$N_GPU_LAYERS)..." >&2
"$DEVICEFARM_DIR/build_devicefarm.sh" --skip-xcframework

IPA_PATH="${IPA_PATH:-$BUILD_DIR/llama.swiftui_bench.ipa}"
TEST_ZIP="${TEST_ZIP:-$BUILD_DIR/llama.swiftui_benchUITests.zip}"

if [[ ! -f "$IPA_PATH" || ! -f "$TEST_ZIP" ]]; then
  echo "Missing build artifacts after build_devicefarm.sh" >&2
  exit 1
fi

cp "$DEVICEFARM_DIR/devicefarm_bench.json" "$RESULTS_DIR/devicefarm_bench.json"

wait_for_upload() {
  local arn="$1"
  if [[ -z "$arn" || "$arn" == "None" ]]; then
    echo "Upload did not return an ARN (check AWS credentials)" >&2
    exit 1
  fi
  local status=""
  while [[ "$status" != "SUCCEEDED" ]]; do
    if ! status="$(aws devicefarm get-upload --region "$REGION" --arn "$arn" --query upload.status --output text 2>&1)"; then
      handle_aws_error "$AWS_PROFILE" "Device Farm upload" "$status" || true
      exit 1
    fi
    if [[ "$status" == "FAILED" ]]; then
      echo "Upload failed: $arn" >&2
      aws devicefarm get-upload --region "$REGION" --arn "$arn" >&2
      exit 1
    fi
    sleep 5
  done
}

upload_file() {
  local file_path="$1"
  local upload_type="$2"
  local name
  name="$(basename "$file_path")"
  local arn url
  local create_out
  if ! create_out="$(aws devicefarm create-upload \
    --region "$REGION" \
    --project-arn "$PROJECT_ARN" \
    --name "$name" \
    --type "$upload_type" \
    --query 'upload.[arn,url]' \
    --output text 2>&1)"; then
    handle_aws_error "$AWS_PROFILE" "Device Farm upload" "$create_out" || true
    exit 1
  fi
  read -r arn url <<< "$create_out"
  echo "Uploading $name ..." >&2
  curl --fail --silent --show-error -T "$file_path" "$url" >/dev/null
  wait_for_upload "$arn"
  aws devicefarm get-upload --region "$REGION" --arn "$arn" --query upload.arn --output text
}

echo "Uploading app IPA..." >&2
APP_UPLOAD_ARN="$(upload_file "$IPA_PATH" "IOS_APP")"
echo "Uploading XCUITest zip..." >&2
TEST_UPLOAD_ARN="$(upload_file "$TEST_ZIP" "XCTEST_UI_TEST_PACKAGE")"

export PROJECT_ARN DEVICE_ARN APP_UPLOAD_ARN TEST_UPLOAD_ARN TIMEOUT_MIN

SCHEDULE_JSON="$RESULTS_DIR/schedule_input.json"
python3 - <<'PY' > "$SCHEDULE_JSON"
import json
import os

print(json.dumps({
    "projectArn": os.environ["PROJECT_ARN"],
    "appArn": os.environ["APP_UPLOAD_ARN"],
    "deviceSelectionConfiguration": {
        "filters": [{
            "attribute": "ARN",
            "operator": "EQUALS",
            "values": [os.environ["DEVICE_ARN"]],
        }],
        "maxDevices": 1,
    },
    "test": {
        "type": "XCTEST_UI",
        "testPackageArn": os.environ["TEST_UPLOAD_ARN"],
    },
    "executionConfiguration": {
        "jobTimeoutMinutes": int(os.environ["TIMEOUT_MIN"]),
        "videoCapture": True,
    },
}))
PY

echo "Scheduling Device Farm run (model=$MODEL_PATH, n_gpu_layers=$N_GPU_LAYERS)..." >&2
RUN_JSON="$RESULTS_DIR/schedule_run.json"
if ! aws devicefarm schedule-run \
  --region "$REGION" \
  --cli-input-json "file://$SCHEDULE_JSON" \
  > "$RUN_JSON" 2> "$RESULTS_DIR/schedule_error.txt"; then
  schedule_err="$(cat "$RESULTS_DIR/schedule_error.txt")"
  handle_aws_error "$AWS_PROFILE" "Device Farm schedule" "$schedule_err" || true
  cat "$RESULTS_DIR/schedule_error.txt" >&2
  exit 1
fi

RUN_ARN="$(python3 -c "import json; print(json.load(open('$RUN_JSON'))['run']['arn'])")"
echo "Run ARN: $RUN_ARN"
echo "$RUN_ARN" > "$RESULTS_DIR/run_arn.txt"

if [[ "$WAIT_FOR_RUN" -eq 0 ]]; then
  echo ""
  echo "Run scheduled. Script exiting immediately."
  echo "Fetch results when the run completes:"
  echo "  ./devicefarm/fetch_devicefarm_results.sh $RUN_ARN $RESULTS_DIR"
  exit 0
fi

echo "Waiting for run to complete (timeout ${TIMEOUT_MIN}m)..."
DEADLINE=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
STATUS=""
while [[ "$STATUS" != "COMPLETED" ]]; do
  if (( $(date +%s) > DEADLINE )); then
    echo "Timed out waiting for run" >&2
    exit 1
  fi
  if ! poll="$(aws devicefarm get-run --region "$REGION" --arn "$RUN_ARN" --query 'run.[status,result]' --output text 2>&1)"; then
    handle_aws_error "$AWS_PROFILE" "Device Farm poll" "$poll" || true
    echo "Run may still finish on Device Farm. Fetch later:" >&2
    echo "  ./devicefarm/fetch_devicefarm_results.sh $RUN_ARN $RESULTS_DIR" >&2
    exit 1
  fi
  read -r STATUS RESULT <<< "$poll"
  echo "  status=$STATUS result=$RESULT"
  if [[ "$STATUS" == "COMPLETED" ]]; then
    break
  fi
  sleep 30
done

RESULT="$(aws devicefarm get-run --region "$REGION" --arn "$RUN_ARN" --query run.result --output text)"
echo "Final result: $RESULT"
echo "$RESULT" > "$RESULTS_DIR/run_result.txt"

JOB_ARN="$(aws devicefarm list-jobs --region "$REGION" --arn "$RUN_ARN" --query 'jobs[0].arn' --output text)"
if [[ -z "$JOB_ARN" || "$JOB_ARN" == "None" ]]; then
  echo "No jobs found for run (result may be SKIPPED). Check device ARN in config.env" >&2
  exit 1
fi
echo "$JOB_ARN" > "$RESULTS_DIR/job_arn.txt"

echo "Downloading artifacts..."
aws devicefarm list-artifacts \
  --region "$REGION" \
  --arn "$JOB_ARN" \
  --type FILE \
  --query 'artifacts[*].[name,type,url]' \
  --output text > "$RESULTS_DIR/artifacts.txt"

artifact_idx=0
while IFS=$'\t' read -r name type url; do
  [[ -z "${name:-}" ]] && continue
  artifact_idx=$((artifact_idx + 1))
  safe_name="${name//\//_}"
  echo "  $name"
  curl --fail --silent --show-error -L "$url" -o "$RESULTS_DIR/${artifact_idx}_${safe_name}"
done < "$RESULTS_DIR/artifacts.txt"

python3 "$DEVICEFARM_DIR/benchmark_devicefarm.py" \
  --results-dir "$RESULTS_DIR" \
  --output "$RESULTS_DIR/benchmark.xlsx"

echo "Results: $RESULTS_DIR"
echo "Excel:   $RESULTS_DIR/benchmark.xlsx"

if [[ "$RESULT" != "PASSED" ]]; then
  exit 1
fi
