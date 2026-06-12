#!/usr/bin/env bash
set -euo pipefail

DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$DEVICEFARM_DIR/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/devicefarm"

if [[ -f "$DEVICEFARM_DIR/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$DEVICEFARM_DIR/config.env"
fi

REGION="${AWS_REGION:-us-west-2}"
PROJECT_ARN="${AWS_PROJECT_ARN:?Set AWS_PROJECT_ARN in devicefarm/config.env}"
DEVICE_ARN="${AWS_DEVICE_ARN:?Set AWS_DEVICE_ARN in devicefarm/config.env}"
REPETITIONS="${REPETITIONS:-${DEFAULT_REPETITIONS:-5}}"
TIMEOUT_MIN="${TEST_TIMEOUT_MINUTES:-60}"

MODEL_PATH="${1:-qwen3-0.6B/Qwen3-0.6B-TQ2_0_Tether.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-${DEFAULT_N_GPU_LAYERS:-99}}"

IPA_PATH="${IPA_PATH:-$BUILD_DIR/llama.swiftui_bench.ipa}"
TEST_ZIP="${TEST_ZIP:-$BUILD_DIR/llama.swiftui_benchUITests.zip}"

if [[ ! -f "$IPA_PATH" || ! -f "$TEST_ZIP" ]]; then
  echo "Missing build artifacts. Run ./devicefarm/build_devicefarm.sh first." >&2
  exit 1
fi

SAFE_MODEL="${MODEL_PATH//\//_}"
if [[ "$N_GPU_LAYERS" == "0" ]]; then
  SAFE_MODEL="${SAFE_MODEL}_cpu"
else
  SAFE_MODEL="${SAFE_MODEL}_gpu"
fi
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$DEVICEFARM_DIR/devicefarm_results/${SAFE_MODEL}_${RUN_ID}"
mkdir -p "$RESULTS_DIR"

wait_for_upload() {
  local arn="$1"
  if [[ -z "$arn" || "$arn" == "None" ]]; then
    echo "Upload did not return an ARN (check AWS credentials)" >&2
    exit 1
  fi
  local status=""
  while [[ "$status" != "SUCCEEDED" ]]; do
    if ! status="$(aws devicefarm get-upload --region "$REGION" --arn "$arn" --query upload.status --output text 2>&1)"; then
      echo "$status" >&2
      if [[ "$status" == *ExpiredToken* ]]; then
        echo "AWS credentials expired. Update access keys in ~/.aws/credentials and retry." >&2
      fi
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
    echo "$create_out" >&2
    if [[ "$create_out" == *ExpiredToken* ]]; then
      echo "AWS credentials expired. Refresh credentials and retry." >&2
    fi
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

SCHEDULE_JSON="$RESULTS_DIR/schedule_input.json"
python3 - <<PY > "$SCHEDULE_JSON"
import json
print(json.dumps({
    "projectArn": "$PROJECT_ARN",
    "appArn": "$APP_UPLOAD_ARN",
    "deviceSelectionConfiguration": {
        "filters": [{
            "attribute": "ARN",
            "operator": "EQUALS",
            "values": ["$DEVICE_ARN"],
        }],
        "maxDevices": 1,
    },
    "test": {
        "type": "XCTEST_UI",
        "testPackageArn": "$TEST_UPLOAD_ARN",
        "parameters": {
            "model_path": "$MODEL_PATH",
            "n_gpu_layers": "$N_GPU_LAYERS",
            "repetitions": "$REPETITIONS",
            "skip_download": "false",
        },
    },
    "executionConfiguration": {
        "jobTimeoutMinutes": int("$TIMEOUT_MIN"),
        "videoCapture": True,
    },
}))
PY

echo "Scheduling Device Farm run (model=$MODEL_PATH, n_gpu_layers=$N_GPU_LAYERS)..." >&2
RUN_JSON="$RESULTS_DIR/schedule_run.json"
aws devicefarm schedule-run \
  --region "$REGION" \
  --cli-input-json "file://$SCHEDULE_JSON" \
  > "$RUN_JSON"

RUN_ARN="$(python3 -c "import json; print(json.load(open('$RUN_JSON'))['run']['arn'])")"
echo "Run ARN: $RUN_ARN"
echo "$RUN_ARN" > "$RESULTS_DIR/run_arn.txt"

echo "Waiting for run to complete (timeout ${TIMEOUT_MIN}m)..."
DEADLINE=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
STATUS=""
while [[ "$STATUS" != "COMPLETED" ]]; do
  if (( $(date +%s) > DEADLINE )); then
    echo "Timed out waiting for run" >&2
    exit 1
  fi
  if ! poll="$(aws devicefarm get-run --region "$REGION" --arn "$RUN_ARN" --query 'run.[status,result]' --output text 2>&1)"; then
    echo "$poll" >&2
    if [[ "$poll" == *ExpiredToken* ]]; then
      echo "AWS credentials expired while waiting. Run may still finish on Device Farm." >&2
      echo "Refresh access keys in ~/.aws/credentials, then fetch results:" >&2
      echo "  ./devicefarm/fetch_devicefarm_results.sh $RUN_ARN" >&2
    fi
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

while IFS=$'\t' read -r name type url; do
  [[ -z "${name:-}" ]] && continue
  safe_name="${name//\//_}"
  echo "  $name"
  curl --fail --silent --show-error -L "$url" -o "$RESULTS_DIR/${safe_name}"
done < "$RESULTS_DIR/artifacts.txt"

python3 "$DEVICEFARM_DIR/benchmark_devicefarm.py" \
  --results-dir "$RESULTS_DIR" \
  --output "$RESULTS_DIR/benchmark.xlsx"

echo "Results: $RESULTS_DIR"
echo "Excel:   $RESULTS_DIR/benchmark.xlsx"

if [[ "$RESULT" != "PASSED" ]]; then
  exit 1
fi
