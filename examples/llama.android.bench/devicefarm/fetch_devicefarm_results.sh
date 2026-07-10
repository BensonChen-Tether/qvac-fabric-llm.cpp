#!/usr/bin/env bash
set -euo pipefail

DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: fetch_devicefarm_results.sh RUN_ARN [OUTPUT_DIR]

Download Device Farm job artifacts and generate benchmark.xlsx for one run.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

if [[ -f "$DEVICEFARM_DIR/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$DEVICEFARM_DIR/config.env"
fi

REGION="${AWS_REGION:-us-west-2}"
RUN_ARN="$1"
OUTPUT_DIR="${2:-$DEVICEFARM_DIR/devicefarm_results/fetch_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"

JOB_ARN="$(aws devicefarm list-jobs --region "$REGION" --arn "$RUN_ARN" --query 'jobs[0].arn' --output text)"
echo "$RUN_ARN" > "$OUTPUT_DIR/run_arn.txt"
echo "$JOB_ARN" > "$OUTPUT_DIR/job_arn.txt"

aws devicefarm list-artifacts \
  --region "$REGION" \
  --arn "$JOB_ARN" \
  --type FILE \
  --query 'artifacts[*].[name,type,url]' \
  --output text > "$OUTPUT_DIR/artifacts.txt"

artifact_idx=0
while IFS=$'\t' read -r name type url; do
  [[ -z "${name:-}" ]] && continue
  artifact_idx=$((artifact_idx + 1))
  safe_name="${name//\//_}"
  curl --fail --silent --show-error -L "$url" -o "$OUTPUT_DIR/${artifact_idx}_${safe_name}"
done < "$OUTPUT_DIR/artifacts.txt"

python3 "$DEVICEFARM_DIR/benchmark_devicefarm.py" \
  --results-dir "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/benchmark.xlsx"

echo "Wrote $OUTPUT_DIR/benchmark.xlsx"
