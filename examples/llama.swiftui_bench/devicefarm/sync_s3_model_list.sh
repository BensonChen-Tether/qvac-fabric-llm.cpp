#!/usr/bin/env bash
# List .gguf models from the S3 prefix in config.env and write devicefarm/s3-model-list.txt.
set -euo pipefail

DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$DEVICEFARM_DIR/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$DEVICEFARM_DIR/config.env"
fi

S3_MODEL_BUCKET="${S3_MODEL_BUCKET:-tether-ai-dev}"
S3_MODEL_PREFIX="${S3_MODEL_PREFIX:-models/qwen3-checkpoints/gguf/}"
S3_REGION="${S3_REGION:-eu-central-1}"
AWS_S3_PROFILE="${AWS_S3_PROFILE:-833707431398_Tether_S3_RW_tether-ai-dev}"
OUTPUT="${1:-$DEVICEFARM_DIR/s3-model-list.txt}"

AWS_PROFILE="$AWS_S3_PROFILE" aws s3 ls \
  "s3://${S3_MODEL_BUCKET}/${S3_MODEL_PREFIX}" \
  --region "$S3_REGION" \
  | awk '/\.gguf$/ { print $4 }' \
  | sort > "$OUTPUT"

count="$(wc -l < "$OUTPUT" | tr -d ' ')"
echo "Wrote $count model(s) to $OUTPUT"
cat "$OUTPUT"
