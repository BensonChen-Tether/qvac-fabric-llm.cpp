#!/usr/bin/env bash
# Generate a presigned download URL for a qwen3-checkpoints GGUF on S3.
set -euo pipefail

DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$DEVICEFARM_DIR/aws_creds.sh"

MODEL_FILENAME="${1:?Usage: presign_s3_model.sh <model-filename.gguf>}"
S3_MODEL_BUCKET="${S3_MODEL_BUCKET:-tether-ai-dev}"
S3_MODEL_PREFIX="${S3_MODEL_PREFIX:-models/qwen3-checkpoints/gguf/}"
S3_REGION="${S3_REGION:-eu-central-1}"
AWS_S3_PROFILE="${AWS_S3_PROFILE:-833707431398_Tether_S3_RW_tether-ai-dev}"
PRESIGN_EXPIRES="${PRESIGN_EXPIRES:-7200}"

require_aws_profile "$AWS_S3_PROFILE" "S3 presign" >/dev/null

MODEL_FILENAME="$(basename "$MODEL_FILENAME")"
S3_URI="s3://${S3_MODEL_BUCKET}/${S3_MODEL_PREFIX}${MODEL_FILENAME}"

if ! URL="$(AWS_PROFILE="$AWS_S3_PROFILE" aws s3 presign "$S3_URI" --expires-in "$PRESIGN_EXPIRES" --region "$S3_REGION" 2>&1)"; then
  echo "Cannot access model due to device farm error: failed to presign $S3_URI" >&2
  handle_aws_error "$AWS_S3_PROFILE" "S3 presign" "$URL" || true
  exit 1
fi

echo "$URL"
