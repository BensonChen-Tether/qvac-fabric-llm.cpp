#!/usr/bin/env bash
# Shared AWS credential checks and expiry notifications.

notify_user() {
  local title="$1"
  local message="$2"
  echo "" >&2
  echo "============================================" >&2
  echo "$title" >&2
  echo "$message" >&2
  echo "============================================" >&2
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"${message//\"/\\\"}\" with title \"${title//\"/\\\"}\"" 2>/dev/null || true
  fi
}

fail_expired_creds() {
  local profile="$1"
  local purpose="$2"
  local aws_output="${3:-}"

  local message="AWS credentials expired or invalid for profile '${profile}' (${purpose})."
  message="${message} Update ~/.aws/credentials and retry."
  if [[ "$aws_output" == *ExpiredToken* ]]; then
    message="AWS credentials EXPIRED for profile '${profile}' (${purpose}). Update ~/.aws/credentials and retry."
  fi

  notify_user "Device Farm — Credentials Expired" "$message"
  [[ -n "$aws_output" ]] && echo "$aws_output" >&2
  exit 1
}

require_aws_profile() {
  local profile="$1"
  local purpose="$2"
  local out

  if ! out="$(AWS_PROFILE="$profile" aws sts get-caller-identity 2>&1)"; then
    fail_expired_creds "$profile" "$purpose" "$out"
  fi
  echo "$out"
}

handle_aws_error() {
  local profile="$1"
  local purpose="$2"
  local aws_output="$3"

  if [[ "$aws_output" == *ExpiredToken* || "$aws_output" == *NoCredentials* ]]; then
    fail_expired_creds "$profile" "$purpose" "$aws_output"
  fi
  echo "$aws_output" >&2
  return 1
}
