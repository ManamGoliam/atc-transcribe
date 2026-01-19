#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${ROOT_DIR}/.env" ]; then
  echo "Loading .env..."
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
fi

if [ ! -d "${ROOT_DIR}/.venv" ]; then
  echo "Error: .venv not found. Create it with: python -m venv .venv" >&2
  exit 1
fi

echo "Activating venv..."
source "${ROOT_DIR}/.venv/bin/activate"

export PYTHONPATH="${ROOT_DIR}/src"
export CHUNK_SECONDS="${CHUNK_SECONDS:-15}"
export MODEL_SIZE="${MODEL_SIZE:-base}"
export STREAM_URL="${STREAM_URL:-https://d.liveatc.net/redir.php/okbk2?nocache=2026011914421769105}"
export RNNOISE_MODEL_PATH="${RNNOISE_MODEL_PATH:-${ROOT_DIR}/models/rnnoise.rnnn}"
export STREAM_USER_AGENT="${STREAM_USER_AGENT:-Mozilla/5.0}"
export STREAM_HEADERS="${STREAM_HEADERS:-Referer: https://www.liveatc.net/\r\n}"

if [ -z "${STREAM_URL:-}" ]; then
  echo "Error: STREAM_URL is required. Set it in your shell or .env." >&2
  exit 2
fi

echo "Using RNNoise model: ${RNNOISE_MODEL_PATH}"

echo "Starting pipeline..."
python -m transcribe.pipeline
