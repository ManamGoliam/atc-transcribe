#!/usr/bin/env bash
set -euo pipefail

MODEL_URL="${MODEL_URL:-https://raw.githubusercontent.com/GregorR/rnnoise-models/master/somnolent-hogwash-2018-09-01/sh.rnnn}"
MODEL_DIR="${MODEL_DIR:-models}"
MODEL_NAME="${MODEL_NAME:-rnnoise.rnnn}"
OUT_PATH="${MODEL_DIR}/${MODEL_NAME}"

echo "RNNoise model download"
echo "  URL: ${MODEL_URL}"
echo "  Dir: ${MODEL_DIR}"
echo "  Out: ${OUT_PATH}"

mkdir -p "${MODEL_DIR}"

if command -v curl >/dev/null 2>&1; then
  echo "Using curl..."
  curl -fL --retry 3 --retry-delay 2 --connect-timeout 10 "${MODEL_URL}" -o "${OUT_PATH}"
elif command -v wget >/dev/null 2>&1; then
  echo "Using wget..."
  wget -O "${OUT_PATH}" "${MODEL_URL}"
else
  echo "Error: need curl or wget installed." >&2
  exit 1
fi

if [ ! -s "${OUT_PATH}" ]; then
  echo "Error: download failed or file is empty: ${OUT_PATH}" >&2
  exit 1
fi

echo "Downloaded model to ${OUT_PATH}"
echo "export RNNOISE_MODEL_PATH=\"${OUT_PATH}\""
