# LiveATC Transcribe Pipeline

Minimal CPU-first pipeline to capture a LiveATC stream, optionally denoise it with RNNoise, and transcribe audio chunks with `faster-whisper`.

## Requirements
- Python 3.10+
- `ffmpeg` on `PATH`
- LiveATC stream URL (from the `<audio src="...">` tag)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
export STREAM_URL="https://d.liveatc.net/redir.php/okbk2?nocache=..."
export RNNOISE_MODEL_PATH="/path/to/rnnoise_model.rnnn"  # optional
export CHUNK_SECONDS="15"                                # optional, default 15
export MODEL_SIZE="base"                                 # optional, default base
export PYTHONPATH=src
export STREAM_USER_AGENT="Mozilla/5.0"                   # optional
export STREAM_HEADERS=$'Referer: https://www.liveatc.net/\r\n'  # optional
export OUTPUT_DIR="outputs"                              # optional, default outputs
export SAVE_CHUNKS="1"                                   # optional, default 1 (save)
export SAVE_RAW_CHUNKS="1"                               # optional, default 1 (save)
export RAW_OUTPUT_DIR="outputs/raw"                      # optional, default outputs/raw

python -m transcribe.pipeline
```

### Run with helper script
Create a `.env` file (optional) and run:
```bash
cat > .env <<'EOF'
STREAM_URL="https://d.liveatc.net/redir.php/okbk2?nocache=..."
RNNOISE_MODEL_PATH="models/rnnoise.rnnn"
CHUNK_SECONDS=15
MODEL_SIZE=base
STREAM_USER_AGENT="Mozilla/5.0"
STREAM_HEADERS="Referer: https://www.liveatc.net/\r\n"
OUTPUT_DIR="outputs"
SAVE_CHUNKS=1
SAVE_RAW_CHUNKS=1
RAW_OUTPUT_DIR="outputs/raw"
EOF

./run.sh
```

## Denoising Model
RNNoise uses a pretrained model file (`.rnnn`). Download one with:
```bash
./scripts/get_rnnoise_model.sh
```
Then export the path it prints:
```bash
export RNNOISE_MODEL_PATH="models/rnnoise.rnnn"
```

## Notes
- If `RNNOISE_MODEL_PATH` is not set, denoising is disabled.
- The `STREAM_URL` can be the LiveATC `redir.php` URL; `ffmpeg` will follow redirects.
- Output is plain text with chunk timestamps for downstream post-processing.
