import os
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # s16le


def _format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hrs:
        return f"{hrs:02d}:{mins:02d}:{secs:05.2f}"
    return f"{mins:02d}:{secs:05.2f}"


def _format_timestamp_for_filename(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hrs:
        return f"{hrs:02d}-{mins:02d}-{secs:05.2f}"
    return f"{mins:02d}-{secs:05.2f}"


def _build_ffmpeg_cmd(stream_url: str, denoise_model: Optional[str]) -> list[str]:
    filters = []
    if denoise_model:
        # RNNoise model for ffmpeg's arnndn filter.
        filters.append(f"arnndn=m={denoise_model}")
    filter_chain = ",".join(filters) if filters else None
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-user_agent",
        os.getenv("STREAM_USER_AGENT", "Mozilla/5.0"),
        "-headers",
        os.getenv("STREAM_HEADERS", "Referer: https://www.liveatc.net/\r\n"),
        "-i",
        stream_url,
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
    ]
    if filter_chain:
        cmd += ["-af", filter_chain]
    cmd.append("pipe:1")
    return cmd


def _read_chunk(stream, chunk_bytes: int) -> bytes:
    data = bytearray()
    while len(data) < chunk_bytes:
        block = stream.read(chunk_bytes - len(data))
        if not block:
            break
        data.extend(block)
    return bytes(data)


def main() -> int:
    stream_url = os.getenv("STREAM_URL")
    if not stream_url:
        print("STREAM_URL is required.", file=sys.stderr)
        return 2

    chunk_seconds = int(os.getenv("CHUNK_SECONDS", "15"))
    model_size = os.getenv("MODEL_SIZE", "base")
    denoise_model = os.getenv("RNNOISE_MODEL_PATH")
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
    save_chunks = os.getenv("SAVE_CHUNKS", "1") not in {"0", "false", "False"}
    save_raw_chunks = os.getenv("SAVE_RAW_CHUNKS", "1") not in {"0", "false", "False"}
    raw_output_dir = Path(os.getenv("RAW_OUTPUT_DIR", output_dir / "raw"))

    if save_chunks:
        output_dir.mkdir(parents=True, exist_ok=True)
    if save_raw_chunks:
        raw_output_dir.mkdir(parents=True, exist_ok=True)

    if not denoise_model:
        print(
            "RNNOISE_MODEL_PATH not set; denoising disabled.",
            file=sys.stderr,
        )

    cmd = _build_ffmpeg_cmd(stream_url, denoise_model)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    raw_process = None
    if save_raw_chunks and denoise_model:
        raw_cmd = _build_ffmpeg_cmd(stream_url, None)
        raw_process = subprocess.Popen(
            raw_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    chunk_bytes = SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_seconds
    chunk_index = 0

    def shutdown(*_args) -> None:
        process.terminate()
        if raw_process:
            raw_process.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            raw = _read_chunk(process.stdout, chunk_bytes)  # type: ignore[arg-type]
            if not raw:
                break
            if len(raw) < chunk_bytes:
                # Avoid partial trailing chunks to keep timestamps stable.
                break
            raw_pre = raw
            if raw_process:
                raw_pre = _read_chunk(raw_process.stdout, chunk_bytes)  # type: ignore[arg-type]
                if not raw_pre or len(raw_pre) < chunk_bytes:
                    break

            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            chunk_start = chunk_index * chunk_seconds
            chunk_index += 1

            segments, _info = model.transcribe(
                audio,
                language="en",
                task="transcribe",
                vad_filter=False,
            )

            text = " ".join(segment.text.strip() for segment in segments).strip()
            ts = _format_timestamp(chunk_start)
            if text:
                print(f"[{ts}] {text}", flush=True)

            if save_chunks:
                ts_file = _format_timestamp_for_filename(chunk_start)
                base = output_dir / f"chunk_{chunk_index:06d}_{ts_file}"
                wav_path = f"{base}.wav"
                txt_path = f"{base}.txt"
                with wave.open(wav_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(BYTES_PER_SAMPLE)
                    wav_file.setframerate(SAMPLE_RATE)
                    wav_file.writeframes(raw)
                with open(txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(f"[{ts}] {text}\n")
            if save_raw_chunks:
                ts_file = _format_timestamp_for_filename(chunk_start)
                raw_base = raw_output_dir / f"chunk_{chunk_index:06d}_{ts_file}_raw"
                raw_wav_path = f"{raw_base}.wav"
                with wave.open(raw_wav_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(BYTES_PER_SAMPLE)
                    wav_file.setframerate(SAMPLE_RATE)
                    wav_file.writeframes(raw_pre)

    finally:
        process.terminate()
        if raw_process:
            raw_process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        if raw_process:
            try:
                raw_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                raw_process.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
