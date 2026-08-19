#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes POST /v1/audio/speech compatible with OpenAI's TTS API, enabling
integration with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

Usage:
    pip install "faster-qwen3-tts[server]"

    # Single default voice:
    python openai_server.py \
        --ref-audio voice.wav --ref-text "Reference transcription" \
        --language English

    # Multiple named voices from a JSON config:
    python openai_server.py --voices voices/voices.json

    # Custom model and port:
    python openai_server.py \
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --ref-audio voice.wav --ref-text "transcript" \
        --port 8000

Voices config (voices/voices.json) — two supported formats:

    # Precomputed speaker embedding (recommended — fastest inference):
    {
        "alloy": {"spk_embedding": "alloy.pt", "language": "English"},
        "echo":  {"spk_embedding": "echo.pt",  "language": "English"}
    }

    # Legacy WAV reference (embedding is auto-extracted on first use):
    {
        "alloy": {"ref_audio": "voice.wav", "ref_text": "...", "language": "English"}
    }

API usage:
    curl -s http://localhost:8000/v1/audio/speech \
        -H "Content-Type: application/json" \
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav"}' \
        --output speech.wav
"""
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import struct
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

# Browser clients (including the Gradio UI) send an OPTIONS preflight before
# cross-origin JSON, PATCH, DELETE, and multipart requests. Starlette's CORS
# middleware answers those preflights with the required 2xx response and CORS
# headers before request routing reaches the API handlers.
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("TTS_CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

tts_model = None
voices: dict = {}
default_voice: Optional[str] = None
voices_file: Optional[str] = None  # path to voices.json, for write-back
SAMPLE_RATE = 24000  # updated once the model loads
_model_lock = threading.Lock()  # prevent concurrent GPU inference

TARGET_SAMPLE_RATE = 24000
SSE_FINAL_PADDING_MS = int(os.environ.get("TTS_SSE_FINAL_PADDING_MS", "120"))
SSE_DONE_DELAY_MS = int(os.environ.get("TTS_SSE_DONE_DELAY_MS", "150"))
_BASE_DIR = Path(os.environ.get("TTS_DATA_DIR", Path(__file__).resolve().parent))
VOICE_STORAGE_DIR = (_BASE_DIR / "voices").resolve()
_VOICE_FILE_EXTENSIONS = {".wav", ".mp3", ".flac", ".aac", ".opus", ".ogg", ".m4a", ".pcm"}
_EMBEDDING_EXTENSION = ".pt"  # precomputed speaker x-vector

def ensure_voice_storage():
    VOICE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

def _iter_voice_files():
    """Yield audio files in VOICE_STORAGE_DIR (WAV, MP3, etc.)."""
    ensure_voice_storage()
    for path in sorted(VOICE_STORAGE_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in _VOICE_FILE_EXTENSIONS:
            yield path

def _iter_embedding_files():
    """Yield .pt speaker embedding files in VOICE_STORAGE_DIR."""
    ensure_voice_storage()
    for path in sorted(VOICE_STORAGE_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() == _EMBEDDING_EXTENSION:
            yield path

def resolve_voice_file(voice_id: str):
    """Return the audio or embedding file for a voice_id, or None."""
    # Prefer embedding files (.pt) over audio files
    for path in _iter_embedding_files():
        if path.stem == voice_id:
            return path
        json_path = path.with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                    if meta.get("name") == voice_id:
                        return path
            except Exception:
                pass
    for path in _iter_voice_files():
        if path.stem == voice_id:
            return path
        json_path = path.with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                    if meta.get("name") == voice_id:
                        return path
            except Exception:
                pass
    return None

def _read_voice_meta(json_path: Path) -> dict:
    """Safely read a sidecar .json metadata file."""
    if json_path.exists():
        try:
            with open(json_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def list_uploaded_voices():
    """List all uploaded voices (embedding .pt files take precedence over WAV)."""
    seen_stems = set()
    result = []

    # Embeddings first
    for path in _iter_embedding_files():
        stem = path.stem
        seen_stems.add(stem)
        stat = path.stat()
        meta = _read_voice_meta(path.with_suffix(".json"))
        voice_name = meta.get("name") or stem
        result.append({
            "id": voice_name,
            "voice_id": stem,
            "name": voice_name,
            "object": "voice",
            "created": int(stat.st_mtime),
            "owned_by": "faster-qwen3-tts",
            "filename": path.name,
            "ref_text": meta.get("ref_text", ""),
            "embedding": True,
        })

    # Audio files (skip those already covered by an embedding)
    for path in _iter_voice_files():
        stem = path.stem
        if stem in seen_stems:
            continue
        stat = path.stat()
        meta = _read_voice_meta(path.with_suffix(".json"))
        voice_name = meta.get("name") or stem
        result.append({
            "id": voice_name,
            "voice_id": stem,
            "name": voice_name,
            "object": "voice",
            "created": int(stat.st_mtime),
            "owned_by": "faster-qwen3-tts",
            "filename": path.name,
            "ref_text": meta.get("ref_text", ""),
            "embedding": False,
        })

    return result

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _load_embedding_vcp(pt_path: str, device: str = "cuda") -> dict:
    """Load a precomputed .pt x-vector and return a voice_clone_prompt dict."""
    spk_emb = torch.load(pt_path, weights_only=True).to(device)
    return dict(
        ref_code=[None],
        ref_spk_embedding=[spk_emb],
        x_vector_only_mode=[True],
        icl_mode=[False],
    )


def _extract_and_save_embedding(ref_audio_path: str, pt_path: str) -> None:
    """Extract speaker x-vector from a WAV file and save it as a .pt file."""
    prompt_items = tts_model.model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text="",
        x_vector_only_mode=True,
    )
    spk_emb = prompt_items[0].ref_spk_embedding.cpu()
    torch.save(spk_emb, pt_path)
    logger.info("Saved speaker embedding: %s (shape=%s)", pt_path, tuple(spk_emb.shape))


def _warm_embedding_cache(pt_path: str, ref_text: str = "") -> None:
    """Pre-populate the model's voice_prompt_cache for a .pt embedding path.

    The cache key used by model._prepare_generation is
    (str(ref_audio), ref_text, xvec_only, append_silence).  We register the
    .pt file path under that key so the first call hits the cache directly,
    skipping the speaker encoder entirely.
    """
    if tts_model is None:
        return
    device = tts_model.device
    vcp = _load_embedding_vcp(pt_path, device=device)
    ref_ids = [None]
    # xvec_only=True, append_silence arg is unused in xvec path but still part of key
    for append_silence in (True, False):
        cache_key = (pt_path, ref_text, True, append_silence)
        tts_model._voice_prompt_cache[cache_key] = (vcp, ref_ids)


def preload_embedding_voices() -> None:
    """Load all .pt embedding files from VOICE_STORAGE_DIR into the model cache."""
    if tts_model is None:
        return
    count = 0
    for path in _iter_embedding_files():
        pt_path = str(path.absolute())
        meta = _read_voice_meta(path.with_suffix(".json"))
        ref_text = meta.get("ref_text", "")
        _warm_embedding_cache(pt_path, ref_text)
        count += 1
    if count:
        logger.info("Pre-loaded %d speaker embedding(s) into model cache", count)


async def _download_voice(voice_url: str) -> str:
    import urllib.request
    def _dl():
        parsed = urlparse(voice_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("voice_url must use http or https")
        suffix = Path(parsed.path).suffix or ".wav"
        req = urllib.request.Request(
            voice_url, headers={"User-Agent": "faster-qwen3-tts", "Accept": "*/*"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(resp.read())
                return tmp.name
    return await asyncio.to_thread(_dl)

def load_audio_to_mono_24k(path, max_seconds=30.0):
    import subprocess
    def _ffmpeg_decode(src):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), tmp.name],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return tmp.name

    cleanup = None
    try:
        try:
            audio, sr = sf.read(str(path), always_2d=False)
        except Exception:
            cleanup = _ffmpeg_decode(path)
            audio, sr = sf.read(cleanup, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != TARGET_SAMPLE_RATE:
            n = int(round(len(audio) * TARGET_SAMPLE_RATE / sr))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
            ).astype(np.float32)
        if max_seconds:
            audio = audio[: int(TARGET_SAMPLE_RATE * max_seconds)]
        return audio, len(audio) / TARGET_SAMPLE_RATE
    finally:
        if cleanup and os.path.exists(cleanup):
            os.remove(cleanup)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"  # wav | pcm | mp3
    speed: float = 1.0           # accepted but not yet applied
    stream_format: Optional[str] = None
    language: Optional[str] = None
    reference_text: Optional[str] = None
    ref_text: Optional[str] = None
    voice_url: Optional[str] = None
    chunk_size: Optional[int] = None
    instructions: Optional[str] = None


class VoiceUpdate(BaseModel):
    """Editable metadata for an uploaded voice."""

    name: Optional[str] = None
    ref_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Use data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to a complete WAV file in memory."""
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to MP3 bytes (requires pydub + ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


def _encode_audio_bytes(pcm: np.ndarray, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    """Encode audio into the requested wire format and return bytes plus MIME type."""
    fmt = response_format.lower()
    if fmt == "wav":
        return _to_wav_bytes(pcm, sample_rate), "audio/wav"
    if fmt == "pcm":
        return _to_pcm16(pcm), "audio/pcm"
    if fmt == "mp3":
        return _to_mp3_bytes(pcm, sample_rate), "audio/mpeg"
    raise HTTPException(
        status_code=400,
        detail=f"response_format {response_format!r} not supported. Use: wav, pcm, mp3",
    )


def _pad_audio_tail(pcm: np.ndarray, sample_rate: int, pad_ms: int) -> np.ndarray:
    """Append a short silence tail to the final SSE chunk to avoid cutoff on client teardown."""
    if pad_ms <= 0:
        return pcm
    if pcm.ndim > 1:
        pcm = pcm.squeeze()
    pad_samples = max(1, int(sample_rate * pad_ms / 1000))
    silence = np.zeros(pad_samples, dtype=pcm.dtype)
    return np.concatenate((pcm, silence))


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def _resolve_embedding_path(raw: str) -> Optional[str]:
    """Resolve a .pt path from voices.json (relative → absolute via VOICE_STORAGE_DIR)."""
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = VOICE_STORAGE_DIR / p.name
    if candidate.exists():
        return str(candidate.absolute())
    return None


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    if voice_name in voices:
        vcfg = dict(voices[voice_name])

        # --- Embedding-backed voice (spk_embedding key) ---
        if "spk_embedding" in vcfg:
            pt_path = _resolve_embedding_path(vcfg["spk_embedding"])
            if pt_path is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Speaker embedding file not found for voice {voice_name!r}: {vcfg['spk_embedding']}",
                )
            vcfg["spk_embedding"] = pt_path
            return vcfg

        # --- Legacy WAV-backed voice (ref_audio key) ---
        ref_path = Path(vcfg.get("ref_audio", ""))
        if not ref_path.exists() and (VOICE_STORAGE_DIR / ref_path.name).exists():
            vcfg["ref_audio"] = str((VOICE_STORAGE_DIR / ref_path.name).absolute())
        elif ref_path.exists():
            vcfg["ref_audio"] = str(ref_path.absolute())
        return vcfg

    # --- Uploaded voices (scan disk) ---
    candidate = resolve_voice_file(voice_name)
    if candidate is not None:
        meta = _read_voice_meta(candidate.with_suffix(".json"))
        if candidate.suffix.lower() == _EMBEDDING_EXTENSION:
            return {
                "spk_embedding": str(candidate.absolute()),
                "ref_text": meta.get("ref_text", ""),
                "language": "Auto",
            }
        return {
            "ref_audio": str(candidate.absolute()),
            "ref_text": meta.get("ref_text", ""),
            "language": "Auto",
        }

    if default_voice and default_voice in voices:
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            default_voice,
        )
        return dict(voices[default_voice])
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available predefined voices: {list(voices.keys())}. Also supports uploaded voice IDs."
        ),
    )


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


def _ensure_embedding_cached(voice_cfg: dict) -> None:
    """If voice_cfg uses a .pt embedding, make sure it's in the model cache.

    This is a no-op if the embedding is already cached (e.g. from preload or a
    previous request). Thread-safe: the model lock is held by the caller.
    """
    pt_path = voice_cfg.get("spk_embedding")
    if pt_path is None:
        return
    ref_text = voice_cfg.get("ref_text", "")
    # Check both append_silence variants
    for append_silence in (True, False):
        cache_key = (pt_path, ref_text, True, append_silence)
        if cache_key not in tts_model._voice_prompt_cache:
            _warm_embedding_cache(pt_path, ref_text)
            return


async def _generate_chunks_queue(voice_cfg: dict, text: str):
    """Run generator in background and return queue."""
    q: queue.Queue = queue.Queue()
    _DONE = object()

    # Determine ref_audio path: for embedding voices we pass the .pt path so
    # _prepare_generation's cache lookup hits immediately (cache is keyed by path).
    is_embedding = "spk_embedding" in voice_cfg
    ref_audio_arg = voice_cfg["spk_embedding"] if is_embedding else voice_cfg["ref_audio"]

    def producer():
        try:
            with _model_lock:
                if is_embedding:
                    _ensure_embedding_cached(voice_cfg)
                for chunk, _sr, _timing in tts_model.generate_voice_clone_streaming(
                    text=text,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=ref_audio_arg,
                    ref_text=voice_cfg.get("ref_text", ""),
                    chunk_size=voice_cfg.get("chunk_size", 12),
                    non_streaming_mode=False,
                ):
                    q.put((chunk, _sr, _timing))
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    return q, _DONE


async def _stream_chunks(voice_cfg: dict, text: str) -> AsyncGenerator[bytes, None]:
    """Yield raw PCM bytes chunks."""
    q, _DONE = await _generate_chunks_queue(voice_cfg, text)
    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        chunk, _sr, _timing = item
        yield _to_pcm16(chunk)


async def _stream_chunks_sse(
    voice_cfg: dict,
    text: str,
    response_format: str,
) -> AsyncGenerator[str, None]:
    """Yield SSE JSON chunks encoded in the requested response format."""
    import time
    import base64

    q, _DONE = await _generate_chunks_queue(voice_cfg, text)
    loop = asyncio.get_event_loop()

    t0 = time.perf_counter()
    total_audio_s = 0.0
    voice_clone_ms = 0.0
    total_gen_ms = 0.0
    ttfa_ms = None
    first_chunk_wall_ms = None
    first_chunk_model_ms = 0.0

    def _build_audio_event(
        audio_chunk: np.ndarray,
        sr: int,
        timing: dict,
        *,
        final: bool,
    ) -> str:
        nonlocal total_audio_s, voice_clone_ms, total_gen_ms, ttfa_ms

        if final:
            audio_chunk = _pad_audio_tail(audio_chunk, sr, SSE_FINAL_PADDING_MS)

        if ttfa_ms is None:
            wall_first_ms = first_chunk_wall_ms
            if wall_first_ms is None:
                wall_first_ms = (time.perf_counter() - t0) * 1000
            model_ms = first_chunk_model_ms or (
                timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
            )
            voice_clone_ms = max(0.0, wall_first_ms - model_ms)
            total_gen_ms += timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
            ttfa_ms = total_gen_ms
        else:
            total_gen_ms += timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)

        dur = len(audio_chunk) / sr
        total_audio_s += dur
        rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

        encoded_audio, mime_type = _encode_audio_bytes(audio_chunk, sr, response_format)
        payload = {
            "type": "audio.chunk",
            "data": base64.b64encode(encoded_audio).decode("ascii"),
            "format": response_format,
            "mime_type": mime_type,
            "sample_rate": sr,
            "ttfa_ms": round(ttfa_ms),
            "voice_clone_ms": round(voice_clone_ms),
            "rtf": round(rtf, 3),
            "total_audio_s": round(total_audio_s, 3),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            "final": final,
        }
        return f"data: {json.dumps(payload)}\n\n"

    pending_item = None

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item

            if first_chunk_wall_ms is None:
                _audio_chunk, _sr, timing = item
                first_chunk_wall_ms = (time.perf_counter() - t0) * 1000
                first_chunk_model_ms = timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)

            if pending_item is None:
                pending_item = item
                continue

            audio_chunk, sr, timing = pending_item
            yield _build_audio_event(audio_chunk, sr, timing, final=False)
            pending_item = item

        if pending_item is not None:
            audio_chunk, sr, timing = pending_item
            yield _build_audio_event(audio_chunk, sr, timing, final=True)
    except Exception as exc:
        err = {"type": "error", "message": str(exc)}
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
        return

    if total_audio_s > 0 and SSE_DONE_DELAY_MS > 0:
        await asyncio.sleep(SSE_DONE_DELAY_MS / 1000)

    rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0
    done_payload = {
        "type": "done",
        "ttfa_ms": round(ttfa_ms) if ttfa_ms else 0,
        "voice_clone_ms": round(voice_clone_ms),
        "rtf": round(rtf, 3),
        "total_audio_s": round(total_audio_s, 3),
        "total_ms": round((time.perf_counter() - t0) * 1000),
    }
    yield f"data: {json.dumps(done_payload)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None}

@app.get("/v1/models")
@app.get("/v1/audio/models")
async def list_models():
    """List available models — OpenAI-compatible endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-tts",
                "object": "model",
                "created": 0,
                "owned_by": "faster-qwen3-tts",
            }
        ],
    }


@app.get("/v1/audio/voices")
async def list_voices():
    """List uploaded and configured voices."""
    voices_list = []

    # Add pre-configured static voices from voices.json
    for v_name, v_cfg in voices.items():
        if "spk_embedding" in v_cfg:
            filename = Path(v_cfg["spk_embedding"]).name
            embedding = True
        else:
            filename = Path(v_cfg.get("ref_audio", "")).name
            embedding = False
        voices_list.append({
            "id": v_name,
            "voice_id": v_name,
            "name": v_name,
            "object": "voice",
            "created": 0,
            "owned_by": "faster-qwen3-tts",
            "filename": filename,
            "ref_text": v_cfg.get("ref_text", ""),
            "embedding": embedding,
        })

    # Add uploaded voices from disk
    voices_list.extend(list_uploaded_voices())

    return {
        "object": "list",
        "data": voices_list,
    }


def _uploaded_voice_path_or_404(voice_id: str) -> Path:
    """Resolve an uploaded voice, rejecting configured read-only voices."""
    if voice_id in voices:
        raise HTTPException(status_code=403, detail="Configured voices cannot be modified")
    candidate = resolve_voice_file(voice_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Uploaded voice {voice_id!r} not found")
    return candidate


def _voice_response_for_path(path: Path) -> dict:
    meta = _read_voice_meta(path.with_suffix(".json"))
    stem = path.stem
    name = meta.get("name") or stem
    return {
        "id": name,
        "voice_id": stem,
        "name": name,
        "object": "voice",
        "owned_by": "faster-qwen3-tts",
        "filename": path.name,
        "ref_text": meta.get("ref_text", ""),
        "embedding": path.suffix.lower() == _EMBEDDING_EXTENSION,
    }


@app.patch("/v1/audio/voices/{voice_id}")
async def update_voice(voice_id: str, update: VoiceUpdate):
    """Update the name and/or reference transcript of an uploaded voice."""
    candidate = _uploaded_voice_path_or_404(voice_id)
    meta_path = candidate.with_suffix(".json")
    meta = _read_voice_meta(meta_path)

    if "name" in update.model_fields_set:
        new_name = (update.name or "").strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="Voice name cannot be empty")
        if new_name in voices:
            raise HTTPException(status_code=409, detail="Voice name conflicts with a configured voice")
        for item in list_uploaded_voices():
            if item["voice_id"] != candidate.stem and item["name"] == new_name:
                raise HTTPException(status_code=409, detail="Voice name is already in use")
        meta["name"] = new_name

    if "ref_text" in update.model_fields_set:
        ref_text = (update.ref_text or "").strip()
        if ref_text:
            meta["ref_text"] = ref_text
        else:
            meta.pop("ref_text", None)

    if meta:
        with open(meta_path, "w") as jf:
            json.dump(meta, jf)
    elif meta_path.exists():
        meta_path.unlink()

    return _voice_response_for_path(candidate)


@app.get("/v1/audio/voices/{voice_id}")
async def get_voice(voice_id: str):
    """Return one configured or uploaded voice."""
    if voice_id in voices:
        config = voices[voice_id]
        return {
            "id": voice_id,
            "voice_id": voice_id,
            "name": voice_id,
            "object": "voice",
            "owned_by": "faster-qwen3-tts",
            "filename": Path(config.get("spk_embedding", config.get("ref_audio", ""))).name,
            "ref_text": config.get("ref_text", ""),
            "embedding": "spk_embedding" in config,
        }
    candidate = _uploaded_voice_path_or_404(voice_id)
    return _voice_response_for_path(candidate)


@app.delete("/v1/audio/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete an uploaded voice and all files generated for it."""
    candidate = _uploaded_voice_path_or_404(voice_id)
    stem = candidate.stem
    removed = []
    for path in VOICE_STORAGE_DIR.iterdir():
        if path.is_file() and path.stem == stem and path.suffix.lower() in (
            _VOICE_FILE_EXTENSIONS | {_EMBEDDING_EXTENSION, ".json"}
        ):
            path.unlink()
            removed.append(path.name)

    if tts_model is not None:
        for cache_key in list(tts_model._voice_prompt_cache):
            if isinstance(cache_key, tuple) and cache_key and str(cache_key[0]) in {
                str(candidate), str(candidate.with_suffix(".pt")), str(candidate.with_suffix(".wav"))
            }:
                tts_model._voice_prompt_cache.pop(cache_key, None)

    return {"deleted": True, "voice_id": stem, "files": removed}


@app.post("/upload_voice")
async def upload_voice(
    voice_file: UploadFile = File(None),
    voice_url: str = Form(None),
    name: str = Form(None),
    voice_name: str = Form(None),
    ref_text: str = Form(None),
    reference_text: str = Form(None),
    data: str = Form(None),
):
    """Upload a voice sample → normalise to 24 kHz mono WAV → store locally → return voice_id."""
    ensure_voice_storage()

    # Support metadata provided as a JSON string in the 'data' field
    meta_from_data = {}
    if data:
        try:
            meta_from_data = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Failed to parse 'data' form field as JSON: %r", data)

    if voice_url is None and voice_file is None:
        raise HTTPException(status_code=400, detail="voice_url or voice_file is required")

    final_ref_text = (ref_text or reference_text or meta_from_data.get("ref_text") or "").strip()
    # ``name`` is the canonical API field. Accept the older/client-specific
    # ``voice_name`` field as a compatibility fallback so uploads retain the
    # supplied display name instead of falling back to the UUID filename.
    final_name = (name or voice_name or meta_from_data.get("name") or "").strip()
    source_path = None
    download_tmp = None
    dest_path = None

    try:
        if voice_url:
            download_tmp = await _download_voice(voice_url)
            source_path = download_tmp
        else:
            suffix = Path(voice_file.filename or "").suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await voice_file.read())
                source_path = tmp.name

        audio, duration = await asyncio.to_thread(
            load_audio_to_mono_24k, source_path, 600.0
        )

        voice_id = uuid.uuid4().hex
        dest_path = VOICE_STORAGE_DIR / f"{voice_id}.wav"
        sf.write(str(dest_path), audio, samplerate=TARGET_SAMPLE_RATE, subtype="PCM_16")

        if final_ref_text or final_name:
            json_path = dest_path.with_suffix(".json")
            meta = {}
            if final_ref_text:
                meta["ref_text"] = final_ref_text
            if final_name:
                meta["name"] = final_name
            with open(json_path, "w") as jf:
                json.dump(meta, jf)

        # Extract speaker embedding immediately after saving the WAV.
        # This pre-populates the model cache so the first inference request
        # skips the speaker encoder entirely.
        pt_path = VOICE_STORAGE_DIR / f"{voice_id}{_EMBEDDING_EXTENSION}"
        if tts_model is not None:
            try:
                await asyncio.to_thread(
                    _extract_and_save_embedding,
                    str(dest_path),
                    str(pt_path),
                )
                _warm_embedding_cache(str(pt_path.absolute()), final_ref_text)
                logger.info("Speaker embedding extracted: %s", pt_path.name)
            except Exception as emb_err:
                logger.warning("Failed to extract speaker embedding: %s", emb_err)

        logger.info(f"Voice uploaded: voice_id={voice_id}, duration={duration:.1f}s")
        return JSONResponse({"voice_id": voice_id})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process voice: {e}")
    finally:
        for p in {download_tmp, source_path}:
            if p and dest_path and os.path.abspath(str(p)) == str(dest_path):
                continue
            if p and os.path.exists(str(p)):
                try:
                    os.remove(p)
                except Exception:
                    pass

from fastapi import BackgroundTasks

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, background_tasks: BackgroundTasks):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    voice_cfg = resolve_voice(req.voice)

    # Process request overrides
    if req.language is not None:
        voice_cfg["language"] = req.language
    if req.reference_text is not None or req.ref_text is not None:
        voice_cfg["ref_text"] = req.reference_text or req.ref_text
    if req.chunk_size is not None:
        voice_cfg["chunk_size"] = req.chunk_size
    if req.instructions is not None:
        voice_cfg["instructions"] = req.instructions

    # voice_url overrides voice source entirely (download → WAV path)
    if req.voice_url:
        try:
            dl_tmp = await _download_voice(req.voice_url)
            # Remove embedding key if present so we use the downloaded audio directly
            voice_cfg.pop("spk_embedding", None)
            voice_cfg["ref_audio"] = dl_tmp
            def _cleanup():
                if dl_tmp and os.path.exists(dl_tmp):
                    os.remove(dl_tmp)
            background_tasks.add_task(_cleanup)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download voice_url: {e}")

    fmt = req.response_format.lower()

    _CONTENT_TYPES = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
    }
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3",
        )
    content_type = _CONTENT_TYPES[fmt]

    # --- MP3: generate all audio, then encode (non-streaming) ---
    if fmt == "mp3":
        loop = asyncio.get_event_loop()
        is_embedding = "spk_embedding" in voice_cfg
        ref_audio_arg = voice_cfg["spk_embedding"] if is_embedding else voice_cfg["ref_audio"]

        def _generate():
            with _model_lock:
                if is_embedding:
                    _ensure_embedding_cached(voice_cfg)
                return tts_model.generate_voice_clone(
                    text=req.input,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=ref_audio_arg,
                    ref_text=voice_cfg.get("ref_text", ""),
                )

        audio_arrays, sr = await loop.run_in_executor(None, _generate)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        return Response(content=_to_mp3_bytes(audio, sr), media_type=content_type)

    # --- WAV / PCM: stream chunks as they are generated ---
    stream_fmt = (req.stream_format or "").strip().lower()

    if stream_fmt == "sse":
        return StreamingResponse(
            _stream_chunks_sse(voice_cfg, req.input, fmt),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)  # stream with unknown data length
        async for raw_chunk in _stream_chunks(voice_cfg, req.input):
            yield raw_chunk

    return StreamingResponse(audio_stream(), media_type=content_type)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        help="HuggingFace model ID or local path (default: Qwen/Qwen3-TTS-12Hz-0.6B-Base)",
    )
    p.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES", "voices/voices.json"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    p.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    p.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
        help="Target language (English, French, Auto, …) when --voices is not used",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    return p.parse_args()


def _auto_extract_voices_json_embeddings() -> None:
    """For any voices.json entry with ref_audio but no spk_embedding, extract the
    embedding on startup, warm the model cache, and update voices.json to use
    spk_embedding.  On subsequent restarts the entry already has spk_embedding so
    only the cache warm step runs.
    """
    changed = False
    for v_name, v_cfg in voices.items():
        if "spk_embedding" in v_cfg:
            # Already using an embedding — just warm the cache
            pt_path = _resolve_embedding_path(v_cfg["spk_embedding"])
            if pt_path:
                _warm_embedding_cache(pt_path, v_cfg.get("ref_text", ""))
            continue
        if "ref_audio" not in v_cfg:
            continue

        ref_audio_path_raw = v_cfg["ref_audio"]
        ref_audio_path = Path(ref_audio_path_raw)
        if not ref_audio_path.exists():
            ref_audio_path = VOICE_STORAGE_DIR / ref_audio_path_raw
        if not ref_audio_path.exists():
            logger.warning("Voice %r: ref_audio not found, skipping embedding extraction", v_name)
            continue

        # Derive .pt path alongside the WAV file
        pt_path = ref_audio_path.with_suffix(_EMBEDDING_EXTENSION)
        if not pt_path.exists():
            logger.info("Extracting speaker embedding for voice %r → %s", v_name, pt_path.name)
            try:
                _extract_and_save_embedding(str(ref_audio_path.absolute()), str(pt_path))
            except Exception as e:
                logger.warning("Embedding extraction failed for voice %r: %s", v_name, e)
                continue

        _warm_embedding_cache(str(pt_path.absolute()), v_cfg.get("ref_text", ""))
        logger.info("Voice %r: embedding cached from %s", v_name, pt_path.name)

        # Update the in-memory config and mark for write-back
        v_cfg["spk_embedding"] = pt_path.name  # store relative filename
        v_cfg.pop("ref_audio", None)
        v_cfg.pop("ref_text", None)  # ref_text not used in x-vector mode
        changed = True

    if changed and voices_file:
        try:
            with open(voices_file, "w") as f:
                json.dump(voices, f, indent=4)
            logger.info("Updated %s to use precomputed speaker embeddings", voices_file)
        except Exception as e:
            logger.warning("Failed to write back voices.json: %s", e)


def main():
    global tts_model, voices, default_voice, voices_file, SAMPLE_RATE

    args = _parse_args()

    # Build voice registry
    if args.voices and os.path.exists(args.voices):
        voices_file = os.path.abspath(args.voices)
        with open(voices_file) as f:
            voices = json.load(f)
        if voices:
            default_voice = next(iter(voices))
        logger.info("Loaded %d voice(s) from %s", len(voices), voices_file)
    elif args.ref_audio:
        voices = {
            "default": {
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "language": args.language,
            }
        }
        default_voice = "default"
        logger.info("Using single voice from --ref-audio: %s", args.ref_audio)
    else:
        logger.info("No static voices loaded from config or --ref-audio. Relying on dynamically uploaded voices.")

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("Loading model %s on %s …", args.model, args.device)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    SAMPLE_RATE = tts_model.sample_rate
    logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)

    # Pre-warm model cache from precomputed embeddings
    _auto_extract_voices_json_embeddings()  # voices.json WAV → .pt (one-time)
    preload_embedding_voices()              # all .pt files in VOICE_STORAGE_DIR

    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
