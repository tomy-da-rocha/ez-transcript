"""
Transcription engine: auto-selects faster-whisper (CUDA/CPU) or whisper.cpp fallback.
Handles model loading, transcription, progress reporting, and VRAM fallback.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from backend.gpu_utils import GPUInfo, detect_gpu, select_model_size, estimate_batch_size

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = ""
    duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    engine: str = ""
    model: str = ""
    device: str = ""


# Global model cache to avoid reloading
_model_cache: dict = {}


ProgressCallback = Callable[[float, str], None]  # (percent, message)


class _CudaPermanentError(Exception):
    """Raised when CUDA libraries are missing — no point retrying on GPU."""
    pass


def _load_faster_whisper(model_size: str, gpu_info: GPUInfo):
    """Load a faster-whisper model with batched inference pipeline for max GPU utilization."""
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    cache_key = f"faster_whisper_{model_size}_{gpu_info.device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    device = gpu_info.device if gpu_info.device in ("cuda", "cpu") else "cpu"
    compute_type = gpu_info.compute_type

    # Use multiple CPU threads for preprocessing (mel-spectrogram, audio decoding)
    # so the GPU doesn't starve waiting for data
    try:
        import os
        cpu_threads = max(4, os.cpu_count() or 4)
    except Exception:
        cpu_threads = 4

    logger.info(f"Loading faster-whisper model '{model_size}' on {device} (compute: {compute_type}, cpu_threads: {cpu_threads})")

    model_kwargs = dict(
        device=device,
        compute_type=compute_type,
        download_root=str(Path(__file__).resolve().parent / "models"),
        cpu_threads=cpu_threads,
        num_workers=4 if device == "cuda" else 1,
    )

    try:
        base_model = WhisperModel(model_size, **model_kwargs)
    except ValueError as e:
        # Compute type not supported — try int8 on same device, then CPU fallback
        err_msg = str(e).lower()
        if "float16" in err_msg or "compute type" in err_msg or "int8_float16" in err_msg:
            logger.warning(f"compute_type '{compute_type}' not supported on {device}, retrying with int8: {e}")
            model_kwargs["compute_type"] = "int8"
            try:
                base_model = WhisperModel(model_size, **model_kwargs)
            except Exception:
                logger.warning(f"int8 on {device} also failed, falling back to CPU")
                device = "cpu"
                model_kwargs["device"] = "cpu"
                base_model = WhisperModel(model_size, **model_kwargs)
        else:
            raise

    # Wrap in BatchedInferencePipeline for parallel segment processing on GPU
    actual_device = model_kwargs.get("device", device)
    if actual_device == "cuda":
        model = BatchedInferencePipeline(model=base_model)
        logger.info("Using BatchedInferencePipeline for GPU-accelerated parallel decoding")
    else:
        model = base_model

    _model_cache[cache_key] = model
    return model


def transcribe_audio(
    audio_path: Path,
    language: str | None = None,
    progress_callback: ProgressCallback | None = None,
    model_size: str | None = None,
) -> TranscriptionResult:
    """
    Transcribe an audio file using the best available engine.
    Uses an aggressive GPU retry cascade before ever falling back to CPU:
      1. Smart batch_size estimated from VRAM, with one halved retry
      2. Downgrade model on GPU (large-v3 → medium → small)
      3. CPU fallback only as last resort
    """
    from backend.gpu_utils import AVAILABLE_MODELS

    gpu_info = detect_gpu()
    if not model_size:
        model_size = select_model_size(gpu_info)

    if progress_callback:
        progress_callback(0.0, f"Chargement du modèle {model_size}...")

    # --- GPU retry cascade ---
    if gpu_info.device == "cuda":
        vram = gpu_info.vram_free_mb if gpu_info.vram_free_mb > 0 else gpu_info.vram_total_mb

        # Build model fallback list: start from selected, go down
        model_idx = AVAILABLE_MODELS.index(model_size) if model_size in AVAILABLE_MODELS else 0
        models_to_try = AVAILABLE_MODELS[model_idx:]

        gpu_permanently_failed = False

        for try_model in models_to_try:
            if gpu_permanently_failed:
                break

            optimal_batch = estimate_batch_size(try_model, vram)
            # Try optimal, then half, then 1
            batch_sizes = sorted(set([optimal_batch, max(1, optimal_batch // 2), 1]), reverse=True)

            for try_batch in batch_sizes:
                logger.info(f"GPU attempt: model={try_model}, batch_size={try_batch}")
                if progress_callback and (try_model != model_size or try_batch != batch_sizes[0]):
                    progress_callback(
                        0.0,
                        f"Retry GPU: modèle {try_model}, batch {try_batch}..."
                    )

                try:
                    result = _try_transcribe_faster_whisper(
                        audio_path, try_model, gpu_info, language, progress_callback,
                        batch_size=try_batch,
                    )
                except _CudaPermanentError as e:
                    logger.error(f"CUDA libraries missing, skipping all GPU attempts: {e}")
                    if progress_callback:
                        progress_callback(0.0, "CUDA indisponible, basculement sur CPU...")
                    gpu_permanently_failed = True
                    break

                if result is not None:
                    return result

                _clear_gpu_cache(try_model, gpu_info)

            if not gpu_permanently_failed:
                logger.warning(f"All batch sizes failed for {try_model} on GPU, trying smaller model...")

    # --- CPU fallback (last resort) ---
    logger.warning("All GPU attempts exhausted, falling back to CPU")
    if progress_callback:
        progress_callback(0.0, "Basculement sur CPU...")

    cpu_info = GPUInfo(
        available=False, device="cpu", name="CPU",
        vram_total_mb=0, vram_free_mb=0, compute_type="int8",
    )
    # Use the best model that fits in RAM
    cpu_model_size = select_model_size(cpu_info)
    logger.info(f"CPU fallback: using model {cpu_model_size}")
    if progress_callback:
        progress_callback(0.0, f"CPU: chargement du modèle {cpu_model_size}...")

    result = _try_transcribe_faster_whisper(
        audio_path, cpu_model_size, cpu_info, language, progress_callback,
        batch_size=0,
    )

    if result is None:
        raise RuntimeError(
            "La transcription a échoué. Vérifiez que le fichier audio est valide "
            "et que vous avez suffisamment de mémoire disponible."
        )

    return result


def _clear_gpu_cache(model_size: str, gpu_info: GPUInfo):
    """Evict a model from the cache and free GPU memory."""
    cache_key = f"faster_whisper_{model_size}_{gpu_info.device}"
    _model_cache.pop(cache_key, None)
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _try_transcribe_faster_whisper(
    audio_path: Path,
    model_size: str,
    gpu_info: GPUInfo,
    language: str | None,
    progress_callback: ProgressCallback | None,
    batch_size: int = 0,
) -> TranscriptionResult | None:
    """Attempt transcription with faster-whisper. Returns None on VRAM/OOM errors.

    Args:
        batch_size: Number of segments to process in parallel on GPU.
                    0 means sequential (CPU mode, uses beam_size=5).
    """
    try:
        model = _load_faster_whisper(model_size, gpu_info)

        if progress_callback:
            progress_callback(5.0, "Transcription en cours...")

        start_time = time.time()

        kwargs = {
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
        }
        if language:
            kwargs["language"] = language

        if batch_size > 0:
            # BatchedInferencePipeline: process multiple segments in parallel
            kwargs["batch_size"] = batch_size
        else:
            kwargs["beam_size"] = 5

        segments_gen, info = model.transcribe(str(audio_path), **kwargs)

        segments = []
        full_text_parts = []
        duration = info.duration if info.duration else 1.0
        last_progress_time = 0.0

        for seg in segments_gen:
            segments.append(TranscriptionSegment(
                start=round(seg.start, 2),
                end=round(seg.end, 2),
                text=seg.text.strip(),
            ))
            full_text_parts.append(seg.text.strip())

            # Throttle progress updates to max 1/sec to avoid cross-thread sync overhead
            if progress_callback and duration > 0:
                now = time.time()
                if now - last_progress_time >= 1.0:
                    last_progress_time = now
                    pct = min(95.0, 5.0 + (seg.end / duration) * 90.0)
                    progress_callback(pct, f"Transcription... {_fmt_time(seg.end)} / {_fmt_time(duration)}")

        elapsed = time.time() - start_time

        if progress_callback:
            progress_callback(100.0, "Transcription terminée !")

        return TranscriptionResult(
            text=" ".join(full_text_parts),
            segments=segments,
            language=info.language if info.language else "",
            duration_seconds=round(duration, 2),
            processing_time_seconds=round(elapsed, 2),
            engine="faster-whisper",
            model=model_size,
            device=gpu_info.device,
        )

    except Exception as e:
        err_str = str(e).lower()
        # Permanent CUDA failures: missing libraries — no point retrying on GPU
        if any(kw in err_str for kw in ("not found", "cannot be loaded", "no kernel image")):
            logger.error(f"Permanent CUDA failure (missing libraries): {e}")
            raise _CudaPermanentError(str(e))
        # Transient CUDA failures: OOM — worth retrying with smaller batch/model
        if any(kw in err_str for kw in ("out of memory", "cuda", "oom", "cudnn", "cublas")):
            logger.warning(f"VRAM/CUDA error during transcription (model={model_size}, batch={batch_size}): {e}")
            return None
        else:
            logger.error(f"Transcription error: {e}")
            raise


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def generate_srt(segments: list[TranscriptionSegment]) -> str:
    """Generate SRT subtitle content from segments."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start_srt = _seconds_to_srt_time(seg.start)
        end_srt = _seconds_to_srt_time(seg.end)
        lines.append(f"{i}")
        lines.append(f"{start_srt} --> {end_srt}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def result_to_json(result: TranscriptionResult) -> dict:
    """Convert a TranscriptionResult to a JSON-serializable dict."""
    return {
        "text": result.text,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in result.segments
        ],
        "language": result.language,
        "duration_seconds": result.duration_seconds,
        "processing_time_seconds": result.processing_time_seconds,
        "engine": result.engine,
        "model": result.model,
        "device": result.device,
    }
