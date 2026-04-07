"""
File upload handling, validation, and FFmpeg audio extraction/segmentation.
"""

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".mp3", ".mp4", ".wav", ".ogg", ".flac",
    ".mkv", ".avi", ".webm", ".m4a", ".wma", ".aac",
}

ALLOWED_MIME_PREFIXES = ("audio/", "video/")


def validate_file(filename: str, content_type: str | None) -> str | None:
    """Validate file extension and MIME type. Returns error message or None."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"Format non supporté : {ext}. Formats acceptés : {', '.join(sorted(ALLOWED_EXTENSIONS))}"
    if content_type and not any(content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        if content_type != "application/octet-stream":
            return f"Type MIME non supporté : {content_type}"
    return None


async def save_upload(file, filename: str) -> Path:
    """Save an uploaded file to the uploads directory with a unique name."""
    ext = Path(filename).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / safe_name

    # Stream write for large files
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            f.write(chunk)

    logger.info(f"Saved upload: {filename} -> {dest} ({dest.stat().st_size} bytes)")
    return dest


def get_media_duration(filepath: Path) -> float:
    """Get media duration in seconds using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe introuvable. Veuillez installer FFmpeg.")
    try:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Failed to get duration for {filepath}: {e}")
        raise RuntimeError("Impossible de lire la durée du fichier. Vérifiez que le fichier est valide.")


def extract_audio(input_path: Path, start: float | None = None, end: float | None = None) -> Path:
    """
    Extract audio from a media file as WAV (16kHz mono) for Whisper.
    Optionally trim to [start, end] seconds.
    Returns path to the extracted WAV file.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg introuvable. Veuillez installer FFmpeg.")

    output_name = f"{uuid.uuid4().hex}.wav"
    output_path = UPLOAD_DIR / output_name

    cmd = [ffmpeg, "-y", "-threads", "0", "-i", str(input_path)]

    if start is not None:
        cmd.extend(["-ss", str(start)])
    if end is not None:
        cmd.extend(["-to", str(end)])

    # Convert to 16kHz mono WAV for Whisper
    cmd.extend([
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ])

    logger.info(f"Extracting audio: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise RuntimeError("Erreur lors de l'extraction audio. Vérifiez que le fichier est valide.")

    return output_path


def extract_segments(input_path: Path, segments: list[dict]) -> list[Path]:
    """
    Extract multiple segments from a media file.
    Each segment dict has: start (float), end (float), label (str, optional).
    Returns list of WAV file paths.
    """
    paths = []
    for seg in segments:
        wav = extract_audio(input_path, start=seg.get("start"), end=seg.get("end"))
        paths.append(wav)
    return paths


def cleanup_file(filepath: Path):
    """Remove a temporary file."""
    try:
        if filepath.exists():
            filepath.unlink()
    except Exception as e:
        logger.warning(f"Failed to clean up {filepath}: {e}")


def get_upload_info(filepath: Path) -> dict:
    """Get info about an uploaded file."""
    duration = get_media_duration(filepath)
    ext = filepath.suffix.lower()
    is_video = ext in {".mp4", ".mkv", ".avi", ".webm"}

    return {
        "path": str(filepath),
        "filename": filepath.name,
        "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
        "duration_seconds": round(duration, 2),
        "duration_display": _format_duration(duration),
        "is_video": is_video,
        "extension": ext,
    }


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
