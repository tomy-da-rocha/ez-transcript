"""
EZ Transcript — FastAPI backend entry point.
Serves the frontend, handles file uploads, runs transcription with SSE progress,
and provides export/download endpoints.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.gpu_utils import get_system_info
from backend.file_handler import (
    validate_file, save_upload, get_upload_info,
    extract_segments, cleanup_file, UPLOAD_DIR, OUTPUT_DIR,
)
from backend.transcriber import (
    transcribe_audio, generate_srt, result_to_json, TranscriptionResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="EZ Transcript", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# In-memory job tracking
jobs: dict[str, dict] = {}


# --- Frontend ---

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# --- System info ---

@app.get("/api/system-info")
async def system_info():
    return get_system_info()


# --- File upload ---

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    error = validate_file(file.filename or "unknown", file.content_type)
    if error:
        raise HTTPException(status_code=400, detail=error)

    try:
        filepath = await save_upload(file, file.filename or "unknown")
        info = get_upload_info(filepath)
        file_id = filepath.stem

        # Store file info for later use
        jobs[file_id] = {
            "file_id": file_id,
            "original_name": file.filename,
            "filepath": str(filepath),
            "info": info,
            "status": "uploaded",
            "result": None,
        }

        return {"file_id": file_id, **info}

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'upload du fichier.")


# --- Media streaming for preview ---

@app.get("/api/media/{file_id}")
async def serve_media(file_id: str):
    """Serve an uploaded media file for preview in the browser."""
    job = jobs.get(file_id)
    if not job:
        raise HTTPException(status_code=404, detail="Fichier non trouvé")

    filepath = Path(job["filepath"])
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Fichier non trouvé sur le disque")

    return FileResponse(
        str(filepath),
        media_type=_guess_media_type(filepath.suffix),
        filename=job.get("original_name", filepath.name),
    )


# --- Transcription with SSE progress ---

@app.post("/api/transcribe/{file_id}")
async def start_transcription(file_id: str, request: Request):
    """Start transcription and stream progress via SSE."""
    job = jobs.get(file_id)
    if not job:
        raise HTTPException(status_code=404, detail="Fichier non trouvé")

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    segments = body.get("segments", [])
    language = body.get("language") or None
    model = body.get("model") or None

    filepath = Path(job["filepath"])

    async def event_stream():
        loop = asyncio.get_event_loop()
        progress_queue: asyncio.Queue = asyncio.Queue()

        def progress_callback(percent: float, message: str):
            try:
                loop.call_soon_threadsafe(progress_queue.put_nowait, (percent, message))
            except Exception:
                pass

        async def run_transcription():
            try:
                # Extract audio segments or pass original file directly
                if segments:
                    audio_paths = await loop.run_in_executor(
                        None, extract_segments, filepath, segments
                    )
                    temp_files = list(audio_paths)
                else:
                    # No extraction needed — faster-whisper reads all formats directly
                    audio_paths = [filepath]
                    temp_files = []

                all_results = []
                total_segments = len(audio_paths)

                for i, audio_path in enumerate(audio_paths):
                    seg_label = segments[i].get("label", f"Segment {i+1}") if segments else "Complet"

                    def seg_progress(pct, msg, _idx=i, _total=total_segments):
                        base = (_idx / _total) * 100 if _total > 1 else 0
                        scaled = (pct / _total) if _total > 1 else pct
                        progress_callback(base + scaled, f"[{seg_label}] {msg}")

                    result = await loop.run_in_executor(
                        None, transcribe_audio, audio_path, language, seg_progress, model
                    )
                    all_results.append(result)

                    # Cleanup temp audio (only extracted segments)
                    if audio_path in temp_files:
                        cleanup_file(audio_path)

                # Merge results
                merged = _merge_results(all_results, segments)

                # Save to output
                output_data = result_to_json(merged)
                output_data["original_name"] = job.get("original_name", "")
                output_data["timestamp"] = datetime.now().isoformat()
                output_data["file_id"] = file_id

                output_file = OUTPUT_DIR / f"{file_id}.json"
                output_file.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")

                job["status"] = "completed"
                job["result"] = output_data

                return output_data

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                return {"error": str(e)}

        # Start transcription in background
        task = asyncio.create_task(run_transcription())

        # Stream progress events
        while not task.done():
            try:
                percent, message = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield f"data: {json.dumps({'type': 'progress', 'percent': percent, 'message': message})}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

        # Drain remaining progress messages
        while not progress_queue.empty():
            percent, message = progress_queue.get_nowait()
            yield f"data: {json.dumps({'type': 'progress', 'percent': percent, 'message': message})}\n\n"

        result = await task

        if "error" in result:
            yield f"data: {json.dumps({'type': 'error', 'message': result['error']})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'complete', 'result': result})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Results & Export ---

@app.get("/api/result/{file_id}")
async def get_result(file_id: str):
    """Get transcription result for a file."""
    output_file = OUTPUT_DIR / f"{file_id}.json"
    if output_file.exists():
        data = json.loads(output_file.read_text(encoding="utf-8"))
        return data

    job = jobs.get(file_id)
    if job and job.get("result"):
        return job["result"]

    raise HTTPException(status_code=404, detail="Résultat non trouvé")


@app.get("/api/export/{file_id}/{format}")
async def export_result(file_id: str, format: str):
    """Export transcription result in txt, srt, or json format."""
    output_file = OUTPUT_DIR / f"{file_id}.json"
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Résultat non trouvé")

    data = json.loads(output_file.read_text(encoding="utf-8"))
    original_name = Path(data.get("original_name", "transcription")).stem

    if format == "txt":
        content = data.get("text", "")
        return StreamingResponse(
            iter([content]),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{original_name}.txt"'},
        )

    elif format == "srt":
        from backend.transcriber import TranscriptionSegment
        segs = [TranscriptionSegment(**s) for s in data.get("segments", [])]
        srt_content = generate_srt(segs)
        return StreamingResponse(
            iter([srt_content]),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{original_name}.srt"'},
        )

    elif format == "json":
        return StreamingResponse(
            iter([json.dumps(data, ensure_ascii=False, indent=2)]),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{original_name}.json"'},
        )

    else:
        raise HTTPException(status_code=400, detail=f"Format non supporté : {format}")


# --- History ---

@app.get("/api/history")
async def get_history():
    """List all previous transcriptions from the output directory."""
    history = []
    for f in sorted(OUTPUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            history.append({
                "file_id": f.stem,
                "original_name": data.get("original_name", f.stem),
                "timestamp": data.get("timestamp", ""),
                "language": data.get("language", ""),
                "duration_seconds": data.get("duration_seconds", 0),
                "model": data.get("model", ""),
                "device": data.get("device", ""),
            })
        except Exception:
            continue
    return history


@app.delete("/api/history/{file_id}")
async def delete_history_entry(file_id: str):
    """Delete a transcription result."""
    output_file = OUTPUT_DIR / f"{file_id}.json"
    if output_file.exists():
        output_file.unlink()
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Entrée non trouvée")


# --- Helpers ---

def _merge_results(
    results: list[TranscriptionResult],
    segments_config: list[dict],
) -> TranscriptionResult:
    """Merge multiple transcription results into one."""
    if len(results) == 1:
        return results[0]

    all_text = []
    all_segments = []
    total_duration = 0.0
    total_processing = 0.0

    for i, result in enumerate(results):
        offset = segments_config[i].get("start", 0) if segments_config else 0
        all_text.append(result.text)
        for seg in result.segments:
            all_segments.append(type(seg)(
                start=round(seg.start + offset, 2),
                end=round(seg.end + offset, 2),
                text=seg.text,
            ))
        total_duration += result.duration_seconds
        total_processing += result.processing_time_seconds

    first = results[0]
    return TranscriptionResult(
        text="\n\n".join(all_text),
        segments=all_segments,
        language=first.language,
        duration_seconds=round(total_duration, 2),
        processing_time_seconds=round(total_processing, 2),
        engine=first.engine,
        model=first.model,
        device=first.device,
    )


def _guess_media_type(ext: str) -> str:
    mapping = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".wma": "audio/x-ms-wma",
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
    }
    return mapping.get(ext.lower(), "application/octet-stream")


# Mount static files AFTER all API routes to avoid shadowing
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
