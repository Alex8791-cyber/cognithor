"""/api/media/upload + /api/media/thumb FastAPI routes.

Separate from the MediaUploadServer (which only serves vLLM fetches). These
endpoints are invoked by the Flutter client and run on the main Cognithor API
port. They delegate the actual storage to MediaUploadServer.save_upload and
run ffprobe/ffmpeg via asyncio.to_thread so the uvicorn event loop stays
responsive even for 500 MB uploads or slow remote URLs.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from cognithor.core.llm_backend import (
    MediaUploadError,
    MediaUploadQuotaExceededError,
    MediaUploadTooLargeError,
    MediaUploadUnsupportedFormatError,
)
from cognithor.core.video_sampling import resolve_sampling
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.channels.media_server import MediaUploadServer
    from cognithor.config import CognithorConfig

log = get_logger(__name__)

media_router = APIRouter(prefix="/api/media", tags=["media"])


@media_router.post("/upload")
async def upload_video(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    config: CognithorConfig = request.app.state.config
    media_server: MediaUploadServer = request.app.state.media_server

    data = await file.read()
    filename = file.filename or "video.mp4"
    ext = Path(filename).suffix.lstrip(".")

    try:
        uuid = media_server.save_upload(data, ext)
    except MediaUploadTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail={"message": str(exc), "recovery_hint": exc.recovery_hint},
        ) from exc
    except MediaUploadUnsupportedFormatError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
    except MediaUploadQuotaExceededError as exc:
        raise HTTPException(
            status_code=507,
            detail={"message": str(exc), "recovery_hint": exc.recovery_hint},
        ) from exc
    except MediaUploadError as exc:
        raise HTTPException(status_code=507, detail={"message": str(exc)}) from exc

    saved_path = media_server._media_dir / f"{uuid}.{ext.lower()}"

    await asyncio.to_thread(
        _extract_thumbnail,
        saved_path,
        media_server._media_dir / f"{uuid}.jpg",
        ffmpeg_path="ffmpeg",
    )

    sampling = await asyncio.to_thread(
        resolve_sampling,
        str(saved_path),
        ffprobe_path=config.vllm.video_ffprobe_path,
        timeout_seconds=config.vllm.video_ffprobe_timeout_seconds,
        http_timeout_seconds=config.vllm.video_ffprobe_http_timeout_seconds,
        override=config.vllm.video_sampling_mode,
    )

    return {
        "uuid": uuid,
        "url": media_server.public_url(uuid, ext),
        "duration_sec": sampling.duration_sec,
        "sampling": sampling.as_mm_kwargs(),
        "thumb_url": f"/api/media/thumb/{uuid}.jpg",
    }


@media_router.get("/thumb/{filename}")
async def thumb(request: Request, filename: str) -> FileResponse:
    media_server: MediaUploadServer = request.app.state.media_server
    media_dir = media_server._media_dir
    path = media_dir / filename
    try:
        resolved = path.resolve()
        if not resolved.is_relative_to(media_dir.resolve()):
            raise HTTPException(status_code=400, detail="invalid filename")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid filename") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="image/jpeg")


def _extract_thumbnail(source: Path, dest: Path, *, ffmpeg_path: str = "ffmpeg") -> bool:
    """Best-effort: extract the first frame as JPEG. Returns True on success."""
    if not shutil.which(ffmpeg_path) and not Path(ffmpeg_path).is_file():
        return False
    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-ss",
                "0",
                "-i",
                str(source),
                "-vframes",
                "1",
                "-vf",
                "scale=192:108",
                str(dest),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        log.warning("thumbnail_extract_failed", error=str(exc))
        return False
    return dest.is_file()


def build_media_app(
    *,
    config: CognithorConfig,
    media_server: MediaUploadServer,
) -> FastAPI:
    """Standalone FastAPI app for tests. In production the media_router is
    included into the main APIChannel app via app.include_router()."""
    app = FastAPI()
    app.state.config = config
    app.state.media_server = media_server
    app.include_router(media_router)
    return app
