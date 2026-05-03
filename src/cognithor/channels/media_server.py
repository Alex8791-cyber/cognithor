"""MediaUploadServer — local HTTP file-server for vLLM to fetch user uploads.

vLLM inside the Cognithor-managed Docker container reaches this server via
``http://host.docker.internal:<port>/media/<uuid>.<ext>``. The server binds
only on ``127.0.0.1``, so only processes on the host machine can reach it.

This file contains the storage + quota logic. The FastAPI app + port binding
live in companion methods `start()` / `stop()` added in Task 7.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import uuid as _uuid
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from cognithor.core.llm_backend import (
    MediaUploadQuotaExceededError,
    MediaUploadTooLargeError,
    MediaUploadUnsupportedFormatError,
)
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from cognithor.config import CognithorConfig

log = get_logger(__name__)

_ALLOWED_EXTS = frozenset({"mp4", "webm", "mov", "mkv", "avi"})


class MediaUploadServer:
    """Local-loopback static file server for vLLM media fetches.

    Lifecycle: instantiate with the live CognithorConfig, call ``await start()``
    to bind the ephemeral port (returns the port number), ``await stop()`` at
    shutdown. In between, call ``save_upload(data, ext) -> uuid`` to store
    bytes and ``public_url(uuid, ext) -> str`` to get the URL vLLM should fetch.
    """

    def __init__(self, config: CognithorConfig) -> None:
        self._config = config
        self._media_dir = config.cognithor_home / "media" / "vllm-uploads"
        self._media_dir.mkdir(parents=True, exist_ok=True)
        self._max_per_file_bytes = config.vllm.video_max_upload_mb * 1024 * 1024
        self._quota_bytes = config.vllm.video_quota_gb * 1024 * 1024 * 1024
        self._port: int | None = None
        self._server = None  # filled by start() in Task 7
        self._serve_task: asyncio.Task | None = None
        # Guards the evict+write critical section in save_upload. save_upload
        # is called from async FastAPI handlers via asyncio.to_thread, so
        # threading.Lock (not asyncio.Lock) is the right primitive here.
        self._lock = threading.Lock()

    def save_upload(self, data: bytes, ext: str) -> str:
        """Store ``data`` under ``<uuid>.<ext>`` in the media dir, return uuid.

        Raises MediaUploadTooLargeError / MediaUploadUnsupportedFormatError /
        MediaUploadQuotaExceededError on the respective failure modes. LRU-
        evicts older files if the new upload would push total size over quota.
        """
        ext_lower = ext.lower().lstrip(".")
        if ext_lower not in _ALLOWED_EXTS:
            raise MediaUploadUnsupportedFormatError(
                f"Unsupported extension: {ext!r}. Allowed: {sorted(_ALLOWED_EXTS)}",
                status_code=400,
            )
        if len(data) > self._max_per_file_bytes:
            mb = len(data) / 1024 / 1024
            cap = self._max_per_file_bytes / 1024 / 1024
            raise MediaUploadTooLargeError(
                f"Upload is {mb:.1f} MB, max per file is {cap:.0f} MB",
                status_code=413,
                recovery_hint="Shorten or downscale the clip before uploading.",
            )
        if len(data) > self._quota_bytes:
            raise MediaUploadQuotaExceededError(
                f"Upload alone ({len(data) / 1024 / 1024:.1f} MB) exceeds the full quota"
                f" ({self._quota_bytes / 1024 / 1024 / 1024:.1f} GB)",
                status_code=413,
                recovery_hint="Raise config.vllm.video_quota_gb or shrink the file.",
            )

        # LRU eviction + write must be atomic relative to other save_upload
        # calls, otherwise two concurrent uploads can both pass the quota
        # check and both write, leaving the directory above quota.
        with self._lock:
            self._evict_until_fits(len(data))
            uuid_str = _uuid.uuid4().hex
            path = self._media_dir / f"{uuid_str}.{ext_lower}"
            path.write_bytes(data)
        log.info(
            "video_upload_saved",
            uuid=uuid_str,
            ext=ext_lower,
            bytes=len(data),
        )
        return uuid_str

    def _evict_until_fits(self, incoming_bytes: int) -> None:
        """Delete oldest files (by mtime) until adding ``incoming_bytes`` fits under quota.

        Snapshots (path, size, mtime) once per file to avoid racy double-stat
        calls (the file could vanish between two stat() invocations on a
        busy directory).
        """
        entries: list[tuple[Path, int, float]] = []
        for p in self._media_dir.iterdir():
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((p, st.st_size, st.st_mtime))

        current = sum(size for _, size, _ in entries)
        if current + incoming_bytes <= self._quota_bytes:
            return

        entries.sort(key=lambda e: e[2])  # oldest first
        for f, size, _ in entries:
            if current + incoming_bytes <= self._quota_bytes:
                break
            try:
                f.unlink()
            except OSError as exc:
                log.warning("video_evict_failed", file=str(f), error=str(exc))
                continue
            # Also drop sidecar thumbnail if present
            thumb = f.with_suffix(".jpg")
            if thumb.exists():
                with contextlib.suppress(OSError):
                    thumb.unlink()
            current -= size
            log.info("video_evicted_lru", file=f.name, freed_bytes=size)

    def delete(self, uuid: str, ext: str) -> None:
        """Remove a specific upload (and its thumbnail). Noop if missing."""
        ext_lower = ext.lower().lstrip(".")
        path = self._media_dir / f"{uuid}.{ext_lower}"
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                log.warning("video_delete_failed", uuid=uuid, error=str(exc))
        thumb = self._media_dir / f"{uuid}.jpg"
        if thumb.exists():
            with contextlib.suppress(OSError):
                thumb.unlink()

    def public_url(self, uuid: str, ext: str) -> str:
        """Return the URL vLLM should fetch: ``http://host.docker.internal:<port>/media/<uuid>.<ext>``.

        Requires ``start()`` to have been called (or ``_port`` manually set in tests).
        """
        if self._port is None:
            raise RuntimeError("MediaUploadServer not started; call await start() first")
        ext_lower = ext.lower().lstrip(".")
        return f"http://host.docker.internal:{self._port}/media/{uuid}.{ext_lower}"

    async def start(self) -> int:
        """Bind to 127.0.0.1:<ephemeral>, serve /media/<uuid>.<ext> as static files.

        Returns the bound port. Call ``await stop()`` at shutdown.
        """
        # Pick an ephemeral port ourselves so we can tell the orchestrator before
        # uvicorn actually binds (uvicorn accepts 0 but only reports after start).
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            self._port = probe.getsockname()[1]

        app = FastAPI(title="Cognithor MediaUploadServer", openapi_url=None, docs_url=None)

        @app.get("/media/{filename}")
        async def serve(filename: str) -> FileResponse:
            # Defense-in-depth: resolve the full path and verify it's still
            # inside media_dir. Substring checks like `"/" in filename` miss
            # Windows absolute paths (e.g. `C:\Windows\...`) because pathlib
            # treats absolute args as replacing the base.
            media_dir = self._media_dir
            path = media_dir / filename
            try:
                resolved = path.resolve()
                if not resolved.is_relative_to(media_dir.resolve()):
                    raise HTTPException(status_code=400, detail="invalid filename")
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="invalid filename") from exc
            if not path.is_file():
                raise HTTPException(status_code=404, detail="not found")
            return FileResponse(path)

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._serve_task = asyncio.create_task(self._server.serve())
        # Wait for startup (server.started flips true when uvicorn is ready)
        for _ in range(50):
            if self._server.started:
                break
            await asyncio.sleep(0.05)
        log.info("media_server_started", port=self._port)
        return self._port

    async def stop(self) -> None:
        """Shut down the uvicorn serving loop. Idempotent."""
        if self._server is None:
            return
        self._server.should_exit = True
        try:
            await self._serve_task
        except Exception as exc:
            log.warning("media_server_stop_error", error=str(exc))
        self._server = None
        self._serve_task = None
        log.info("media_server_stopped")
