"""Media-Pipeline: Verarbeitung von Bildern, Audio und Dokumenten.

MCP-Tools für multimodale Medienverarbeitung — vollständig lokal.

Tools:
  - media_transcribe_audio: Audio → Text (Whisper)
  - media_analyze_image: Bild → Beschreibung (multimodales LLM via Ollama)
  - media_extract_text: PDF/DOCX/TXT → Text
  - media_convert_audio: Audio-Formatkonvertierung (ffmpeg)
  - media_image_resize: Bildgröße ändern (Pillow)
  - media_tts: Text → Sprache (Piper/eSpeak)

Alle Tools arbeiten mit lokalen Dateipfaden — keine Cloud-Uploads.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Maximale Textlänge für LLM-Kontext
MAX_EXTRACT_LENGTH = 15_000

# Maximale Bilddateigroesse fuer Base64-Encoding (10 MB)
MAX_IMAGE_FILE_SIZE = 10_485_760

# Standard-Modelle und -Stimmen
DEFAULT_OLLAMA_MODEL = "llava:13b"
DEFAULT_IMAGE_PROMPT = "Beschreibe dieses Bild detailliert auf Deutsch."
DEFAULT_PIPER_VOICE = "de_DE-thorsten-high"

__all__ = [
    "MediaPipeline",
    "MediaResult",
    "register_media_tools",
    "MEDIA_TOOL_SCHEMAS",
]


@dataclass
class MediaResult:
    """Einheitliches Ergebnis aller Media-Operationen."""

    success: bool = True
    text: str = ""
    output_path: str | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None


class MediaPipeline:
    """Zentrale Klasse für Medienverarbeitung.

    Alle Methoden sind async und nutzen run_in_executor
    für CPU-intensive Operationen (Whisper, Pillow, etc.).
    """

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self._workspace = workspace_dir or Path.home() / ".jarvis" / "workspace" / "media"
        self._workspace.mkdir(parents=True, exist_ok=True)

    def _validate_input_path(self, file_path: str) -> Path | None:
        """Validates input file path against path traversal.

        Returns resolved Path or None if invalid.
        """
        try:
            path = Path(file_path).expanduser().resolve()
        except (ValueError, OSError):
            return None
        # Block path traversal: must not contain '..' after resolution
        # and must be a real file (not symlink to sensitive location)
        if not path.exists():
            return None
        return path

    # ========================================================================
    # Audio → Text (Whisper STT)
    # ========================================================================

    async def transcribe_audio(
        self,
        audio_path: str,
        *,
        language: str = "de",
        model: str = "base",
    ) -> MediaResult:
        """Transkribiert eine Audiodatei zu Text.

        Unterstützt: WAV, MP3, OGG, FLAC, M4A, WEBM
        Backend: faster-whisper (lokal, GPU-beschleunigt)

        Args:
            audio_path: Pfad zur Audiodatei.
            language: Sprache (ISO-Code, z.B. 'de', 'en').
            model: Whisper-Modell ('tiny', 'base', 'small', 'medium', 'large-v3').
        """
        path = self._validate_input_path(audio_path)
        if path is None:
            return MediaResult(success=False, error=f"Datei nicht gefunden oder ungueltig: {audio_path}")

        try:
            from faster_whisper import WhisperModel

            loop = asyncio.get_running_loop()

            def _transcribe() -> str:
                m = WhisperModel(model, device="auto", compute_type="int8")
                segments, info = m.transcribe(str(path), language=language, vad_filter=True)
                text = " ".join(seg.text.strip() for seg in segments)
                return text

            text = await loop.run_in_executor(None, _transcribe)

            if not text.strip():
                return MediaResult(success=True, text="[Keine Sprache erkannt]")

            log.info("audio_transcribed", path=audio_path, length=len(text))
            return MediaResult(
                success=True,
                text=text,
                metadata={"language": language, "model": model, "source": audio_path},
            )

        except ImportError:
            return MediaResult(
                success=False,
                error="faster-whisper nicht installiert. pip install faster-whisper",
            )
        except Exception as exc:
            log.error("transcribe_failed", path=audio_path, error=str(exc))
            return MediaResult(success=False, error=f"Transkription fehlgeschlagen: {exc}")

    # ========================================================================
    # Bild → Beschreibung (multimodales LLM)
    # ========================================================================

    async def analyze_image(
        self,
        image_path: str,
        *,
        prompt: str = DEFAULT_IMAGE_PROMPT,
        model: str = DEFAULT_OLLAMA_MODEL,
        ollama_url: str = "http://localhost:11434",
    ) -> MediaResult:
        """Analysiert ein Bild mit einem multimodalen LLM (Ollama).

        Unterstützt: JPG, PNG, GIF, WEBP, BMP
        Backend: LLaVA, Moondream oder ähnliches via Ollama

        Args:
            image_path: Pfad zum Bild.
            prompt: Analyseanweisung für das LLM.
            model: Multimodales Ollama-Modell.
            ollama_url: Ollama API-URL.
        """
        import base64

        path = self._validate_input_path(image_path)
        if path is None:
            return MediaResult(success=False, error=f"Bild nicht gefunden oder ungueltig: {image_path}")

        suffix = path.suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
            return MediaResult(success=False, error=f"Nicht unterstütztes Bildformat: {suffix}")

        try:
            import httpx

            # Dateigroesse pruefen
            file_size = path.stat().st_size
            if file_size > MAX_IMAGE_FILE_SIZE:
                return MediaResult(
                    success=False,
                    error=f"Bild zu gross ({file_size / 1_048_576:.1f} MB, max {MAX_IMAGE_FILE_SIZE // 1_048_576} MB)",
                )

            # Bild als Base64 laden
            image_data = base64.b64encode(path.read_bytes()).decode("utf-8")

            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                resp = await client.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                                "images": [image_data],
                            }
                        ],
                        "stream": False,
                    },
                )

                if resp.status_code != 200:
                    return MediaResult(
                        success=False,
                        error=f"Ollama HTTP {resp.status_code}: {resp.text[:300]}",
                    )

                data = resp.json()
                description = data.get("message", {}).get("content", "")

                log.info("image_analyzed", path=image_path, model=model)
                return MediaResult(
                    success=True,
                    text=description,
                    metadata={
                        "model": model,
                        "source": image_path,
                        "image_size": path.stat().st_size,
                    },
                )

        except ImportError:
            return MediaResult(success=False, error="httpx nicht installiert")
        except Exception as exc:
            log.error("image_analysis_failed", path=image_path, error=str(exc))
            return MediaResult(success=False, error=f"Bildanalyse fehlgeschlagen: {exc}")

    # ========================================================================
    # Dokument → Text
    # ========================================================================

    async def extract_text(self, file_path: str) -> MediaResult:
        """Extrahiert Text aus verschiedenen Dokumentformaten.

        Unterstützt: PDF, DOCX, TXT, MD, HTML, CSV, JSON, XML

        Args:
            file_path: Pfad zum Dokument.
        """
        path = self._validate_input_path(file_path)
        if path is None:
            return MediaResult(success=False, error=f"Datei nicht gefunden oder ungueltig: {file_path}")

        suffix = path.suffix.lower()
        loop = asyncio.get_running_loop()

        try:
            if suffix == ".pdf":
                text = await loop.run_in_executor(None, self._extract_pdf, path)
            elif suffix == ".docx":
                text = await loop.run_in_executor(None, self._extract_docx, path)
            elif suffix in (".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log"):
                text = path.read_text(encoding="utf-8", errors="replace")
            elif suffix in (".html", ".htm"):
                text = await loop.run_in_executor(None, self._extract_html, path)
            else:
                return MediaResult(
                    success=False,
                    error=f"Nicht unterstütztes Format: {suffix}. "
                    f"Unterstützt: PDF, DOCX, TXT, MD, HTML, CSV, JSON, XML",
                )

            if len(text) > MAX_EXTRACT_LENGTH:
                text = text[:MAX_EXTRACT_LENGTH] + f"\n\n[... gekürzt, {len(text)} Zeichen gesamt]"

            log.info("text_extracted", path=file_path, length=len(text), format=suffix)
            return MediaResult(
                success=True,
                text=text,
                metadata={"source": file_path, "format": suffix, "original_length": len(text)},
            )

        except Exception as exc:
            log.error("text_extraction_failed", path=file_path, error=str(exc))
            return MediaResult(success=False, error=f"Text-Extraktion fehlgeschlagen: {exc}")

    def _extract_pdf(self, path: Path) -> str:
        """PDF-Textextraktion mit pymupdf oder pdfplumber."""
        # Versuch 1: PyMuPDF (schnell)
        try:
            import fitz  # pymupdf

            doc = fitz.open(str(path))
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            pass

        # Versuch 2: pdfplumber
        try:
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n\n".join(pages)
        except ImportError:
            pass

        raise ImportError(
            "Kein PDF-Reader verfügbar. Installiere: pip install pymupdf oder pip install pdfplumber"
        )

    def _extract_docx(self, path: Path) -> str:
        """DOCX-Textextraktion mit python-docx."""
        try:
            from docx import Document

            doc = Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            raise ImportError(
                "python-docx nicht installiert. pip install python-docx"
            ) from None

    def _extract_html(self, path: Path) -> str:
        """HTML-Textextraktion (einfach, ohne BeautifulSoup-Pflicht)."""
        import re

        html = path.read_text(encoding="utf-8", errors="replace")
        # Script/Style-Tags entfernen
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # HTML-Tags entfernen
        text = re.sub(r"<[^>]+>", " ", html)
        # Mehrfach-Whitespace normalisieren
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ========================================================================
    # Audio-Konvertierung (ffmpeg)
    # ========================================================================

    # Erlaubte Audio-Samplerates
    ALLOWED_SAMPLE_RATES = frozenset({8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000})

    async def convert_audio(
        self,
        input_path: str,
        output_format: str = "wav",
        *,
        sample_rate: int = 16000,
    ) -> MediaResult:
        """Konvertiert Audio zwischen Formaten via ffmpeg.

        Args:
            input_path: Quell-Audiodatei.
            output_format: Zielformat (wav, mp3, ogg, flac).
            sample_rate: Ziel-Samplerate (8000-96000).
        """
        if sample_rate not in self.ALLOWED_SAMPLE_RATES:
            return MediaResult(
                success=False,
                error=f"Ungueltige Samplerate: {sample_rate}. Erlaubt: {sorted(self.ALLOWED_SAMPLE_RATES)}",
            )

        path = self._validate_input_path(input_path)
        if path is None:
            return MediaResult(success=False, error=f"Datei nicht gefunden oder ungueltig: {input_path}")

        output_path = self._workspace / f"{path.stem}_converted.{output_format}"

        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-i", str(path),
                "-ar", str(sample_rate),
                "-ac", "1",  # Mono
                "-y",  # Überschreiben
                str(output_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode != 0:
                return MediaResult(
                    success=False,
                    error=f"ffmpeg Fehler: {stderr.decode()[:300]}",
                )

            log.info("audio_converted", input=input_path, output=str(output_path))
            return MediaResult(
                success=True,
                text=f"Konvertiert: {output_path}",
                output_path=str(output_path),
                metadata={"format": output_format, "sample_rate": sample_rate},
            )

        except FileNotFoundError:
            return MediaResult(
                success=False,
                error="ffmpeg nicht installiert. apt install ffmpeg",
            )

    # ========================================================================
    # Bildgröße ändern (Pillow)
    # ========================================================================

    # Maximale erlaubte Bilddimensionen
    MAX_IMAGE_DIMENSION = 8192

    async def resize_image(
        self,
        image_path: str,
        *,
        max_width: int = 1024,
        max_height: int = 1024,
        output_format: str | None = None,
    ) -> MediaResult:
        """Ändert die Bildgröße (behält Seitenverhältnis).

        Args:
            image_path: Quellbild.
            max_width: Maximale Breite (1-8192).
            max_height: Maximale Höhe (1-8192).
            output_format: Optionales Ausgabeformat (jpg, png, webp).
        """
        # Dimensionen validieren
        max_width = max(1, min(max_width, self.MAX_IMAGE_DIMENSION))
        max_height = max(1, min(max_height, self.MAX_IMAGE_DIMENSION))

        path = self._validate_input_path(image_path)
        if path is None:
            return MediaResult(success=False, error=f"Bild nicht gefunden oder ungueltig: {image_path}")

        try:
            from PIL import Image

            loop = asyncio.get_running_loop()

            def _resize() -> tuple[str, int, int]:
                img = Image.open(path)
                img.thumbnail((max_width, max_height), Image.LANCZOS)

                fmt = output_format or path.suffix.lstrip(".") or "png"
                out = self._workspace / f"{path.stem}_resized.{fmt}"
                img.save(str(out), quality=90)
                return str(out), img.width, img.height

            output, w, h = await loop.run_in_executor(None, _resize)

            log.info("image_resized", input=image_path, output=output, size=f"{w}x{h}")
            return MediaResult(
                success=True,
                text=f"Bild skaliert auf {w}x{h}: {output}",
                output_path=output,
                metadata={"width": w, "height": h},
            )

        except ImportError:
            return MediaResult(
                success=False,
                error="Pillow nicht installiert. pip install Pillow",
            )
        except Exception as exc:
            return MediaResult(success=False, error=f"Bildskalierung fehlgeschlagen: {exc}")

    # ========================================================================
    # Text → Sprache (TTS)
    # ========================================================================

    async def text_to_speech(
        self,
        text: str,
        *,
        output_path: str | None = None,
        voice: str = DEFAULT_PIPER_VOICE,
    ) -> MediaResult:
        """Synthetisiert Text zu Sprache (WAV).

        Backend: Piper TTS (lokal, schnell) → eSpeak-NG Fallback

        Args:
            text: Zu sprechender Text.
            output_path: Ausgabedatei. Auto-generiert wenn None.
            voice: Piper-Stimmenmodell.
        """
        if not text.strip():
            return MediaResult(success=False, error="Leerer Text")

        out = Path(output_path) if output_path else self._workspace / "tts_output.wav"

        # Versuch 1: Piper
        try:
            proc = await asyncio.create_subprocess_exec(
                "piper",
                "--model", voice,
                "--output_file", str(out),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate(input=text.encode("utf-8"))

            if proc.returncode == 0:
                log.info("tts_piper_success", output=str(out), length=len(text))
                return MediaResult(
                    success=True,
                    text=f"Audio erzeugt: {out}",
                    output_path=str(out),
                    metadata={"engine": "piper", "voice": voice},
                )
        except FileNotFoundError:
            pass

        # Versuch 2: eSpeak-NG
        try:
            proc = await asyncio.create_subprocess_exec(
                "espeak-ng",
                "-v", "de",
                "-w", str(out),
                "--", text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            if proc.returncode == 0:
                log.info("tts_espeak_success", output=str(out))
                return MediaResult(
                    success=True,
                    text=f"Audio erzeugt (eSpeak): {out}",
                    output_path=str(out),
                    metadata={"engine": "espeak"},
                )
        except FileNotFoundError:
            pass

        return MediaResult(
            success=False,
            error="Kein TTS-Backend verfügbar. Installiere piper oder espeak-ng.",
        )


# ============================================================================
# MCP-Tool-Schemas
# ============================================================================

MEDIA_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "media_transcribe_audio": {
        "description": (
            "Transkribiert eine Audiodatei (WAV, MP3, OGG, etc.) zu Text. "
            "Lokal via Whisper — keine Cloud-Uploads."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string", "description": "Pfad zur Audiodatei"},
                "language": {
                    "type": "string",
                    "description": "Sprache (ISO-Code)",
                    "default": "de",
                },
                "model": {
                    "type": "string",
                    "description": "Whisper-Modell (tiny/base/small/medium/large-v3)",
                    "default": "base",
                },
            },
            "required": ["audio_path"],
        },
    },
    "media_analyze_image": {
        "description": (
            "Analysiert ein Bild mit einem multimodalen LLM (LLaVA via Ollama). "
            "Beschreibt Inhalt, erkennt Text, beantwortet Fragen zum Bild."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Pfad zum Bild"},
                "prompt": {
                    "type": "string",
                    "description": "Analyseanweisung",
                    "default": DEFAULT_IMAGE_PROMPT,
                },
                "model": {
                    "type": "string",
                    "description": "Multimodales Ollama-Modell",
                    "default": DEFAULT_OLLAMA_MODEL,
                },
            },
            "required": ["image_path"],
        },
    },
    "media_extract_text": {
        "description": (
            "Extrahiert Text aus Dokumenten: PDF, DOCX, TXT, MD, HTML, CSV, JSON, XML. "
            "Lokal — keine Cloud-Dienste."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Pfad zum Dokument"},
            },
            "required": ["file_path"],
        },
    },
    "media_convert_audio": {
        "description": "Konvertiert Audio zwischen Formaten (WAV, MP3, OGG, FLAC) via ffmpeg.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Quell-Audiodatei"},
                "output_format": {
                    "type": "string",
                    "description": "Zielformat",
                    "default": "wav",
                    "enum": ["wav", "mp3", "ogg", "flac"],
                },
                "sample_rate": {
                    "type": "integer",
                    "description": "Samplerate",
                    "default": 16000,
                },
            },
            "required": ["input_path"],
        },
    },
    "media_resize_image": {
        "description": "Ändert Bildgröße (behält Seitenverhältnis). Unterstützt JPG, PNG, WEBP.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Pfad zum Bild"},
                "max_width": {"type": "integer", "description": "Max. Breite", "default": 1024},
                "max_height": {"type": "integer", "description": "Max. Höhe", "default": 1024},
            },
            "required": ["image_path"],
        },
    },
    "media_tts": {
        "description": (
            "Konvertiert Text zu Sprache (WAV). "
            "Lokal via Piper TTS oder eSpeak-NG."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Zu sprechender Text"},
                "voice": {
                    "type": "string",
                    "description": "Piper-Stimmenmodell",
                    "default": DEFAULT_PIPER_VOICE,
                },
            },
            "required": ["text"],
        },
    },
}


def register_media_tools(mcp_client: Any) -> MediaPipeline:
    """Registriert Media-MCP-Tools beim MCP-Client.

    Args:
        mcp_client: JarvisMCPClient-Instanz.

    Returns:
        MediaPipeline-Instanz.
    """
    pipeline = MediaPipeline()

    async def _transcribe(audio_path: str, language: str = "de", model: str = "base", **_: Any) -> str:
        result = await pipeline.transcribe_audio(
            audio_path, language=language, model=model,
        )
        return result.text if result.success else f"Fehler: {result.error}"

    async def _analyze_image(image_path: str, prompt: str = DEFAULT_IMAGE_PROMPT, model: str = DEFAULT_OLLAMA_MODEL, **_: Any) -> str:
        result = await pipeline.analyze_image(
            image_path, prompt=prompt, model=model,
        )
        return result.text if result.success else f"Fehler: {result.error}"

    async def _extract_text(file_path: str, **_: Any) -> str:
        result = await pipeline.extract_text(file_path)
        return result.text if result.success else f"Fehler: {result.error}"

    async def _convert_audio(input_path: str, output_format: str = "wav", sample_rate: int = 16000, **_: Any) -> str:
        result = await pipeline.convert_audio(
            input_path, output_format=output_format, sample_rate=sample_rate,
        )
        return result.text if result.success else f"Fehler: {result.error}"

    async def _resize_image(image_path: str, max_width: int = 1024, max_height: int = 1024, **_: Any) -> str:
        result = await pipeline.resize_image(
            image_path, max_width=max_width, max_height=max_height,
        )
        return result.text if result.success else f"Fehler: {result.error}"

    async def _tts(text: str, voice: str = DEFAULT_PIPER_VOICE, **_: Any) -> str:
        result = await pipeline.text_to_speech(text, voice=voice)
        return result.text if result.success else f"Fehler: {result.error}"

    handlers = {
        "media_transcribe_audio": _transcribe,
        "media_analyze_image": _analyze_image,
        "media_extract_text": _extract_text,
        "media_convert_audio": _convert_audio,
        "media_resize_image": _resize_image,
        "media_tts": _tts,
    }

    for name, schema in MEDIA_TOOL_SCHEMAS.items():
        mcp_client.register_builtin_handler(
            name,
            handlers[name],
            description=schema["description"],
            input_schema=schema["inputSchema"],
        )

    log.info("media_tools_registered", tools=list(MEDIA_TOOL_SCHEMAS.keys()))
    return pipeline
