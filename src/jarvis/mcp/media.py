"""Media-Pipeline: Verarbeitung von Bildern, Audio und Dokumenten.

MCP-Tools für multimodale Medienverarbeitung -- vollständig lokal.

Tools:
  - media_transcribe_audio: Audio → Text (Whisper)
  - media_analyze_image: Bild → Beschreibung (multimodales LLM via Ollama)
  - media_extract_text: PDF/DOCX/TXT → Text
  - media_convert_audio: Audio-Formatkonvertierung (ffmpeg)
  - media_image_resize: Bildgröße ändern (Pillow)
  - media_tts: Text → Sprache (Piper/eSpeak)

Alle Tools arbeiten mit lokalen Dateipfaden -- keine Cloud-Uploads.
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

# Maximale Dateigroessen fuer Security-Limits
MAX_EXTRACT_FILE_SIZE = 52_428_800   # 50 MB für Dokument-Extraktion
MAX_AUDIO_FILE_SIZE = 104_857_600    # 100 MB für Audio-Transkription

# Standard-Modelle und -Stimmen (Fallbacks, wenn kein Config verfügbar)
_DEFAULT_VISION_MODEL = "openbmb/minicpm-v4.5"
_DEFAULT_VISION_MODEL_DETAIL = "qwen3-vl:32b"
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
        # LLM + Vault injection (gesetzt via _set_llm_fn / _set_vault)
        self._llm_fn: Any = None
        self._llm_model: str = ""
        self._vault: Any = None

    def _set_llm_fn(self, llm_fn: Any, model_name: str = "") -> None:
        """Injiziert eine LLM-Funktion für Dokument-Analyse.

        Args:
            llm_fn: Async-Funktion mit Signatur (prompt: str, model: str) -> str
            model_name: Name des zu verwendenden Modells.
        """
        self._llm_fn = llm_fn
        self._llm_model = model_name

    def _set_vault(self, vault: Any) -> None:
        """Injiziert VaultTools-Referenz für optionales Speichern.

        Args:
            vault: VaultTools-Instanz mit vault_save-Methode.
        """
        self._vault = vault

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

        file_size = path.stat().st_size
        if file_size > MAX_AUDIO_FILE_SIZE:
            return MediaResult(
                success=False,
                error=f"Audiodatei zu gross ({file_size / 1_048_576:.1f} MB, max {MAX_AUDIO_FILE_SIZE // 1_048_576} MB)",
            )

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
        model: str = _DEFAULT_VISION_MODEL,
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

            async with httpx.AsyncClient(timeout=180.0, trust_env=False) as client:
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

        file_size = path.stat().st_size
        if file_size > MAX_EXTRACT_FILE_SIZE:
            return MediaResult(
                success=False,
                error=f"Datei zu gross ({file_size / 1_048_576:.1f} MB, max {MAX_EXTRACT_FILE_SIZE // 1_048_576} MB)",
            )

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

        if path.stat().st_size > MAX_EXTRACT_FILE_SIZE:
            raise ValueError(
                f"HTML-Datei zu gross ({path.stat().st_size / 1_048_576:.1f} MB, "
                f"max {MAX_EXTRACT_FILE_SIZE // 1_048_576} MB)"
            )
        html = path.read_text(encoding="utf-8", errors="replace")
        # Script/Style-Tags entfernen
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # HTML-Tags entfernen
        text = re.sub(r"<[^>]+>", " ", html)
        # Mehrfach-Whitespace normalisieren
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ========================================================================
    # Dokument-Analyse (LLM-gestützt)
    # ========================================================================

    async def analyze_document(
        self,
        path: str,
        analysis_type: str = "full",
        language: str = "de",
        save_to_vault: bool = False,
    ) -> str:
        """Analysiert ein Dokument strukturiert mit LLM-Unterstützung.

        Extrahiert Text, sendet ihn an das LLM mit einem strukturierten
        Analyse-Prompt und liefert eine Analyse mit 6 Abschnitten.

        Args:
            path: Pfad zum Dokument (PDF, DOCX, TXT, HTML, etc.).
            analysis_type: Art der Analyse:
                'full' = vollständige Analyse (6 Abschnitte)
                'summary' = nur Zusammenfassung
                'risks' = nur Risiken & Bedenken
                'todos' = nur Handlungsbedarf / To-Dos
            language: Sprache der Analyse ('de' oder 'en').
            save_to_vault: Wenn True, Analyse im Vault speichern.

        Returns:
            Strukturierte Analyse als Markdown-Text.
        """
        if self._llm_fn is None:
            return (
                "Fehler: Dokument-Analyse nicht verfügbar. "
                "Kein LLM konfiguriert. Verwende stattdessen media_extract_text."
            )

        # 1. Text extrahieren
        extract_result = await self.extract_text(path)
        if not extract_result.success:
            return f"Fehler bei Text-Extraktion: {extract_result.error}"

        text = extract_result.text
        if not text.strip():
            return "Fehler: Kein Text im Dokument gefunden. Möglicherweise ein gescanntes PDF (OCR erforderlich)."

        # Hinweis bei sehr kurzem Text
        ocr_warning = ""
        if len(text.strip()) < 100:
            ocr_warning = (
                "\n\n⚠ **Hinweis**: Sehr wenig Text extrahiert. "
                "Falls es sich um ein gescanntes PDF handelt, wird OCR-Software benötigt."
            )

        # 2. Analyse-Prompt bauen
        filename = Path(path).name
        prompt = _build_analysis_prompt(text, analysis_type, language, filename)

        # 3. LLM aufrufen
        try:
            analysis = await self._llm_fn(prompt, self._llm_model)
        except Exception as exc:
            log.error("document_analysis_llm_failed", path=path, error=str(exc))
            return f"Fehler bei LLM-Analyse: {exc}"

        result = analysis + ocr_warning

        # 4. Optional im Vault speichern
        if save_to_vault and self._vault is not None:
            try:
                vault_title = f"Analyse: {filename}"
                await self._vault.vault_save(
                    title=vault_title,
                    content=result,
                    tags="analyse, dokument",
                    folder="research",
                    sources=path,
                )
                result += f"\n\n✓ Analyse im Vault gespeichert als '{vault_title}'."
            except Exception as vault_exc:
                log.warning("vault_save_failed_during_analysis", error=str(vault_exc))
                result += f"\n\n⚠ Vault-Speicherung fehlgeschlagen: {vault_exc}"

        log.info("document_analyzed", path=path, type=analysis_type, chars=len(result))
        return result

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

    # ========================================================================
    # Dokument-Export (PDF / DOCX)
    # ========================================================================

    async def export_document(
        self,
        content: str,
        *,
        fmt: str = "pdf",
        title: str = "",
        author: str = "",
        filename: str = "dokument",
    ) -> MediaResult:
        """Exportiert Text als PDF- oder DOCX-Dokument.

        Args:
            content: Text-Inhalt (Absätze durch \\n\\n getrennt).
            fmt: Ausgabeformat ('pdf' oder 'docx').
            title: Optionaler Titel/Betreff.
            author: Optionaler Absender/Autor.
            filename: Dateiname ohne Endung.
        """
        fmt = fmt.lower().strip()
        if fmt not in ("pdf", "docx"):
            return MediaResult(success=False, error=f"Nicht unterstütztes Format: {fmt}. Erlaubt: pdf, docx")

        if not content.strip():
            return MediaResult(success=False, error="Leerer Inhalt")

        # Sicheres Verzeichnis
        doc_dir = Path.home() / ".jarvis" / "workspace" / "documents"
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Dateinamen bereinigen
        safe_name = "".join(c for c in filename if c.isalnum() or c in "-_ ").strip() or "dokument"
        output_path = doc_dir / f"{safe_name}.{fmt}"

        loop = asyncio.get_running_loop()

        try:
            if fmt == "pdf":
                await loop.run_in_executor(None, self._generate_pdf, output_path, content, title, author)
            else:
                await loop.run_in_executor(None, self._generate_docx, output_path, content, title, author)

            log.info("document_exported", path=str(output_path), format=fmt)
            return MediaResult(
                success=True,
                text=f"Dokument erstellt: {output_path}",
                output_path=str(output_path),
                metadata={"format": fmt, "title": title, "filename": safe_name},
            )
        except ImportError as exc:
            return MediaResult(success=False, error=str(exc))
        except Exception as exc:
            log.error("document_export_failed", error=str(exc))
            return MediaResult(success=False, error=f"Dokument-Export fehlgeschlagen: {exc}")

    def _generate_pdf(self, output_path: Path, content: str, title: str, author: str) -> None:
        """Generiert ein PDF-Dokument mit fpdf2."""
        try:
            from fpdf import FPDF
        except ImportError:
            raise ImportError("fpdf2 nicht installiert. pip install fpdf2") from None

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=25)
        pdf.add_page()

        # Unicode-Font einbetten (für Umlaute etc.), sonst Helvetica
        font_name = "Helvetica"
        has_bold = True  # Built-in Fonts haben immer Bold
        try:
            # Suche einen Unicode-fähigen TrueType-Font
            font_candidates = [
                # DejaVu Sans (Regular + Bold)
                ("DejaVu", [
                    Path("C:/Windows/Fonts/DejaVuSans.ttf"),
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                    Path.home() / ".fonts" / "DejaVuSans.ttf",
                ], [
                    Path("C:/Windows/Fonts/DejaVuSans-Bold.ttf"),
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                    Path.home() / ".fonts" / "DejaVuSans-Bold.ttf",
                ]),
                # Arial als Fallback (Windows)
                ("ArialUni", [
                    Path("C:/Windows/Fonts/arial.ttf"),
                ], [
                    Path("C:/Windows/Fonts/arialbd.ttf"),
                ]),
            ]
            for fname, regular_paths, bold_paths in font_candidates:
                regular = next((p for p in regular_paths if p.exists()), None)
                if regular:
                    pdf.add_font(fname, "", str(regular), uni=True)
                    font_name = fname
                    bold = next((p for p in bold_paths if p.exists()), None)
                    if bold:
                        pdf.add_font(fname, "B", str(bold), uni=True)
                        has_bold = True
                    else:
                        has_bold = False
                    break
        except Exception:
            pass  # Fallback auf Helvetica

        # Autor/Absender
        if author:
            pdf.set_font(font_name, size=10)
            pdf.multi_cell(0, 6, author)
            pdf.ln(10)

        # Titel
        if title:
            pdf.set_font(font_name, "B" if has_bold else "", size=14)
            pdf.multi_cell(0, 8, title)
            pdf.ln(8)

        # Inhalt
        pdf.set_font(font_name, size=11)
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if para:
                pdf.multi_cell(0, 6, para)
                pdf.ln(4)

        pdf.output(str(output_path))

    def _generate_docx(self, output_path: Path, content: str, title: str, author: str) -> None:
        """Generiert ein DOCX-Dokument mit python-docx."""
        try:
            from docx import Document
            from docx.shared import Pt
        except ImportError:
            raise ImportError("python-docx nicht installiert. pip install python-docx") from None

        doc = Document()

        # Standardschriftart auf Calibri setzen
        style = doc.styles["Normal"]
        font = style.font
        font.name = "Calibri"
        font.size = Pt(11)

        # Autor/Absender
        if author:
            p = doc.add_paragraph(author)
            p.style.font.size = Pt(10)

        # Titel
        if title:
            doc.add_heading(title, level=1)

        # Inhalt
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if para:
                doc.add_paragraph(para)

        doc.save(str(output_path))

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
# Analyse-Prompt-Builder
# ============================================================================


def _build_analysis_prompt(
    text: str,
    analysis_type: str,
    language: str,
    filename: str,
) -> str:
    """Baut den LLM-Prompt für die Dokument-Analyse.

    Args:
        text: Extrahierter Dokumenttext.
        analysis_type: full/summary/risks/todos.
        language: Sprache der Analyse.
        filename: Name der Quelldatei.

    Returns:
        Fertiger LLM-Prompt.
    """
    lang_instruction = "Antworte auf Deutsch." if language == "de" else "Answer in English."

    if analysis_type == "summary":
        task = (
            "Erstelle eine prägnante Zusammenfassung des Dokuments in 2-3 Sätzen. "
            "Nenne die wichtigsten Kernaussagen (max. 5 Punkte)."
        )
    elif analysis_type == "risks":
        task = (
            "Identifiziere alle Risiken und Bedenken im Dokument. "
            "Bewerte jedes Risiko als HOCH, MITTEL oder NIEDRIG mit Begründung."
        )
    elif analysis_type == "todos":
        task = (
            "Extrahiere alle Handlungspunkte, To-Dos und offenen Aufgaben aus dem Dokument. "
            "Ordne jedem Punkt eine Priorität zu (Hoch/Mittel/Niedrig)."
        )
    else:  # full
        task = (
            "Erstelle eine vollständige strukturierte Analyse mit genau diesen 6 Abschnitten:\n\n"
            "## Zusammenfassung\n"
            "2-3 Sätze, die den Kern des Dokuments erfassen.\n\n"
            "## Kernaussagen\n"
            "Maximal 7 priorisierte Punkte.\n\n"
            "## Risiken & Bedenken\n"
            "Bewertung: HOCH / MITTEL / NIEDRIG mit Begründung.\n\n"
            "## Handlungsbedarf / To-Dos\n"
            "Konkrete Aktionspunkte mit Priorität.\n\n"
            "## Entscheidungsprotokoll\n"
            "- Bereits getroffene Entscheidungen\n"
            "- Offene Entscheidungen\n\n"
            "## Metadaten\n"
            "- Dokumenttyp (Vertrag/Angebot/Bericht/Protokoll/Rechnung/Sonstiges)\n"
            "- Datum (falls erkennbar)\n"
            "- Beteiligte Parteien\n"
            "- Umfang"
        )

    return (
        f"Du bist ein Experte für Dokumentenanalyse. {lang_instruction}\n\n"
        f"Analysiere das folgende Dokument (Datei: {filename}).\n\n"
        f"AUFGABE:\n{task}\n\n"
        "Falls es sich um ein Rechtsdokument (Vertrag, AGB, etc.) handelt, "
        "füge am Ende diesen Disclaimer an: "
        '"Dies ist eine automatische Analyse und ersetzt keine rechtliche Beratung."\n\n'
        f"DOKUMENT-INHALT:\n---\n{text}\n---"
    )


# ============================================================================
# MCP-Tool-Schemas
# ============================================================================

MEDIA_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "media_transcribe_audio": {
        "description": (
            "Transkribiert eine Audiodatei (WAV, MP3, OGG, etc.) zu Text. "
            "Lokal via Whisper -- keine Cloud-Uploads."
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
            "Analysiert ein Bild mit einem multimodalen Vision-LLM via Ollama. "
            "Beschreibt Inhalt, erkennt Text (OCR), beantwortet Fragen zum Bild. "
            "Standard-Modell: schnell und gut. Mit detail=true: höchste Qualität "
            "(langsamer, für komplexe Bilder, Dokumente, schwer lesbare Texte)."
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
                "detail": {
                    "type": "boolean",
                    "description": "true = Detail-Modell (höchste Qualität, langsamer)",
                    "default": False,
                },
            },
            "required": ["image_path"],
        },
    },
    "media_extract_text": {
        "description": (
            "Extrahiert Text aus Dokumenten: PDF, DOCX, TXT, MD, HTML, CSV, JSON, XML. "
            "Lokal -- keine Cloud-Dienste."
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
    "analyze_document": {
        "description": (
            "Analysiert ein Dokument (PDF, DOCX, TXT, HTML) strukturiert mit LLM. "
            "Liefert Zusammenfassung, Kernaussagen, Risiken, To-Dos, Entscheidungen und Metadaten. "
            "Optional im Knowledge Vault speichern."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Pfad zum Dokument"},
                "analysis_type": {
                    "type": "string",
                    "enum": ["full", "summary", "risks", "todos"],
                    "description": "Art der Analyse: full (vollständig), summary, risks, todos",
                    "default": "full",
                },
                "language": {
                    "type": "string",
                    "description": "Sprache der Analyse (de/en)",
                    "default": "de",
                },
                "save_to_vault": {
                    "type": "boolean",
                    "description": "Analyse im Knowledge Vault speichern",
                    "default": False,
                },
            },
            "required": ["path"],
        },
    },
    "document_export": {
        "description": (
            "Exportiert Text als PDF- oder DOCX-Dokument (Briefform, Schreiben, etc.). "
            "Der Inhalt wird als Fließtext übergeben, Absätze durch doppelte Zeilenumbrüche getrennt."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Text-Inhalt des Dokuments"},
                "format": {
                    "type": "string",
                    "enum": ["pdf", "docx"],
                    "description": "Ausgabeformat",
                    "default": "pdf",
                },
                "title": {"type": "string", "description": "Titel/Betreff", "default": ""},
                "author": {"type": "string", "description": "Absender/Autor", "default": ""},
                "filename": {
                    "type": "string",
                    "description": "Dateiname (ohne Endung)",
                    "default": "dokument",
                },
            },
            "required": ["content"],
        },
    },
}


def register_media_tools(mcp_client: Any, config: Any = None) -> MediaPipeline:
    """Registriert Media-MCP-Tools beim MCP-Client.

    Args:
        mcp_client: JarvisMCPClient-Instanz.
        config: JarvisConfig-Instanz (optional, für Vision-Modell-Auswahl).

    Returns:
        MediaPipeline-Instanz.
    """
    # Vision-Modelle aus Config oder Fallback-Defaults
    vision_model = getattr(config, "vision_model", _DEFAULT_VISION_MODEL) if config else _DEFAULT_VISION_MODEL
    vision_model_detail = getattr(config, "vision_model_detail", _DEFAULT_VISION_MODEL_DETAIL) if config else _DEFAULT_VISION_MODEL_DETAIL
    ollama_url = getattr(getattr(config, "ollama", None), "base_url", "http://localhost:11434") if config else "http://localhost:11434"

    pipeline = MediaPipeline()

    async def _transcribe(audio_path: str, language: str = "de", model: str = "base", **_: Any) -> str:
        result = await pipeline.transcribe_audio(
            audio_path, language=language, model=model,
        )
        return result.text if result.success else f"Fehler: {result.error}"

    async def _analyze_image(image_path: str, prompt: str = DEFAULT_IMAGE_PROMPT, detail: bool = False, **_: Any) -> str:
        model = vision_model_detail if detail else vision_model
        log.info("vision_model_selected", model=model, detail=detail)
        result = await pipeline.analyze_image(
            image_path, prompt=prompt, model=model, ollama_url=ollama_url,
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

    async def _analyze_document(
        path: str,
        analysis_type: str = "full",
        language: str = "de",
        save_to_vault: bool = False,
        **_: Any,
    ) -> str:
        result = await pipeline.analyze_document(
            path, analysis_type=analysis_type, language=language,
            save_to_vault=save_to_vault,
        )
        return result

    async def _export_document(
        content: str,
        format: str = "pdf",
        title: str = "",
        author: str = "",
        filename: str = "dokument",
        **_: Any,
    ) -> str:
        result = await pipeline.export_document(
            content, fmt=format, title=title, author=author, filename=filename,
        )
        if result.success:
            return result.output_path or result.text
        return f"Fehler: {result.error}"

    handlers = {
        "media_transcribe_audio": _transcribe,
        "media_analyze_image": _analyze_image,
        "media_extract_text": _extract_text,
        "media_convert_audio": _convert_audio,
        "media_resize_image": _resize_image,
        "media_tts": _tts,
        "analyze_document": _analyze_document,
        "document_export": _export_document,
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
