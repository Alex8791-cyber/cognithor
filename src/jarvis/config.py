"""
Jarvis · Configuration system.

Loads configuration from:
  1. Defaults (defined here)
  2. ~/.jarvis/config.yaml (overrides defaults)
  3. Environment variables JARVIS_* (overrides everything)

Automatically creates the ~/.jarvis/ directory structure on first start.
Architecture Bible: §4.9, §8, §12
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from jarvis.models import ModelConfig, SandboxConfig

log = logging.getLogger(__name__)

# ============================================================================
# Konfigurationsmodelle
# ============================================================================


class OllamaConfig(BaseModel):
    """Ollama-Server Konfiguration."""

    base_url: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    timeout_seconds: int = Field(default=180, ge=10, le=600)
    keep_alive: str = "30m"  # Wie lange Modelle im VRAM bleiben


class ModelsConfig(BaseModel):
    """Modell-Zuordnung. [B§8.1]"""

    planner: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="qwen3:32b",
            context_window=32768,
            vram_gb=20.0,
            strengths=["reasoning", "planning", "reflection", "german"],
            speed="medium",
        )
    )
    executor: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="qwen3:8b",
            context_window=32768,
            vram_gb=6.0,
            strengths=["tool-calling", "simple-tasks"],
            speed="fast",
        )
    )
    coder: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="qwen3-coder:30b",
            context_window=32768,
            vram_gb=20.0,
            strengths=["code-generation", "debugging", "testing"],
            speed="medium",
        )
    )
    coder_fast: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="qwen2.5-coder:7b",
            context_window=32768,
            vram_gb=5.0,
            strengths=["code-generation", "real-time-coding"],
            speed="fast",
        )
    )
    embedding: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="qwen3-embedding:0.6b",
            context_window=8192,
            vram_gb=0.5,
            strengths=["semantic-search", "multilingual"],
            speed="fast",
            embedding_dimensions=1024,
        )
    )


class GatekeeperConfig(BaseModel):
    """Gatekeeper-Einstellungen. [B§3.2]"""

    policies_dir: str = "policies"  # Relativ zu jarvis_home
    default_risk_level: Literal["green", "yellow", "orange", "red"] = "yellow"
    max_blocked_retries: int = Field(default=3, ge=1, le=10)


class PlannerConfig(BaseModel):
    """Planner-Einstellungen. [B§3.1, §3.4]"""

    max_iterations: int = Field(default=25, ge=1, le=50)
    escalation_after: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    response_token_budget: int = Field(default=4000, ge=500, le=8000)


class WebConfig(BaseModel):
    """Web-Tools Konfiguration. [B§5.3]

    Suchbackends (Prioritaet: SearXNG > Brave > Google CSE > DuckDuckGo):
      - SearXNG: Self-hosted, keine API-Keys noetig
      - Brave Search: Schnell, 2000 Anfragen/Monat kostenlos
      - Google CSE: Custom Search Engine, 100 Anfragen/Tag kostenlos
      - DuckDuckGo: Immer verfuegbar, kein API-Key noetig (Standard-Fallback)
    """

    searxng_url: str = ""
    """URL der SearXNG-Instanz (z.B. 'http://localhost:8888')."""

    brave_api_key: str = ""
    """Brave Search API Key (https://brave.com/search/api/)."""

    google_cse_api_key: str = ""
    """Google Custom Search Engine API Key (https://developers.google.com/custom-search)."""

    google_cse_cx: str = ""
    """Google Custom Search Engine ID (cx Parameter)."""

    jina_api_key: str = ""
    """Jina AI Reader API Key (optional, Free-Tier funktioniert ohne Key)."""

    duckduckgo_enabled: bool = True
    """DuckDuckGo als kostenloser Fallback wenn kein anderes Backend konfiguriert."""

    domain_blocklist: list[str] = Field(default_factory=list)
    """Liste blockierter Domains (z.B. ['example.com']). Fetch wird verweigert."""

    domain_allowlist: list[str] = Field(default_factory=list)
    """Wenn nicht leer: NUR diese Domains sind erlaubt (Whitelist-Modus)."""

    # ── Limits (bisher hardcoded in web.py) ────────────────────────────────
    max_fetch_bytes: int = Field(default=500_000, ge=10_000, le=10_000_000)
    """Maximale Antwortgroesse beim URL-Fetch (Bytes)."""

    max_text_chars: int = Field(default=20_000, ge=1000, le=200_000)
    """Maximale Zeichenzahl des extrahierten Textes."""

    fetch_timeout_seconds: int = Field(default=15, ge=5, le=120)
    """HTTP-Timeout fuer URL-Fetch (Sekunden)."""

    search_timeout_seconds: int = Field(default=10, ge=5, le=60)
    """Timeout fuer Suchmaschinen-Anfragen (Sekunden)."""

    max_search_results: int = Field(default=10, ge=1, le=50)
    """Maximale Anzahl Suchergebnisse."""

    ddg_min_delay_seconds: float = Field(default=2.0, ge=0.5, le=10.0)
    """Mindestabstand zwischen DuckDuckGo-Suchen (Sekunden)."""

    ddg_ratelimit_wait_seconds: int = Field(default=30, ge=5, le=120)
    """Wartezeit bei DuckDuckGo Rate-Limiting (Sekunden)."""

    ddg_cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    """Cache-TTL fuer DuckDuckGo-Ergebnisse (Sekunden)."""

    search_and_read_max_chars: int = Field(default=5000, ge=1000, le=50_000)
    """Maximale Zeichenzahl pro Seite bei search_and_read."""

    # ── HTTP Request Tool ──────────────────────────────────────────────────
    http_request_max_body_bytes: int = Field(default=1_048_576, ge=1024, le=10_485_760)
    """Maximale Body-Groesse fuer http_request (Bytes). Standard: 1 MB."""

    http_request_timeout_seconds: int = Field(default=30, ge=1, le=120)
    """Standard-Timeout fuer http_request (Sekunden)."""

    http_request_rate_limit_seconds: float = Field(default=1.0, ge=0.0, le=30.0)
    """Mindestabstand zwischen http_request-Aufrufen (Sekunden). 0 = kein Limit."""


class BrowserConfig(BaseModel):
    """Browser-Automation Konfiguration (Playwright)."""

    max_text_length: int = Field(default=8000, ge=1000, le=100_000)
    """Maximale Textlaenge die ans LLM zurueckgegeben wird."""

    max_js_length: int = Field(default=50_000, ge=1000, le=500_000)
    """Maximale JavaScript-Scriptlaenge (Zeichen)."""

    default_timeout_ms: int = Field(default=30_000, ge=5000, le=120_000)
    """Standard-Browser-Timeout (Millisekunden)."""

    default_viewport_width: int = Field(default=1280, ge=320, le=3840)
    """Standard-Viewport-Breite (Pixel)."""

    default_viewport_height: int = Field(default=720, ge=240, le=2160)
    """Standard-Viewport-Hoehe (Pixel)."""


class FilesystemConfig(BaseModel):
    """Dateisystem-Tools Konfiguration."""

    max_tree_entries: int = Field(default=200, ge=10, le=10_000)
    """Maximale Eintraege im Verzeichnisbaum-Listing."""


class ShellConfig(BaseModel):
    """Shell-Execution Konfiguration."""

    default_timeout_seconds: int = Field(default=30, ge=5, le=600)
    """Standard-Timeout fuer Shell-Befehle (Sekunden)."""

    max_log_command_length: int = Field(default=200, ge=50, le=2000)
    """Maximale Befehlslaenge im Log."""

    max_redacted_log_prefix: int = Field(default=50, ge=10, le=500)
    """Maximale Praefixlaenge fuer geschwaerzte Log-Eintraege."""


class ToolsConfig(BaseModel):
    """Konfiguration fuer Desktop-Automation und Desktop-Tools."""

    computer_use_enabled: bool = Field(
        default=False,
        description=(
            "Desktop-Automation via Screenshot + Koordinaten-Klick"
            " (pyautogui, mss). Erfordert pip install cognithor[desktop]."
        ),
    )
    """Aktiviert Computer-Use (Screenshot + Klick)."""

    desktop_tools_enabled: bool = Field(
        default=False,
        description=(
            "Clipboard-Zugriff und Screenshot-Tools. Erfordert pip install cognithor[desktop]."
        ),
    )
    """Aktiviert Desktop-Tools (Clipboard, Screenshot)."""


class MediaConfig(BaseModel):
    """Media-Pipeline Konfiguration."""

    max_extract_length: int = Field(default=15_000, ge=1000, le=100_000)
    """Maximale Textlaenge fuer LLM-Kontext bei Extraktion."""

    max_image_file_size: int = Field(default=10_485_760, ge=1_048_576, le=104_857_600)
    """Maximale Bilddateigroesse fuer Base64-Encoding (Bytes)."""

    max_extract_file_size: int = Field(default=52_428_800, ge=1_048_576, le=524_288_000)
    """Maximale Dateigroesse fuer Dokument-Extraktion (Bytes)."""

    max_audio_file_size: int = Field(default=104_857_600, ge=1_048_576, le=1_073_741_824)
    """Maximale Audio-Dateigroesse (Bytes)."""

    max_image_dimension: int = Field(default=8192, ge=256, le=16384)
    """Maximale Bilddimension in Pixel."""

    default_max_width: int = Field(default=1024, ge=64, le=8192)
    """Standard-Maximalbreite bei Bild-Resize."""

    default_max_height: int = Field(default=1024, ge=64, le=8192)
    """Standard-Maximalhoehe bei Bild-Resize."""


class SynthesisConfig(BaseModel):
    """Knowledge-Synthesis Konfiguration."""

    max_source_chars: int = Field(default=4000, ge=500, le=50_000)
    """Maximale Zeichenzahl pro Quelle fuer LLM-Kontext."""

    max_context_chars: int = Field(default=25_000, ge=5000, le=200_000)
    """Maximale Gesamtgroesse des Kontexts fuer das LLM."""


class EmailConfig(BaseModel):
    """E-Mail-Tools Konfiguration (IMAP/SMTP).

    Passwort wird NIE in der Config gespeichert, sondern aus einer
    Umgebungsvariable gelesen (``password_env``).
    """

    enabled: bool = False
    """Aktiviert oder deaktiviert die E-Mail-Tools."""

    imap_host: str = ""
    """IMAP-Server Hostname (z.B. 'imap.gmail.com')."""

    imap_port: int = Field(default=993, ge=1, le=65535)
    """IMAP-Server Port (Standard: 993 fuer SSL)."""

    smtp_host: str = ""
    """SMTP-Server Hostname (z.B. 'smtp.gmail.com')."""

    smtp_port: int = Field(default=465, ge=1, le=65535)
    """SMTP-Server Port (Standard: 465 fuer SSL, 587 fuer STARTTLS)."""

    username: str = ""
    """E-Mail-Benutzername (oft die E-Mail-Adresse)."""

    password_env: str = "JARVIS_EMAIL_PASSWORD"
    """Name der Umgebungsvariable mit dem E-Mail-Passwort."""


class CalendarConfig(BaseModel):
    """Kalender-Tools Konfiguration (ICS/CalDAV).

    Primaer: Lokale ICS-Datei (immer verfuegbar).
    Optional: CalDAV-Client wenn ``caldav``-Bibliothek installiert.
    """

    enabled: bool = False
    """Aktiviert oder deaktiviert die Kalender-Tools."""

    ics_path: str = ""
    """Pfad zur lokalen ICS-Datei (Default: ~/.jarvis/calendar.ics)."""

    caldav_url: str = ""
    """CalDAV-Server URL (optional, z.B. 'https://caldav.example.com/dav/')."""

    username: str = ""
    """CalDAV-Benutzername (optional)."""

    password_env: str = "JARVIS_CALENDAR_PASSWORD"
    """Name der Umgebungsvariable mit dem CalDAV-Passwort."""

    timezone: str = ""
    """Zeitzone (z.B. 'Europe/Berlin'). Leer = System-Zeitzone."""


class CodeConfig(BaseModel):
    """Code-Execution Konfiguration."""

    max_code_size: int = Field(default=1_048_576, ge=1024, le=10_485_760)
    """Maximale Code-Groesse (Bytes)."""

    default_timeout_seconds: int = Field(default=60, ge=5, le=600)
    """Standard-Timeout fuer Python-Ausfuehrung (Sekunden)."""


class PersonalityConfig(BaseModel):
    """Personality Engine Konfiguration."""

    warmth: float = Field(default=0.7, ge=0.0, le=1.0)
    """Waerme-Level: 0.0 = neutral/sachlich, 1.0 = sehr warm und empathisch."""

    humor: float = Field(default=0.3, ge=0.0, le=1.0)
    """Humor-Level: 0.0 = ernst, 1.0 = spielerisch."""

    follow_up_questions: bool = True
    """Soll Jarvis am Ende Nachfragen stellen?"""

    success_celebration: bool = True
    """Soll Jarvis erfolgreiche Aktionen positiv bestaetigen?"""

    greeting_enabled: bool = True
    """Soll Jarvis Tageszeit-abhaengige Gruesse verwenden?"""


class IdentityConfig(BaseModel):
    """Immortal Mind Protocol — Kognitive Identitaetsschicht."""

    enabled: bool = True
    """Identity Layer aktivieren/deaktivieren."""

    identity_id: str = "jarvis"
    """Standard-Identitaets-ID."""

    checkpoint_every_n: int = Field(default=5, ge=1, le=50)
    """Konsolidierung alle N Interaktionen."""

    checkpoint_interval_minutes: int = Field(default=10, ge=1, le=120)
    """Konsolidierung alle N Minuten."""

    narrative_reflect_every_n: int = Field(default=50, ge=10, le=500)
    """Narrative Selbstreflexion alle N Interaktionen."""

    max_active_memories: int = Field(default=10000, ge=100, le=100000)
    """Maximale Anzahl aktiver Erinnerungen."""

    reality_check_enabled: bool = True
    """Halluzinationsschutz aktiviert."""

    blockchain_enabled: bool = False
    """Blockchain-Ankerung (opt-in)."""

    blockchain_chain: str = "base_sepolia"
    """Blockchain-Netzwerk."""

    arweave_enabled: bool = False
    """Arweave permanente Speicherung (opt-in)."""


class ExecutorConfig(BaseModel):
    """Executor Konfiguration."""

    default_timeout_seconds: int = Field(default=30, ge=5, le=600)
    """Standard-Timeout fuer Tool-Ausfuehrung (Sekunden)."""

    max_output_chars: int = Field(default=10_000, ge=1000, le=100_000)
    """Maximale Tool-Output-Laenge (Zeichen)."""

    max_retries: int = Field(default=3, ge=0, le=10)
    """Maximale Wiederholungsversuche bei transienten Fehlern."""

    backoff_base_delay_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
    """Basis-Verzoegerung fuer exponentiellen Backoff (Sekunden)."""

    max_parallel_tools: int = Field(default=4, ge=1, le=16)
    """Maximale Anzahl parallel ausgefuehrter Tools (DAG-Execution)."""

    # Tool-spezifische Timeouts
    media_analyze_image_timeout: int = Field(default=180, ge=30, le=600)
    """Timeout fuer Bildanalyse (Sekunden)."""

    media_transcribe_audio_timeout: int = Field(default=120, ge=30, le=600)
    """Timeout fuer Audio-Transkription (Sekunden)."""

    media_extract_text_timeout: int = Field(default=120, ge=30, le=600)
    """Timeout fuer Text-Extraktion (Sekunden)."""

    media_tts_timeout: int = Field(default=120, ge=30, le=600)
    """Timeout fuer Text-to-Speech (Sekunden)."""

    run_python_timeout: int = Field(default=120, ge=30, le=600)
    """Timeout fuer Python-Code-Ausfuehrung (Sekunden)."""


class VaultConfig(BaseModel):
    """Knowledge Vault Konfiguration — Obsidian-kompatibles Markdown-Vault.

    Persistente Wissensablage fuer Recherchen, Meeting-Notizen, Projektnotizen
    und taegliche Zusammenfassungen. Notizen verwenden Obsidian-kompatibles
    YAML-Frontmatter und [[Backlinks]].
    """

    enabled: bool = True
    """Aktiviert oder deaktiviert das Vault-System."""

    path: str = "~/.jarvis/vault"
    """Pfad zum Vault-Verzeichnis. Wird bei Bedarf automatisch erstellt."""

    auto_save_research: bool = False
    """Wenn True, werden Web-Recherche-Ergebnisse automatisch im Vault gespeichert."""

    default_folders: dict[str, str] = Field(
        default_factory=lambda: {
            "research": "recherchen",
            "meetings": "meetings",
            "knowledge": "wissen",
            "projects": "projekte",
            "daily": "daily",
        }
    )
    """Mapping von logischen Ordnernamen zu Verzeichnisnamen im Vault."""

    encrypt_files: bool = False
    """Encrypt vault .md files at rest (Fernet/AES-256).

    Default: False (Obsidian-compatible plaintext).
    Set to True for maximum security — vault files become unreadable
    by Obsidian but are protected against disk cloning.
    Databases are ALWAYS encrypted regardless of this setting.
    """


class OsintConfig(BaseModel):
    """OSINT / Human Investigation Module configuration."""
    enabled: bool = True
    github_token: str = ""
    default_depth: str = "standard"
    collector_timeout: int = Field(default=30, ge=5, le=120)
    report_ttl_days: int = Field(default=30, ge=1, le=365)
    vault_folder: str = "recherchen/osint"


class ContextPipelineConfig(BaseModel):
    """Adaptive Context Pipeline — automatische Kontext-Anreicherung vor dem Planner."""

    enabled: bool = True
    """Pipeline aktivieren/deaktivieren."""

    memory_top_k: int = 8
    """Anzahl Memory-Ergebnisse (BM25-only, sync, ~5-20ms)."""

    vault_top_k: int = 5
    """Anzahl Vault-Suchergebnisse (~10-50ms)."""

    episode_days: int = 2
    """Anzahl Tage fuer Episoden-Kontext (heute + gestern)."""

    min_query_length: int = 8
    """Mindestlaenge der User-Nachricht fuer Kontext-Suche."""

    max_context_chars: int = 8000
    """Maximale Zeichenzahl des injizierten Kontexts."""

    smalltalk_patterns: list[str] = [
        "hallo",
        "hi",
        "hey",
        "guten morgen",
        "guten tag",
        "guten abend",
        "danke",
        "tschüss",
        "bye",
        "ok",
        "ja",
        "nein",
        "alles klar",
    ]
    """Patterns die als Smalltalk erkannt werden (keine Kontext-Suche)."""


class MemoryConfig(BaseModel):
    """Memory-System Konfiguration. [B§4]"""

    chunk_size_tokens: int = Field(default=400, ge=100, le=2000)
    chunk_overlap_tokens: int = Field(default=80, ge=0, le=500)
    search_top_k: int = Field(default=6, ge=1, le=20)
    # Hybrid-Suche Gewichtung [B§4.7]
    weight_vector: float = Field(default=0.50, ge=0.0, le=1.0)
    weight_bm25: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_graph: float = Field(default=0.20, ge=0.0, le=1.0)
    # Recency Decay [B§4.7]
    recency_half_life_days: int = Field(default=30, ge=1, le=365)
    # Working Memory [B§4.6]
    compaction_threshold: float = Field(default=0.80, ge=0.5, le=0.95)
    compaction_keep_last_n: int = Field(default=8, ge=2, le=20)
    # Token-Budget Verteilung (statische Anteile, geschaetzte Tokens)
    budget_core_memory: int = Field(default=500, ge=100, le=5000)
    budget_system_prompt: int = Field(default=800, ge=200, le=5000)
    budget_procedures: int = Field(default=600, ge=100, le=5000)
    budget_injected_memories: int = Field(default=2500, ge=200, le=10000)
    budget_tool_descriptions: int = Field(default=1200, ge=200, le=10000)
    budget_response_reserve: int = Field(default=3000, ge=500, le=15000)

    # Episodic Memory: Wie viele Tage an Tageslogs sollen behalten werden?
    # Older files are deleted when initializing the memory system.
    episodic_retention_days: int = Field(default=365, ge=1, le=3650)

    # Dynamische Gewichtung der Hybrid-Suche.
    # Wenn aktiviert, passt Jarvis die Gewichtungsfaktoren (Vektor/BM25/Graph)
    # zur Laufzeit basierend auf Eigenschaften der Suchanfrage an.
    # For short queries, lexical and graph hits are weighted more heavily,
    # bei langen oder komplexen Anfragen erhalten semantische (Vektor-)Treffer
    # mehr Gewicht. Ist diese Option deaktiviert, werden die statischen
    # Gewichtungen (weight_vector, weight_bm25, weight_graph) aus der
    # Konfiguration verwendet.
    dynamic_weighting: bool = False

    @model_validator(mode="after")
    def validate_weights(self) -> MemoryConfig:
        """Validator um sicherzustellen, dass die Gewichtungen der Hybrid-Suche
        sich sinnvoll verhalten.

        Falls die Summe der Gewichte von Vektor-, BM25- und Graph-Kanal groesser
        als 1.0 ist, werden alle drei Gewichte so skaliert, dass ihre Summe
        1.0 ergibt. Dies verhindert ungewollte Ueberskalierung und sorgt fuer
        konsistente Scores.
        """
        total = self.weight_vector + self.weight_bm25 + self.weight_graph
        if total > 1.0 and total > 0.0:
            self.weight_vector = self.weight_vector / total
            self.weight_bm25 = self.weight_bm25 / total
            self.weight_graph = self.weight_graph / total
        return self


# --------------------------------------------------------------------------
# Heartbeat- und Plugin-Konfigurationen
# --------------------------------------------------------------------------


class HeartbeatConfig(BaseModel):
    """Konfiguration fuer den Heartbeat-Mechanismus.

    Wenn aktiviert, fuehrt Jarvis in regelmaessigen Abstaenden einen
    "Heartbeat" aus. Dabei werden Aufgaben aus einer Checklist-Datei
    (standardmaessig ``HEARTBEAT.md``) gelesen und als Systemnachricht
    an den Gateway gesendet. Auf diese Weise kann Jarvis proaktiv
    ueberpruefen, ob neue E-Mails, Kalendertermine oder Aufgaben
    Aufmerksamkeit erfordern, ohne dass der Nutzer eine Anfrage stellt.
    """

    enabled: bool = False
    """Aktiviert oder deaktiviert den Heartbeat. Wenn ``False``, wird
    keine periodische Heartbeat-Nachricht gesendet."""

    interval_minutes: int = Field(default=30, ge=1, le=1440)
    """Interval in Minuten zwischen zwei Heartbeat-Laeufen.
    Standardwert sind 30 Minuten. Der zulaessige Wertebereich liegt
    zwischen 1 und 1440 Minuten (24 Stunden)."""

    checklist_file: str = "HEARTBEAT.md"
    """Dateiname der Checklist im ``jarvis_home``. Diese Datei enthaelt
    Text oder Bullet-Points, die beim Heartbeat an den Agenten
    uebermittelt werden. Falls die Datei nicht existiert, wird eine
    leere Nachricht gesendet."""

    channel: str = "cli"
    """Name des Kanals, ueber den Heartbeat-Meldungen gesendet werden.
    Standard ist ``cli``; weitere gueltige Werte sind die Namen der
    registrierten Channels (z. B. ``telegram``, ``webui``)."""

    model: str = "qwen3:8b"
    """Name des Modells, das fuer Heartbeat-Kommunikation verwendet
    werden soll. Dieser Wert wird im Cron-Job ignoriert, ist aber
    vorhanden, damit sich Heartbeats semantisch wie CronJobs verhalten."""


class PluginsConfig(BaseModel):
    """Konfiguration fuer das Plugin-Oekosystem.

    Plugins stellen zusaetzliche Skills (Prozeduren, Tools oder
    Channel-Erweiterungen) bereit. Sie werden in einem separaten
    Verzeichnis abgelegt und zur Laufzeit geladen. Automatische
    Updates koennen deaktiviert werden, wenn der Nutzer volle Kontrolle
    ueber installierte Plugins wuenscht.
    """

    skills_dir: str = "skills"
    """Relativer Name des Verzeichnisses im ``jarvis_home``, in dem
    zusaetzliche Skills installiert werden. Der Standardwert ist
    ``skills``. Dies fuehrt dazu, dass externe Prozeduren in
    ``~/.jarvis/skills`` abgelegt werden."""

    auto_update: bool = False
    """Legt fest, ob Jarvis beim Start automatisch nach Updates fuer
    installierte Plugins sucht und diese einspielt. Standardmaessig
    deaktiviert, um ungewollte Aenderungen zu verhindern."""


class MarketplaceConfig(BaseModel):
    """Konfiguration fuer den Skill Marketplace.

    Der Marketplace bietet eine zentrale Anlaufstelle fuer Browse, Search
    und Install von Skills. Daten werden in einer lokalen SQLite-Datenbank
    persistiert. Der Marketplace ist optional und kann deaktiviert werden.
    """

    enabled: bool = Field(
        default=True,
        description="Skill Marketplace aktivieren",
    )
    """Aktiviert oder deaktiviert den Marketplace. Wenn ``False``, werden
    keine Marketplace-API-Endpoints registriert und keine DB angelegt."""

    db_path: str = Field(
        default="",
        description="Pfad zur Marketplace-DB (leer = ~/.jarvis/marketplace.db)",
    )
    """Pfad zur SQLite-Datenbank. Wenn leer, wird ``~/.jarvis/marketplace.db``
    verwendet."""

    auto_update: bool = Field(
        default=False,
        description="Skills automatisch aktualisieren",
    )
    """Wenn ``True``, prueft der Marketplace beim Start ob Updates fuer
    installierte Skills verfuegbar sind und installiert diese automatisch."""

    require_signatures: bool = Field(
        default=True,
        description="Nur signierte Skills installieren",
    )
    """Wenn ``True``, werden nur Skills mit gueltiger Signatur installiert.
    Unsignierte Skills werden abgelehnt."""

    auto_seed: bool = Field(
        default=True,
        description="Marketplace beim ersten Start mit Built-in-Prozeduren fuellen",
    )
    """Wenn ``True`` und die Marketplace-DB leer ist, werden die
    Built-in-Prozeduren aus ``data/procedures/`` als Seed-Daten eingefuegt."""


class CommunityMarketplaceConfig(BaseModel):
    """Community Skill Marketplace Konfiguration.

    Steuert den Zugriff auf das oeffentliche Community-Skill-Registry
    (GitHub-Repo ``Alex8791-cyber/skill-registry``).  Community-Skills sind
    architektonisch malware-sicher: Skills sind Daten (Markdown), nicht Code.
    """

    enabled: bool = Field(
        default=True,
        description="Community Marketplace aktivieren",
    )
    """Aktiviert oder deaktiviert den Community Marketplace."""

    registry_url: str = Field(
        default="https://raw.githubusercontent.com/Alex8791-cyber/skill-registry/main",
        description="URL des Community-Skill-Registry (GitHub Raw Content)",
    )
    """Basis-URL fuer das Registry-Repo.  Kann auf einen Fork zeigen."""

    auto_recall_check_interval: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Intervall fuer automatische Recall-Checks (Sekunden)",
    )
    """Wie oft nach zurueckgerufenen Skills gesucht wird (Default: 1h)."""

    min_publisher_reputation: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimaler Publisher-Reputation-Score fuer Installation",
    )
    """Skills von Publishern unter diesem Score werden mit Warnung installiert."""

    require_verified_publisher: bool = Field(
        default=False,
        description="Nur Skills von verifizierten Publishern installieren",
    )
    """Wenn True, werden nur Skills von Publishern mit TrustLevel >= VERIFIED
    installiert.  Schraenkt die Auswahl stark ein."""

    max_tool_calls_default: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default max Tool-Calls pro Community-Skill-Aufruf",
    )
    """Globaler Default fuer max_tool_calls wenn kein Manifest-Wert."""

    auto_sync: bool = Field(
        default=True,
        description="Registry automatisch beim Start synchronisieren",
    )
    """Wenn True, wird die Registry beim Start und periodisch synchronisiert."""


# --------------------------------------------------------------------------
# Dashboard- und Modell-Override-Konfigurationen
# --------------------------------------------------------------------------


class DashboardConfig(BaseModel):
    """Konfiguration fuer das optionale Web-Dashboard.

    Das Dashboard bietet eine grafische Oberflaeche zur Ueberwachung von
    Cron-Jobs, Heartbeats, Skills und Speicherzustaenden. Es kann
    aktiviert werden, wenn FastAPI oder ein kompatibler Web-Server
    installiert ist. Der Standard-Port ist 9090. Das Dashboard ist
    standardmaessig deaktiviert, um ungewollte Netzwerkschnittstellen zu
    vermeiden.
    """

    enabled: bool = False
    """Legt fest, ob das Dashboard beim Start automatisch geladen wird."""

    port: int = Field(default=9090, ge=1024, le=65535)
    """Port auf dem das Dashboard lauschen soll."""


class ModelOverrideConfig(BaseModel):
    """Konfiguration fuer Modell-Overrides pro Skill.

    Mit diesem Mapping koennen Nutzer fuer einzelne Skills alternative
    Modelle definieren. Der Schluessel ist der Skill-Name (Dateiname ohne
    Erweiterung), der Wert der interne Modell-Name (z. B. "qwen3:32b").
    """

    skill_models: dict[str, str] = Field(default_factory=dict)


# ── Provider-spezifische Modell-Defaults ──────────────────────────
# Wenn ein Nutzer auf ein anderes LLM-Backend wechselt (z.B. OpenAI oder
# Anthropic), werden die Ollama-Modellnamen (qwen3:32b etc.) automatisch
# durch passende Modelle des jeweiligen Providers ersetzt -- aber nur, wenn
# the user has not explicitly overridden the model names.

_OLLAMA_DEFAULT_MODEL_NAMES = {
    "qwen3:32b",
    "qwen3:8b",
    "qwen3-coder:30b",
    "qwen2.5-coder:7b",
    "nomic-embed-text",
    "qwen3-embedding:0.6b",
    "llava:13b",
    "openbmb/minicpm-v4.5",
    # Legacy detection for upgrades from older versions
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-20250514",
}

_PROVIDER_MODEL_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    "openai": {
        "planner": {
            "name": "gpt-5.2",
            "context_window": 400000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "gpt-5-mini",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "o3",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing", "architecture"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "o4-mini",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        "embedding": {
            "name": "text-embedding-3-large",
            "context_window": 8191,
            "vram_gb": 0.0,
            "strengths": ["semantic-search"],
            "speed": "fast",
            "embedding_dimensions": 3072,
        },
        "vision": {
            "name": "gpt-5.2",
        },
    },
    "anthropic": {
        "planner": {
            "name": "claude-opus-4-6",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "claude-haiku-4-5-20251001",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "claude-sonnet-4-6",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "claude-haiku-4-5-20251001",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        # Anthropic hat keine Embedding-API → Ollama-Fallback bleibt
        "embedding": {
            "name": "qwen3-embedding:0.6b",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search", "multilingual"],
            "speed": "fast",
        },
        "vision": {
            "name": "claude-sonnet-4-6",
        },
    },
    "gemini": {
        "planner": {
            "name": "gemini-2.5-pro",
            "context_window": 1000000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "gemini-2.5-flash",
            "context_window": 1000000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "gemini-2.5-pro",
            "context_window": 1000000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "gemini-2.5-flash",
            "context_window": 1000000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        "embedding": {
            "name": "gemini-embedding-001",
            "context_window": 8192,
            "vram_gb": 0.0,
            "strengths": ["semantic-search"],
            "speed": "fast",
            "embedding_dimensions": 768,
        },
        "vision": {
            "name": "gemini-2.5-pro",
        },
    },
    "groq": {
        "planner": {
            "name": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "llama-3.1-8b-instant",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "llama-3.3-70b-versatile",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "llama-3.1-8b-instant",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        # Groq has no embedding API -> Ollama fallback
        "embedding": {
            "name": "qwen3-embedding:0.6b",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search", "multilingual"],
            "speed": "fast",
        },
        "vision": {
            "name": "meta-llama/llama-4-scout-17b-16e-instruct",
        },
    },
    "deepseek": {
        "planner": {
            "name": "deepseek-chat",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "deepseek-chat",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "deepseek-chat",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "deepseek-chat",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        # DeepSeek has no embedding API -> Ollama fallback
        "embedding": {
            "name": "qwen3-embedding:0.6b",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search", "multilingual"],
            "speed": "fast",
        },
        # DeepSeek has no vision API -> Ollama fallback
        "vision": {
            "name": "llava:13b",
        },
    },
    "mistral": {
        "planner": {
            "name": "mistral-large-latest",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "mistral-small-latest",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "codestral-latest",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "codestral-latest",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        "embedding": {
            "name": "mistral-embed",
            "context_window": 8192,
            "vram_gb": 0.0,
            "strengths": ["semantic-search"],
            "speed": "fast",
            "embedding_dimensions": 1024,
        },
        "vision": {
            "name": "pixtral-large-latest",
        },
    },
    "together": {
        "planner": {
            "name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        # Together has no embedding API -> Ollama fallback
        "embedding": {
            "name": "qwen3-embedding:0.6b",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search", "multilingual"],
            "speed": "fast",
        },
        "vision": {
            "name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        },
    },
    "openrouter": {
        "planner": {
            "name": "anthropic/claude-opus-4.6",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "google/gemini-2.5-flash",
            "context_window": 1000000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "anthropic/claude-sonnet-4.6",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "google/gemini-2.5-flash",
            "context_window": 1000000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        # OpenRouter hat keine Embedding-API → Ollama-Fallback bleibt
        "embedding": {
            "name": "qwen3-embedding:0.6b",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search", "multilingual"],
            "speed": "fast",
        },
        "vision": {
            "name": "anthropic/claude-sonnet-4.6",
        },
    },
    "xai": {
        "planner": {
            "name": "grok-4-1-fast-reasoning",
            "context_window": 2000000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "grok-4-1-fast-non-reasoning",
            "context_window": 2000000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "grok-code-fast-1",
            "context_window": 256000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "grok-code-fast-1",
            "context_window": 256000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        # xAI has no embedding API -> Ollama fallback
        "embedding": {
            "name": "qwen3-embedding:0.6b",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search", "multilingual"],
            "speed": "fast",
        },
        "vision": {
            "name": "grok-4-1-fast-reasoning",
        },
    },
    "cerebras": {
        "planner": {
            "name": "gpt-oss-120b",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "fast",
        },
        "executor": {
            "name": "llama3.1-8b",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "gpt-oss-120b",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "fast",
        },
        "coder_fast": {
            "name": "llama3.1-8b",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        "embedding": {
            "name": "qwen3-embedding:0.6b",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search", "multilingual"],
            "speed": "fast",
        },
        "vision": {
            "name": "llama-4-scout-17b-16e-instruct",
        },
    },
    "github": {
        "planner": {
            "name": "gpt-4.1",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "gpt-4.1-mini",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "gpt-4.1",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "gpt-4.1-mini",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        "embedding": {
            "name": "text-embedding-3-large",
            "context_window": 8191,
            "vram_gb": 0.0,
            "strengths": ["semantic-search"],
            "speed": "fast",
            "embedding_dimensions": 3072,
        },
        "vision": {
            "name": "gpt-4.1",
        },
    },
    "bedrock": {
        "planner": {
            "name": "us.anthropic.claude-opus-4-6-v1:0",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "us.anthropic.claude-sonnet-4-6-v1:0",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        "embedding": {
            "name": "amazon.titan-embed-text-v2:0",
            "context_window": 8192,
            "vram_gb": 0.0,
            "strengths": ["semantic-search"],
            "speed": "fast",
            "embedding_dimensions": 1024,
        },
        "vision": {
            "name": "us.anthropic.claude-sonnet-4-6-v1:0",
        },
    },
    "huggingface": {
        "planner": {
            "name": "meta-llama/Llama-3.3-70B-Instruct",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        "embedding": {
            "name": "qwen3-embedding:0.6b",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search", "multilingual"],
            "speed": "fast",
        },
        "vision": {
            "name": "llava:13b",
        },
    },
    "moonshot": {
        "planner": {
            "name": "kimi-k2.5",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german"],
            "speed": "medium",
        },
        "executor": {
            "name": "kimi-k2-turbo-preview",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks"],
            "speed": "fast",
        },
        "coder": {
            "name": "kimi-k2.5",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "kimi-k2-turbo-preview",
            "context_window": 128000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        "embedding": {
            "name": "qwen3-embedding:0.6b",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search", "multilingual"],
            "speed": "fast",
        },
        "vision": {
            "name": "kimi-k2.5",
        },
    },
    "claude-code": {
        "planner": {
            "name": "opus",
            "context_window": 1000000,
            "vram_gb": 0.0,
            "strengths": ["reasoning", "planning", "reflection", "german", "multi-agent"],
            "speed": "medium",
        },
        "executor": {
            "name": "haiku",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["tool-calling", "simple-tasks", "fast-responses"],
            "speed": "fast",
        },
        "coder": {
            "name": "sonnet",
            "context_window": 1000000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing", "long-context"],
            "speed": "medium",
        },
        "coder_fast": {
            "name": "haiku",
            "context_window": 200000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "real-time-coding"],
            "speed": "fast",
        },
        # Claude Code CLI hat keine Embedding-API -- Ollama-Fallback bleibt
        "embedding": {
            "name": "qwen3-embedding:0.6b",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search", "multilingual"],
            "speed": "fast",
        },
        "vision": {
            "name": "sonnet",
        },
    },
}

# Base URLs for OpenAI-compatible providers
_PROVIDER_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "xai": "https://api.x.ai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "github": "https://models.inference.ai.azure.com",
    "bedrock": "https://bedrock-runtime.us-east-1.amazonaws.com/v1",
    "huggingface": "https://api-inference.huggingface.co/v1",
    "moonshot": "https://api.moonshot.cn/v1",
}


class VoiceConfig(BaseModel):
    """Voice-spezifische Konfiguration (TTS/STT/Wake Word)."""

    tts_backend: str = "piper"  # "piper" | "espeak" | "elevenlabs"
    piper_voice: str = "de_DE-thorsten_emotional-medium"  # Piper-Stimme (HuggingFace-ID)
    piper_length_scale: float = Field(default=1.0, ge=0.5, le=2.0)  # Sprechgeschwindigkeit
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "hJAaR77ekN23CNyp0byH"
    elevenlabs_model: str = "eleven_multilingual_v2"
    wake_word_enabled: bool = True
    wake_word: str = "jarvis"
    wake_word_backend: str = "browser"  # "browser" | "vosk" | "porcupine"
    talk_mode_enabled: bool = False
    talk_mode_auto_listen: bool = False


class ChannelConfig(BaseModel):
    """Channel-Konfiguration. [B§9]"""

    cli_enabled: bool = True
    telegram_enabled: bool = False
    telegram_whitelist: list[str] = Field(default_factory=list)
    telegram_use_webhook: bool = Field(
        default=False,
        description="Telegram Webhook statt Polling verwenden",
    )
    telegram_webhook_url: str = Field(
        default="",
        description="Externe Webhook-URL (z.B. https://jarvis.example.com/telegram/webhook)",
    )
    telegram_webhook_port: int = Field(
        default=8443,
        ge=1024,
        le=65535,
        description="Lokaler Port fuer Telegram-Webhook-Server",
    )
    telegram_webhook_host: str = Field(
        default="0.0.0.0",
        description="Lokaler Bind-Host fuer Webhook-Server",
    )
    webui_enabled: bool = False
    webui_port: int = Field(default=8080, ge=1024, le=65535)
    voice_enabled: bool = False

    # Additional chat channels
    slack_enabled: bool = False
    slack_default_channel: str = ""
    discord_enabled: bool = False
    discord_channel_id: str = ""

    @field_validator("discord_channel_id", mode="before")
    @classmethod
    def _coerce_discord_id(cls, v: object) -> str:
        """Accept int (e.g. 0) from YAML and coerce to str."""
        return str(v) if v is not None else ""

    # Extended messaging channels
    whatsapp_enabled: bool = False
    whatsapp_default_chat: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_webhook_port: int = Field(default=8443, ge=1024, le=65535)
    whatsapp_verify_token: str = ""
    whatsapp_allowed_numbers: list[str] = Field(default_factory=list)
    signal_enabled: bool = False
    signal_default_user: str = ""
    matrix_enabled: bool = False
    matrix_homeserver: str = ""
    matrix_user_id: str = ""
    teams_enabled: bool = False
    teams_default_channel: str = ""
    imessage_enabled: bool = False
    imessage_device_id: str = ""

    # v22: Google Chat
    google_chat_enabled: bool = False
    google_chat_credentials_path: str = ""
    google_chat_allowed_spaces: list[str] = Field(default_factory=list)

    # v22: Mattermost
    mattermost_enabled: bool = False
    mattermost_url: str = ""
    mattermost_token: str = ""
    mattermost_channel: str = ""

    # v22: Feishu/Lark
    feishu_enabled: bool = False
    feishu_app_id: str = ""
    feishu_app_secret: str = ""

    # v22: IRC
    irc_enabled: bool = False
    irc_server: str = ""
    irc_port: int = Field(default=6667, ge=1, le=65535)
    irc_nick: str = "JarvisBot"
    irc_channels: list[str] = Field(default_factory=list)

    # v22: Twitch
    twitch_enabled: bool = False
    twitch_token: str = ""
    twitch_channel: str = ""
    twitch_allowed_users: list[str] = Field(default_factory=list)

    # v22: Voice Erweiterungen
    voice_config: VoiceConfig = Field(default_factory=VoiceConfig)


class LoggingConfig(BaseModel):
    """Logging-Konfiguration."""

    level: str = "INFO"
    json_logs: bool = False
    console: bool = True


class MtlsConfig(BaseModel):
    """Mutual TLS Konfiguration fuer Frontend-Backend-Kommunikation."""

    enabled: bool = Field(default=False, description="mTLS fuer WebUI-API aktivieren")
    certs_dir: str = Field(
        default="", description="Zertifikats-Verzeichnis (Standard: ~/.jarvis/certs/)"
    )
    auto_generate: bool = Field(default=True, description="Zertifikate automatisch generieren")


class SecurityConfig(BaseModel):
    """Sicherheits-Konfiguration. [B§11]"""

    # Maximale Agent-Loop Iterationen pro Anfrage
    max_iterations: int = Field(default=25, ge=1, le=50)
    # Allowed file paths (gatekeeper checks against these)
    # Projekt-Verzeichnis wird automatisch in gatekeeper.initialize() hinzugefuegt
    allowed_paths: list[str] = Field(
        default_factory=lambda: [
            "~/.jarvis/",
            str(Path(tempfile.gettempdir()) / "jarvis") + os.sep,
        ]
    )
    # Wenn True, wird das Projektverzeichnis (jarvis_home Parent) automatisch
    # zu allowed_paths hinzugefuegt, damit Cognithor in seine eigene Codebase
    # schreiben kann (z.B. Skripte erstellen, Code integrieren).
    allow_project_dir: bool = True
    # Regex patterns for destructive shell commands [B§3.2]
    blocked_commands: list[str] = Field(
        default_factory=lambda: [
            r"rm\s+-rf\s+/",
            r"mkfs\b",
            r"dd\s+if=/dev",
            r":\(\)\{\s*:\|:&\s*\};:",
            r"\bformat\s+",
            r"\bdel\s+/f\b",
            r"\bshutdown\b",
            r"\breboot\b",
        ]
    )
    # Regex patterns for credential detection [B§11]
    credential_patterns: list[str] = Field(
        default_factory=lambda: [
            r"sk-[a-zA-Z0-9]{20,}",
            r"token_[a-zA-Z0-9]+",
            r"password\s*[:=]\s*\S+",
            r"secret\s*[:=]\s*\S+",
            r"api_key\s*[:=]\s*\S+",
        ]
    )
    # Maximum recursion depth for sub-agent delegations
    max_sub_agent_depth: int = Field(default=3, ge=1, le=10)
    """Maximale Verschachtelungstiefe fuer Sub-Agent-Aufrufe via handle_message."""

    # TLS configuration for webhook server and API
    ssl_certfile: str = Field(default="", description="Pfad zum SSL-Zertifikat (PEM)")
    ssl_keyfile: str = Field(default="", description="Pfad zum SSL-Privat-Key (PEM)")

    # Mutual TLS fuer WebUI-API
    mtls: MtlsConfig = Field(default_factory=MtlsConfig)

    # TTLDict defaults for channel dicts
    channel_dict_ttl_seconds: int = Field(default=86400, ge=300, le=604800)
    channel_dict_max_size: int = Field(default=10000, ge=100, le=100000)

    # DNS-Cache
    dns_cache_ttl_seconds: int = Field(default=300, ge=30, le=3600)
    dns_cache_max_size: int = Field(default=1000, ge=100, le=10000)

    # Circuit Breaker
    circuit_breaker_failure_threshold: int = Field(default=5, ge=2, le=50)
    circuit_breaker_recovery_timeout: int = Field(default=60, ge=10, le=600)

    # Shell-Pfad-Validierung
    shell_validate_paths: bool = Field(default=True)


class AuditConfig(BaseModel):
    """Audit-Trail Konfiguration fuer Compliance."""

    hmac_enabled: bool = Field(
        default=True,
        description="HMAC-SHA256 Signaturen auf Audit-Eintraege",
    )
    hmac_key_file: str = Field(
        default="",
        description="Pfad zur HMAC-Key-Datei (leer = ~/.jarvis/audit_key)",
    )
    ed25519_enabled: bool = Field(
        default=False,
        description="Ed25519 asymmetrische Signaturen auf Audit-Eintraege (erfordert cryptography)",
    )
    ed25519_key_file: str = Field(
        default="",
        description="Pfad zur Ed25519-Key-Datei (leer = ~/.jarvis/audit_ed25519.key)",
    )
    breach_notification_enabled: bool = Field(
        default=True,
        description="Automatische Breach-Erkennung und Benachrichtigung",
    )
    breach_cooldown_hours: int = Field(
        default=1,
        ge=1,
        le=72,
        description="Mindestabstand zwischen Breach-Benachrichtigungen in Stunden",
    )
    retention_days: int = Field(
        default=90,
        ge=7,
        le=3650,
        description="Aufbewahrungsfrist fuer Audit-Logs in Tagen",
    )

    # RFC 3161 Timestamp Authority
    tsa_enabled: bool = Field(
        default=False,
        description="Taegliche RFC 3161 Timestamps auf Audit-Anchor-Hash",
    )
    tsa_url: str = Field(
        default="https://freetsa.org/tsr",
        description="URL des TSA-Servers",
    )

    # WORM Storage (future — prepared config fields)
    worm_backend: Literal["none", "s3", "minio"] = Field(
        default="none",
        description="WORM-Backend: none (lokal), s3 (AWS Object Lock), minio (Self-Hosted)",
    )
    worm_bucket: str = Field(
        default="",
        description="S3/MinIO Bucket-Name fuer WORM-Storage",
    )
    worm_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="WORM Retention-Lock in Tagen",
    )


class RecoveryConfig(BaseModel):
    """Smart Recovery & Transparency Konfiguration."""

    pre_flight_enabled: bool = Field(
        default=True,
        description="Plan-Vorschau vor komplexen Aktionen anzeigen",
    )
    pre_flight_timeout_seconds: int = Field(
        default=3,
        ge=1,
        le=30,
        description="Auto-Execute nach N Sekunden (agentic-first)",
    )
    pre_flight_min_steps: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Pre-Flight nur bei Plaenen mit N+ Schritten",
    )
    correction_learning_enabled: bool = Field(
        default=True,
        description="Aus User-Korrekturen lernen",
    )
    correction_proactive_threshold: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Nach N gleichen Korrekturen proaktiv fragen",
    )


# ============================================================================
# Autonomous Evolution Engine
# ============================================================================


class EvolutionConfig(BaseModel):
    """Autonomous Evolution Engine configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable autonomous evolution during idle time",
    )
    idle_minutes: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Minutes of inactivity before evolution starts",
    )
    max_cycles_per_day: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum evolution cycles per day",
    )
    cycle_cooldown_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Cooldown between cycles in seconds",
    )
    agent_budgets: dict[str, float] = Field(
        default_factory=dict,
        description="Per-agent daily budget in USD, e.g. {'scout': 0.50, 'skill_builder': 0.30}",
    )
    learning_goals: list[str] = Field(
        default_factory=list,
        description="User-defined learning topics, e.g. ['Python async patterns', 'Kubernetes deployment']",
    )
    # Deep Learning (Phase 5)
    deep_learning_enabled: bool = Field(
        default=True,
        description="Enable deep learning plans (auto-promotes complex goals)",
    )
    max_concurrent_plans: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum simultaneously active learning plans",
    )
    max_pages_per_crawl: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Maximum pages to fetch per sitemap crawl",
    )
    quality_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Minimum quality score to pass a SubGoal (0.0-1.0)",
    )
    coverage_threshold: float = Field(
        default=0.7,
        ge=0.3,
        le=1.0,
        description="Minimum coverage score to pass a SubGoal (0.0-1.0)",
    )
    auto_expand: bool = Field(
        default=True,
        description="HorizonScanner automatically adds new SubGoals",
    )


# ============================================================================
# Datenbank-Konfiguration
# ============================================================================


class DatabaseConfig(BaseModel):
    """Datenbank-Konfiguration."""

    backend: str = Field(default="sqlite", description="'sqlite' oder 'postgresql'")
    pg_host: str = "localhost"
    pg_port: int = Field(default=5432, ge=1, le=65535)
    pg_dbname: str = "jarvis"
    pg_user: str = "jarvis"
    pg_password: str = ""
    pg_pool_min: int = Field(default=2, ge=1, le=50)
    pg_pool_max: int = Field(default=10, ge=1, le=100)
    encryption_enabled: bool = Field(
        default=False,
        description="SQLite-Datenbanken mit SQLCipher verschluesseln",
    )
    encryption_backend: str = Field(
        default="keyring",
        description="Schluessel-Backend: 'keyring' (OS Credential Store)",
    )
    sqlite_max_retries: int = Field(default=5, ge=0, le=20)
    sqlite_retry_base_delay: float = Field(default=0.1, ge=0.01, le=5.0)


class QueueConfig(BaseModel):
    """Konfiguration fuer die Durable Message Queue."""

    enabled: bool = Field(default=False, description="Durable message queue aktivieren")
    max_size: int = Field(default=10_000, ge=100, le=1_000_000)
    ttl_hours: int = Field(default=24, ge=1, le=168)
    max_retries: int = Field(default=3, ge=0, le=10)
    priority_boost_channels: list[str] = Field(
        default_factory=lambda: ["api", "telegram"],
        description="Channels die automatisch höhere Priorität bekommen",
    )


class ImprovementGovernanceConfig(BaseModel):
    """Steuerung der Self-Improvement-Domains (SAFE_DOMAINS)."""

    enabled: bool = True
    auto_domains: list[str] = Field(
        default_factory=lambda: ["prompt_tuning", "tool_parameters", "workflow_order"],
    )
    hitl_domains: list[str] = Field(
        default_factory=lambda: ["memory_weights", "model_selection"],
    )
    blocked_domains: list[str] = Field(
        default_factory=lambda: ["code_generation"],
    )
    cooldown_minutes: int = Field(default=30, ge=5, le=1440)
    max_changes_per_hour: int = Field(default=5, ge=1, le=50)


class PromptEvolutionConfig(BaseModel):
    """A/B-Test-basierte Prompt-Evolution."""

    enabled: bool = False  # Opt-in Feature
    min_sessions_per_arm: int = Field(default=20, ge=5, le=200)
    significance_threshold: float = Field(default=0.05, ge=0.01, le=0.5)
    evolution_interval_hours: int = Field(default=6, ge=1, le=168)
    max_concurrent_tests: int = Field(default=1, ge=1, le=3)


class GEPAConfig(BaseModel):
    """GEPA — Guided Evolution through Pattern Analysis."""

    enabled: bool = True  # Opt-out (enabled by default)
    evolution_interval_hours: int = Field(default=6, ge=1, le=168)
    min_traces_for_proposal: int = Field(default=10, ge=3, le=100)
    max_active_optimizations: int = Field(default=1, ge=1, le=3)
    auto_rollback_threshold: float = Field(default=0.10, ge=0.01, le=0.5)
    auto_apply: bool = False  # If True, apply proposals automatically


# ============================================================================
# Hashline Guard
# ============================================================================


class HashlineGuardConfig(BaseModel):
    """Configuration for the Hashline Guard line-level integrity system."""

    enabled: bool = True
    hash_algorithm: str = "xxhash64"
    tag_length: int = Field(default=2, ge=2, le=4)
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    max_line_length: int = Field(default=10000, ge=100)
    stale_threshold_seconds: int = Field(default=300, ge=10)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=0.5, ge=0)
    cache_max_files: int = Field(default=100, ge=10, le=1000)
    binary_detection: bool = True
    audit_enabled: bool = True
    excluded_patterns: list[str] = Field(
        default_factory=lambda: [
            "*.pyc",
            "__pycache__/**",
            ".git/**",
            "*.lock",
            "node_modules/**",
        ]
    )
    protected_paths: list[str] = Field(default_factory=list)


class RetentionConfig(BaseModel):
    """Data retention periods (days)."""

    episodic_days: int = Field(default=90, ge=1, le=3650)
    processing_log_days: int = Field(default=90, ge=1, le=3650)
    model_usage_log_days: int = Field(default=180, ge=1, le=3650)
    him_report_days: int = Field(default=30, ge=1, le=365)
    vault_osint_days: int = Field(default=30, ge=1, le=365)
    session_days: int = Field(default=180, ge=1, le=3650)


class ComplianceConfig(BaseModel):
    """GDPR compliance configuration."""

    consent_required: bool = True
    compliance_engine_enabled: bool = True
    privacy_mode: bool = False
    privacy_notice_version: str = "1.0"
    cloud_consent_required: bool = True


class SessionConfig(BaseModel):
    """Session lifecycle settings."""

    inactivity_timeout_minutes: int = Field(
        default=30,
        description=(
            "Nach dieser Inaktivitaetszeit (Minuten) wird automatisch eine neue Session erstellt."
        ),
    )
    chat_history_limit: int = Field(
        default=100,
        description="Maximale Anzahl Chat-Nachrichten die beim Session-Resume geladen werden.",
    )


# ============================================================================
# Haupt-Konfiguration
# ============================================================================


class JarvisConfig(BaseModel):
    """Complete Jarvis configuration. [B§12, §4.9]

    Loaded once at startup and then used throughout the entire system.
    """

    # Meta — version is always read from the package's __version__
    version: str = __import__("jarvis").__version__
    language: str = Field(
        default="de",
        description="UI language for error messages, greetings, and status texts. "
        "Supports any installed i18n language pack (e.g., 'en', 'de', 'fr'). "
        "Also settable via JARVIS_LANGUAGE env var.",
    )
    owner_name: str = Field(
        default="User",
        description="Name des Besitzers/Benutzers. Wird in Prompts und CORE.md verwendet.",
    )

    # Betriebsmodus
    operation_mode: Literal["offline", "online", "hybrid", "auto"] = Field(
        default="auto",
        description=(
            "Betriebsmodus: 'offline', 'online', 'hybrid', 'auto' (auto-detect aus API-Keys)"
        ),
    )

    # LLM-Backend
    llm_backend_type: Literal[
        "ollama",
        "openai",
        "anthropic",
        "gemini",
        "groq",
        "deepseek",
        "mistral",
        "together",
        "openrouter",
        "xai",
        "cerebras",
        "github",
        "bedrock",
        "huggingface",
        "moonshot",
        "lmstudio",
        "claude-code",
    ] = Field(
        default="ollama",
        description=(
            "LLM-Backend: 'ollama', 'openai', 'anthropic', "
            "'gemini', 'groq', 'deepseek', 'mistral', "
            "'together', 'openrouter', 'xai', 'cerebras', "
            "'github', 'bedrock', 'huggingface', "
            "'moonshot', 'lmstudio', 'claude-code'"
        ),
    )
    openai_api_key: str = Field(default="", description="API-Key für OpenAI-kompatibles Backend")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base-URL für OpenAI-kompatibles Backend (auch für Together, Groq, vLLM)",
    )
    anthropic_api_key: str = Field(default="", description="API-Key für Anthropic Claude")
    anthropic_max_tokens: int = Field(
        default=4096, ge=1, le=1_000_000, description="Max Output-Tokens für Claude"
    )
    gemini_api_key: str = Field(default="", description="API-Key für Google Gemini")
    groq_api_key: str = Field(default="", description="API-Key für Groq")
    deepseek_api_key: str = Field(default="", description="API-Key für DeepSeek")
    mistral_api_key: str = Field(default="", description="API-Key für Mistral AI")
    together_api_key: str = Field(default="", description="API-Key für Together AI")
    openrouter_api_key: str = Field(default="", description="API-Key für OpenRouter")
    xai_api_key: str = Field(default="", description="API-Key für xAI (Grok)")
    cerebras_api_key: str = Field(default="", description="API-Key für Cerebras")
    github_api_key: str = Field(default="", description="API-Key/Token für GitHub Models")
    bedrock_api_key: str = Field(
        default="", description="API-Key für AWS Bedrock (OpenAI-kompatibel via Gateway)"
    )
    huggingface_api_key: str = Field(default="", description="API-Key für Hugging Face Inference")
    moonshot_api_key: str = Field(default="", description="API-Key für Moonshot/Kimi")
    lmstudio_api_key: str = Field(
        default="lm-studio", description="API-Key für LM Studio (beliebiger Wert, da lokal)"
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1", description="Base-URL für LM Studio API"
    )
    vision_model: str = Field(
        default="openbmb/minicpm-v4.5", description="Standard-Vision-Modell (schnell)"
    )
    vision_model_detail: str = Field(
        default="qwen3-vl:32b", description="Detail-Vision-Modell (höchste Qualität)"
    )

    # Basis-Pfade
    jarvis_home: Path = Field(default_factory=lambda: Path.home() / ".jarvis")

    # Subsystem-Konfigurationen
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    gatekeeper: GatekeeperConfig = Field(default_factory=GatekeeperConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    channels: ChannelConfig = Field(default_factory=ChannelConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    filesystem: FilesystemConfig = Field(default_factory=FilesystemConfig)
    shell: ShellConfig = Field(default_factory=ShellConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    code: CodeConfig = Field(default_factory=CodeConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
    osint: OsintConfig = Field(default_factory=OsintConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    context_pipeline: ContextPipelineConfig = Field(default_factory=ContextPipelineConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    marketplace: MarketplaceConfig = Field(default_factory=MarketplaceConfig)
    community_marketplace: CommunityMarketplaceConfig = Field(
        default_factory=CommunityMarketplaceConfig
    )
    improvement: ImprovementGovernanceConfig = Field(default_factory=ImprovementGovernanceConfig)
    prompt_evolution: PromptEvolutionConfig = Field(default_factory=PromptEvolutionConfig)
    gepa: GEPAConfig = Field(default_factory=GEPAConfig)
    hashline: HashlineGuardConfig = Field(default_factory=HashlineGuardConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)

    # Heartbeat- und Plugin-Konfigurationen
    # Die HeartbeatConfig steuert einen periodischen Check (Heartbeat), der
    # Aufgaben aus einer Checklist-Datei liest und als Systemnachricht
    # an den Gateway-Handler sendet. Die PluginsConfig definiert, in
    # which directory additional skills (procedures) are installed in
    # werden und ob automatische Updates erlaubt sind.
    heartbeat: HeartbeatConfig = Field(default_factory=lambda: HeartbeatConfig())
    plugins: PluginsConfig = Field(default_factory=lambda: PluginsConfig())

    # Cost Tracking
    cost_tracking_enabled: bool = True
    daily_budget_usd: float = Field(
        default=0.0, ge=0.0, description="Tageslimit in USD (0 = kein Limit)"
    )
    monthly_budget_usd: float = Field(
        default=0.0, ge=0.0, description="Monatslimit in USD (0 = kein Limit)"
    )

    dashboard: DashboardConfig = Field(default_factory=lambda: DashboardConfig())
    model_overrides: ModelOverrideConfig = Field(default_factory=lambda: ModelOverrideConfig())

    # Multi-Instance / Distributed Locking
    lock_backend: Literal["local", "file", "redis"] = Field(
        default="local",
        description="Lock-Backend für Multi-Instance: local, file, redis",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        pattern=r"^rediss?://",
        description="Redis-URL für distributed locking und queuing",
    )

    @field_validator(
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "groq_api_key",
        "deepseek_api_key",
        "mistral_api_key",
        "together_api_key",
        "openrouter_api_key",
        "xai_api_key",
        "cerebras_api_key",
        "github_api_key",
        "bedrock_api_key",
        "huggingface_api_key",
        "moonshot_api_key",
        mode="before",
    )
    @classmethod
    def _validate_api_key_length(cls, v: str) -> str:
        v = v.strip()
        if v and v != "***" and len(v) < 8:
            raise ValueError(
                f"API-Key zu kurz ({len(v)} Zeichen, mind. 8). "
                "Prüfe ob der Key korrekt kopiert wurde."
            )
        return v

    # ---- Auto-Adaptation: Modelle an LLM-Backend anpassen ----

    def model_post_init(self, __context: Any) -> None:
        """Automatically adjusts model names to the chosen LLM backend.

        Wenn der Nutzer ein anderes Backend als Ollama waehlt (z.B. durch
        Setzen von llm_backend_type oder Eingabe eines API-Keys), werden
        die Standard-Ollama-Modellnamen (qwen3:32b, qwen3:8b, etc.)
        automatisch durch passende Provider-Modelle ersetzt.

        Explizit vom Nutzer gesetzte Modellnamen bleiben erhalten.
        """
        # Backend-Typ bestimmen: explizit gesetzt oder aus API-Key ableiten
        backend = self.llm_backend_type
        if backend == "ollama" and self.operation_mode not in ("offline", "hybrid"):
            # Auto-Detection: Wenn ein API-Key vorhanden ist aber der
            # Backend-Typ noch auf "ollama" steht, Backend automatisch setzen
            # Priority: anthropic > openai > gemini > groq > deepseek > mistral > together
            # NICHT wenn operation_mode="offline" — dann bleibt Ollama.
            # NICHT wenn operation_mode="hybrid" — Hybrid nutzt Ollama als
            # Standard-Backend. Cloud-APIs nur fuer Frontier Agent / komplexe Tasks.
            if self.anthropic_api_key:
                backend = "anthropic"
                object.__setattr__(self, "llm_backend_type", "anthropic")
            elif self.openai_api_key:
                backend = "openai"
                object.__setattr__(self, "llm_backend_type", "openai")
            elif self.gemini_api_key:
                backend = "gemini"
                object.__setattr__(self, "llm_backend_type", "gemini")
            elif self.groq_api_key:
                backend = "groq"
                object.__setattr__(self, "llm_backend_type", "groq")
            elif self.deepseek_api_key:
                backend = "deepseek"
                object.__setattr__(self, "llm_backend_type", "deepseek")
            elif self.mistral_api_key:
                backend = "mistral"
                object.__setattr__(self, "llm_backend_type", "mistral")
            elif self.together_api_key:
                backend = "together"
                object.__setattr__(self, "llm_backend_type", "together")
            elif self.openrouter_api_key:
                backend = "openrouter"
                object.__setattr__(self, "llm_backend_type", "openrouter")
            elif self.xai_api_key:
                backend = "xai"
                object.__setattr__(self, "llm_backend_type", "xai")
            elif self.cerebras_api_key:
                backend = "cerebras"
                object.__setattr__(self, "llm_backend_type", "cerebras")
            elif self.github_api_key:
                backend = "github"
                object.__setattr__(self, "llm_backend_type", "github")
            elif self.bedrock_api_key:
                backend = "bedrock"
                object.__setattr__(self, "llm_backend_type", "bedrock")
            elif self.huggingface_api_key:
                backend = "huggingface"
                object.__setattr__(self, "llm_backend_type", "huggingface")
            elif self.moonshot_api_key:
                backend = "moonshot"
                object.__setattr__(self, "llm_backend_type", "moonshot")

        # Auto-detect OperationMode (VOR dem fruehen Return)
        from jarvis.models import OperationMode

        _has_any_api_key = any(
            [
                self.openai_api_key,
                self.anthropic_api_key,
                self.gemini_api_key,
                self.groq_api_key,
                self.deepseek_api_key,
                self.mistral_api_key,
                self.together_api_key,
                self.openrouter_api_key,
                self.xai_api_key,
                self.cerebras_api_key,
                self.github_api_key,
                self.bedrock_api_key,
                self.huggingface_api_key,
                self.moonshot_api_key,
            ]
        )
        if self.operation_mode == "auto":
            if _has_any_api_key:
                object.__setattr__(self, "_resolved_operation_mode", OperationMode.ONLINE)
            else:
                object.__setattr__(self, "_resolved_operation_mode", OperationMode.OFFLINE)
        else:
            try:
                object.__setattr__(
                    self, "_resolved_operation_mode", OperationMode(self.operation_mode)
                )
            except ValueError:
                log.warning(
                    "Unbekannter operation_mode '%s', fallback auf OFFLINE", self.operation_mode
                )
                object.__setattr__(self, "_resolved_operation_mode", OperationMode.OFFLINE)

        if backend == "ollama" or backend not in _PROVIDER_MODEL_DEFAULTS:
            return

        provider_defaults = _PROVIDER_MODEL_DEFAULTS[backend]

        # Check each model role and adjust if necessary
        for role in ("planner", "executor", "coder", "coder_fast", "embedding"):
            current_model: ModelConfig = getattr(self.models, role)
            if current_model.name in _OLLAMA_DEFAULT_MODEL_NAMES:
                role_defaults = provider_defaults.get(role)
                if role_defaults:
                    new_model = ModelConfig(**role_defaults)
                    object.__setattr__(current_model, "name", new_model.name)
                    object.__setattr__(current_model, "context_window", new_model.context_window)
                    object.__setattr__(current_model, "vram_gb", new_model.vram_gb)
                    object.__setattr__(current_model, "strengths", new_model.strengths)
                    object.__setattr__(current_model, "speed", new_model.speed)
                    if "embedding_dimensions" in role_defaults:
                        object.__setattr__(
                            current_model, "embedding_dimensions", new_model.embedding_dimensions
                        )

        # Heartbeat-Modell ebenfalls anpassen wenn noch auf Ollama-Default
        if self.heartbeat.model in _OLLAMA_DEFAULT_MODEL_NAMES:
            executor_default = provider_defaults.get("executor", {})
            if executor_default:
                object.__setattr__(self.heartbeat, "model", executor_default["name"])

        # Vision-Modell anpassen (einfacher String, kein ModelConfig)
        if self.vision_model in _OLLAMA_DEFAULT_MODEL_NAMES:
            vision_default = provider_defaults.get("vision", {})
            if vision_default:
                object.__setattr__(self, "vision_model", vision_default["name"])

        # Timeout Cross-Validation (#48 Optimierung)
        self._cross_validate_timeouts()

    def _cross_validate_timeouts(self) -> None:
        """Checks timeout consistency between subsystems."""
        base = self.executor.default_timeout_seconds
        issues: list[str] = []

        # Tool-specific timeouts must be >= default_timeout
        for field_name in (
            "media_analyze_image_timeout",
            "media_transcribe_audio_timeout",
            "media_extract_text_timeout",
            "media_tts_timeout",
            "run_python_timeout",
        ):
            val = getattr(self.executor, field_name, base)
            if val < base:
                issues.append(
                    f"executor.{field_name} ({val}s) < executor.default_timeout_seconds ({base}s)"
                )

        # Shell timeout should not be larger than executor timeout
        shell_timeout = self.shell.default_timeout_seconds
        if shell_timeout > base * 10:
            issues.append(
                f"shell.default_timeout_seconds ({shell_timeout}s) ist unverhältnismäßig groß "
                f"vs executor.default_timeout_seconds ({base}s)"
            )

        # Ollama-Timeout sollte >= Executor-Timeout sein
        ollama_timeout = self.ollama.timeout_seconds
        if ollama_timeout < base:
            issues.append(
                f"ollama.timeout_seconds ({ollama_timeout}s) < "
                f"executor.default_timeout_seconds ({base}s)"
                f" — LLM-Calls könnten vom Executor "
                f"vorzeitig abgebrochen werden"
            )

        for issue in issues:
            log.warning("config_timeout_warning: %s", issue)

    # ---- Convenience Properties ----

    @property
    def resolved_operation_mode(self) -> Any:
        """Gibt den aufgeloesten Betriebsmodus zurueck (OperationMode Enum)."""
        from jarvis.models import OperationMode

        return getattr(self, "_resolved_operation_mode", OperationMode.OFFLINE)

    @property
    def log_level(self) -> str:
        """Shortcut for logging.level."""
        return self.logging.level

    @property
    def core_memory_path(self) -> Path:
        """Alias for core_memory_file."""
        return self.core_memory_file

    # ---- Abgeleitete Pfade (per Property, nicht manuell konfigurierbar) ----

    @property
    def config_file(self) -> Path:
        """Pfad zur Jarvis-Konfigurationsdatei."""
        return self.jarvis_home / "config.yaml"

    @property
    def policies_dir(self) -> Path:
        """Directory for security policies."""
        return self.jarvis_home / self.gatekeeper.policies_dir

    @property
    def memory_dir(self) -> Path:
        """Wurzelverzeichnis des Memory-Systems."""
        return self.jarvis_home / "memory"

    @property
    def core_memory_file(self) -> Path:
        """Pfad zur CORE.md (Core Memory)."""
        return self.memory_dir / "CORE.md"

    @property
    def episodes_dir(self) -> Path:
        """Directory for episodic daily log files."""
        return self.memory_dir / "episodes"

    @property
    def knowledge_dir(self) -> Path:
        """Directory for semantic knowledge."""
        return self.memory_dir / "knowledge"

    @property
    def procedures_dir(self) -> Path:
        """Directory for procedural skills."""
        return self.memory_dir / "procedures"

    @property
    def sessions_dir(self) -> Path:
        """Directory for session data."""
        return self.memory_dir / "sessions"

    @property
    def index_dir(self) -> Path:
        """Directory for BM25/FTS index files."""
        return self.jarvis_home / "index"

    @property
    def db_path(self) -> Path:
        """Pfad zur SQLite-Datenbank des Indexers."""
        return self.index_dir / "memory.db"

    @property
    def embeddings_cache_path(self) -> Path:
        """Pfad zum Embeddings-Cache."""
        return self.index_dir / "embeddings.cache"

    @property
    def workspace_dir(self) -> Path:
        """Working directory for temporary files."""
        return self.jarvis_home / "workspace"

    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        return self.jarvis_home / "logs"

    @property
    def mcp_config_file(self) -> Path:
        """Pfad zur MCP-Server-Konfiguration."""
        return self.jarvis_home / "mcp" / "config.yaml"

    @property
    def cron_config_file(self) -> Path:
        """Pfad zur Cron-Konfiguration."""
        return self.jarvis_home / "cron" / "jobs.yaml"

    # ---- Verzeichnisstruktur-Management ----

    def ensure_directories(self) -> list[str]:
        """Erstellt ~/.jarvis/ Verzeichnisstruktur."""
        return ensure_directory_structure(self)

    def ensure_default_files(self) -> list[str]:
        """Alias -- ensure_directories() erstellt auch Default-Dateien."""
        # ensure_directory_structure erstellt bereits Dirs + Files
        # Only check here if default files are missing
        return ensure_directory_structure(self)


# ============================================================================
# Config-Laden
# ============================================================================


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Tiefes Mergen von zwei Dicts. Override gewinnt bei Konflikten."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Wendet JARVIS_* Umgebungsvariablen an.

    Konvention: JARVIS_SECTION_KEY → data["section"]["key"]
    Beispiel: JARVIS_OLLAMA_BASE_URL → data["ollama"]["base_url"]
    """
    prefix = "JARVIS_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("_")
        if len(parts) >= 2:
            # Recursive descent: walk into existing dict sections,
            # then set the remaining parts (joined with _) as leaf key.
            node = data
            consumed = 0
            for i in range(len(parts) - 1):
                candidate = parts[i]
                if candidate in node and isinstance(node[candidate], dict):
                    node = node[candidate]
                    consumed = i + 1
                else:
                    break
            if consumed == 0:
                # No existing section found -- use first part as section
                section = parts[0]
                if section not in node:
                    node[section] = {}
                if isinstance(node[section], dict):
                    node = node[section]
                    consumed = 1
            leaf_key = "_".join(parts[consumed:])
            if leaf_key:
                node[leaf_key] = value
        elif len(parts) == 1:
            data[parts[0]] = value
    return data


def load_config(config_path: Path | None = None) -> JarvisConfig:
    """Loads the configuration.

    Order (later overrides earlier):
      1. Defaults (in den Pydantic-Modellen)
      2. config.yaml (wenn vorhanden)
      3. JARVIS_* Umgebungsvariablen

    Args:
        config_path: Expliziter Pfad zur config.yaml. Wenn None: ~/.jarvis/config.yaml

    Returns:
        Fully validated JarvisConfig.
    """
    data: dict[str, Any] = {}

    # 1. Config-Datei laden (wenn vorhanden)
    if config_path is None:
        config_path = Path.home() / ".jarvis" / "config.yaml"

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                file_data = yaml.safe_load(f) or {}
            if isinstance(file_data, dict):
                data = file_data
        except yaml.YAMLError as exc:
            import logging

            logging.getLogger("jarvis.config").warning(
                "Fehlerhafte config.yaml wird ignoriert: %s",
                exc,
            )

    # 2. Umgebungsvariablen anwenden
    data = _apply_env_overrides(data)

    # 3. Version aus YAML ignorieren — immer aus dem Package nehmen
    data.pop("version", None)

    # 4. Model-Strings aus Env-Vars in ModelConfig-Dicts konvertieren
    #    JARVIS_MODELS_PLANNER=qwen3.5:9b → {"name": "qwen3.5:9b"}
    _MODEL_ROLES = {"planner", "executor", "coder", "coder_fast", "embedding"}
    if "models" in data and isinstance(data["models"], dict):
        for role in _MODEL_ROLES:
            val = data["models"].get(role)
            if isinstance(val, str):
                data["models"][role] = {"name": val}

    # 5. Pydantic validates and fills defaults
    return JarvisConfig(**data)


# ============================================================================
# Create directory structure
# ============================================================================


_DEFAULT_CORE_MEMORY = """\
# Identitaet

Ich bin Jarvis, das lokale, autonome Agent-Betriebssystem von {owner_name}.
Ich laufe vollstaendig auf dem lokalen Rechner -- keine Cloud, keine externen APIs,
und damit voll DSGVO-konform. Mein Zuhause ist `~/.jarvis/`.

{owner_name} ist der Besitzer und Benutzer dieses Systems.

## Persoenlichkeit

Ich bin kompetent, direkt und effizient. Ich kommuniziere praegnant und
respektvoll, ohne unnoetige Floskeln. Wenn etwas nicht funktioniert,
formuliere ich das klar und mache konstruktive Vorschlaege zur Verbesserung.
Ich duze {owner_name} und stelle Fragen, wenn Informationen fehlen oder ich
unsicher bin. Ich rate nicht -- ich frage nach.

## Fachgebiet

Jarvis ist nicht auf eine bestimmte Branche beschraenkt. Ich unterstuetze
{owner_name} bei einer Vielzahl von Aufgaben wie Recherche, Projekt- und
Organisationsmanagement, Dateiverwaltung, Notizen und Planung. Neue
Faehigkeiten koennen jederzeit durch Prozeduren hinzugefuegt oder angepasst
werden.

## Harte Regeln -- IMMER einhalten

1. DATENSCHUTZ: Niemals persoenliche Informationen (Namen, Adressen,
   Geburtsdaten, Vertragsnummern oder Gesundheitsdaten) in Logs, Shell-Ausgaben
   oder unverschluesselte Dateien schreiben.
2. DATENBLEIBEN: Alle Daten bleiben lokal. Kein Upload, kein Cloud-Sync.
3. E-MAILS: E-Mails IMMER als Entwurf vorlegen. Niemals automatisch
   versenden, es sei denn {owner_name} bestaetigt es ausdruecklich.
4. SHELL: Keine destruktiven Befehle (rm -rf, mkfs, dd). Im Zweifel nachfragen.
5. PLAN-LIMIT: Maximal 25 Iterationen pro Anfrage. Danach zusammenfassen
   und nachfragen.
6. SICHERHEIT: Keine illegalen, unsicheren oder gegen Policies verstossenden
   Handlungen ausfuehren.

## Technisches Umfeld

-- Hardware: Haengt vom System ab (z. B. leistungsfaehige GPU empfohlen fuer
  grosse Modelle)
-- LLM: Standard-Modelle via Ollama (lokal)
-- Planner: z. B. „qwen3:32b" fuer umfangreiche Planung
-- Executor: z. B. „qwen3:8b" fuer schnelle Tool-Aufrufe
-- Coder: Modell fuer Code-Generierung (optional)
-- Embeddings: Modell fuer Hybrid-Suche (z. B. „nomic-embed-text")

## Praeferenzen

-- Sprache: Deutsch (Code-Kommentare auf Deutsch, Variablennamen auf Englisch)
-- Codesprache: Python
-- Zeitzone: Europe/Berlin
-- Anrede: {owner_name} (Du)
-- Kommunikation: Direkt, substanziell, ohne Fuellwoerter
-- Bei Unsicherheit: Lieber nachfragen als raten
-- Ausgabeformat: Markdown fuer strukturierte Inhalte, Plaintext fuer kurze
  Antworten
"""

_DEFAULT_POLICY = """\
# Jarvis · Default policies
# Architektur-Bibel §3.2

rules:
  - name: no_destructive_shell
    match:
      tool: exec_command
      params:
        command:
          regex: "rm -rf|mkfs|dd if=/dev|:(){ :|:& };:|format |del /f|shutdown|reboot"
    action: BLOCK
    reason: "Destruktiver Shell-Befehl erkannt"

  - name: email_requires_approval
    match:
      tool: email_send
    action: APPROVE
    reason: "E-Mail-Versand erfordert Bestaetigung"

  - name: file_delete_requires_approval
    match:
      tool: delete_file
    action: APPROVE
    reason: "Datei loeschen erfordert Bestaetigung"

  - name: file_write_inform
    match:
      tool: write_file
    action: INFORM
    reason: "Datei wird geschrieben"

  - name: memory_read_allow
    match:
      tool: search_memory
    action: ALLOW
    reason: "Memory-Lesen ist sicher"

  - name: credential_masking
    match:
      tool: "*"
      params:
        contains_pattern: "(sk-|token_|password|secret|api_key)"
    action: MASK
    reason: "Credential in Parameter erkannt -- wird maskiert"
"""

_DEFAULT_CONFIG = """\
# Jarvis · Main configuration
# Generated on first start. Customize as needed.

# Name of the user -- used in prompts and greetings.
owner_name: User

ollama:
  base_url: http://localhost:11434
  timeout_seconds: 120
  keep_alive: 30m

planner:
  max_iterations: 25
  temperature: 0.7

memory:
  chunk_size_tokens: 400
  chunk_overlap_tokens: 80
  search_top_k: 6

channels:
  cli_enabled: true
  telegram_enabled: false
  slack_enabled: false
  slack_default_channel: ""
  discord_enabled: false
  discord_channel_id: ""
  whatsapp_enabled: false
  whatsapp_default_chat: ""
  signal_enabled: false
  signal_default_user: ""
  matrix_enabled: false
  matrix_homeserver: ""
  matrix_user_id: ""
  teams_enabled: false
  teams_default_channel: ""
  imessage_enabled: false
  imessage_device_id: ""
  # v22: Neue Channels
  google_chat_enabled: false
  google_chat_credentials_path: ""
  mattermost_enabled: false
  mattermost_url: ""
  mattermost_token: ""
  mattermost_channel: ""
  feishu_enabled: false
  feishu_app_id: ""
  feishu_app_secret: ""
  irc_enabled: false
  irc_server: ""
  irc_port: 6667
  irc_nick: JarvisBot
  twitch_enabled: false
  twitch_token: ""
  twitch_channel: ""

# ── Tool Toggles ─────────────────────────────────────────────────
# Steuert welche Tool-Gruppen aktiv sind.
# Desktop-Tools sind aus Sicherheitsgruenden standardmaessig deaktiviert.

tools:
  computer_use_enabled: false   # Desktop-Automation (Screenshot + Klick)
  desktop_tools_enabled: false  # Clipboard und Screenshots

heartbeat:
  enabled: false
  interval_minutes: 30
  checklist_file: HEARTBEAT.md
  channel: cli
  model: qwen3:8b

plugins:
  skills_dir: skills
  auto_update: false

dashboard:
  enabled: false
  port: 9090

model_overrides:
  skill_models: {}

logging:
  level: INFO
  json_logs: false
  console: true
"""

_DEFAULT_CRON_JOBS = """\
# Jarvis · Geplante Aufgaben
# Architektur-Bibel §10.1

jobs:
  morning_briefing:
    schedule: "0 7 * * 1-5"
    prompt: |
      Erstelle mein Morning Briefing:
      1. Heutige Termine
      2. Ungelesene E-Mails (Zusammenfassung)
      3. Offene Aufgaben aus gestern
      4. Wetter fuer Nuernberg
    channel: telegram
    model: qwen3:8b
    enabled: false

  weekly_review:
    schedule: "0 18 * * 5"
    prompt: |
      Wochenrueckblick:
      - Was wurde diese Woche erledigt?
      - Welche neuen Prozeduren wurden gelernt?
      - Was ist noch offen?
    channel: telegram
    model: qwen3:32b
    enabled: false
"""

_DEFAULT_MCP_CONFIG = """\
# Jarvis · MCP-Server Konfiguration
#
# Builtin tools (automatically active, no configuration needed):
#   Filesystem: read_file, write_file, edit_file, list_directory, delete_file
#   Shell:      exec_command
#   Code:       run_python, analyze_code
#   Memory:     search_memory, save_to_memory, get_entity, add_entity,
#               get_recent_episodes, search_procedures, get_core_memory,
#               update_core_section, save_episode, get_working_context
#   Web:        web_search, web_fetch
#
# Register external MCP servers here.
# These are automatically connected on gateway start.

servers: {}
  # Example: Custom MCP server (Python)
  # mein_server:
  #   transport: stdio
  #   command: python
  #   args: ["-m", "mein_mcp_modul"]
  #   enabled: true
  #
  # Example: NPX-based MCP server
  # github_server:
  #   transport: stdio
  #   command: npx
  #   args: ["-y", "@modelcontextprotocol/server-github"]
  #   env:
  #     GITHUB_TOKEN: "ghp_..."
  #   enabled: false

# ── MCP-Server-Modus (OPTIONAL) ─────────────────────────────────
# Exposes Jarvis itself as an MCP server so external clients
# (Claude Desktop, Cursor, VS Code, other agents) can connect.
#
# Mode: disabled (default), stdio, http, both
# WARNING: Disabled by default! Only enable when needed.

server_mode:
  mode: disabled
  # http_host: "127.0.0.1"
  # http_port: 3001
  # server_name: "jarvis"
  # require_auth: false
  # auth_token: ""
  # expose_tools: true
  # expose_resources: true
  # expose_prompts: true
  # enable_sampling: false

# ── A2A Protocol (OPTIONAL) ──────────────────────────────────────
# Agent-zu-Agent-Kommunikation nach Linux Foundation A2A RC v1.0.
# Allows Jarvis to receive tasks from other agents and
# selbst Tasks an Remote-Agenten zu delegieren.
# JSON-RPC 2.0 over HTTP, SSE streaming, push notifications.
#
# WARNING: Disabled by default! Only enable when needed.

a2a:
  enabled: false
  # host: "127.0.0.1"
  # port: 3002
  # agent_name: "Jarvis"
  # agent_description: ""
  # require_auth: false
  # auth_token: ""
  # max_tasks: 100
  # task_timeout_seconds: 3600
  # enable_streaming: false
  # enable_push: false
  # remotes: []  # Liste von Remote-Agenten: [{endpoint: "http://...", auth_token: ""}]
"""

# Heartbeat checklist -- created on first start in ``~/.jarvis/HEARTBEAT.md``
# if no existing file is found. Users can customize this file to define
# tasks that Jarvis should periodically check. Each line represents a task
# or checklist item. Jarvis sends the content unchanged as a heartbeat
# message.
_DEFAULT_HEARTBEAT_MD = """\
# Heartbeat Checkliste

Dies ist die Standard-Heartbeat-Datei. Du kannst diese Liste beliebig
anpassen, um periodische Erinnerungen oder Checks zu definieren. Jede
Zeile stellt einen zu ueberpruefenden Punkt dar. Beispiel:

- 📬 Neue E-Mails pruefen
- 📅 Kalendertermine fuer heute zusammenfassen
- 📝 Offene Aufgaben aus der To-Do-Liste anzeigen
- 🔔 Benachrichtigungen aus externen Diensten abrufen

Wenn keine relevanten Punkte gefunden werden, antwortet Jarvis mit
"HEARTBEAT_OK".
"""


def ensure_directory_structure(config: JarvisConfig) -> list[str]:
    """Creates the complete ~/.jarvis/ directory structure. [B§4.9]

    Idempotent -- kann beliebig oft aufgerufen werden.
    Creates only what is missing, never overwrites existing files.

    Raises:
        PermissionError: Wenn Verzeichnisse/Dateien nicht erstellt werden
            koennen (fehlende Berechtigungen). Die Fehlermeldung enthaelt
            den Fix-Befehl (z.B. sudo chown).
        OSError: Wenn die Festplatte voll ist oder ein anderes
            Dateisystem-Problem vorliegt.

    Returns:
        List of newly created paths (for logging).
    """
    created: list[str] = []

    # Verzeichnisse
    dirs = [
        config.jarvis_home,
        config.policies_dir,
        config.memory_dir,
        config.episodes_dir,
        config.knowledge_dir,
        config.knowledge_dir / "kunden",
        config.knowledge_dir / "produkte",
        config.knowledge_dir / "projekte",
        config.procedures_dir,
        config.sessions_dir,
        config.index_dir,
        config.workspace_dir,
        config.workspace_dir / "tmp",
        config.logs_dir,
        config.jarvis_home / "mcp",
        config.jarvis_home / "mcp" / "servers",
        config.jarvis_home / "cron",
        config.jarvis_home / "locks",
        # Directory for plugins/skills
        config.jarvis_home / config.plugins.skills_dir,
    ]

    for d in dirs:
        if not d.exists():
            _safe_mkdir(d)
            created.append(str(d))

    # Default-Dateien (nur wenn nicht vorhanden)
    default_files: list[tuple[Path, str]] = [
        (config.config_file, _DEFAULT_CONFIG),
        (config.core_memory_file, _DEFAULT_CORE_MEMORY.format(owner_name=config.owner_name)),
        (config.policies_dir / "default.yaml", _DEFAULT_POLICY),
        (config.cron_config_file, _DEFAULT_CRON_JOBS),
        (config.mcp_config_file, _DEFAULT_MCP_CONFIG),
        # Heartbeat-Checkliste
        (config.jarvis_home / config.heartbeat.checklist_file, _DEFAULT_HEARTBEAT_MD),
    ]

    for path, content in default_files:
        if not path.exists():
            _safe_mkdir(path.parent)
            _safe_write(path, content)
            created.append(str(path))

    # Starter-Prozeduren aus data/procedures/ kopieren (nur wenn Verzeichnis leer)
    _install_starter_procedures(config.procedures_dir, created)

    return created


def _safe_mkdir(d: Path) -> None:
    """mkdir mit PermissionError/disk-full Handling und nutzerfreundlichen Meldungen."""
    try:
        d.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        home = d
        # Finde das hoechste nicht-existierende Verzeichnis fuer den Fix-Befehl
        while home.parent != home and not home.parent.exists():
            home = home.parent
        fix = (
            f"Verzeichnis '{d}' konnte nicht erstellt werden (fehlende Berechtigung).\n"
            f"  Beheben mit:\n"
            f"    sudo mkdir -p {d}\n"
            f"    sudo chown -R $(whoami) {home}"
        )
        log.error("directory_permission_error: %s", d)
        raise PermissionError(fix) from None
    except OSError as exc:
        # errno 28 = ENOSPC (disk full), errno 30 = EROFS (read-only FS)
        if exc.errno == 28:
            fix = (
                f"Festplatte voll -- Verzeichnis '{d}' konnte nicht erstellt werden.\n"
                f"  Freien Speicher pruefen: df -h {d.parent}\n"
                f"  Mindestens 500 MB freier Speicher werden benoetigt."
            )
            log.error("disk_full: %s", d)
            raise OSError(fix) from None
        log.error("directory_creation_failed: %s — %s", d, exc)
        raise


def _safe_write(path: Path, content: str) -> None:
    """write_text mit PermissionError/disk-full Handling."""
    try:
        path.write_text(content, encoding="utf-8")
    except PermissionError:
        fix = (
            f"Datei '{path}' konnte nicht geschrieben werden (fehlende Berechtigung).\n"
            f"  Beheben mit:\n"
            f"    sudo chown $(whoami) {path.parent}"
        )
        log.error("file_permission_error: %s", path)
        raise PermissionError(fix) from None
    except OSError as exc:
        if exc.errno == 28:
            fix = (
                f"Festplatte voll -- Datei '{path}' konnte nicht geschrieben werden.\n"
                f"  Freien Speicher pruefen: df -h {path.parent}\n"
                f"  Mindestens 500 MB freier Speicher werden benoetigt."
            )
            log.error("disk_full_write: %s", path)
            raise OSError(fix) from None
        log.error("file_write_failed: %s — %s", path, exc)
        raise


def _install_starter_procedures(procedures_dir: Path, created: list[str]) -> None:
    """Kopiert Starter-Prozeduren wenn das Verzeichnis leer ist."""
    # Nur installieren wenn noch keine Prozeduren existieren
    existing = list(procedures_dir.glob("*.md"))
    if existing:
        return

    # data/procedures/ relativ zum Package suchen
    try:
        data_dir = Path(__file__).parent.parent.parent / "data" / "procedures"
        if not data_dir.exists():
            return

        for proc_file in data_dir.glob("*.md"):
            target = procedures_dir / proc_file.name
            if not target.exists():
                target.write_text(proc_file.read_text(encoding="utf-8"), encoding="utf-8")
                created.append(str(target))
    except Exception:
        log.debug("starter_procedures_copy_skipped", exc_info=True)
