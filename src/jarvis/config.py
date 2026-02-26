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
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from jarvis.models import ModelConfig, SandboxConfig

log = logging.getLogger(__name__)

# ============================================================================
# Konfigurationsmodelle
# ============================================================================


class OllamaConfig(BaseModel):
    """Ollama-Server Konfiguration."""

    base_url: str = "http://localhost:11434"
    timeout_seconds: int = Field(default=120, ge=10, le=600)
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
            name="qwen3-coder:32b",
            context_window=32768,
            vram_gb=20.0,
            strengths=["code-generation", "debugging", "testing"],
            speed="medium",
        )
    )
    embedding: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="nomic-embed-text",
            context_window=8192,
            vram_gb=0.5,
            strengths=["semantic-search"],
            speed="fast",
        )
    )


class GatekeeperConfig(BaseModel):
    """Gatekeeper-Einstellungen. [B§3.2]"""

    policies_dir: str = "policies"  # Relativ zu jarvis_home
    default_risk_level: Literal["green", "yellow", "orange", "red"] = "yellow"
    max_blocked_retries: int = Field(default=3, ge=1, le=10)


class PlannerConfig(BaseModel):
    """Planner-Einstellungen. [B§3.1, §3.4]"""

    max_iterations: int = Field(default=10, ge=1, le=50)
    escalation_after: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    response_token_budget: int = Field(default=3000, ge=500, le=8000)


class WebConfig(BaseModel):
    """Web-Tools Konfiguration. [B§5.3]

    Suchbackends (Priorität: SearXNG > Brave > DuckDuckGo):
      - SearXNG: Self-hosted, keine API-Keys nötig
      - Brave Search: Schnell, 2000 Anfragen/Monat kostenlos
      - DuckDuckGo: Immer verfügbar, kein API-Key nötig (Standard-Fallback)
    """

    searxng_url: str = ""
    """URL der SearXNG-Instanz (z.B. 'http://localhost:8888')."""

    brave_api_key: str = ""
    """Brave Search API Key (https://brave.com/search/api/)."""

    duckduckgo_enabled: bool = True
    """DuckDuckGo als kostenloser Fallback wenn kein anderes Backend konfiguriert."""


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
    compaction_keep_last_n: int = Field(default=4, ge=2, le=20)

    # Episodic Memory: Wie viele Tage an Tageslogs sollen behalten werden?
    # Ältere Dateien werden beim Initialisieren des Memory-Systems gelöscht.
    episodic_retention_days: int = Field(default=365, ge=1, le=3650)

    # Dynamische Gewichtung der Hybrid-Suche.
    # Wenn aktiviert, passt Jarvis die Gewichtungsfaktoren (Vektor/BM25/Graph)
    # zur Laufzeit basierend auf Eigenschaften der Suchanfrage an.
    # Bei kurzen Anfragen werden lexikalische und Graph-Treffer stärker gewichtet,
    # bei langen oder komplexen Anfragen erhalten semantische (Vektor-)Treffer
    # mehr Gewicht. Ist diese Option deaktiviert, werden die statischen
    # Gewichtungen (weight_vector, weight_bm25, weight_graph) aus der
    # Konfiguration verwendet.
    dynamic_weighting: bool = False

    @model_validator(mode="after")
    def validate_weights(self) -> "MemoryConfig":
        """Validator um sicherzustellen, dass die Gewichtungen der Hybrid-Suche
        sich sinnvoll verhalten.

        Falls die Summe der Gewichte von Vektor-, BM25- und Graph-Kanal größer
        als 1.0 ist, werden alle drei Gewichte so skaliert, dass ihre Summe
        1.0 ergibt. Dies verhindert ungewollte Überskalierung und sorgt für
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
    """Konfiguration für den Heartbeat-Mechanismus.

    Wenn aktiviert, führt Jarvis in regelmäßigen Abständen einen
    "Heartbeat" aus. Dabei werden Aufgaben aus einer Checklist-Datei
    (standardmäßig ``HEARTBEAT.md``) gelesen und als Systemnachricht
    an den Gateway gesendet. Auf diese Weise kann Jarvis proaktiv
    überprüfen, ob neue E-Mails, Kalendertermine oder Aufgaben
    Aufmerksamkeit erfordern, ohne dass der Nutzer eine Anfrage stellt.
    """

    enabled: bool = False
    """Aktiviert oder deaktiviert den Heartbeat. Wenn ``False``, wird
    keine periodische Heartbeat-Nachricht gesendet."""

    interval_minutes: int = Field(default=30, ge=1, le=1440)
    """Interval in Minuten zwischen zwei Heartbeat-Läufen.
    Standardwert sind 30 Minuten. Der zulässige Wertebereich liegt
    zwischen 1 und 1440 Minuten (24 Stunden)."""

    checklist_file: str = "HEARTBEAT.md"
    """Dateiname der Checklist im ``jarvis_home``. Diese Datei enthält
    Text oder Bullet-Points, die beim Heartbeat an den Agenten
    übermittelt werden. Falls die Datei nicht existiert, wird eine
    leere Nachricht gesendet."""

    channel: str = "cli"
    """Name des Kanals, über den Heartbeat-Meldungen gesendet werden.
    Standard ist ``cli``; weitere gültige Werte sind die Namen der
    registrierten Channels (z. B. ``telegram``, ``webui``)."""

    model: str = "qwen3:8b"
    """Name des Modells, das für Heartbeat-Kommunikation verwendet
    werden soll. Dieser Wert wird im Cron-Job ignoriert, ist aber
    vorhanden, damit sich Heartbeats semantisch wie CronJobs verhalten."""


class PluginsConfig(BaseModel):
    """Konfiguration für das Plugin-Ökosystem.

    Plugins stellen zusätzliche Skills (Prozeduren, Tools oder
    Channel-Erweiterungen) bereit. Sie werden in einem separaten
    Verzeichnis abgelegt und zur Laufzeit geladen. Automatische
    Updates können deaktiviert werden, wenn der Nutzer volle Kontrolle
    über installierte Plugins wünscht.
    """

    skills_dir: str = "skills"
    """Relativer Name des Verzeichnisses im ``jarvis_home``, in dem
    zusätzliche Skills installiert werden. Der Standardwert ist
    ``skills``. Dies führt dazu, dass externe Prozeduren in
    ``~/.jarvis/skills`` abgelegt werden."""

    auto_update: bool = False
    """Legt fest, ob Jarvis beim Start automatisch nach Updates für
    installierte Plugins sucht und diese einspielt. Standardmäßig
    deaktiviert, um ungewollte Änderungen zu verhindern."""


# --------------------------------------------------------------------------
# Dashboard- und Modell-Override-Konfigurationen
# --------------------------------------------------------------------------

class DashboardConfig(BaseModel):
    """Konfiguration für das optionale Web-Dashboard.

    Das Dashboard bietet eine grafische Oberfläche zur Überwachung von
    Cron-Jobs, Heartbeats, Skills und Speicherzuständen. Es kann
    aktiviert werden, wenn FastAPI oder ein kompatibler Web-Server
    installiert ist. Der Standard-Port ist 9090. Das Dashboard ist
    standardmäßig deaktiviert, um ungewollte Netzwerkschnittstellen zu
    vermeiden.
    """

    enabled: bool = False
    """Legt fest, ob das Dashboard beim Start automatisch geladen wird."""

    port: int = Field(default=9090, ge=1024, le=65535)
    """Port auf dem das Dashboard lauschen soll."""


class ModelOverrideConfig(BaseModel):
    """Konfiguration für Modell-Overrides pro Skill.

    Mit diesem Mapping können Nutzer für einzelne Skills alternative
    Modelle definieren. Der Schlüssel ist der Skill-Name (Dateiname ohne
    Erweiterung), der Wert der interne Modell-Name (z. B. "qwen3:32b").
    """

    skill_models: dict[str, str] = Field(default_factory=dict)


# ── Provider-spezifische Modell-Defaults ──────────────────────────
# Wenn ein Nutzer auf ein anderes LLM-Backend wechselt (z.B. OpenAI oder
# Anthropic), werden die Ollama-Modellnamen (qwen3:32b etc.) automatisch
# durch passende Modelle des jeweiligen Providers ersetzt -- aber nur, wenn
# der Nutzer die Modellnamen nicht explizit überschrieben hat.

_OLLAMA_DEFAULT_MODEL_NAMES = {
    "qwen3:32b",
    "qwen3:8b",
    "qwen3-coder:32b",
    "nomic-embed-text",
    "llava:13b",
    # Legacy-Erkennung für Upgrades von älteren Versionen
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
            "name": "gpt-5.1-codex",
            "context_window": 400000,
            "vram_gb": 0.0,
            "strengths": ["code-generation", "debugging", "testing"],
            "speed": "medium",
        },
        "embedding": {
            "name": "text-embedding-3-large",
            "context_window": 8191,
            "vram_gb": 0.0,
            "strengths": ["semantic-search"],
            "speed": "fast",
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
        # Anthropic hat keine Embedding-API → Ollama-Fallback bleibt
        "embedding": {
            "name": "nomic-embed-text",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search"],
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
        "embedding": {
            "name": "gemini-embedding-001",
            "context_window": 8192,
            "vram_gb": 0.0,
            "strengths": ["semantic-search"],
            "speed": "fast",
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
        # Groq has no embedding API -> Ollama fallback
        "embedding": {
            "name": "nomic-embed-text",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search"],
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
        # DeepSeek has no embedding API -> Ollama fallback
        "embedding": {
            "name": "nomic-embed-text",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search"],
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
        "embedding": {
            "name": "mistral-embed",
            "context_window": 8192,
            "vram_gb": 0.0,
            "strengths": ["semantic-search"],
            "speed": "fast",
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
        # Together has no embedding API -> Ollama fallback
        "embedding": {
            "name": "nomic-embed-text",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search"],
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
        # OpenRouter hat keine Embedding-API → Ollama-Fallback bleibt
        "embedding": {
            "name": "nomic-embed-text",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search"],
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
        # xAI has no embedding API -> Ollama fallback
        "embedding": {
            "name": "nomic-embed-text",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search"],
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
        "embedding": {
            "name": "nomic-embed-text",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search"],
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
        "embedding": {
            "name": "text-embedding-3-large",
            "context_window": 8191,
            "vram_gb": 0.0,
            "strengths": ["semantic-search"],
            "speed": "fast",
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
        "embedding": {
            "name": "amazon.titan-embed-text-v2:0",
            "context_window": 8192,
            "vram_gb": 0.0,
            "strengths": ["semantic-search"],
            "speed": "fast",
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
        "embedding": {
            "name": "nomic-embed-text",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search"],
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
        "embedding": {
            "name": "nomic-embed-text",
            "context_window": 8192,
            "vram_gb": 0.5,
            "strengths": ["semantic-search"],
            "speed": "fast",
        },
        "vision": {
            "name": "kimi-k2.5",
        },
    },
}

# Base-URLs für OpenAI-kompatible Provider
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
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "hJAaR77ekN23CNyp0byH"
    elevenlabs_model: str = "eleven_multilingual_v2"
    wake_word_enabled: bool = False
    wake_word: str = "jarvis"
    wake_word_backend: str = "vosk"  # "vosk" | "porcupine"
    talk_mode_enabled: bool = False
    talk_mode_auto_listen: bool = False


class ChannelConfig(BaseModel):
    """Channel-Konfiguration. [B§9]"""

    cli_enabled: bool = True
    telegram_enabled: bool = False
    telegram_whitelist: list[str] = Field(default_factory=list)
    webui_enabled: bool = False
    webui_port: int = Field(default=8080, ge=1024, le=65535)
    voice_enabled: bool = False

    # Zusätzliche Chat-Kanäle
    slack_enabled: bool = False
    slack_default_channel: str = ""
    discord_enabled: bool = False
    discord_channel_id: int = 0

    # Erweiterte Messaging-Kanäle
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


class SecurityConfig(BaseModel):
    """Sicherheits-Konfiguration. [B§11]"""

    # Maximale Agent-Loop Iterationen pro Anfrage
    max_iterations: int = Field(default=10, ge=1, le=50)
    # Erlaubte Dateipfade (Gatekeeper prüft dagegen)
    allowed_paths: list[str] = Field(default_factory=lambda: ["~/.jarvis/", "/tmp/jarvis/"])
    # Regex-Patterns für destruktive Shell-Befehle [B§3.2]
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
    # Regex-Patterns für Credential-Erkennung [B§11]
    credential_patterns: list[str] = Field(
        default_factory=lambda: [
            r"sk-[a-zA-Z0-9]{20,}",
            r"token_[a-zA-Z0-9]+",
            r"password\s*[:=]\s*\S+",
            r"secret\s*[:=]\s*\S+",
            r"api_key\s*[:=]\s*\S+",
        ]
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


# ============================================================================
# Haupt-Konfiguration
# ============================================================================


class JarvisConfig(BaseModel):
    """Complete Jarvis configuration. [B§12, §4.9]

    Loaded once at startup and then used throughout the entire system.
    """

    # Meta
    version: str = "0.22.0"
    owner_name: str = Field(
        default="User",
        description="Name des Besitzers/Benutzers. Wird in Prompts und CORE.md verwendet.",
    )

    # Betriebsmodus
    operation_mode: Literal["offline", "online", "hybrid", "auto"] = Field(
        default="auto",
        description="Betriebsmodus: 'offline', 'online', 'hybrid', 'auto' (auto-detect aus API-Keys)",
    )

    # LLM-Backend
    llm_backend_type: Literal[
        "ollama", "openai", "anthropic", "gemini",
        "groq", "deepseek", "mistral", "together",
        "openrouter", "xai", "cerebras", "github", "bedrock",
        "huggingface", "moonshot",
    ] = Field(
        default="ollama",
        description="LLM-Backend: 'ollama', 'openai', 'anthropic', 'gemini', 'groq', 'deepseek', 'mistral', 'together', 'openrouter', 'xai', 'cerebras', 'github', 'bedrock', 'huggingface', 'moonshot'",
    )
    openai_api_key: str = Field(default="", description="API-Key für OpenAI-kompatibles Backend")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base-URL für OpenAI-kompatibles Backend (auch für Together, Groq, vLLM)",
    )
    anthropic_api_key: str = Field(default="", description="API-Key für Anthropic Claude")
    anthropic_max_tokens: int = Field(default=4096, description="Max Output-Tokens für Claude")
    gemini_api_key: str = Field(default="", description="API-Key für Google Gemini")
    groq_api_key: str = Field(default="", description="API-Key für Groq")
    deepseek_api_key: str = Field(default="", description="API-Key für DeepSeek")
    mistral_api_key: str = Field(default="", description="API-Key für Mistral AI")
    together_api_key: str = Field(default="", description="API-Key für Together AI")
    openrouter_api_key: str = Field(default="", description="API-Key für OpenRouter")
    xai_api_key: str = Field(default="", description="API-Key für xAI (Grok)")
    cerebras_api_key: str = Field(default="", description="API-Key für Cerebras")
    github_api_key: str = Field(default="", description="API-Key/Token für GitHub Models")
    bedrock_api_key: str = Field(default="", description="API-Key für AWS Bedrock (OpenAI-kompatibel via Gateway)")
    huggingface_api_key: str = Field(default="", description="API-Key für Hugging Face Inference")
    moonshot_api_key: str = Field(default="", description="API-Key für Moonshot/Kimi")
    vision_model: str = Field(default="llava:13b", description="Vision-Model für Screenshot-Analyse")

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
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # Heartbeat- und Plugin-Konfigurationen
    # Die HeartbeatConfig steuert einen periodischen Check (Heartbeat), der
    # Aufgaben aus einer Checklist-Datei liest und als Systemnachricht
    # an den Gateway-Handler sendet. Die PluginsConfig definiert, in
    # welchem Verzeichnis zusätzliche Skills (Prozeduren) installiert
    # werden und ob automatische Updates erlaubt sind.
    heartbeat: 'HeartbeatConfig' = Field(default_factory=lambda: HeartbeatConfig())
    plugins: 'PluginsConfig' = Field(default_factory=lambda: PluginsConfig())

    # Cost Tracking
    cost_tracking_enabled: bool = True
    daily_budget_usd: float = Field(default=0.0, ge=0.0, description="Tageslimit in USD (0 = kein Limit)")
    monthly_budget_usd: float = Field(default=0.0, ge=0.0, description="Monatslimit in USD (0 = kein Limit)")

    dashboard: 'DashboardConfig' = Field(default_factory=lambda: DashboardConfig())
    model_overrides: 'ModelOverrideConfig' = Field(default_factory=lambda: ModelOverrideConfig())

    # ---- Auto-Adaptation: Modelle an LLM-Backend anpassen ----

    def model_post_init(self, __context: Any) -> None:
        """Passt Modellnamen automatisch an das gewählte LLM-Backend an.

        Wenn der Nutzer ein anderes Backend als Ollama wählt (z.B. durch
        Setzen von llm_backend_type oder Eingabe eines API-Keys), werden
        die Standard-Ollama-Modellnamen (qwen3:32b, qwen3:8b, etc.)
        automatisch durch passende Provider-Modelle ersetzt.

        Explizit vom Nutzer gesetzte Modellnamen bleiben erhalten.
        """
        # Backend-Typ bestimmen: explizit gesetzt oder aus API-Key ableiten
        backend = self.llm_backend_type
        if backend == "ollama":
            # Auto-Detection: Wenn ein API-Key vorhanden ist aber der
            # Backend-Typ noch auf "ollama" steht, Backend automatisch setzen
            # Priorität: anthropic > openai > gemini > groq > deepseek > mistral > together
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
        _has_any_api_key = any([
            self.openai_api_key, self.anthropic_api_key, self.gemini_api_key,
            self.groq_api_key, self.deepseek_api_key, self.mistral_api_key,
            self.together_api_key, self.openrouter_api_key, self.xai_api_key,
            self.cerebras_api_key, self.github_api_key, self.bedrock_api_key,
            self.huggingface_api_key, self.moonshot_api_key,
        ])
        if self.operation_mode == "auto":
            if _has_any_api_key:
                object.__setattr__(self, "_resolved_operation_mode", OperationMode.ONLINE)
            else:
                object.__setattr__(self, "_resolved_operation_mode", OperationMode.OFFLINE)
        else:
            try:
                object.__setattr__(self, "_resolved_operation_mode", OperationMode(self.operation_mode))
            except ValueError:
                log.warning("Unbekannter operation_mode '%s', fallback auf OFFLINE", self.operation_mode)
                object.__setattr__(self, "_resolved_operation_mode", OperationMode.OFFLINE)

        if backend == "ollama" or backend not in _PROVIDER_MODEL_DEFAULTS:
            return

        provider_defaults = _PROVIDER_MODEL_DEFAULTS[backend]

        # Jede Modell-Rolle prüfen und ggf. anpassen
        for role in ("planner", "executor", "coder", "embedding"):
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

    # ---- Convenience Properties ----

    @property
    def resolved_operation_mode(self) -> Any:
        """Gibt den aufgeloesten Betriebsmodus zurueck (OperationMode Enum)."""
        from jarvis.models import OperationMode
        return getattr(self, "_resolved_operation_mode", OperationMode.OFFLINE)

    @property
    def log_level(self) -> str:
        """Shortcut für logging.level."""
        return self.logging.level

    @property
    def core_memory_path(self) -> Path:
        """Alias für core_memory_file."""
        return self.core_memory_file

    # ---- Abgeleitete Pfade (per Property, nicht manuell konfigurierbar) ----

    @property
    def config_file(self) -> Path:
        """Pfad zur Jarvis-Konfigurationsdatei."""
        return self.jarvis_home / "config.yaml"

    @property
    def policies_dir(self) -> Path:
        """Verzeichnis für Sicherheits-Policies."""
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
        """Verzeichnis für episodische Tageslog-Dateien."""
        return self.memory_dir / "episodes"

    @property
    def knowledge_dir(self) -> Path:
        """Verzeichnis für semantisches Wissen."""
        return self.memory_dir / "knowledge"

    @property
    def procedures_dir(self) -> Path:
        """Verzeichnis für prozedurale Skills."""
        return self.memory_dir / "procedures"

    @property
    def sessions_dir(self) -> Path:
        """Verzeichnis für Session-Daten."""
        return self.memory_dir / "sessions"

    @property
    def index_dir(self) -> Path:
        """Verzeichnis für BM25/FTS-Indexdateien."""
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
        """Arbeitsverzeichnis für temporäre Dateien."""
        return self.jarvis_home / "workspace"

    @property
    def logs_dir(self) -> Path:
        """Verzeichnis für Log-Dateien."""
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
        # Hier nur noch prüfen ob Default-Dateien fehlen
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
        parts = key[len(prefix):].lower().split("_")
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
    """Lädt die Konfiguration.

    Reihenfolge (spätere überschreiben frühere):
      1. Defaults (in den Pydantic-Modellen)
      2. config.yaml (wenn vorhanden)
      3. JARVIS_* Umgebungsvariablen

    Args:
        config_path: Expliziter Pfad zur config.yaml. Wenn None: ~/.jarvis/config.yaml

    Returns:
        Vollständig validierte JarvisConfig.
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
                "Fehlerhafte config.yaml wird ignoriert: %s", exc,
            )

    # 2. Umgebungsvariablen anwenden
    data = _apply_env_overrides(data)

    # 3. Pydantic validiert und füllt Defaults
    return JarvisConfig(**data)


# ============================================================================
# Verzeichnisstruktur erstellen
# ============================================================================


_DEFAULT_CORE_MEMORY = """\
# Identität

Ich bin Jarvis, das lokale, autonome Agent-Betriebssystem von {owner_name}.
Ich laufe vollständig auf dem lokalen Rechner -- keine Cloud, keine externen APIs,
und damit voll DSGVO-konform. Mein Zuhause ist `~/.jarvis/`.

{owner_name} ist der Besitzer und Benutzer dieses Systems.

## Persönlichkeit

Ich bin kompetent, direkt und effizient. Ich kommuniziere prägnant und
respektvoll, ohne unnötige Floskeln. Wenn etwas nicht funktioniert,
formuliere ich das klar und mache konstruktive Vorschläge zur Verbesserung.
Ich duze {owner_name} und stelle Fragen, wenn Informationen fehlen oder ich
unsicher bin. Ich rate nicht -- ich frage nach.

## Fachgebiet

Jarvis ist nicht auf eine bestimmte Branche beschränkt. Ich unterstütze
{owner_name} bei einer Vielzahl von Aufgaben wie Recherche, Projekt- und
Organisationsmanagement, Dateiverwaltung, Notizen und Planung. Neue
Fähigkeiten können jederzeit durch Prozeduren hinzugefügt oder angepasst
werden.

## Harte Regeln -- IMMER einhalten

1. DATENSCHUTZ: Niemals persönliche Informationen (Namen, Adressen,
   Geburtsdaten, Vertragsnummern oder Gesundheitsdaten) in Logs, Shell-Ausgaben
   oder unverschlüsselte Dateien schreiben.
2. DATENBLEIBEN: Alle Daten bleiben lokal. Kein Upload, kein Cloud-Sync.
3. E-MAILS: E-Mails IMMER als Entwurf vorlegen. Niemals automatisch
   versenden, es sei denn {owner_name} bestätigt es ausdrücklich.
4. SHELL: Keine destruktiven Befehle (rm -rf, mkfs, dd). Im Zweifel nachfragen.
5. PLAN-LIMIT: Maximal 10 Iterationen pro Anfrage. Danach zusammenfassen
   und nachfragen.
6. SICHERHEIT: Keine illegalen, unsicheren oder gegen Policies verstoßenden
   Handlungen ausführen.

## Technisches Umfeld

-- Hardware: Hängt vom System ab (z. B. leistungsfähige GPU empfohlen für
  große Modelle)
-- LLM: Standard-Modelle via Ollama (lokal)
-- Planner: z. B. „qwen3:32b" für umfangreiche Planung
-- Executor: z. B. „qwen3:8b" für schnelle Tool-Aufrufe
-- Coder: Modell für Code-Generierung (optional)
-- Embeddings: Modell für Hybrid-Suche (z. B. „nomic-embed-text")

## Präferenzen

-- Sprache: Deutsch (Code-Kommentare auf Deutsch, Variablennamen auf Englisch)
-- Codesprache: Python
-- Zeitzone: Europe/Berlin
-- Anrede: {owner_name} (Du)
-- Kommunikation: Direkt, substanziell, ohne Füllwörter
-- Bei Unsicherheit: Lieber nachfragen als raten
-- Ausgabeformat: Markdown für strukturierte Inhalte, Plaintext für kurze
  Antworten
"""

_DEFAULT_POLICY = """\
# Jarvis · Standard-Policies
# Architektur-Bibel §3.2

rules:
  - name: no_destructive_shell
    match:
      tool: exec_command
      params.command:
        regex: "rm -rf|mkfs|dd if=/dev|:(){ :|:& };:|format |del /f|shutdown|reboot"
    action: BLOCK
    reason: "Destruktiver Shell-Befehl erkannt"

  - name: email_requires_approval
    match:
      tool: email_send
    action: APPROVE
    reason: "E-Mail-Versand erfordert Bestätigung"

  - name: file_delete_requires_approval
    match:
      tool: delete_file
    action: APPROVE
    reason: "Datei löschen erfordert Bestätigung"

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
# Jarvis · Hauptkonfiguration
# Generiert beim ersten Start. Anpassen nach Bedarf.

# Name des Benutzers -- wird in Prompts und Begrüßungen verwendet.
owner_name: User

ollama:
  base_url: http://localhost:11434
  timeout_seconds: 120
  keep_alive: 30m

planner:
  max_iterations: 10
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
  discord_channel_id: 0
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
      4. Wetter für Nürnberg
    channel: telegram
    model: qwen3:8b
    enabled: false

  weekly_review:
    schedule: "0 18 * * 5"
    prompt: |
      Wochenrückblick:
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
# Builtin-Tools (automatisch aktiv, keine Konfiguration nötig):
#   Filesystem: read_file, write_file, edit_file, list_directory, delete_file
#   Shell:      exec_command
#   Memory:     search_memory, save_to_memory, get_entity, add_entity,
#               get_recent_episodes, search_procedures, get_core_memory,
#               update_core_section, save_episode, get_working_context
#   Web:        web_search, web_fetch
#
# Externe MCP-Server hier registrieren.
# Diese werden beim Gateway-Start automatisch verbunden.

servers: {}
  # Beispiel: Eigener MCP-Server (Python)
  # mein_server:
  #   transport: stdio
  #   command: python
  #   args: ["-m", "mein_mcp_modul"]
  #   enabled: true
  #
  # Beispiel: NPX-basierter MCP-Server
  # github_server:
  #   transport: stdio
  #   command: npx
  #   args: ["-y", "@modelcontextprotocol/server-github"]
  #   env:
  #     GITHUB_TOKEN: "ghp_..."
  #   enabled: false

# ── MCP-Server-Modus (OPTIONAL) ─────────────────────────────────
# Exponiert Jarvis selbst als MCP-Server, damit externe Clients
# (Claude Desktop, Cursor, VS Code, andere Agenten) sich verbinden können.
#
# Modus: disabled (Standard), stdio, http, both
# ACHTUNG: Standardmäßig deaktiviert! Nur aktivieren wenn benötigt.

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
# Ermöglicht Jarvis Tasks von anderen Agenten zu empfangen und
# selbst Tasks an Remote-Agenten zu delegieren.
# JSON-RPC 2.0 über HTTP, SSE-Streaming, Push Notifications.
#
# ACHTUNG: Standardmäßig deaktiviert! Nur aktivieren wenn benötigt.

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

# Heartbeat-Checkliste -- wird beim ersten Start in ``~/.jarvis/HEARTBEAT.md``
# angelegt, falls keine vorhandene Datei existiert. Nutzer können diese
# Datei anpassen, um Aufgaben zu definieren, die Jarvis periodisch prüfen
# soll. Jede Zeile repräsentiert eine Aufgabe oder ein Element der
# Checkliste. Jarvis sendet den Inhalt unverändert als Heartbeat-Nachricht.
_DEFAULT_HEARTBEAT_MD = """\
# Heartbeat Checkliste

Dies ist die Standard-Heartbeat-Datei. Du kannst diese Liste beliebig
anpassen, um periodische Erinnerungen oder Checks zu definieren. Jede
Zeile stellt einen zu überprüfenden Punkt dar. Beispiel:

- 📬 Neue E-Mails prüfen
- 📅 Kalendertermine für heute zusammenfassen
- 📝 Offene Aufgaben aus der To-Do-Liste anzeigen
- 🔔 Benachrichtigungen aus externen Diensten abrufen

Wenn keine relevanten Punkte gefunden werden, antwortet Jarvis mit
"HEARTBEAT_OK".
"""


def ensure_directory_structure(config: JarvisConfig) -> list[str]:
    """Erstellt die vollständige ~/.jarvis/ Verzeichnisstruktur. [B§4.9]

    Idempotent -- kann beliebig oft aufgerufen werden.
    Erstellt nur was fehlt, überschreibt nie vorhandene Dateien.

    Returns:
        Liste der neu erstellten Pfade (für Logging).
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
        # Verzeichnis für Plugins/Skills
        config.jarvis_home / config.plugins.skills_dir,
    ]

    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
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
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            created.append(str(path))

    # Starter-Prozeduren aus data/procedures/ kopieren (nur wenn Verzeichnis leer)
    _install_starter_procedures(config.procedures_dir, created)

    return created


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
