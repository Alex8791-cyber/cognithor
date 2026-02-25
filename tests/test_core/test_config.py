"""
Tests für jarvis.config – Konfigurationssystem.

Testet:
  - Config-Laden mit Defaults
  - Config aus YAML-Datei
  - Umgebungsvariablen-Override
  - Verzeichnisstruktur-Erstellung
  - Idempotenz (doppeltes Erstellen ist sicher)
  - Pfad-Properties
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jarvis.config import (
    JarvisConfig,
    ensure_directory_structure,
    load_config,
)

if TYPE_CHECKING:
    import pytest


class TestJarvisConfigDefaults:
    """Config mit reinen Defaults (kein YAML, keine Env-Vars)."""

    def test_default_home(self) -> None:
        config = JarvisConfig()
        assert config.jarvis_home == Path.home() / ".jarvis"

    def test_custom_home(self, tmp_jarvis_home: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_jarvis_home)
        assert config.jarvis_home == tmp_jarvis_home

    def test_ollama_defaults(self) -> None:
        config = JarvisConfig()
        assert config.ollama.base_url == "http://localhost:11434"
        assert config.ollama.timeout_seconds == 120

    def test_model_defaults(self) -> None:
        config = JarvisConfig()
        assert config.models.planner.name == "qwen3:32b"
        assert config.models.executor.name == "qwen3:8b"
        assert config.models.embedding.name == "nomic-embed-text"

    def test_planner_defaults(self) -> None:
        config = JarvisConfig()
        assert config.planner.max_iterations == 10
        assert config.planner.escalation_after == 3
        assert config.planner.temperature == 0.7

    def test_memory_defaults(self) -> None:
        config = JarvisConfig()
        assert config.memory.chunk_size_tokens == 400
        assert config.memory.weight_vector == 0.50
        assert config.memory.compaction_threshold == 0.80

    def test_gatekeeper_defaults(self) -> None:
        config = JarvisConfig()
        assert config.gatekeeper.max_blocked_retries == 3


class TestConfigPaths:
    """Alle abgeleiteten Pfade sind korrekt."""

    def test_paths_relative_to_home(self, config: JarvisConfig) -> None:
        home = config.jarvis_home
        assert config.config_file == home / "config.yaml"
        assert config.memory_dir == home / "memory"
        assert config.core_memory_file == home / "memory" / "CORE.md"
        assert config.episodes_dir == home / "memory" / "episodes"
        assert config.knowledge_dir == home / "memory" / "knowledge"
        assert config.procedures_dir == home / "memory" / "procedures"
        assert config.sessions_dir == home / "memory" / "sessions"
        assert config.index_dir == home / "index"
        assert config.db_path == home / "index" / "memory.db"
        assert config.workspace_dir == home / "workspace"
        assert config.logs_dir == home / "logs"


class TestLoadConfig:
    """Config-Laden aus YAML und Env-Vars."""

    def test_load_defaults_when_no_file(self, tmp_path: Path) -> None:
        # Nicht-existierender Pfad → reine Defaults
        config = load_config(tmp_path / "nonexistent" / "config.yaml")
        assert config.ollama.base_url == "http://localhost:11434"

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
ollama:
  base_url: http://gpu-server:11434
  timeout_seconds: 60

planner:
  max_iterations: 5
  temperature: 0.3

logging:
  level: DEBUG
"""
        )
        config = load_config(config_file)
        assert config.ollama.base_url == "http://gpu-server:11434"
        assert config.ollama.timeout_seconds == 60
        assert config.planner.max_iterations == 5
        assert config.planner.temperature == 0.3
        assert config.logging.level == "DEBUG"
        # Nicht überschriebene Werte bleiben Default
        assert config.models.planner.name == "qwen3:32b"

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        config = load_config(config_file)
        # Alles Default
        assert config.ollama.base_url == "http://localhost:11434"

    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JARVIS_OLLAMA_BASE_URL", "http://remote:11434")
        config = load_config(tmp_path / "nonexistent.yaml")
        # Env-Var sollte überschreiben
        # Hinweis: Der aktuelle Parser unterstützt einfache section_key Paare
        assert config.ollama.base_url == "http://remote:11434"


class TestDirectoryStructure:
    """Verzeichnisstruktur-Erstellung."""

    def test_creates_all_directories(self, config: JarvisConfig) -> None:
        created = ensure_directory_structure(config)
        assert len(created) > 0

        # Prüfe wichtige Verzeichnisse
        assert config.memory_dir.is_dir()
        assert config.episodes_dir.is_dir()
        assert config.knowledge_dir.is_dir()
        assert config.procedures_dir.is_dir()
        assert config.sessions_dir.is_dir()
        assert config.index_dir.is_dir()
        assert config.workspace_dir.is_dir()
        assert config.logs_dir.is_dir()
        assert config.policies_dir.is_dir()

    def test_creates_default_files(self, config: JarvisConfig) -> None:
        ensure_directory_structure(config)

        # CORE.md existiert
        assert config.core_memory_file.exists()
        core_content = config.core_memory_file.read_text()
        assert "Jarvis" in core_content
        assert "User" in core_content  # Default owner_name

        # Default Policy existiert
        policy_file = config.policies_dir / "default.yaml"
        assert policy_file.exists()
        policy_content = policy_file.read_text()
        assert "no_destructive_shell" in policy_content

        # Config-Datei existiert
        assert config.config_file.exists()

    def test_idempotent(self, config: JarvisConfig) -> None:
        """Doppeltes Aufrufen ist sicher und überschreibt nichts."""
        created_1 = ensure_directory_structure(config)
        assert len(created_1) > 0

        # Zweiter Aufruf: nichts neues
        created_2 = ensure_directory_structure(config)
        assert len(created_2) == 0

    def test_does_not_overwrite_existing_files(self, config: JarvisConfig) -> None:
        ensure_directory_structure(config)

        # User ändert CORE.md
        custom_content = "# Meine angepasste Identität"
        config.core_memory_file.write_text(custom_content)

        # Nochmal erstellen
        ensure_directory_structure(config)

        # Datei wurde NICHT überschrieben
        assert config.core_memory_file.read_text() == custom_content

    def test_knowledge_subdirectories(self, config: JarvisConfig) -> None:
        ensure_directory_structure(config)
        assert (config.knowledge_dir / "kunden").is_dir()
        assert (config.knowledge_dir / "produkte").is_dir()
        assert (config.knowledge_dir / "projekte").is_dir()


class TestConfigSerialization:
    """Config lässt sich serialisieren und wiederherstellen."""

    def test_json_round_trip(self, config: JarvisConfig) -> None:
        data = config.model_dump_json()
        restored = JarvisConfig.model_validate_json(data)
        assert str(restored.jarvis_home) == str(config.jarvis_home)
        assert restored.ollama.base_url == config.ollama.base_url
        assert restored.models.planner.name == config.models.planner.name


# ============================================================================
# SecurityConfig
# ============================================================================


class TestSecurityConfig:
    """Tests für SecurityConfig."""

    def test_defaults(self, config: JarvisConfig) -> None:
        assert config.security.max_iterations == 10
        assert len(config.security.allowed_paths) >= 2
        assert len(config.security.blocked_commands) >= 6
        assert len(config.security.credential_patterns) >= 3

    def test_custom_max_iterations(self, tmp_path: Path) -> None:
        cfg = JarvisConfig(jarvis_home=tmp_path, security={"max_iterations": 20})
        assert cfg.security.max_iterations == 20


# ============================================================================
# Neue JarvisConfig Properties
# ============================================================================


class TestJarvisConfigExtended:
    """Tests für erweiterte JarvisConfig Properties und Methoden."""

    def test_version(self, config: JarvisConfig) -> None:
        assert config.version == "0.22.0"

    def test_log_level(self, config: JarvisConfig) -> None:
        assert config.log_level == "INFO"

    def test_core_memory_path(self, config: JarvisConfig) -> None:
        assert config.core_memory_path == config.core_memory_file

    def test_ensure_directories_method(self, config: JarvisConfig) -> None:
        config.ensure_directories()
        assert config.workspace_dir.is_dir()
        assert config.logs_dir.is_dir()
        assert config.policies_dir.is_dir()

    def test_ensure_default_files_method(self, config: JarvisConfig) -> None:
        config.ensure_default_files()
        assert config.core_memory_file.exists()
        assert (config.policies_dir / "default.yaml").exists()


# ============================================================================
# Auto-Adaptation: Modelle passen sich an LLM-Backend an
# ============================================================================


class TestModelAutoAdaptation:
    """Modellnamen passen sich automatisch an das LLM-Backend an."""

    def test_ollama_defaults_unchanged(self) -> None:
        """Ohne API-Key bleiben Ollama-Defaults bestehen."""
        config = JarvisConfig()
        assert config.llm_backend_type == "ollama"
        assert config.models.planner.name == "qwen3:32b"
        assert config.models.executor.name == "qwen3:8b"

    def test_openai_backend_adapts_models(self, tmp_path: Path) -> None:
        """Bei llm_backend_type='openai' werden Modelle automatisch angepasst."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            llm_backend_type="openai",
            openai_api_key="sk-test-key",
        )
        assert config.models.planner.name == "gpt-5.2"
        assert config.models.executor.name == "gpt-5-mini"
        assert config.models.coder.name == "gpt-5.1-codex"
        assert config.models.embedding.name == "text-embedding-3-large"
        assert config.models.planner.context_window == 400000

    def test_anthropic_backend_adapts_models(self, tmp_path: Path) -> None:
        """Bei llm_backend_type='anthropic' werden Modelle automatisch angepasst."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            llm_backend_type="anthropic",
            anthropic_api_key="sk-ant-test-key",
        )
        assert config.models.planner.name == "claude-opus-4-6"
        assert config.models.executor.name == "claude-haiku-4-5-20251001"
        assert config.models.coder.name == "claude-sonnet-4-6"
        # Anthropic hat kein Embedding → bleibt bei Ollama-Default
        assert config.models.embedding.name == "nomic-embed-text"
        assert config.models.planner.context_window == 200000

    def test_api_key_auto_detects_backend(self, tmp_path: Path) -> None:
        """Nur API-Key gesetzt (ohne expliziten Backend-Typ) → Backend wird erkannt."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            openai_api_key="sk-test-key",
        )
        assert config.llm_backend_type == "openai"
        assert config.models.planner.name == "gpt-5.2"

    def test_anthropic_key_auto_detects_backend(self, tmp_path: Path) -> None:
        """Anthropic API-Key → Backend wird automatisch auf 'anthropic' gesetzt."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            anthropic_api_key="sk-ant-test-key",
        )
        assert config.llm_backend_type == "anthropic"
        assert config.models.planner.name == "claude-opus-4-6"

    def test_explicit_model_names_preserved(self, tmp_path: Path) -> None:
        """Explizit gesetzte Modellnamen werden NICHT überschrieben."""
        from jarvis.models import ModelConfig

        config = JarvisConfig(
            jarvis_home=tmp_path,
            llm_backend_type="openai",
            openai_api_key="sk-test-key",
            models={
                "planner": ModelConfig(name="my-custom-model"),
            },
        )
        # Benutzerdefinierter Name bleibt erhalten
        assert config.models.planner.name == "my-custom-model"
        # Nicht überschriebene Rolle wird angepasst
        assert config.models.executor.name == "gpt-5-mini"

    def test_heartbeat_model_adapts(self, tmp_path: Path) -> None:
        """Heartbeat-Modell wird ebenfalls angepasst."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            llm_backend_type="openai",
            openai_api_key="sk-test-key",
        )
        assert config.heartbeat.model == "gpt-5-mini"

    def test_anthropic_prioritized_over_openai(self, tmp_path: Path) -> None:
        """Wenn beide API-Keys vorhanden: Anthropic hat Priorität."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            anthropic_api_key="sk-ant-test",
            openai_api_key="sk-test",
        )
        assert config.llm_backend_type == "anthropic"
        assert config.models.planner.name == "claude-opus-4-6"


class TestVisionModelAutoAdaptation:
    """Vision-Modell passt sich automatisch an das LLM-Backend an."""

    def test_vision_model_default_llava(self) -> None:
        """Default Vision-Model ist llava:13b (Ollama)."""
        config = JarvisConfig()
        assert config.vision_model == "llava:13b"

    def test_vision_model_auto_openai(self, tmp_path: Path) -> None:
        """OpenAI-Key → vision_model wird gpt-5.2."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            openai_api_key="sk-test-key",
        )
        assert config.vision_model == "gpt-5.2"

    def test_vision_model_auto_anthropic(self, tmp_path: Path) -> None:
        """Anthropic-Key → vision_model wird claude-sonnet-4-6."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            anthropic_api_key="sk-ant-test-key",
        )
        assert config.vision_model == "claude-sonnet-4-6"

    def test_vision_model_explicit_not_overridden(self, tmp_path: Path) -> None:
        """Explizit gesetztes Vision-Model wird NICHT überschrieben."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            openai_api_key="sk-test-key",
            vision_model="my-custom-vision-model",
        )
        assert config.vision_model == "my-custom-vision-model"


# ============================================================================
# Multi-Provider Auto-Adaptation
# ============================================================================


class TestMultiProviderAutoAdaptation:
    """Tests für die automatische Backend-Erkennung neuer Provider."""

    def test_gemini_key_auto_detects_backend(self, tmp_path: Path) -> None:
        """Gemini API-Key → Backend wird automatisch auf 'gemini' gesetzt."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            gemini_api_key="AIza-test-key",
        )
        assert config.llm_backend_type == "gemini"
        assert config.models.planner.name == "gemini-2.5-pro"

    def test_groq_key_auto_detects_backend(self, tmp_path: Path) -> None:
        """Groq API-Key → Backend wird automatisch auf 'groq' gesetzt."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            groq_api_key="gsk_test-key",
        )
        assert config.llm_backend_type == "groq"
        assert config.models.planner.name == "meta-llama/llama-4-maverick-17b-128e-instruct"

    def test_deepseek_key_auto_detects_backend(self, tmp_path: Path) -> None:
        """DeepSeek API-Key → Backend wird automatisch auf 'deepseek' gesetzt."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            deepseek_api_key="sk-deepseek-test",
        )
        assert config.llm_backend_type == "deepseek"
        assert config.models.planner.name == "deepseek-chat"

    def test_mistral_key_auto_detects_backend(self, tmp_path: Path) -> None:
        """Mistral API-Key → Backend wird automatisch auf 'mistral' gesetzt."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            mistral_api_key="mistral-test-key",
        )
        assert config.llm_backend_type == "mistral"
        assert config.models.planner.name == "mistral-large-latest"

    def test_together_key_auto_detects_backend(self, tmp_path: Path) -> None:
        """Together API-Key → Backend wird automatisch auf 'together' gesetzt."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            together_api_key="together-test-key",
        )
        assert config.llm_backend_type == "together"
        assert config.models.planner.name == "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

    def test_gemini_model_defaults(self, tmp_path: Path) -> None:
        """Gemini-Backend setzt korrekte Modellnamen."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            llm_backend_type="gemini",
            gemini_api_key="AIza-test",
        )
        assert config.models.planner.name == "gemini-2.5-pro"
        assert config.models.executor.name == "gemini-2.5-flash"
        assert config.models.coder.name == "gemini-2.5-pro"
        assert config.models.embedding.name == "gemini-embedding-001"
        assert config.models.planner.context_window == 1000000
        assert config.vision_model == "gemini-2.5-pro"

    def test_groq_model_defaults(self, tmp_path: Path) -> None:
        """Groq-Backend setzt korrekte Modellnamen."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            llm_backend_type="groq",
            groq_api_key="gsk_test",
        )
        assert config.models.planner.name == "meta-llama/llama-4-maverick-17b-128e-instruct"
        assert config.models.executor.name == "llama-3.1-8b-instant"
        assert config.models.coder.name == "llama-3.3-70b-versatile"
        # Embedding bleibt bei Ollama-Fallback
        assert config.models.embedding.name == "nomic-embed-text"

    def test_deepseek_model_defaults(self, tmp_path: Path) -> None:
        """DeepSeek-Backend setzt korrekte Modellnamen."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            llm_backend_type="deepseek",
            deepseek_api_key="sk-ds-test",
        )
        assert config.models.planner.name == "deepseek-chat"
        assert config.models.executor.name == "deepseek-chat"
        assert config.models.coder.name == "deepseek-chat"

    def test_mistral_model_defaults(self, tmp_path: Path) -> None:
        """Mistral-Backend setzt korrekte Modellnamen."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            llm_backend_type="mistral",
            mistral_api_key="mistral-test",
        )
        assert config.models.planner.name == "mistral-large-latest"
        assert config.models.executor.name == "mistral-small-latest"
        assert config.models.coder.name == "codestral-latest"
        assert config.models.embedding.name == "mistral-embed"
        assert config.vision_model == "pixtral-large-latest"

    def test_together_model_defaults(self, tmp_path: Path) -> None:
        """Together-Backend setzt korrekte Modellnamen."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            llm_backend_type="together",
            together_api_key="together-test",
        )
        assert config.models.planner.name == "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        assert config.models.executor.name == "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        assert config.models.coder.name == "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

    def test_openrouter_key_auto_detects_backend(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, openrouter_api_key="sk-or-test")
        assert config.llm_backend_type == "openrouter"
        assert config.models.planner.name == "anthropic/claude-opus-4.6"

    def test_xai_key_auto_detects_backend(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, xai_api_key="xai-test")
        assert config.llm_backend_type == "xai"
        assert config.models.planner.name == "grok-4-1-fast-reasoning"

    def test_cerebras_key_auto_detects_backend(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, cerebras_api_key="csk-test")
        assert config.llm_backend_type == "cerebras"
        assert config.models.planner.name == "gpt-oss-120b"

    def test_github_key_auto_detects_backend(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, github_api_key="ghp_test")
        assert config.llm_backend_type == "github"
        assert config.models.planner.name == "gpt-4.1"

    def test_bedrock_key_auto_detects_backend(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, bedrock_api_key="bedrock-test")
        assert config.llm_backend_type == "bedrock"
        assert config.models.planner.name == "us.anthropic.claude-opus-4-6-v1:0"

    def test_huggingface_key_auto_detects_backend(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, huggingface_api_key="hf_test")
        assert config.llm_backend_type == "huggingface"
        assert config.models.planner.name == "meta-llama/Llama-3.3-70B-Instruct"

    def test_moonshot_key_auto_detects_backend(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, moonshot_api_key="sk-moon-test")
        assert config.llm_backend_type == "moonshot"
        assert config.models.planner.name == "kimi-k2.5"

    def test_openrouter_model_defaults(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, llm_backend_type="openrouter", openrouter_api_key="sk-or-test")
        assert config.models.planner.name == "anthropic/claude-opus-4.6"
        assert config.models.executor.name == "google/gemini-2.5-flash"
        assert config.vision_model == "anthropic/claude-sonnet-4.6"

    def test_xai_model_defaults(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, llm_backend_type="xai", xai_api_key="xai-test")
        assert config.models.planner.name == "grok-4-1-fast-reasoning"
        assert config.models.executor.name == "grok-4-1-fast-non-reasoning"
        assert config.vision_model == "grok-4-1-fast-reasoning"

    def test_cerebras_model_defaults(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, llm_backend_type="cerebras", cerebras_api_key="csk-test")
        assert config.models.planner.name == "gpt-oss-120b"
        assert config.models.executor.name == "llama3.1-8b"

    def test_github_model_defaults(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, llm_backend_type="github", github_api_key="ghp_test")
        assert config.models.planner.name == "gpt-4.1"
        assert config.models.embedding.name == "text-embedding-3-large"

    def test_bedrock_model_defaults(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, llm_backend_type="bedrock", bedrock_api_key="test")
        assert config.models.planner.name == "us.anthropic.claude-opus-4-6-v1:0"
        assert config.models.embedding.name == "amazon.titan-embed-text-v2:0"

    def test_moonshot_model_defaults(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, llm_backend_type="moonshot", moonshot_api_key="test")
        assert config.models.planner.name == "kimi-k2.5"
        assert config.models.executor.name == "kimi-k2-turbo-preview"
