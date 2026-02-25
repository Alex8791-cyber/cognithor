"""Jarvis · Installations-Assistent & Ersteinrichtung.

Grafischer Einrichtungs-Assistent für neue Installationen:

  - HardwareDetector:      Erkennt CPU, GPU, RAM, Speicherplatz
  - ModelRecommender:       Empfiehlt LLM-Modelle basierend auf Hardware
  - PresetConfig:           Vorkonfigurierte Setups (Minimal, Standard, Power, Enterprise)
  - ChannelConfigurator:    Konfiguriert Kommunikationskanäle
  - SetupWizard:            Hauptklasse -- schrittweiser Einrichtungsprozess

Architektur-Bibel: §2.1 (Installation), §2.2 (First-Run-Experience)
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Hardware Detection
# ============================================================================


@dataclass
class GPUInfo:
    """Erkannte GPU-Informationen."""

    name: str = "None"
    vram_gb: float = 0
    cuda_available: bool = False
    driver_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vram_gb": self.vram_gb,
            "cuda": self.cuda_available,
        }


@dataclass
class HardwareProfile:
    """Erkanntes Hardware-Profil."""

    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_gb: float = 0
    disk_free_gb: float = 0
    gpu: GPUInfo = field(default_factory=GPUInfo)
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""

    @property
    def tier(self) -> str:
        """Hardware-Tier: minimal, standard, power, enterprise."""
        if self.ram_gb >= 64 and self.cpu_cores >= 16 and self.gpu.vram_gb >= 48:
            return "enterprise"
        if self.gpu.vram_gb >= 16 and self.ram_gb >= 32:
            return "power"
        if self.gpu.vram_gb >= 8 and self.ram_gb >= 16:
            return "standard"
        return "minimal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu": self.cpu_name,
            "cores": self.cpu_cores,
            "threads": self.cpu_threads,
            "ram_gb": self.ram_gb,
            "disk_free_gb": self.disk_free_gb,
            "gpu": self.gpu.to_dict(),
            "os": f"{self.os_name} {self.os_version}",
            "tier": self.tier,
        }


class HardwareDetector:
    """Erkennt System-Hardware für optimale Konfiguration."""

    def detect(self) -> HardwareProfile:
        """Erkennt aktuelle Hardware."""
        profile = HardwareProfile(
            cpu_name=platform.processor() or platform.machine(),
            cpu_cores=os.cpu_count() or 1,
            cpu_threads=(os.cpu_count() or 1),
            os_name=platform.system(),
            os_version=platform.release(),
            python_version=platform.python_version(),
        )

        # Disk erkennen
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            profile.disk_free_gb = round(free / (1024**3), 1)
        except Exception:
            logger.debug("disk_detection_skipped", exc_info=True)

        # RAM erkennen
        try:
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                mem = ctypes.c_ulonglong(0)
                kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem))
                profile.ram_gb = round(mem.value / (1024 * 1024), 1)
            elif sys.platform == "darwin":
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    profile.ram_gb = round(int(result.stdout.strip()) / (1024**3), 1)
            else:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            profile.ram_gb = round(kb / (1024**2), 1)
                            break
        except Exception:
            logger.debug("ram_detection_skipped, using fallback", exc_info=True)
            profile.ram_gb = 8.0  # Fallback

        # GPU erkennen (simuliert für Portabilität)
        profile.gpu = self._detect_gpu()

        return profile

    def _detect_gpu(self) -> GPUInfo:
        """Versucht GPU zu erkennen."""
        try:
            # Versuche nvidia-smi
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                return GPUInfo(
                    name=parts[0].strip(),
                    vram_gb=round(float(parts[1].strip()) / 1024, 1),
                    cuda_available=True,
                    driver_version=parts[2].strip() if len(parts) > 2 else "",
                )
        except Exception:
            logger.debug("gpu_detection_skipped", exc_info=True)

        return GPUInfo()

    @staticmethod
    def from_specs(
        cpu: str = "", cores: int = 4, ram_gb: float = 16,
        gpu_name: str = "", vram_gb: float = 0, disk_gb: float = 100,
    ) -> HardwareProfile:
        """Erstellt Profil aus manuellen Angaben."""
        return HardwareProfile(
            cpu_name=cpu,
            cpu_cores=cores,
            cpu_threads=cores * 2,
            ram_gb=ram_gb,
            disk_free_gb=disk_gb,
            gpu=GPUInfo(name=gpu_name, vram_gb=vram_gb, cuda_available=vram_gb > 0),
        )


# ============================================================================
# Model Recommender
# ============================================================================


@dataclass
class ModelRecommendation:
    """Eine LLM-Modellempfehlung."""

    model_name: str
    model_size: str           # "7B", "13B", "70B"
    quantization: str         # "Q4_K_M", "Q8_0", "FP16"
    vram_required_gb: float
    ram_required_gb: float
    quality_score: int        # 1-10
    speed_score: int          # 1-10
    use_case: str
    provider: str = "ollama"  # ollama, llama.cpp, vllm

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "size": self.model_size,
            "quant": self.quantization,
            "vram_gb": self.vram_required_gb,
            "quality": self.quality_score,
            "speed": self.speed_score,
            "use_case": self.use_case,
        }


class ModelRecommender:
    """Empfiehlt LLM-Modelle basierend auf Hardware."""

    MODELS = [
        # Kleine Modelle (< 4GB VRAM)
        ModelRecommendation("gemma2:2b", "2B", "Q4_K_M", 2.0, 4.0, 5, 9, "Chat, einfache Aufgaben"),
        ModelRecommendation("phi3:mini", "3.8B", "Q4_K_M", 3.0, 6.0, 6, 8, "Reasoning, Code"),
        ModelRecommendation("llama3.2:3b", "3B", "Q4_K_M", 2.5, 5.0, 6, 9, "Allgemein, schnell"),

        # Mittlere Modelle (4-8GB VRAM)
        ModelRecommendation("llama3.1:8b", "8B", "Q4_K_M", 5.0, 8.0, 7, 7, "Allgemein, gut"),
        ModelRecommendation("mistral:7b", "7B", "Q4_K_M", 4.5, 8.0, 7, 7, "Chat, Reasoning"),
        ModelRecommendation("deepseek-coder:6.7b", "6.7B", "Q4_K_M", 4.0, 8.0, 8, 7, "Code-Spezialist"),
        ModelRecommendation("command-r:7b", "7B", "Q4_K_M", 5.0, 8.0, 7, 7, "Tool-Use, RAG"),

        # Große Modelle (8-16GB VRAM)
        ModelRecommendation("llama3.1:8b", "8B", "Q8_0", 8.0, 12.0, 8, 6, "Höchste 8B-Qualität"),
        ModelRecommendation("mixtral:8x7b", "47B", "Q4_K_M", 14.0, 24.0, 8, 5, "MoE, vielseitig"),
        ModelRecommendation("qwen2.5:14b", "14B", "Q4_K_M", 9.0, 14.0, 8, 6, "Mehrsprachig, Deutsch"),

        # Power-Modelle (16-24GB+ VRAM)
        ModelRecommendation("llama3.1:70b", "70B", "Q4_K_M", 20.0, 48.0, 9, 4, "Frontier-Qualität"),
        ModelRecommendation("qwen2.5:32b", "32B", "Q4_K_M", 18.0, 32.0, 9, 5, "Deutsch exzellent"),
        ModelRecommendation("deepseek-r1:32b", "32B", "Q4_K_M", 18.0, 32.0, 9, 4, "Reasoning-Champion"),

        # CPU-only
        ModelRecommendation("gemma2:2b", "2B", "Q4_K_M", 0, 4.0, 5, 6, "CPU-only, kompakt", "llama.cpp"),
        ModelRecommendation("phi3:mini", "3.8B", "Q4_K_M", 0, 6.0, 6, 5, "CPU-only, Reasoning", "llama.cpp"),
    ]

    def recommend(self, hardware: HardwareProfile, top_n: int = 3) -> list[ModelRecommendation]:
        """Empfiehlt Modelle basierend auf verfügbarer Hardware."""
        vram = hardware.gpu.vram_gb
        ram = hardware.ram_gb

        candidates = []
        for model in self.MODELS:
            if vram > 0 and model.vram_required_gb <= vram:
                candidates.append(model)
            elif vram == 0 and model.vram_required_gb == 0 and model.ram_required_gb <= ram:
                candidates.append(model)

        # Sortiere nach Qualität (absteigend), dann Speed
        candidates.sort(key=lambda m: (m.quality_score, m.speed_score), reverse=True)

        # Dedupliziere nach Modellname
        seen = set()
        unique = []
        for c in candidates:
            if c.model_name not in seen:
                seen.add(c.model_name)
                unique.append(c)

        return unique[:top_n]

    def recommend_for_use_case(self, hardware: HardwareProfile, use_case: str) -> ModelRecommendation | None:
        """Empfiehlt das beste Modell für einen bestimmten Anwendungsfall."""
        recs = self.recommend(hardware, top_n=20)
        uc_lower = use_case.lower()
        for r in recs:
            if uc_lower in r.use_case.lower():
                return r
        return recs[0] if recs else None


# ============================================================================
# Preset Configs
# ============================================================================


class PresetLevel(Enum):
    MINIMAL = "minimal"         # Raspberry Pi, alte Laptops
    STANDARD = "standard"       # Gaming-PC mit GPU
    POWER = "power"            # Workstation (RTX 4090/5090)
    ENTERPRISE = "enterprise"   # Server, Multi-GPU


@dataclass
class PresetConfig:
    """Vorkonfiguriertes Setup."""

    preset_id: str
    level: PresetLevel
    name: str
    description: str
    model: str
    max_agents: int
    max_concurrent: int
    memory_limit_mb: int
    enable_rag: bool
    enable_federation: bool
    enable_cron: bool
    channels: list[str]
    estimated_latency_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.name,
            "level": self.level.value,
            "model": self.model,
            "agents": self.max_agents,
            "rag": self.enable_rag,
            "channels": self.channels,
            "latency_ms": self.estimated_latency_ms,
        }


PRESETS = {
    PresetLevel.MINIMAL: PresetConfig(
        "PRESET-MIN", PresetLevel.MINIMAL,
        "Minimal", "Für ältere Hardware (4-8 GB RAM, keine GPU)",
        "gemma2:2b", max_agents=1, max_concurrent=1,
        memory_limit_mb=2048, enable_rag=False, enable_federation=False,
        enable_cron=False, channels=["telegram"],
        estimated_latency_ms=3000,
    ),
    PresetLevel.STANDARD: PresetConfig(
        "PRESET-STD", PresetLevel.STANDARD,
        "Standard", "Für Gaming-PCs (16 GB RAM, 8 GB VRAM)",
        "llama3.1:8b", max_agents=3, max_concurrent=2,
        memory_limit_mb=8192, enable_rag=True, enable_federation=False,
        enable_cron=True, channels=["telegram", "slack", "web"],
        estimated_latency_ms=800,
    ),
    PresetLevel.POWER: PresetConfig(
        "PRESET-PWR", PresetLevel.POWER,
        "Power", "Für Workstations (32+ GB RAM, 16+ GB VRAM)",
        "qwen2.5:32b", max_agents=10, max_concurrent=4,
        memory_limit_mb=32768, enable_rag=True, enable_federation=True,
        enable_cron=True, channels=["telegram", "slack", "web", "matrix", "teams"],
        estimated_latency_ms=400,
    ),
    PresetLevel.ENTERPRISE: PresetConfig(
        "PRESET-ENT", PresetLevel.ENTERPRISE,
        "Enterprise", "Für Server (64+ GB RAM, Multi-GPU)",
        "llama3.1:70b", max_agents=50, max_concurrent=10,
        memory_limit_mb=65536, enable_rag=True, enable_federation=True,
        enable_cron=True, channels=["telegram", "slack", "web", "matrix", "teams", "email", "api"],
        estimated_latency_ms=200,
    ),
}


# ============================================================================
# Channel Configurator
# ============================================================================


class ChannelType(Enum):
    TELEGRAM = "telegram"
    SLACK = "slack"
    WEB = "web"
    MATRIX = "matrix"
    TEAMS = "teams"
    EMAIL = "email"
    API = "api"
    IMESSAGE = "imessage"
    WHATSAPP = "whatsapp"


@dataclass
class ChannelConfig:
    """Konfiguration eines Kommunikationskanals."""

    channel_type: ChannelType
    enabled: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "unconfigured"     # unconfigured, configured, connected, error

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.channel_type.value,
            "enabled": self.enabled,
            "status": self.status,
            "config_keys": list(self.config.keys()),
        }


class ChannelConfigurator:
    """Konfiguriert Kommunikationskanäle."""

    REQUIRED_CONFIG: dict[ChannelType, list[str]] = {
        ChannelType.TELEGRAM: ["bot_token"],
        ChannelType.SLACK: ["bot_token", "app_token"],
        ChannelType.WEB: ["port", "host"],
        ChannelType.MATRIX: ["homeserver", "username", "password"],
        ChannelType.TEAMS: ["tenant_id", "client_id", "client_secret"],
        ChannelType.EMAIL: ["imap_host", "smtp_host", "username", "password"],
        ChannelType.API: ["api_key"],
        ChannelType.IMESSAGE: ["applescript_enabled"],
        ChannelType.WHATSAPP: ["session_path"],
    }

    def __init__(self) -> None:
        self._channels: dict[ChannelType, ChannelConfig] = {}

    def configure(self, channel_type: ChannelType, config: dict[str, Any]) -> ChannelConfig:
        """Konfiguriert einen Kanal."""
        required = self.REQUIRED_CONFIG.get(channel_type, [])
        missing = [k for k in required if k not in config]

        ch = ChannelConfig(
            channel_type=channel_type,
            config=config,
            enabled=not missing,
            status="configured" if not missing else "unconfigured",
        )
        self._channels[channel_type] = ch
        return ch

    def enable(self, channel_type: ChannelType) -> bool:
        ch = self._channels.get(channel_type)
        if not ch:
            return False
        ch.enabled = True
        return True

    def configured_channels(self) -> list[ChannelConfig]:
        return [c for c in self._channels.values() if c.status == "configured"]

    @property
    def channel_count(self) -> int:
        return len(self._channels)

    def stats(self) -> dict[str, Any]:
        chs = list(self._channels.values())
        return {
            "total": len(chs),
            "configured": sum(1 for c in chs if c.status == "configured"),
            "enabled": sum(1 for c in chs if c.enabled),
        }


# ============================================================================
# Setup Wizard (Hauptklasse)
# ============================================================================


class SetupStep(Enum):
    WELCOME = "welcome"
    HARDWARE_DETECT = "hardware_detect"
    MODEL_SELECT = "model_select"
    PRESET_SELECT = "preset_select"
    CHANNEL_CONFIG = "channel_config"
    SECURITY_SETUP = "security_setup"
    FIRST_AGENT = "first_agent"
    COMPLETE = "complete"


@dataclass
class SetupState:
    """Zustand des Einrichtungsprozesses."""

    current_step: SetupStep = SetupStep.WELCOME
    hardware: HardwareProfile | None = None
    selected_model: str = ""
    selected_preset: PresetLevel | None = None
    channels: list[ChannelType] = field(default_factory=list)
    admin_password_set: bool = False
    first_agent_name: str = ""
    started_at: str = ""
    completed_at: str = ""

    @property
    def is_complete(self) -> bool:
        return self.current_step == SetupStep.COMPLETE

    @property
    def progress_percent(self) -> int:
        steps = list(SetupStep)
        idx = steps.index(self.current_step)
        return round(idx / (len(steps) - 1) * 100)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.current_step.value,
            "progress": self.progress_percent,
            "hardware_tier": self.hardware.tier if self.hardware else None,
            "model": self.selected_model,
            "preset": self.selected_preset.value if self.selected_preset else None,
            "channels": [c.value for c in self.channels],
            "complete": self.is_complete,
        }


class SetupWizard:
    """Schrittweiser Einrichtungs-Assistent."""

    def __init__(self) -> None:
        self._detector = HardwareDetector()
        self._recommender = ModelRecommender()
        self._channels = ChannelConfigurator()
        self._state = SetupState(started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @property
    def state(self) -> SetupState:
        return self._state

    @property
    def detector(self) -> HardwareDetector:
        return self._detector

    @property
    def recommender(self) -> ModelRecommender:
        return self._recommender

    @property
    def channels(self) -> ChannelConfigurator:
        return self._channels

    def step_hardware(self, manual: HardwareProfile | None = None) -> HardwareProfile:
        """Schritt 1: Hardware erkennen."""
        self._state.hardware = manual or self._detector.detect()
        self._state.current_step = SetupStep.HARDWARE_DETECT
        return self._state.hardware

    def step_model(self, override: str = "") -> list[ModelRecommendation]:
        """Schritt 2: Modell empfehlen/auswählen."""
        if not self._state.hardware:
            self.step_hardware()
        recs = self._recommender.recommend(self._state.hardware)
        self._state.selected_model = override or (recs[0].model_name if recs else "gemma2:2b")
        self._state.current_step = SetupStep.MODEL_SELECT
        return recs

    def step_preset(self, level: PresetLevel | None = None) -> PresetConfig:
        """Schritt 3: Preset auswählen."""
        if level is None and self._state.hardware:
            level = PresetLevel(self._state.hardware.tier)
        level = level or PresetLevel.STANDARD
        self._state.selected_preset = level
        self._state.current_step = SetupStep.PRESET_SELECT
        return PRESETS[level]

    def step_channels(self, channel_types: list[ChannelType] | None = None) -> list[ChannelConfig]:
        """Schritt 4: Kanäle konfigurieren."""
        if channel_types is None and self._state.selected_preset:
            preset = PRESETS[self._state.selected_preset]
            channel_types = [ChannelType(c) for c in preset.channels if c in [ct.value for ct in ChannelType]]
        channel_types = channel_types or [ChannelType.WEB]

        configs = []
        for ct in channel_types:
            cfg = self._channels.configure(ct, {"auto_configured": True})
            configs.append(cfg)
        self._state.channels = channel_types
        self._state.current_step = SetupStep.CHANNEL_CONFIG
        return configs

    def step_security(self, admin_password: str = "auto") -> bool:
        """Schritt 5: Sicherheit einrichten."""
        self._state.admin_password_set = bool(admin_password)
        self._state.current_step = SetupStep.SECURITY_SETUP
        return True

    def step_first_agent(self, name: str = "Jarvis") -> str:
        """Schritt 6: Ersten Agenten erstellen."""
        self._state.first_agent_name = name
        self._state.current_step = SetupStep.FIRST_AGENT
        return name

    def complete(self) -> SetupState:
        """Abschluss des Wizards."""
        self._state.current_step = SetupStep.COMPLETE
        self._state.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return self._state

    def auto_setup(self) -> SetupState:
        """One-Click-Setup: Alles automatisch."""
        hw = self.step_hardware()
        self.step_model()
        self.step_preset()
        self.step_channels()
        self.step_security()
        self.step_first_agent()
        return self.complete()

    def generate_config(self) -> dict[str, Any]:
        """Generiert config.yaml aus Wizard-State."""
        preset = PRESETS.get(self._state.selected_preset, PRESETS[PresetLevel.STANDARD])
        return {
            "jarvis": {
                "model": self._state.selected_model,
                "max_agents": preset.max_agents,
                "max_concurrent": preset.max_concurrent,
                "memory_limit_mb": preset.memory_limit_mb,
            },
            "features": {
                "rag": preset.enable_rag,
                "federation": preset.enable_federation,
                "cron": preset.enable_cron,
            },
            "channels": {c.value: {"enabled": True} for c in self._state.channels},
        }

    def stats(self) -> dict[str, Any]:
        return {
            "state": self._state.to_dict(),
            "channels": self._channels.stats(),
        }
