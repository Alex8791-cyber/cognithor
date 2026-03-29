#!/usr/bin/env python3
"""
Cognithor Windows Bootstrap — One-Click Setup & Quick-Start.

Erster Start:  Hardware-Erkennung, Deps-Installation, Modell-Download, Shortcut.
Folgestart:    Ollama-Check, Modell-Check, Port-Check, Import-Test (<5s).
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


# ── Version (dynamisch aus pyproject.toml lesen) ──────────────────────────
def _read_version() -> str:
    """Liest die Version aus pyproject.toml oder src/jarvis/__init__.py."""
    # 1. Versuche pyproject.toml im Repo-Root
    for candidate in [
        Path(__file__).resolve().parent.parent / "pyproject.toml",
        Path.cwd() / "pyproject.toml",
    ]:
        if candidate.exists():
            try:
                text = candidate.read_text(encoding="utf-8")
                for line in text.splitlines():
                    if line.strip().startswith("version"):
                        # version = "0.35.5"
                        ver = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if ver:
                            return ver
            except Exception:
                pass
    # 2. Fallback: jarvis.__init__.__version__
    init_path = Path(__file__).resolve().parent.parent / "src" / "jarvis" / "__init__.py"
    if init_path.exists():
        try:
            for line in init_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("__version__"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return "0.0.0"


BOOTSTRAP_VERSION = _read_version()

# ── Pfade ──────────────────────────────────────────────────────────────────
JARVIS_HOME = Path(os.environ.get("JARVIS_HOME", Path.home() / ".jarvis"))
MARKER_FILE = JARVIS_HOME / ".cognithor_initialized"
OLLAMA_URL = "http://localhost:11434"
BACKEND_PORT = 8741

# ── ANSI-Farben (werden bei fehlender Unterstuetzung deaktiviert) ──────────
_COLORS_ENABLED = False
GREEN = ""
RED = ""
YELLOW = ""
BLUE = ""
BOLD = ""
DIM = ""
RESET = ""


def _enable_ansi() -> None:
    """Aktiviert ANSI-Escape-Codes in Windows 10+ Konsolen."""
    global _COLORS_ENABLED, GREEN, RED, YELLOW, BLUE, BOLD, DIM, RESET
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        if kernel32.SetConsoleMode(handle, mode.value | 0x0004):
            _COLORS_ENABLED = True
            GREEN = "\033[92m"
            RED = "\033[91m"
            YELLOW = "\033[93m"
            BLUE = "\033[94m"
            BOLD = "\033[1m"
            DIM = "\033[2m"
            RESET = "\033[0m"
    except Exception:
        pass  # Farben bleiben leer = sicherer Fallback


def _setup_encoding() -> None:
    """Stellt sicher dass stdout UTF-8 kann."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


# ── Output-Helpers (ASCII-safe Symbole) ───────────────────────────────────
def ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET}       {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}[ERROR]{RESET}    {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[WARNING]{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"  {BLUE}[INFO]{RESET}     {msg}")


def header(title: str) -> None:
    line = "-" * 60
    print(f"\n{BOLD}{line}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{line}{RESET}")


# ── BootResult ─────────────────────────────────────────────────────────────
@dataclass
class BootResult:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return len(self.failed) == 0

    def add_pass(self, name: str) -> None:
        self.passed.append(name)
        ok(name)

    def add_fail(self, name: str, detail: str = "") -> None:
        msg = f"{name}: {detail}" if detail else name
        self.failed.append(msg)
        fail(msg)

    def add_warn(self, name: str) -> None:
        self.warnings.append(name)
        warn(name)


# ── GPUInfo ────────────────────────────────────────────────────────────────
@dataclass
class GPUInfo:
    name: str = ""
    vram_gb: float = 0.0
    driver_version: str = ""
    cuda_compute: str = ""
    cuda_available: bool = False


@dataclass
class HardwareProfile:
    gpu: GPUInfo = field(default_factory=GPUInfo)
    ram_gb: float = 8.0
    disk_free_gb: float = 0.0
    cpu_cores: int = os.cpu_count() or 4

    @property
    def tier(self) -> str:
        if self.ram_gb >= 64 and self.cpu_cores >= 16 and self.gpu.vram_gb >= 48:
            return "enterprise"
        if self.gpu.vram_gb >= 16 and self.ram_gb >= 32:
            return "power"
        if self.gpu.vram_gb >= 8 and self.ram_gb >= 16:
            return "standard"
        return "minimal"


# ── Hardware-Erkennung ─────────────────────────────────────────────────────
def detect_gpu() -> GPUInfo:
    """GPU via nvidia-smi erkennen (Multi-GPU-safe: nimmt erste GPU)."""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            # Erste Zeile = erste GPU (bei Multi-GPU-Systemen)
            first_line = r.stdout.strip().splitlines()[0]
            parts = first_line.split(",")
            if len(parts) < 2:
                return GPUInfo()
            gpu = GPUInfo(
                name=parts[0].strip(),
                vram_gb=round(float(parts[1].strip()) / 1024, 1),
                driver_version=parts[2].strip() if len(parts) > 2 else "",
                cuda_available=True,
            )
            # CUDA Compute Capability
            try:
                cc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if cc.returncode == 0 and cc.stdout.strip():
                    gpu.cuda_compute = cc.stdout.strip().splitlines()[0].strip()
            except Exception:
                pass
            return gpu
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return GPUInfo()


def detect_ram() -> float:
    """RAM in GB (Windows-spezifisch)."""
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        mem_kb = ctypes.c_ulonglong(0)
        kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem_kb))
        return round(mem_kb.value / (1024 * 1024), 1)
    except Exception:
        return 8.0


def detect_disk(path: str) -> float:
    """Freier Speicher in GB."""
    try:
        usage = shutil.disk_usage(path)
        return round(usage.free / (1024**3), 1)
    except Exception:
        return 0.0


def detect_hardware(repo_root: str) -> HardwareProfile:
    return HardwareProfile(
        gpu=detect_gpu(),
        ram_gb=detect_ram(),
        disk_free_gb=detect_disk(repo_root),
    )


# ── Ollama ─────────────────────────────────────────────────────────────────
def find_ollama() -> str | None:
    """Findet Ollama Binary."""
    path = shutil.which("ollama")
    if path:
        return path
    # Fallback: Standard-Installationspfade
    local_app = os.environ.get("LOCALAPPDATA", "")
    if local_app:
        candidate = os.path.join(local_app, "Programs", "Ollama", "ollama.exe")
        if os.path.isfile(candidate):
            return candidate
    # Zweiter Fallback: Program Files
    prog_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
    candidate2 = os.path.join(prog_files, "Ollama", "ollama.exe")
    if os.path.isfile(candidate2):
        return candidate2
    return None


def ollama_is_running() -> bool:
    """Prueft ob Ollama-Server laeuft."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def start_ollama(ollama_path: str) -> bool:
    """Startet Ollama im Hintergrund."""
    try:
        create_no_window = 0x08000000
        subprocess.Popen(
            [ollama_path, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=create_no_window,
        )
        # Warte bis Server bereit ist (max 15s)
        for _ in range(30):
            time.sleep(0.5)
            if ollama_is_running():
                return True
        return False
    except Exception:
        return False


def get_installed_models() -> list[str]:
    """Liste installierter Ollama-Modelle."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def pull_model(name: str, ollama_path: str, timeout: int = 1800) -> bool:
    """Zieht ein Ollama-Modell (mit Fortschrittsanzeige)."""
    info(f"Downloading model: {name} (may take a few minutes)...")
    try:
        proc = subprocess.Popen(
            [ollama_path, "pull", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        start = time.time()
        last_line = ""
        while proc.poll() is None:
            if time.time() - start > timeout:
                proc.kill()
                warn(f"Timeout downloading {name}")
                return False
            line = proc.stdout.readline().strip() if proc.stdout else ""
            if line and line != last_line:
                print(f"\r  {DIM}{line[:70]:<70}{RESET}", end="", flush=True)
                last_line = line
        print()  # Newline nach Fortschritt
        return proc.returncode == 0
    except Exception as e:
        warn(f"Error downloading {name}: {e}")
        return False


TIER_MODELS: dict[str, list[str]] = {
    "minimal": ["qwen3:8b", "qwen3-embedding:0.6b"],
    "standard": ["qwen3:8b", "qwen3:32b", "qwen3-embedding:0.6b"],
    "power": ["qwen3:8b", "qwen3:32b", "qwen3-coder:30b", "qwen3-embedding:0.6b"],
    "enterprise": ["qwen3:8b", "qwen3:32b", "qwen3-coder:30b", "qwen3-embedding:0.6b"],
}


def _print_model_pull_commands(tier: str) -> None:
    """Zeigt die manuellen ollama pull Kommandos fuer den Tier an."""
    needed = TIER_MODELS.get(tier, TIER_MODELS["minimal"])
    print()
    info("Download models manually:")
    print()
    for model in needed:
        print(f"    ollama pull {model}")
    print()
    info("Then restart Cognithor.")
    print()


def ensure_models(tier: str, result: BootResult, ollama_path: str) -> list[str]:
    """Stellt sicher dass die Tier-Modelle vorhanden sind."""
    needed = TIER_MODELS.get(tier, TIER_MODELS["minimal"])
    installed = get_installed_models()
    installed_lower = [m.lower() for m in installed]

    pulled: list[str] = []
    for model in needed:
        # Pruefe ob bereits installiert (mit/ohne :latest Tag)
        model_base = model.lower().split(":")[0]
        if any(model.lower() == m or m.startswith(model_base + ":") for m in installed_lower):
            ok(f"Model available: {model}")
            pulled.append(model)
            continue

        if pull_model(model, ollama_path):
            ok(f"Model installed: {model}")
            pulled.append(model)
        else:
            result.add_warn(f"Model {model} could not be downloaded. Manually: ollama pull {model}")

    return pulled


# ── Port-Check ─────────────────────────────────────────────────────────────
def check_port(port: int) -> str:
    """Prueft ob ein Port frei ist. Gibt 'free', 'in_use' oder 'cognithor' zurueck."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        sock.connect(("127.0.0.1", port))
        # Port in Benutzung — pruefen ob Cognithor
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return "cognithor"
        except Exception:
            pass
        return "in_use"
    except (TimeoutError, ConnectionRefusedError, OSError):
        return "free"
    finally:
        with contextlib.suppress(Exception):
            sock.close()


# ── Desktop-Shortcut ──────────────────────────────────────────────────────
def create_desktop_shortcut(bat_path: str) -> bool:
    """Erstellt eine Desktop-Verknuepfung via PowerShell/COM."""
    # Desktop-Pfad finden (OneDrive-kompatibel)
    desktop = None
    for candidate in [
        Path.home() / "Desktop",
        Path.home() / "OneDrive" / "Desktop",
        Path.home() / "OneDrive" / "Schreibtisch",
        Path.home() / "Schreibtisch",
    ]:
        if candidate.is_dir():
            desktop = candidate
            break

    if desktop is None:
        return False

    shortcut_path = desktop / "Cognithor.lnk"
    if shortcut_path.exists():
        return True  # Bereits vorhanden

    # Arbeitsverzeichnis = Ordner der bat-Datei
    working_dir = str(Path(bat_path).parent)

    # Pfade escapen fuer PowerShell (Single-Quotes verdoppeln)
    sc_path_escaped = str(shortcut_path).replace("'", "''")
    bat_path_escaped = bat_path.replace("'", "''")
    wd_escaped = working_dir.replace("'", "''")

    ps_script = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$sc = $ws.CreateShortcut('{sc_path_escaped}'); "
        f"$sc.TargetPath = '{bat_path_escaped}'; "
        f"$sc.WorkingDirectory = '{wd_escaped}'; "
        "$sc.Description = 'Cognithor Control Center starten'; "
        "$sc.WindowStyle = 1; "
        "$sc.Save()"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


# ── Marker-File ────────────────────────────────────────────────────────────
def read_marker() -> dict | None:
    """Liest das Marker-File oder gibt None zurueck."""
    try:
        return json.loads(MARKER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_marker(hw: HardwareProfile, models: list[str], shortcut: bool) -> None:
    """Schreibt das Marker-File."""
    MARKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": BOOTSTRAP_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware_tier": hw.tier,
        "gpu_name": hw.gpu.name,
        "vram_gb": hw.gpu.vram_gb,
        "ram_gb": hw.ram_gb,
        "models_installed": models,
        "shortcut_created": shortcut,
    }
    MARKER_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Piper TTS Voice Download ──────────────────────────────────────────────
def _download_piper_voice(voice: str, dest: Path) -> None:
    """Laedt ein Piper-Voicemodell von HuggingFace herunter."""
    parts = voice.split("-")  # de_DE-pavoque-low
    lang = parts[0]  # de_DE
    name = parts[1]  # pavoque
    quality = parts[2] if len(parts) > 2 else "low"
    lang_short = lang.split("_")[0]  # de

    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{lang_short}/{lang}/{name}/{quality}"
    onnx_url = f"{base}/{voice}.onnx?download=true"
    json_url = f"{base}/{voice}.onnx.json?download=true"

    onnx_path = dest / f"{voice}.onnx"
    json_path = dest / f"{voice}.onnx.json"

    info(f"  Download: {onnx_url}")
    urllib.request.urlretrieve(onnx_url, str(onnx_path))
    urllib.request.urlretrieve(json_url, str(json_path))
    ok(f"  Model saved: {onnx_path}")


# ── Installer-Erkennung (uv / pip) ────────────────────────────────────────
def _detect_python_installer(repo_root: str) -> tuple[str, list[str]]:
    """Erkennt uv oder pip und gibt (backend_name, install_command) zurueck.

    uv wird bevorzugt wenn vorhanden (10x schneller als pip).
    """
    uv_path = shutil.which("uv")
    if uv_path:
        try:
            ver = subprocess.run(
                [uv_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if ver.returncode == 0:
                ok(f"uv detected ({ver.stdout.strip()}) -- preferred")
                return "uv", [
                    uv_path,
                    "pip",
                    "install",
                    "-e",
                    ".[all]",
                    "--quiet",
                    "--python",
                    sys.executable,
                ]
        except Exception:
            pass

    # Fallback: pip
    return "pip", [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        ".[all]",
        "--quiet",
        "--disable-pip-version-check",
    ]


# ── Erster Start (14 Schritte) ─────────────────────────────────────────────
def first_start(repo_root: str, *, skip_models: bool = False) -> bool:
    result = BootResult()
    t0 = time.time()

    # ── 1. Python-Version ──────────────────────────────────────────────
    header("1/14  Python Version")
    v = sys.version_info
    if v >= (3, 12):
        result.add_pass(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        result.add_fail("Python-Version", f"{v.major}.{v.minor} < 3.12")
        return False

    # ── 2. Hardware-Erkennung ──────────────────────────────────────────
    header("2/14  Hardware Detection")
    hw = detect_hardware(repo_root)

    if hw.gpu.cuda_available:
        ok(f"GPU: {hw.gpu.name} ({hw.gpu.vram_gb} GB VRAM)")
        if hw.gpu.driver_version:
            info(f"Driver: {hw.gpu.driver_version}")
        if hw.gpu.cuda_compute:
            info(f"CUDA Compute: {hw.gpu.cuda_compute}")
    else:
        nvsmi = shutil.which("nvidia-smi")
        if nvsmi is None:
            result.add_warn(
                "No NVIDIA GPU detected or nvidia-smi missing. "
                "CPU mode active. Drivers: https://www.nvidia.com/drivers/"
            )
        else:
            result.add_warn("nvidia-smi found but GPU query failed")

    ok(f"RAM: {hw.ram_gb} GB")
    ok(f"Disk free: {hw.disk_free_gb} GB")
    ok(f"CPU Cores: {hw.cpu_cores}")
    info(f"Hardware tier: {BOLD}{hw.tier.upper()}{RESET}")

    # ── 3. Ollama pruefen ──────────────────────────────────────────────
    header("3/14  Ollama")
    ollama_path = find_ollama()

    if ollama_path is None:
        # Versuche Ollama via winget zu installieren
        ollama_ready = False
        winget_available = shutil.which("winget") is not None
        if winget_available:
            try:
                answer = input("  Ollama not found. Install now? [Y/n]: ").strip().lower()
            except EOFError:
                answer = "n"
            if answer in ("", "j", "y", "ja", "yes"):
                info("Installing Ollama via winget...")
                try:
                    winget_proc = subprocess.run(
                        [
                            "winget",
                            "install",
                            "--id",
                            "Ollama.Ollama",
                            "-e",
                            "--accept-source-agreements",
                            "--accept-package-agreements",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    if winget_proc.returncode == 0:
                        ok("Ollama installed via winget")
                        ollama_path = find_ollama()
                        if ollama_path:
                            info("Starting Ollama...")
                            if start_ollama(ollama_path):
                                ok("Ollama started")
                                ollama_ready = True
                            else:
                                result.add_warn("Ollama installed but could not be started")
                        else:
                            result.add_warn(
                                "Ollama installed but not in PATH. "
                                "Please close this window and reopen."
                            )
                    else:
                        result.add_warn(
                            "winget installation failed. "
                            "Please install manually: https://ollama.com/download"
                        )
                except Exception as e:
                    result.add_warn(f"Ollama installation failed: {e}")
            else:
                result.add_warn("Ollama not found. Please install: https://ollama.com/download")
        else:
            result.add_warn("Ollama not found. Please install: https://ollama.com/download")
    else:
        ok(f"Ollama found: {ollama_path}")
        if ollama_is_running():
            ok("Ollama server already running")
            ollama_ready = True
        else:
            info("Starting Ollama server...")
            if start_ollama(ollama_path):
                ok("Ollama server started")
                ollama_ready = True
            else:
                result.add_warn("Ollama could not be started")
                ollama_ready = False

    # ── 4. Ollama-Modelle ──────────────────────────────────────────────
    header("4/14  Models")

    # Hardware-Tier und Modell-Empfehlung anzeigen
    _tier_models_display = TIER_MODELS.get(hw.tier, TIER_MODELS["minimal"])
    _vram_str = f"{hw.gpu.vram_gb} GB VRAM" if hw.gpu.cuda_available else "no GPU"
    print(f"  Your system: {_vram_str}, {hw.ram_gb} GB RAM")
    print(f"  Hardware tier: {BOLD}{hw.tier.upper()}{RESET}")
    print(f"  Models: {', '.join(_tier_models_display)}")
    if hw.tier == "minimal":
        print(f"  {DIM}Tip: At least 8 GB VRAM recommended for better quality{RESET}")
    elif hw.tier in ("standard", "power", "enterprise"):
        print(f"  {DIM}Tip: 'cognithor --lite' for only 6 GB VRAM{RESET}")
    print()

    models_installed: list[str] = []
    if skip_models:
        info("Model download skipped (--skip-models)")
        _print_model_pull_commands(hw.tier)
    elif ollama_ready and ollama_path:
        models_installed = ensure_models(hw.tier, result, ollama_path)
    else:
        result.add_warn("Model download skipped (Ollama not ready)")
        _print_model_pull_commands(hw.tier)

    # ── 5. Python-Abhaengigkeiten ──────────────────────────────────────
    header("5/14  Python Dependencies")

    # Pruefe ob jarvis bereits importierbar ist
    jarvis_ok = False
    try:
        check = subprocess.run(
            [sys.executable, "-c", "import jarvis; print(jarvis.__version__)"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=repo_root,
        )
        if check.returncode == 0 and check.stdout.strip():
            ok(f"jarvis already installed (v{check.stdout.strip()})")
            jarvis_ok = True
            # Ensure [all] extras are installed (identity, etc.)
            _identity_check = subprocess.run(
                [sys.executable, "-c", "import chromadb"],
                capture_output=True,
                timeout=10,
            )
            if _identity_check.returncode != 0:
                info("Upgrading to install missing extras ([all])...")
                _upgrade = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", ".[all]", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=repo_root,
                )
                if _upgrade.returncode == 0:
                    ok("Extras installed (identity, etc.)")
                else:
                    result.add_warn("Could not install all extras. Try: pip install -e '.[all]'")
    except Exception:
        pass

    # Check ARC-AGI-3 availability (informational only — not auto-installed)
    if jarvis_ok:
        try:
            _arc_check = subprocess.run(
                [sys.executable, "-c", "import arc_agi"],
                capture_output=True,
                timeout=10,
            )
            if _arc_check.returncode == 0:
                ok("ARC-AGI-3 SDK available.")
            else:
                info('ARC-AGI-3 SDK not installed (optional). Install: pip install -e ".[arc]"')
        except Exception:
            pass

    if not jarvis_ok:
        # uv bevorzugen wenn vorhanden (10x schneller)
        installer_backend, installer_cmd = _detect_python_installer(repo_root)
        info(f"Installing Python dependencies with {installer_backend}...")
        info("This may take a few minutes on first run.")
        inst_proc = subprocess.run(
            installer_cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=repo_root,
        )
        if inst_proc.returncode == 0:
            result.add_pass(f"Python dependencies installed (via {installer_backend})")
        else:
            stderr = inst_proc.stderr.strip()[-300:] if inst_proc.stderr else ""
            result.add_fail(
                f"{installer_backend} install failed",
                f'Run manually: cd "{repo_root}" && pip install -e ".[all]"\n  Error: {stderr}',
            )
            return False

    # Also install into ~/.jarvis/venv if it exists but lacks jarvis
    # (Vite launcher prefers this venv — must have jarvis installed)
    _home_venv_python = (
        JARVIS_HOME
        / "venv"
        / ("Scripts" if sys.platform == "win32" else "bin")
        / ("python.exe" if sys.platform == "win32" else "python")
    )
    if _home_venv_python.exists() and str(_home_venv_python) != sys.executable:
        try:
            _venv_check = subprocess.run(
                [str(_home_venv_python), "-c", "import jarvis"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=repo_root,
            )
            if _venv_check.returncode != 0:
                info(f"Installing jarvis into {JARVIS_HOME / 'venv'}...")
                _venv_inst = subprocess.run(
                    [
                        str(_home_venv_python),
                        "-m",
                        "pip",
                        "install",
                        "-e",
                        ".[all]",
                        "--quiet",
                        "--disable-pip-version-check",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=repo_root,
                )
                if _venv_inst.returncode == 0:
                    ok(f"jarvis also installed in {JARVIS_HOME / 'venv'}")
                else:
                    warn(
                        f"Could not install jarvis into {JARVIS_HOME / 'venv'}: "
                        f"{_venv_inst.stderr[:200]}"
                    )
        except Exception as e:
            warn(f"Venv sync check failed: {e}")

    # ── 6. Flutter Web UI ──────────────────────────────────────────────
    header("6/14  Flutter Web UI")
    flutter_index = os.path.join(
        repo_root,
        "flutter_app",
        "build",
        "web",
        "index.html",
    )
    if os.path.isfile(flutter_index):
        ok("Flutter Web UI present (bundled)")
    else:
        # Try to build if Flutter SDK is available
        flutter_cmd = shutil.which("flutter")
        if flutter_cmd:
            info("Building Flutter Web UI...")
            flutter_dir = os.path.join(repo_root, "flutter_app")
            build_proc = subprocess.run(
                [flutter_cmd, "build", "web", "--release"],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=flutter_dir,
            )
            if build_proc.returncode == 0 and os.path.isfile(flutter_index):
                ok("Flutter Web UI built successfully")
            else:
                result.add_warn("Flutter build failed. UI will run in CLI mode.")
        else:
            result.add_warn(
                "Flutter Web UI not found (flutter_app/build/web/). "
                "Re-clone the repo or install Flutter SDK to build."
            )

    # ── 7. Verzeichnisstruktur ─────────────────────────────────────────
    header("7/14  Directory Structure")
    try:
        init_proc = subprocess.run(
            [sys.executable, "-m", "jarvis", "--init-only"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=repo_root,
        )
        if init_proc.returncode == 0:
            ok("Directory structure initialized")
        else:
            raise RuntimeError("init-only failed")
    except Exception:
        # Fallback: manuell erstellen
        for sub in [
            "memory",
            "memory/episodes",
            "memory/knowledge",
            "memory/procedures",
            "memory/sessions",
            "index",
            "logs",
            "cache",
            "cache/web_search",
        ]:
            (JARVIS_HOME / sub).mkdir(parents=True, exist_ok=True)
        ok("Directory structure created manually")

    # ── 8. Konfiguration ───────────────────────────────────────────────
    header("8/14  Configuration")
    config_dest = JARVIS_HOME / "config.yaml"
    config_src = Path(repo_root) / "config.yaml.example"
    if not config_dest.exists() and config_src.exists():
        shutil.copy2(config_src, config_dest)
        ok("config.yaml created")
    elif config_dest.exists():
        ok("config.yaml already exists")
    else:
        result.add_warn("config.yaml.example not found -- skipped")

    # Locale-basierte Spracherkennung
    if config_dest.exists():
        _cfg_text = config_dest.read_text(encoding="utf-8")
        if "language:" not in _cfg_text:
            import locale as _locale_mod

            try:
                _sys_locale = _locale_mod.getlocale()[0] or ""
                _lang_code = _sys_locale[:2].lower() if len(_sys_locale) >= 2 else "de"
            except Exception:
                _lang_code = "de"
            _detected_lang = "de" if _lang_code == "de" else "en"
            # Sprache an den Anfang der config.yaml schreiben
            config_dest.write_text(
                f'language: "{_detected_lang}"\n' + _cfg_text,
                encoding="utf-8",
            )
            ok(f"Language detected: {_detected_lang} (Locale: {_lang_code})")

    env_dest = JARVIS_HOME / ".env"
    env_src = Path(repo_root) / ".env.example"
    if not env_dest.exists() and env_src.exists():
        shutil.copy2(env_src, env_dest)
        ok(".env created")
    elif env_dest.exists():
        ok(".env already exists")
    else:
        result.add_warn(".env.example not found -- skipped")

    # ── 9. Piper TTS Voice-Modell ────────────────────────────────────
    header("9/14  Piper TTS Voice Model")
    voices_dir = JARVIS_HOME / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    piper_voice = "de_DE-pavoque-low"
    # Stimme aus Config lesen, falls vorhanden
    config_yaml = JARVIS_HOME / "config.yaml"
    if config_yaml.exists():
        try:
            import yaml

            with open(config_yaml, encoding="utf-8") as _cf:
                _ycfg = yaml.safe_load(_cf) or {}
            _vc = (_ycfg.get("channels") or {}).get("voice_config") or {}
            piper_voice = _vc.get("piper_voice", piper_voice)
        except Exception:
            pass
    model_path = voices_dir / f"{piper_voice}.onnx"
    if model_path.exists():
        ok(f"Piper voice available: {piper_voice}")
    else:
        info(f"Downloading Piper voice: {piper_voice}...")
        try:
            _download_piper_voice(piper_voice, voices_dir)
            if model_path.exists():
                result.add_pass(f"Piper voice installed: {piper_voice}")
            else:
                result.add_warn(f"Piper download completed but file missing: {model_path}")
        except Exception as e:
            result.add_warn(f"Piper voice download failed: {e}")

    # ── 10. Schnelltest ─────────────────────────────────────────────────
    header("10/14  Smoke Test")
    try:
        qt = subprocess.run(
            [sys.executable, "-c", "import jarvis; print(f'jarvis v{jarvis.__version__}')"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=repo_root,
        )
        if qt.returncode == 0:
            result.add_pass(f"Import OK: {qt.stdout.strip()}")
        else:
            result.add_warn(f"Import warning: {qt.stderr.strip()[:100]}")
    except Exception as e:
        result.add_warn(f"Import test failed: {e}")

    if ollama_ready:
        ok("Ollama reachable")
    else:
        result.add_warn("Ollama not reachable")

    # ── 11. Desktop-Verknuepfung ───────────────────────────────────────
    header("11/14  Desktop Shortcut")
    bat_path = os.path.join(repo_root, "start_cognithor.bat")
    shortcut_ok = False
    if os.path.isfile(bat_path):
        if create_desktop_shortcut(bat_path):
            result.add_pass("Desktop shortcut created")
            shortcut_ok = True
        else:
            result.add_warn("Could not create desktop shortcut")
    else:
        result.add_warn(f"start_cognithor.bat not found: {bat_path}")

    # ── 12. Marker schreiben ───────────────────────────────────────────
    header("12/14  Marker")
    write_marker(hw, models_installed, shortcut_ok)
    ok(f"Marker written: {MARKER_FILE}")

    # ── 13. LLM-Rauchtest ─────────────────────────────────────────────
    header("13/14  LLM Smoke Test")
    if ollama_ready:
        try:
            _smoke_payload = json.dumps(
                {
                    "model": "qwen3:8b",
                    "messages": [{"role": "user", "content": "Say hello briefly."}],
                    "stream": False,
                }
            ).encode("utf-8")
            _smoke_req = urllib.request.Request(
                f"{OLLAMA_URL}/api/chat",
                data=_smoke_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(_smoke_req, timeout=30) as _smoke_resp:
                _smoke_data = json.loads(_smoke_resp.read().decode())
                _smoke_answer = _smoke_data.get("message", {}).get("content", "").strip()
                if _smoke_answer:
                    # Kuerzen auf max 80 Zeichen fuer Anzeige
                    _display = _smoke_answer[:80] + ("..." if len(_smoke_answer) > 80 else "")
                    ok(f"LLM responds: {_display}")
                else:
                    result.add_warn("LLM responded empty -- model may not be ready yet")
        except Exception as e:
            result.add_warn(f"LLM smoke test failed: {e}")
    else:
        info("LLM smoke test skipped (Ollama not ready)")

    # ── 14. Zusammenfassung ────────────────────────────────────────────
    elapsed = time.time() - t0
    result.timings["total"] = elapsed
    print_summary(result, elapsed, first=True)

    return result.success


# ── Folgestart (5 Schritte) ────────────────────────────────────────────────
def quick_start(repo_root: str, *, skip_models: bool = False) -> bool:
    result = BootResult()
    t0 = time.time()

    info("Subsequent start detected -- Quick Check...")

    # ── 1. Ollama pruefen ──────────────────────────────────────────────
    ollama_path = find_ollama()
    if ollama_path and not ollama_is_running():
        info("Starting Ollama...")
        if start_ollama(ollama_path):
            result.add_pass("Ollama started")
        else:
            result.add_warn("Ollama could not be started")
    elif ollama_is_running():
        result.add_pass("Ollama running")
    else:
        # Auto-fix: Ollama nicht gefunden -- versuche Installation via winget
        info("Ollama not found -- attempting installation via winget...")
        try:
            winget_proc = subprocess.run(
                [
                    "winget",
                    "install",
                    "--id",
                    "Ollama.Ollama",
                    "-e",
                    "--accept-source-agreements",
                    "--accept-package-agreements",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if winget_proc.returncode == 0:
                result.add_pass("Ollama installed via winget")
                ollama_path = find_ollama()
                if ollama_path and not ollama_is_running():
                    info("Starting Ollama...")
                    if start_ollama(ollama_path):
                        result.add_pass("Ollama started")
                    else:
                        result.add_warn("Ollama installed but could not be started")
            else:
                result.add_warn(
                    "Ollama could not be installed via winget. "
                    "Please install manually: https://ollama.com/download"
                )
        except FileNotFoundError:
            result.add_warn(
                "winget not available. Please install Ollama manually: https://ollama.com/download"
            )
        except Exception as e:
            result.add_warn(f"Ollama installation failed: {e}")

    # ── 2. Modelle pruefen ─────────────────────────────────────────────
    if skip_models:
        info("Model check skipped (--skip-models)")
    elif ollama_is_running():
        models = get_installed_models()
        has_qwen = any("qwen3" in m.lower() for m in models)
        if has_qwen:
            result.add_pass(f"Models OK ({len(models)} installed)")
        else:
            result.add_warn("No qwen3 model found.")
            _print_model_pull_commands("minimal")
    else:
        result.add_warn("Model check skipped (Ollama not reachable)")

    # ── 3. Port pruefen ───────────────────────────────────────────────
    port_status = check_port(BACKEND_PORT)
    if port_status == "free":
        result.add_pass(f"Port {BACKEND_PORT} free")
    elif port_status == "cognithor":
        result.add_warn(
            f"Cognithor already running on port {BACKEND_PORT}. "
            f"Open http://localhost:5173 in your browser."
        )
    else:
        result.add_warn(f"Port {BACKEND_PORT} in use by another application")

    # ── 4. Import-Test (system Python + home venv) ──────────────────────
    _home_venv_py = (
        JARVIS_HOME
        / "venv"
        / ("Scripts" if sys.platform == "win32" else "bin")
        / ("python.exe" if sys.platform == "win32" else "python")
    )
    # Check both sys.executable AND the home venv (Vite prefers the venv)
    _pythons_to_check = [sys.executable]
    if _home_venv_py.exists() and str(_home_venv_py) != sys.executable:
        _pythons_to_check.append(str(_home_venv_py))

    for _py in _pythons_to_check:
        try:
            check = subprocess.run(
                [_py, "-c", "import jarvis"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=repo_root,
            )
            if check.returncode == 0:
                result.add_pass(f"Import OK ({Path(_py).parent.parent.name})")
                continue
            # Auto-fix: Import fehlgeschlagen -- automatische Reparatur
            info(f"Import failed in {_py} -- attempting repair...")
            repair_cmd = [
                _py,
                "-m",
                "pip",
                "install",
                "-e",
                ".[all]",
                "--quiet",
                "--disable-pip-version-check",
            ]
            repair_proc = subprocess.run(
                repair_cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=repo_root,
            )
            if repair_proc.returncode == 0:
                result.add_pass(f"Dependencies repaired ({Path(_py).parent.parent.name})")
            else:
                stderr = repair_proc.stderr.strip()[-300:] if repair_proc.stderr else ""
                result.add_fail(
                    f"jarvis import failed in {_py}",
                    f'Manually: cd "{repo_root}" && "{_py}" -m pip install -e ".[all]"\n'
                    f"  Error: {stderr}",
                )
        except Exception as e:
            result.add_fail("Import-Test", str(e))

    # ── 5. Piper TTS Voice-Modell ───────────────────────────────────
    voices_dir = JARVIS_HOME / "voices"
    piper_voice = "de_DE-pavoque-low"
    config_yaml = JARVIS_HOME / "config.yaml"
    if config_yaml.exists():
        try:
            import yaml

            with open(config_yaml, encoding="utf-8") as _cf:
                _ycfg = yaml.safe_load(_cf) or {}
            _vc = (_ycfg.get("channels") or {}).get("voice_config") or {}
            piper_voice = _vc.get("piper_voice", piper_voice)
        except Exception:
            pass
    model_path = voices_dir / f"{piper_voice}.onnx"
    if model_path.exists():
        result.add_pass(f"Piper voice: {piper_voice}")
    else:
        voices_dir.mkdir(parents=True, exist_ok=True)
        try:
            _download_piper_voice(piper_voice, voices_dir)
            result.add_pass(f"Piper voice downloaded: {piper_voice}")
        except Exception as e:
            result.add_warn(f"Piper voice download failed: {e}")

    elapsed = time.time() - t0
    print_summary(result, elapsed, first=False)

    return result.success


# ── Zusammenfassung ────────────────────────────────────────────────────────
def print_summary(result: BootResult, elapsed: float, first: bool = True) -> None:
    mode = "First Start" if first else "Quick Check"
    header(f"Summary ({mode})")

    print(f"  {GREEN}Passed:{RESET}     {len(result.passed)}")
    print(f"  {YELLOW}Warnings:{RESET}   {len(result.warnings)}")
    print(f"  {RED}Errors:{RESET}     {len(result.failed)}")
    print(f"  {DIM}Duration:{RESET}   {elapsed:.1f}s")
    print()

    if result.warnings:
        print(f"  {YELLOW}Warnings:{RESET}")
        for w in result.warnings:
            print(f"    - {w}")
        print()

    if result.failed:
        print(f"  {RED}Errors:{RESET}")
        for entry in result.failed:
            print(f"    - {entry}")
        print()

    if result.success:
        ok("System ready -- starting UI!")
        info("Tip: Use 'python -m jarvis --lite' for minimal VRAM usage (6 GB).")
    else:
        fail("System not ready. Please fix the errors above.")


# ── Versions-Upgrade-Check ─────────────────────────────────────────────────
def needs_reinit(marker: dict) -> bool:
    """Prueft ob Re-Initialisierung noetig ist (Version geaendert)."""
    old_ver = marker.get("version", "0.0.0")
    if old_ver != BOOTSTRAP_VERSION:
        info(f"Version upgrade detected: {old_ver} -> {BOOTSTRAP_VERSION}")
        return True
    return False


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    _enable_ansi()
    _setup_encoding()

    parser = argparse.ArgumentParser(description="Cognithor Windows Bootstrap")
    parser.add_argument("--repo-root", required=True, help="Pfad zum Repository-Root")
    parser.add_argument("--force", action="store_true", help="Erster Start erzwingen")
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Ollama-Modell-Download ueberspringen (manuell mit 'ollama pull' nachholen)",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    if not os.path.isdir(repo_root):
        fail(f"Repository not found: {repo_root}")
        return 1

    print(f"  {DIM}Repo:    {repo_root}{RESET}")
    print(f"  {DIM}Home:    {JARVIS_HOME}{RESET}")
    print(f"  {DIM}Version: {BOOTSTRAP_VERSION}{RESET}")

    marker = read_marker()

    if args.force or marker is None or needs_reinit(marker):
        # Erster Start oder Re-Init
        if marker is not None and needs_reinit(marker):
            info("Re-initialization in progress...")
        success = first_start(repo_root, skip_models=args.skip_models)
    else:
        # Folgestart
        success = quick_start(repo_root, skip_models=args.skip_models)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
