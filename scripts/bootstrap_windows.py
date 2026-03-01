#!/usr/bin/env python3
"""
Cognithor Windows Bootstrap — One-Click Setup & Quick-Start.

Erster Start:  Hardware-Erkennung, Deps-Installation, Modell-Download, Shortcut.
Folgestart:    Ollama-Check, Modell-Check, Port-Check, Import-Test (<5s).
"""

from __future__ import annotations

import argparse
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

# ── Version (muss mit pyproject.toml uebereinstimmen) ─────────────────────
BOOTSTRAP_VERSION = "0.26.0"

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
    print(f"  {RED}[FEHLER]{RESET}   {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[WARNUNG]{RESET}  {msg}")


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
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
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
                    ["nvidia-smi", "--query-gpu=compute_cap",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
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
        return round(usage.free / (1024 ** 3), 1)
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
    prog_files = os.environ.get("ProgramFiles", "C:\\Program Files")
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
        CREATE_NO_WINDOW = 0x08000000
        subprocess.Popen(
            [ollama_path, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
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
    info(f"Lade Modell: {name} (kann einige Minuten dauern)...")
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
                warn(f"Timeout beim Download von {name}")
                return False
            line = proc.stdout.readline().strip() if proc.stdout else ""
            if line and line != last_line:
                print(f"\r  {DIM}{line[:70]:<70}{RESET}", end="", flush=True)
                last_line = line
        print()  # Newline nach Fortschritt
        return proc.returncode == 0
    except Exception as e:
        warn(f"Fehler beim Download von {name}: {e}")
        return False


TIER_MODELS: dict[str, list[str]] = {
    "minimal":    ["qwen3:8b", "nomic-embed-text"],
    "standard":   ["qwen3:8b", "qwen3:32b", "nomic-embed-text"],
    "power":      ["qwen3:8b", "qwen3:32b", "qwen3-coder:32b", "nomic-embed-text"],
    "enterprise": ["qwen3:8b", "qwen3:32b", "qwen3-coder:32b", "nomic-embed-text"],
}


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
            ok(f"Modell vorhanden: {model}")
            pulled.append(model)
            continue

        if pull_model(model, ollama_path):
            ok(f"Modell installiert: {model}")
            pulled.append(model)
        else:
            result.add_warn(
                f"Modell {model} konnte nicht geladen werden. "
                f"Manuell: ollama pull {model}"
            )

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
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/v1/health", method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return "cognithor"
        except Exception:
            pass
        return "in_use"
    except (ConnectionRefusedError, socket.timeout, OSError):
        return "free"
    finally:
        try:
            sock.close()
        except Exception:
            pass


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
            capture_output=True, text=True, timeout=10,
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


# ── Erster Start (12 Schritte) ─────────────────────────────────────────────
def first_start(repo_root: str) -> bool:
    result = BootResult()
    t0 = time.time()

    # ── 1. Python-Version ──────────────────────────────────────────────
    header("1/12  Python-Version")
    v = sys.version_info
    if v >= (3, 12):
        result.add_pass(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        result.add_fail("Python-Version", f"{v.major}.{v.minor} < 3.12")
        return False

    # ── 2. Hardware-Erkennung ──────────────────────────────────────────
    header("2/12  Hardware-Erkennung")
    hw = detect_hardware(repo_root)

    if hw.gpu.cuda_available:
        ok(f"GPU: {hw.gpu.name} ({hw.gpu.vram_gb} GB VRAM)")
        if hw.gpu.driver_version:
            info(f"Treiber: {hw.gpu.driver_version}")
        if hw.gpu.cuda_compute:
            info(f"CUDA Compute: {hw.gpu.cuda_compute}")
    else:
        nvsmi = shutil.which("nvidia-smi")
        if nvsmi is None:
            result.add_warn(
                "Keine NVIDIA GPU erkannt oder nvidia-smi fehlt. "
                "CPU-Modus aktiv. Treiber: https://www.nvidia.com/drivers/"
            )
        else:
            result.add_warn("nvidia-smi vorhanden, aber GPU-Abfrage fehlgeschlagen")

    ok(f"RAM: {hw.ram_gb} GB")
    ok(f"Disk frei: {hw.disk_free_gb} GB")
    ok(f"CPU Cores: {hw.cpu_cores}")
    info(f"Hardware-Tier: {BOLD}{hw.tier.upper()}{RESET}")

    # ── 3. Ollama pruefen ──────────────────────────────────────────────
    header("3/12  Ollama")
    ollama_path = find_ollama()

    if ollama_path is None:
        result.add_warn(
            "Ollama nicht gefunden. Bitte installieren: https://ollama.com/download"
        )
        ollama_ready = False
    else:
        ok(f"Ollama gefunden: {ollama_path}")
        if ollama_is_running():
            ok("Ollama-Server laeuft bereits")
            ollama_ready = True
        else:
            info("Ollama-Server wird gestartet...")
            if start_ollama(ollama_path):
                ok("Ollama-Server gestartet")
                ollama_ready = True
            else:
                result.add_warn("Ollama konnte nicht gestartet werden")
                ollama_ready = False

    # ── 4. Ollama-Modelle ──────────────────────────────────────────────
    header("4/12  Modelle")
    models_installed: list[str] = []
    if ollama_ready and ollama_path:
        models_installed = ensure_models(hw.tier, result, ollama_path)
    else:
        result.add_warn("Modell-Download uebersprungen (Ollama nicht bereit)")

    # ── 5. Python-Abhaengigkeiten ──────────────────────────────────────
    header("5/12  Python-Abhaengigkeiten")

    # Pruefe ob jarvis bereits importierbar ist
    jarvis_ok = False
    try:
        check = subprocess.run(
            [sys.executable, "-c", "import jarvis; print(jarvis.__version__)"],
            capture_output=True, text=True, timeout=15, cwd=repo_root,
        )
        if check.returncode == 0 and check.stdout.strip():
            ok(f"jarvis bereits installiert (v{check.stdout.strip()})")
            jarvis_ok = True
    except Exception:
        pass

    if not jarvis_ok:
        info("Installiere Python-Abhaengigkeiten (pip install -e '.[all]')...")
        info("Das kann beim ersten Mal einige Minuten dauern.")
        pip_proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[all]",
             "--quiet", "--disable-pip-version-check"],
            capture_output=True, text=True, timeout=600, cwd=repo_root,
        )
        if pip_proc.returncode == 0:
            result.add_pass("Python-Abhaengigkeiten installiert")
        else:
            stderr = pip_proc.stderr.strip()[-300:] if pip_proc.stderr else ""
            result.add_fail(
                "pip install fehlgeschlagen",
                f"Manuell ausfuehren: cd \"{repo_root}\" && pip install -e \".[all]\"\n"
                f"  Fehler: {stderr}"
            )
            return False

    # ── 6. Node-Abhaengigkeiten ────────────────────────────────────────
    header("6/12  Node-Abhaengigkeiten")
    ui_dir = os.path.join(repo_root, "ui")
    node_modules = os.path.join(ui_dir, "node_modules")

    if os.path.isdir(node_modules):
        ok("node_modules vorhanden")
    else:
        info("Installiere Node-Abhaengigkeiten (npm install)...")
        npm_proc = subprocess.run(
            ["npm", "install"],
            capture_output=True, text=True, timeout=300,
            cwd=ui_dir, shell=True,
        )
        if npm_proc.returncode == 0:
            result.add_pass("Node-Abhaengigkeiten installiert")
        else:
            stderr = npm_proc.stderr.strip()[-300:] if npm_proc.stderr else ""
            result.add_fail(
                "npm install fehlgeschlagen",
                f"Manuell ausfuehren: cd \"{ui_dir}\" && npm install\n"
                f"  Fehler: {stderr}"
            )
            return False

    # ── 7. Verzeichnisstruktur ─────────────────────────────────────────
    header("7/12  Verzeichnisstruktur")
    try:
        init_proc = subprocess.run(
            [sys.executable, "-m", "jarvis", "--init-only"],
            capture_output=True, text=True, timeout=30, cwd=repo_root,
        )
        if init_proc.returncode == 0:
            ok("Verzeichnisstruktur initialisiert")
        else:
            raise RuntimeError("init-only failed")
    except Exception:
        # Fallback: manuell erstellen
        for sub in ["memory", "logs", "cache", "cache/web_search"]:
            (JARVIS_HOME / sub).mkdir(parents=True, exist_ok=True)
        ok("Verzeichnisstruktur manuell erstellt")

    # ── 8. Konfiguration ───────────────────────────────────────────────
    header("8/12  Konfiguration")
    config_dest = JARVIS_HOME / "config.yaml"
    config_src = Path(repo_root) / "config.yaml.example"
    if not config_dest.exists() and config_src.exists():
        shutil.copy2(config_src, config_dest)
        ok("config.yaml erstellt")
    elif config_dest.exists():
        ok("config.yaml bereits vorhanden")
    else:
        result.add_warn("config.yaml.example nicht gefunden -- uebersprungen")

    env_dest = JARVIS_HOME / ".env"
    env_src = Path(repo_root) / ".env.example"
    if not env_dest.exists() and env_src.exists():
        shutil.copy2(env_src, env_dest)
        ok(".env erstellt")
    elif env_dest.exists():
        ok(".env bereits vorhanden")
    else:
        result.add_warn(".env.example nicht gefunden -- uebersprungen")

    # ── 9. Schnelltest ─────────────────────────────────────────────────
    header("9/12  Schnelltest")
    try:
        qt = subprocess.run(
            [sys.executable, "-c",
             "import jarvis; print(f'jarvis v{jarvis.__version__}')"],
            capture_output=True, text=True, timeout=15, cwd=repo_root,
        )
        if qt.returncode == 0:
            result.add_pass(f"Import OK: {qt.stdout.strip()}")
        else:
            result.add_warn(f"Import-Warnung: {qt.stderr.strip()[:100]}")
    except Exception as e:
        result.add_warn(f"Import-Test fehlgeschlagen: {e}")

    if ollama_ready:
        ok("Ollama erreichbar")
    else:
        result.add_warn("Ollama nicht erreichbar")

    # ── 10. Desktop-Verknuepfung ───────────────────────────────────────
    header("10/12  Desktop-Verknuepfung")
    bat_path = os.path.join(repo_root, "start_cognithor.bat")
    shortcut_ok = False
    if os.path.isfile(bat_path):
        if create_desktop_shortcut(bat_path):
            result.add_pass("Desktop-Verknuepfung erstellt")
            shortcut_ok = True
        else:
            result.add_warn("Desktop-Verknuepfung konnte nicht erstellt werden")
    else:
        result.add_warn(f"start_cognithor.bat nicht gefunden: {bat_path}")

    # ── 11. Marker schreiben ───────────────────────────────────────────
    header("11/12  Marker")
    write_marker(hw, models_installed, shortcut_ok)
    ok(f"Marker geschrieben: {MARKER_FILE}")

    # ── 12. Zusammenfassung ────────────────────────────────────────────
    elapsed = time.time() - t0
    result.timings["total"] = elapsed
    print_summary(result, elapsed, first=True)

    return result.success


# ── Folgestart (4 Schritte) ────────────────────────────────────────────────
def quick_start(repo_root: str) -> bool:
    result = BootResult()
    t0 = time.time()

    info("Folgestart erkannt -- Quick-Check...")

    # ── 1. Ollama pruefen ──────────────────────────────────────────────
    ollama_path = find_ollama()
    if ollama_path and not ollama_is_running():
        info("Ollama wird gestartet...")
        if start_ollama(ollama_path):
            result.add_pass("Ollama gestartet")
        else:
            result.add_warn("Ollama konnte nicht gestartet werden")
    elif ollama_is_running():
        result.add_pass("Ollama laeuft")
    else:
        result.add_warn("Ollama nicht gefunden")

    # ── 2. Modelle pruefen ─────────────────────────────────────────────
    if ollama_is_running():
        models = get_installed_models()
        has_qwen = any("qwen3" in m.lower() for m in models)
        if has_qwen:
            result.add_pass(f"Modelle OK ({len(models)} installiert)")
        else:
            result.add_warn("Kein qwen3-Modell gefunden -- ollama pull qwen3:8b")
    else:
        result.add_warn("Modell-Check uebersprungen (Ollama nicht erreichbar)")

    # ── 3. Port pruefen ───────────────────────────────────────────────
    port_status = check_port(BACKEND_PORT)
    if port_status == "free":
        result.add_pass(f"Port {BACKEND_PORT} frei")
    elif port_status == "cognithor":
        result.add_warn(
            f"Cognithor laeuft bereits auf Port {BACKEND_PORT}. "
            f"Oeffne http://localhost:5173 im Browser."
        )
    else:
        result.add_warn(f"Port {BACKEND_PORT} belegt durch andere Anwendung")

    # ── 4. Import-Test ─────────────────────────────────────────────────
    try:
        check = subprocess.run(
            [sys.executable, "-c", "import jarvis"],
            capture_output=True, text=True, timeout=10, cwd=repo_root,
        )
        if check.returncode == 0:
            result.add_pass("Import OK")
        else:
            result.add_fail(
                "jarvis Import fehlgeschlagen",
                "pip install -e \".[all]\" im Repo-Root ausfuehren"
            )
    except Exception as e:
        result.add_fail("Import-Test", str(e))

    elapsed = time.time() - t0
    print_summary(result, elapsed, first=False)

    return result.success


# ── Zusammenfassung ────────────────────────────────────────────────────────
def print_summary(result: BootResult, elapsed: float, first: bool = True) -> None:
    mode = "Erster Start" if first else "Quick-Check"
    header(f"Zusammenfassung ({mode})")

    print(f"  {GREEN}Bestanden:{RESET}  {len(result.passed)}")
    print(f"  {YELLOW}Warnungen:{RESET}  {len(result.warnings)}")
    print(f"  {RED}Fehler:{RESET}     {len(result.failed)}")
    print(f"  {DIM}Dauer:{RESET}      {elapsed:.1f}s")
    print()

    if result.warnings:
        print(f"  {YELLOW}Warnungen:{RESET}")
        for w in result.warnings:
            print(f"    - {w}")
        print()

    if result.failed:
        print(f"  {RED}Fehler:{RESET}")
        for entry in result.failed:
            print(f"    - {entry}")
        print()

    if result.success:
        ok("System bereit -- UI wird gestartet!")
    else:
        fail("System nicht bereit. Bitte Fehler oben beheben.")


# ── Versions-Upgrade-Check ─────────────────────────────────────────────────
def needs_reinit(marker: dict) -> bool:
    """Prueft ob Re-Initialisierung noetig ist (Version geaendert)."""
    old_ver = marker.get("version", "0.0.0")
    if old_ver != BOOTSTRAP_VERSION:
        info(f"Version-Upgrade erkannt: {old_ver} -> {BOOTSTRAP_VERSION}")
        return True
    return False


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    _enable_ansi()
    _setup_encoding()

    parser = argparse.ArgumentParser(description="Cognithor Windows Bootstrap")
    parser.add_argument("--repo-root", required=True, help="Pfad zum Repository-Root")
    parser.add_argument("--force", action="store_true", help="Erster Start erzwingen")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    if not os.path.isdir(repo_root):
        fail(f"Repository nicht gefunden: {repo_root}")
        return 1

    print(f"  {DIM}Repo:    {repo_root}{RESET}")
    print(f"  {DIM}Home:    {JARVIS_HOME}{RESET}")
    print(f"  {DIM}Version: {BOOTSTRAP_VERSION}{RESET}")

    marker = read_marker()

    if args.force or marker is None or needs_reinit(marker):
        # Erster Start oder Re-Init
        if marker is not None and needs_reinit(marker):
            info("Re-Initialisierung wird durchgefuehrt...")
        success = first_start(repo_root)
    else:
        # Folgestart
        success = quick_start(repo_root)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
