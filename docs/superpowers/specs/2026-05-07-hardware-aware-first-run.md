# Hardware-Aware First-Run — GPU-Detection + Auto-Backend-Empfehlung

> **Status:** Draft v1.0 · **Date:** 2026-05-07 · **Owner:** Alexander Söllner
> **Komplement zu:** `2026-05-07-fast-path-router-spec.md` (Modellempfehlungen).
> **Risk-Class:** USER-VISIBLE — formt die First-Use-Experience neu. Backwards-Compat
> für bestehende Installationen ist hartes Requirement.

## 0. TL;DR

Wenn ein User Cognithor heute via `pip install cognithor` (PyPI) oder
`git clone … && pip install -e .` (GitHub) installiert, läuft beim
ersten Start `scripts/first_boot.py` ab — checkt Python, RAM, GPU, Ollama,
schreibt `~/.cognithor/.cognithor_initialized`. Was **fehlt**: aus den
Erkenntnissen eine **Backend-Empfehlung** ableiten, dem User
**transparent vorschlagen** ("auf deinem System läuft vLLM mit
qwen3.6:27b 3-4× schneller als Ollama — möchtest du das?"), und bei
Bestätigung `~/.cognithor/config.yaml` mit korrekten Backend- und
Modellnamen schreiben.

Default-Code-Wert (`llm_backend_type="ollama"`) bleibt für maximale
Hardware-Kompatibilität, aber der **First-Run-Wizard** kippt User mit
fähiger Hardware automatisch (mit Bestätigung) auf vLLM.

---

## 1. Problem & Motivation

### 1.1 Ist-Zustand (verifiziert 2026-05-07)

```
pip install cognithor
cognithor               # erster Start

→ scripts/first_boot.py:check_system() läuft
  → detect_gpu()     ✅ erfasst NVIDIA name + VRAM + driver
  → detect_ram()     ✅
  → detect_ollama()  ✅
  → detect_disk()    ✅
  → detect_network() ✅

→ SystemProfile.get_tier()             → "enterprise" / "power" / "standard" / "minimal"
→ SystemProfile.get_recommended_mode() → "offline" / "online" / "hybrid"

→ Markiert ~/.cognithor/.cognithor_initialized
→ STARTET MIT DEFAULT-CONFIG (llm_backend_type=ollama)
```

Das `SystemProfile.get_recommended_mode()` empfiehlt nur **Internet-Strategie**
("online"/"offline"/"hybrid"), aber keine **Backend-Wahl** (welches LLM-Backend)
und keine **Modell-Wahl** (welche Modelle zu deiner Hardware passen).

### 1.2 Konkrete User-Story die heute schief geht

**Owner-Maschine:** RTX 5090 (32 GB), Win11 + WSL2 + Docker, Driver 596+.

**Heute:** `pip install cognithor` → Default-Boot → spricht mit Ollama →
qwen3:32b läuft auf CPU oder mit Ollama-CUDA, **3-4× langsamer** als
mit vLLM-cu130-nightly + Qwen3.6-27B-NVFP4 (devnen-Bench: 158 tok/s
auf 5090, vLLM-Recipe: 207 tok/s mit MTP-Speculative). Plus: kein
Multi-Token-Prediction, kein NVFP4-Quant, kein PagedAttention-Throughput.

User merkt nicht, dass er sub-optimal läuft. Premium-Hardware liegt brach.

### 1.3 Dual-Side des Problems

- **Premium-Hardware-User** (RTX 5090, 4090, A6000, …) bekommen ohne
  manuellen Eingriff schlechtere Performance als möglich.
- **Low-End-User** (kein GPU, AMD-GPU, Mac M-Serie) brauchen einen
  Default der auf ihrer Hardware **funktioniert** — Ollama liefert
  das, vLLM nicht.

Eine **statisch gewählte Default-Backend-Konstante** kann diese beiden
Anforderungen nicht gleichzeitig erfüllen. Lösung: **dynamisch
detektieren + empfehlen + bestätigen lassen**.

---

## 2. Goals & Non-Goals

### 2.1 Goals

1. **G1.** First-Run nach `pip install cognithor` (ohne `~/.cognithor/`)
   detektiert die Hardware vollständig (GPU-Vendor, VRAM, Compute-Capability,
   CUDA-Version, Driver, WSL2-Status, Docker, RAM, Plattenplatz, Internet).
2. **G2.** Aus dem Hardware-Profil wird ein **Backend-Tier** abgeleitet
   (`enterprise-vllm-nvfp4` / `power-vllm-fp8` / `standard-ollama-gpu` /
   `minimal-ollama-cpu` / `cloud-only`).
3. **G3.** Pro Tier ist ein vollständiges **Modell-Set** vordefiniert
   (Planner / Executor / Coder / Embedding / Formulate / Fast-Path-Validator).
4. **G4.** Empfehlung wird dem User **transparent angezeigt** (CLI-Wizard
   bei `cognithor` ohne Flutter, Flutter-Onboarding-Screen wenn UI da ist),
   **nicht stillschweigend** angewendet.
5. **G5.** Auf User-Bestätigung wird `~/.cognithor/config.yaml` **idempotent**
   geschrieben — bestehende User-Overrides werden respektiert.
6. **G6.** Re-Detection bei Hardware-Wechsel (neue GPU, Driver-Update,
   GPU entfernt) und Vorschlag zum Re-Config.
7. **G7.** Backwards-Compat: User mit existierender `~/.cognithor/config.yaml`
   sehen den Wizard **nicht** automatisch (opt-in via `cognithor doctor --reconfigure`).

### 2.2 Non-Goals

- **NG1.** Modelle automatisch herunterladen ohne User-Zustimmung
  (Modell-Pulls können 14–60 GB sein).
- **NG2.** Docker / WSL2 automatisch installieren — wird nur
  detektiert + Hinweis gegeben.
- **NG3.** Cloud-API-Keys auto-konfigurieren — bleibt manuell.
- **NG4.** "Smart"-Auto-Switching zur Laufzeit (Backend wechselt sich
  selbst um wenn GPU heiß läuft) — out of scope für v1, ggf. Sprint-N.
- **NG5.** Re-Detection ohne explizites User-Trigger — kein Background-Daemon
  der die Hardware permanent überwacht (Privacy + Trust).

---

## 3. Threat & Edge-Case-Model

| # | Edge-Case / Threat | Severity | Mitigation |
|--|---------------------|---------:|-----------|
| E1 | nvidia-smi vorhanden aber Driver crashed (zombies durch failed update) | HIGH | nvidia-smi exit-code + stderr parsen, Timeout 10s, bei Fehler auf "no_gpu" zurückfallen + Warning loggen, NIEMALS hängen |
| E2 | Mehrere GPUs (Multi-GPU-Box) | MEDIUM | Erfassen ALLER GPUs, Heuristik: "größte einzelne VRAM-GPU als Master, Sum aller VRAMs als Pool". Multi-GPU-vLLM-Setup als Sprint-N (komplex). Default v1: nur erste GPU. |
| E3 | NVIDIA-GPU mit < 8 GB VRAM | MEDIUM | → Tier `standard-ollama-gpu` mit kleineren Modellen, nicht vLLM-27B (würde OOM). Klares "vLLM würde nicht in 6 GB passen"-Hinweis. |
| E4 | Apple Silicon (M2/M3/M4) | HIGH | Unified memory ≠ VRAM. Detection muss `unified_memory=true` flag setzen. Empfehlung: `ollama` (Metal-Backend, läuft hervorragend), kein vLLM (nicht supported). Tier `mac-ollama-metal`. |
| E5 | AMD-GPU (RDNA 3/4) | HIGH | ROCm-Detection (`rocm-smi` oder `/opt/rocm/bin/rocminfo`). vLLM-ROCm ist experimentell, **nicht** als Default empfehlen. Tier `amd-ollama-rocm`. |
| E6 | Intel Arc / oneAPI | LOW | Ollama-IPEX experimentell. v1: erkennen + "experimental" warnen, Default Ollama-CPU. |
| E7 | NVIDIA-GPU OK, aber CUDA 11 / Compute-Capability < 8.0 (Pascal/Volta) | HIGH | NVFP4 braucht sm120 (Blackwell). FP8 braucht sm89+ (Ada/Hopper). Pre-Ampere = nur GPTQ-Int4 oder Ollama. Tier-Auswahl muss Compute-Capability berücksichtigen. |
| E8 | Driver < 596 auf Blackwell-GPU | HIGH | NVFP4 braucht 596+. Detection liefert Driver-Version, Tier-Logik checkt Mindestversion und fällt auf FP8 zurück (oder warnt). |
| E9 | WSL2 nicht installiert auf Windows | MEDIUM | Docker-Desktop-Backend ist Pflicht für vLLM-Container auf Windows. Wenn `wsl --status` 0 zurückgibt + GPU passend → "vLLM via WSL2 möglich, brauchst Docker-Desktop"-Hint, fall back auf Ollama-Default. |
| E10 | Docker installiert aber nicht laufend | MEDIUM | `docker info` mit Timeout 5s. Wenn fail: vLLM-Container nicht direkt verfügbar, Fallback auf vLLM-inprocess-Backend (existiert bereits — `core/vllm_inprocess_backend.py`) oder Ollama. |
| E11 | Cognithor läuft im Container (Cognithor-Docker oder cloud-deployment) | HIGH | `/.dockerenv` und `/proc/1/cgroup` prüfen. Wenn ja: Hardware-Detection zeigt **Host-Hardware nicht zwangsläufig**. NVIDIA-Runtime nur wenn `--gpus all` durchgereicht. Cognithor muss in diesem Fall die "von außen sichtbare" Hardware nehmen, nicht spekulieren. |
| E12 | RAM-Knappheit (16 GB System-RAM) bei vLLM-Auswahl | MEDIUM | vLLM braucht zusätzlich ~12 GB Sys-RAM für KV-Cache + CPU-Offload. Tier-Logik addiert Mindest-Sys-RAM-Constraint. |
| E13 | Plattenplatz für Modelle nicht da (z.B. 30 GB frei, aber 50 GB nötig) | HIGH | `detect_disk()` schon da. Tier-Empfehlung muss free_gb gegen Modell-Größe checken. Bei knapp → kleinere Modelle empfehlen. |
| E14 | User hat schon eine config.yaml (Update-Szenario, nicht Erst-Install) | CRITICAL | `.cognithor_initialized` existiert + config.yaml existiert → **NICHT** auto-overwriten. Nur Hint geben: "Möchtest du `cognithor doctor --reconfigure` laufen lassen?" |
| E15 | User-config.yaml unvollständig (Pydantic-Validation-Fehler) | MEDIUM | Bei Validation-Error: nicht crashen, sondern Wizard erneut anbieten "Config corrupt, neu konfigurieren?" |
| E16 | Hardware-Wechsel nach Install (User tauscht GPU) | LOW | Re-Detection nur auf User-Trigger (`cognithor doctor`). Hardware-Hash im `.cognithor_initialized` storen, beim Boot abgleichen, bei Mismatch sanft hinweisen — nicht aufdrängen. |
| E17 | Cloud-only User (keine Inferenz lokal, nur OpenAI/Anthropic-Keys) | MEDIUM | Tier `cloud-only` ohne lokale-Backend-Empfehlung, dafür Hinweis auf API-Key-Setup-Wizard. |
| E18 | Vollkommen offline (keine Internet während First-Run) | MEDIUM | Network-Detection schon da. Wenn offline + GPU → vLLM-Empfehlung + "lade die Modelle wenn du wieder Netz hast"-Pfad. Wenn offline + kein GPU → Ollama-CPU-only (existing default). |
| E19 | User auf Server-Linux ohne Display (Headless) | LOW | Wizard läuft im CLI (TTY-Detection). Bei kein-TTY: rein non-interactive, schreibt JSON-Empfehlung in `~/.cognithor/recommendation.json`, fängt mit Default an, User kann später bewusst übernehmen. |
| E20 | Sicherheits-Edge: User installiert in Sandbox (CI/CD-Build-Server) wo nvidia-smi falsche Werte liefert | LOW | Tier-Empfehlung ist nur Empfehlung, immer mit Bestätigung. CI/CD-Pfade haben sowieso eigene Configs. Nicht-aufdringliches Default-Behavior. |
| E21 | nvidia-smi liefert Werte mit Komma-Locale (`16,0` statt `16.0` auf DE-Windows) | MEDIUM | `--format=csv,noheader,nounits` ist locale-stabil, aber `float(parts[1].strip())` bricht bei Komma. Nutzen `parts[1].replace(",", ".").strip()`. |
| E22 | Modellnamen zwischen Backends inkompatibel (Ollama: `qwen3:32b` ≠ vLLM: `Qwen/Qwen3.6-27B-FP8`) | HIGH | Tier-Definition liefert PRO-BACKEND den korrekten Namen. Beim Backend-Switch (egal ob Wizard oder Flutter-UI) wird der Modellname aus dem aktuellen Tier-Profil neu aufgelöst, nicht 1:1 übernommen. |
| E23 | Telemetry-Leak: Hardware-Profil wird unbeabsichtigt geloggt / phoned home | HIGH | Hardware-Profil bleibt **lokal** in `~/.cognithor/.system_profile.json`. Kein Telemetry-Endpoint. TRUST-9-Cost-Ledger kriegt KEIN Hardware-Detail (nur Modell-Namen). |
| E24 | Owner-Privacy: Hostname / User-Name landet im Profil | MEDIUM | Profil enthält bewusst keinen Hostname / User. CPU-Modell + GPU-Modell sind weniger fingerprintbar als Hostname. |

---

## 4. What's Already There (re-use)

✅ **`src/cognithor/system/detector.py`** (404 LOC):
  - `SystemDetector.detect_os()` mit WSL-Detection
  - `detect_cpu()`, `detect_ram()`, `detect_disk()`, `detect_network()`
  - `detect_gpu()` — NVIDIA via nvidia-smi, Apple Silicon, "no GPU"
  - `detect_ollama()`, `detect_lmstudio()`
  - `SystemProfile.get_tier()` (enterprise/power/standard/minimal),
    `get_available_modes()`, `get_recommended_mode()`
  - Save/Load von `~/.cognithor/.system_profile.json`

✅ **`scripts/first_boot.py`** (732 LOC):
  - `check_system()`, `check_memory_init()`, `print_summary()`
  - `BootResult` für Aggregation

✅ **`scripts/preflight_check.py`** + `scripts/bootstrap_windows.py`
  (Windows-spezifisch, 1530 LOC)

✅ **`src/cognithor/core/vllm_orchestrator.py`** + `vllm_inprocess_backend.py`
  (Sprint 1.1 — kann vLLM auto-launchen via Docker oder in-Prozess)

✅ **`flutter_app/lib/screens/vllm_setup_screen.dart`** (heute morgen erweitert)
  — VLM-Quality-Dropdown, Backend-Switching-UI

---

## 5. Gaps in Detection-Layer (NEU)

### 5.1 Erweiterte GPU-Detection

```python
# src/cognithor/system/detector.py — additions to detect_gpu()
@dataclass(frozen=True)
class GpuDetail:
    vendor: Literal["nvidia", "amd", "intel", "apple", "none"]
    model: str
    vram_total_gb: float
    vram_free_gb: float
    driver_version: str
    cuda_version: str | None       # NEU — aus `nvcc --version` oder nvidia-smi
    compute_capability: str | None # NEU — z.B. "12.0" für Blackwell, "8.9" für Ada
    architecture: str | None       # NEU — z.B. "blackwell", "ada", "ampere"
    supports_nvfp4: bool            # abgeleitet (compute_capability >= 12.0 + driver >= 596)
    supports_fp8: bool               # abgeleitet (compute_capability >= 8.9)
    multi_gpu_count: int             # >1 wenn mehrere identische GPUs

@dataclass(frozen=True)
class HostDetail:                    # NEU
    is_wsl2: bool
    is_docker_container: bool        # /.dockerenv check
    docker_available: bool           # docker info ok
    docker_running: bool
    docker_compose_available: bool
    rocm_available: bool             # AMD
    intel_oneapi_available: bool
```

### 5.2 Neue Detector-Methoden

| Methode | Liefert | Notes |
|---|---|---|
| `detect_cuda_toolkit()` | CUDA-Version aus `nvcc --version` falls installiert | optional, nicht required |
| `detect_compute_capability()` | sm-Version aus nvidia-smi `compute_cap` | required für NVFP4-Detection |
| `detect_docker()` | Docker-availability + running-state | required für vLLM-Container |
| `detect_wsl2()` | Win-only — `wsl --status` exit code | required für vLLM-on-Windows |
| `detect_rocm()` | AMD-ROCm-Toolkit | LOW-priority |
| `detect_in_container()` | Cognithor selbst läuft in Container? | für E11 |
| `detect_multi_gpu()` | Liste aller GPUs (nicht nur erste) | für E2 |

### 5.3 Latenz-Budget

Voller First-Run-Scan: **≤ 12 s** (User merkt's, soll nicht nerven).
Background-Re-Scan (`cognithor doctor`): kein Budget.

---

## 6. Recommendation Layer (NEU)

### 6.1 Hardware-Tier-Definition

In `src/cognithor/system/recommender.py` (NEU):

```python
@dataclass(frozen=True)
class HardwareTier:
    name: str                                      # "enterprise-vllm-nvfp4"
    backend_type: Literal["ollama", "vllm", "lmstudio", "cloud-only"]
    requires: dict[str, object]                    # constraints für Match
    models_per_backend: dict[str, ModelSet]        # full Model-Set
    rationale: str                                 # für CLI-Wizard-Anzeige
    estimated_setup_minutes: int

TIERS_V1: tuple[HardwareTier, ...] = (
    HardwareTier(
        name="enterprise-vllm-nvfp4",
        backend_type="vllm",
        requires={
            "vendor": "nvidia",
            "vram_min_gb": 24,
            "compute_capability_min": "12.0",
            "driver_min": "596",
            "docker_running": True,
            "ram_min_gb": 32,
            "disk_free_min_gb": 80,
        },
        models_per_backend={
            "vllm": ModelSet(
                planner="sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
                executor="Qwen/Qwen3.5-9B",
                coder="sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
                embedding="qwen3-embedding:0.6b",   # Ollama-served, hybrid
                formulate="Qwen/Qwen3.5-9B",
                fast_path_validator="Qwen/Qwen3.5-9B",
            ),
        },
        rationale="RTX 5090/4090/A6000 mit cu130-Driver — NVFP4 + MTP-Speculative liefert ~3-4× Throughput vs Ollama.",
        estimated_setup_minutes=15,  # Docker pull, model pull, vLLM start
    ),
    HardwareTier(
        name="power-vllm-fp8",
        backend_type="vllm",
        requires={
            "vendor": "nvidia",
            "vram_min_gb": 24,
            "compute_capability_min": "8.9",  # Ada (RTX 4090, A6000-Ada)
            "driver_min": "550",
            "docker_running": True,
            "ram_min_gb": 32,
            "disk_free_min_gb": 80,
        },
        models_per_backend={
            "vllm": ModelSet(
                planner="Qwen/Qwen3.6-27B-FP8",
                executor="Qwen/Qwen3.5-9B",
                coder="Qwen/Qwen3-coder-30B",
                embedding="qwen3-embedding:0.6b",
                formulate="Qwen/Qwen3.5-9B",
                fast_path_validator="Qwen/Qwen3.5-9B",
            ),
        },
        rationale="Ada/Hopper-GPU — FP8 ist optimal, kein NVFP4-Speedup verfügbar.",
        estimated_setup_minutes=15,
    ),
    HardwareTier(
        name="standard-ollama-cuda",
        backend_type="ollama",
        requires={
            "vendor": "nvidia",
            "vram_min_gb": 8,
            "ram_min_gb": 16,
        },
        models_per_backend={
            "ollama": ModelSet(
                planner="qwen3:32b",      # heutiger Default
                executor="qwen3:8b",
                coder="qwen3-coder:30b",
                embedding="qwen3-embedding:0.6b",
                formulate="qwen3:8b",
                fast_path_validator="qwen3:8b",
            ),
        },
        rationale="NVIDIA-GPU mit ≥8 GB VRAM — Ollama-CUDA, kein Docker / vLLM nötig.",
        estimated_setup_minutes=8,
    ),
    HardwareTier(
        name="mac-ollama-metal",
        backend_type="ollama",
        requires={"vendor": "apple"},
        models_per_backend={
            "ollama": ModelSet(
                planner="qwen3:8b",        # 32b auf Mac unrealistisch außer M3 Ultra
                executor="qwen3:4b",
                coder="qwen3-coder:7b",
                embedding="qwen3-embedding:0.6b",
                formulate="qwen3:4b",
                fast_path_validator="qwen3:4b",
            ),
        },
        rationale="Apple-Silicon — Metal-Backend, Modelle gerichtet auf Unified-Memory-Limits.",
        estimated_setup_minutes=8,
    ),
    HardwareTier(
        name="amd-ollama-rocm",
        backend_type="ollama",
        requires={"vendor": "amd", "rocm_available": True},
        models_per_backend={
            "ollama": ModelSet(
                planner="qwen3:8b",
                executor="qwen3:4b",
                coder="qwen3-coder:7b",
                embedding="qwen3-embedding:0.6b",
                formulate="qwen3:4b",
                fast_path_validator="qwen3:4b",
            ),
        },
        rationale="AMD-GPU mit ROCm — Ollama-ROCm. vLLM-ROCm ist experimentell, nicht empfohlen für v1.",
        estimated_setup_minutes=10,
    ),
    HardwareTier(
        name="minimal-ollama-cpu",
        backend_type="ollama",
        requires={"ram_min_gb": 8},
        models_per_backend={
            "ollama": ModelSet(
                planner="qwen3:4b",
                executor="qwen3:1.5b",
                coder="qwen2.5-coder:7b",
                embedding="qwen3-embedding:0.6b",
                formulate="qwen3:1.5b",
                fast_path_validator="qwen3:1.5b",
            ),
        },
        rationale="Kein GPU oder schwache GPU — CPU-Inferenz mit kleinen Modellen, langsam aber funktional.",
        estimated_setup_minutes=5,
    ),
    HardwareTier(
        name="cloud-only",
        backend_type="cloud-only",
        requires={"vram_max_gb": 4, "ram_max_gb": 8},  # nichts lokal sinnvoll
        models_per_backend={},  # User wählt anbieter
        rationale="Hardware reicht nicht für lokale Inferenz — Cloud-API empfohlen (OpenAI, Anthropic, Gemini).",
        estimated_setup_minutes=3,
    ),
)
```

### 6.2 Tier-Matching-Algorithmus

```python
def match_tier(profile: SystemProfile) -> tuple[HardwareTier, ...]:
    """Liefert ALLE Tiers, deren `requires` durch das Profil erfüllt sind,
    sortiert von 'beste' (zuerst) zu 'fallback' (zuletzt)."""
    matches: list[HardwareTier] = []
    for tier in TIERS_V1:
        if _profile_satisfies(profile, tier.requires):
            matches.append(tier)
    return tuple(matches) or (TIERS_V1[-1],)  # cloud-only als ultimative Fallback
```

`match_tier()` liefert eine **rangierte Liste** — der User sieht alle
zutreffenden Optionen, nicht nur die "beste". Beispiel: RTX 5090 matcht
`enterprise-vllm-nvfp4`, `power-vllm-fp8`, `standard-ollama-cuda` —
alle drei werden angeboten, mit Empfehlung auf #1.

### 6.3 Recommendation-Output

```python
@dataclass(frozen=True)
class Recommendation:
    primary_tier: HardwareTier
    alternative_tiers: tuple[HardwareTier, ...]
    profile_snapshot: SystemProfile        # für Audit-Trail
    warnings: tuple[str, ...]              # z.B. "Driver alt — empfehlen Update für NVFP4"
    blockers: tuple[str, ...]              # z.B. "Docker nicht installiert für vLLM"
    estimated_disk_required_gb: float      # für Modell-Pulls
    estimated_first_response_seconds: int  # für Tier-Vergleich
```

---

## 7. First-Run-Wizard Flow

### 7.1 CLI-Pfad (für Headless / SSH / WSL2)

```
$ pip install cognithor
$ cognithor

╭─ Cognithor First-Run ──────────────────────────────────────────╮
│                                                                  │
│  Hardware-Detection läuft … (≤12s)                              │
│                                                                  │
│  ✓ OS: Windows 11 Pro 26200 (WSL2 verfügbar)                    │
│  ✓ CPU: AMD Ryzen 9 7950X3D (16 cores)                          │
│  ✓ RAM: 64 GB                                                    │
│  ✓ GPU: NVIDIA GeForce RTX 5090 (32 GB, sm120, Driver 596.21)   │
│  ✓ Docker Desktop: läuft                                         │
│  ✓ Disk: 412 GB frei                                             │
│  ✓ Internet: ja                                                  │
│  ✓ Ollama: nicht installiert                                     │
│                                                                  │
│  Empfohlener Tier: ENTERPRISE-VLLM-NVFP4                        │
│  ────────────────────────────────────────                       │
│  • Backend: vLLM (Docker, cu130-nightly)                        │
│  • Planner: Qwen3.6-27B-Text-NVFP4-MTP                          │
│  • Executor: Qwen3.5-9B                                          │
│  • Coder: Qwen3.6-27B-Text-NVFP4-MTP (gleicher Container)       │
│                                                                  │
│  Warum: NVFP4 + MTP liefert ~3-4× Throughput vs Ollama auf 5090.│
│  Setup-Zeit: ~15 min (Docker-Image + Modell-Pull ~14 GB)        │
│                                                                  │
│  Alternativen:                                                   │
│    [2] POWER-VLLM-FP8         (FP8 statt NVFP4, ~2× Ollama)     │
│    [3] STANDARD-OLLAMA-CUDA   (kein Docker, läuft direkt)       │
│                                                                  │
│  [1] Übernehmen (empfohlen)  [2/3] Alternativ wählen            │
│  [m] Manuell konfigurieren  [c] Cloud-only  [a] Abbruch         │
│                                                                  │
╰──────────────────────────────────────────────────────────────────╯
> _
```

Auswahl `1` → schreibt config.yaml, startet Modell-Pull, läuft los.

### 7.2 Flutter-Pfad (wenn UI verfügbar)

`flutter_app/lib/screens/onboarding/hardware_wizard_screen.dart` (NEU):

- Stepper-Layout: Detection-Spinner → Profil-Anzeige → Tier-Karten
  (3 Karten nebeneinander, Empfohlene hervorgehoben) → Bestätigung →
  Pull-Progress-View → Done.
- Karten zeigen: Tier-Name, Performance-Estimate (tok/s),
  Setup-Zeit, Modell-Liste, Disk-Bedarf.
- "Manuell konfigurieren"-Button öffnet die bestehende `LLMBackendProvider`-UI.

### 7.3 Non-Interactive-Pfad (CI/CD / Headless ohne TTY)

```
$ COGNITHOR_NONINTERACTIVE=1 cognithor

→ Empfehlung wird nach ~/.cognithor/recommendation.json geschrieben
→ Kein config.yaml-Schreibe
→ Cognithor startet mit Default (Ollama oder Cloud-only je nach env)
→ User kann später `cognithor doctor --apply-recommendation` laufen lassen
```

### 7.4 Wizard-Trigger-Bedingungen

Wizard läuft NUR wenn ALLE wahr:
1. `~/.cognithor/.cognithor_initialized` existiert nicht ODER User explizit `cognithor doctor --reconfigure` aufgerufen hat
2. TTY verfügbar (außer Flutter-Mode)
3. `COGNITHOR_NONINTERACTIVE` env-var nicht gesetzt
4. Cognithor wird **nicht** als Service gestartet (`--service` flag → wizard skip)

---

## 8. Config-Write Semantics

### 8.1 Idempotente Merge-Strategie

```python
def write_config_from_recommendation(
    recommendation: Recommendation,
    existing_yaml: dict | None,
) -> dict:
    """Merge mit Vorsicht. User-Overrides niemals überschreiben."""
    cfg = existing_yaml or {}

    # Backend-Type — nur setzen wenn nicht explizit gesetzt
    if "llm_backend_type" not in cfg or _is_default_value(cfg["llm_backend_type"]):
        cfg["llm_backend_type"] = recommendation.primary_tier.backend_type

    # Modelle — nur setzen wenn nicht explizit gesetzt
    cfg.setdefault("models", {})
    for role, model_name in recommendation.primary_tier.models_per_backend[
        recommendation.primary_tier.backend_type
    ].as_dict().items():
        cfg["models"].setdefault(role, {})
        if "name" not in cfg["models"][role]:
            cfg["models"][role]["name"] = model_name

    # Hardware-Profile-Hash für Re-Detection
    cfg["__system_profile_hash"] = _hash_profile(recommendation.profile_snapshot)
    cfg["__recommended_tier"] = recommendation.primary_tier.name
    cfg["__recommended_at"] = datetime.now(UTC).isoformat()

    return cfg
```

### 8.2 Backup vor Schreibe

`~/.cognithor/config.yaml` → `~/.cognithor/config.yaml.backup-{timestamp}`
vor jedem Wizard-Schreibe-Vorgang. Letzte 5 Backups werden behalten,
ältere gelöscht.

### 8.3 Atomic-Write

`config.yaml.tmp` schreiben → `os.replace()` → atomic.
Bei Schreibfehler: tmp löschen, alte config bleibt unangetastet.

---

## 9. Re-Detection-Triggers

### 9.1 Hardware-Hash-Mismatch beim Boot

```python
def _check_hardware_drift_on_boot(cfg: CognithorConfig) -> None:
    stored_hash = cfg.__dict__.get("__system_profile_hash")
    current_profile = SystemDetector().run_quick_scan()
    current_hash = _hash_profile(current_profile)
    if stored_hash and stored_hash != current_hash:
        log.warning(
            "hardware_drift_detected",
            hint="Hardware seit letztem Setup geändert. "
                 "Optional: 'cognithor doctor --reconfigure' für aktualisierte Empfehlungen.",
        )
```

**Niemals** auto-reconfigure. Nur Hint loggen + Flutter-Banner anzeigen.

### 9.2 Manuelle Re-Detection

`cognithor doctor` (CLI):
- Ohne Flag: nur Detection + Anzeige, kein Schreibe.
- `--reconfigure`: vollen Wizard erneut durchlaufen (mit Backup).
- `--apply-recommendation`: non-interactive, übernimmt die Top-1-Empfehlung mit Backup.
- `--export-profile`: schreibt `~/.cognithor/system_profile.json` für Support-Bug-Reports.

### 9.3 Driver-Update-Hint

Wenn `current_driver_version > stored_driver_version` UND ein höherer
Tier dadurch erreichbar wird → Hint: "Driver-Update detected — neue
Empfehlung verfügbar (`cognithor doctor`)".

---

## 10. Backwards-Compat

### 10.1 Bestehende Installs

User mit existierendem `~/.cognithor/.cognithor_initialized`:
- Wizard läuft NICHT automatisch.
- Beim nächsten `cognithor`-Start: KEIN sichtbares Verhalten-Change.
- Optional: ein einmaliger Hint-Banner ("Hardware-Tier-Empfehlung verfügbar — `cognithor doctor`").

### 10.2 Hint-Banner-Cooldown

Banner wird **maximal 1× pro 30 Tage** angezeigt, dann verstummt.
User kann via `cognithor doctor --silence-banner` permanent stummschalten.

### 10.3 Rollback

Wenn Wizard-Apply fehlschlägt (Pull bricht ab, Docker startet nicht):
- config.yaml.backup wird wiederhergestellt
- `.cognithor_initialized` wird NICHT geschrieben (wizard läuft beim nächsten Start erneut)

### 10.4 v0.99.0-User die schon vLLM manuell nutzen

Wenn config.yaml `llm_backend_type: vllm` UND VLM-Quality-Default
aus heutigem Sprint bereits gesetzt ist → Wizard erkennt das +
schlägt vor "Du nutzt schon vLLM. Modelle modernisieren?" anstatt
Backend zu wechseln.

---

## 11. Test-Strategy

### 11.1 Unit-Tests (`tests/test_system/test_recommender.py`)

- `match_tier()`: pro Tier-Definition mind. 3 Profile-Szenarien (matched, near-match, no-match).
- `Recommendation`-Construction für jedes Tier.
- Idempotente Config-Merge (User-Overrides bleiben).
- Hardware-Profile-Hash deterministisch.
- Negative-Tests: Driver < 596 + sm120 → kein NVFP4-Tier.

### 11.2 Integration-Tests (`tests/test_first_boot/`)

- Mock-`SystemDetector` mit RTX 5090-Profil → Wizard wählt enterprise-vllm-nvfp4.
- Mock-Profil "M3 Mac" → mac-ollama-metal.
- Mock-Profil "no GPU, 8 GB RAM" → minimal-ollama-cpu.
- Wizard-Re-Run schlägt fehl wenn `.cognithor_initialized` da → Hint-Pfad.

### 11.3 E2E-Tests (Playwright)

- Flutter-Onboarding-Screen: Detection-Spinner → Karten-Anzeige → Tier-Auswahl → config.yaml-Schreibe.
- Cancel-Flow: Wizard abbrechen → kein Schreibe.
- Re-Run-Flow: `cognithor doctor` → Detection-Karte zeigt Diff zur letzten Detection.

### 11.4 Property-Tests (Hypothesis)

- `forall valid SystemProfile: match_tier(profile) liefert mindestens 1 Tier`.
- `forall existing config: write_config_from_recommendation respektiert User-Overrides`.

### 11.5 Manuelle Hardware-Matrix (vor Release)

| Hardware | Erwartetes Tier |
|---|---|
| RTX 5090 + Win11 + Docker | enterprise-vllm-nvfp4 |
| RTX 5090 + Linux native | enterprise-vllm-nvfp4 |
| RTX 4090 + Win11 + Docker | power-vllm-fp8 |
| RTX 3090 + Linux | power-vllm-fp8 (FP8-fähig sm86) |
| RTX 3060 12 GB + Win | standard-ollama-cuda |
| GTX 1080 Ti + Linux | standard-ollama-cuda (kein FP8/NVFP4) |
| MacBook Pro M3 Max | mac-ollama-metal |
| Ryzen 5950X + RX 7900 XTX | amd-ollama-rocm |
| Intel NUC + Arc A770 | minimal-ollama-cpu (Intel Arc nicht v1) |
| ThinkPad ohne GPU 16 GB | minimal-ollama-cpu |
| Cloud-VM ohne GPU | cloud-only |

---

## 12. Configuration

### 12.1 Neue Config-Felder

```python
class FirstRunConfig(BaseModel):
    enabled: bool = True
    show_banner_after_drift: bool = True
    banner_cooldown_days: int = 30
    silenced_until: datetime | None = None
    auto_apply_top_tier: bool = False  # für CI/CD-Automation
    download_models_on_apply: bool = False  # gefährlich, default off
```

### 12.2 Env-Vars

| Var | Effekt |
|---|---|
| `COGNITHOR_NONINTERACTIVE` | Wizard skip, schreibe `recommendation.json` |
| `COGNITHOR_FORCE_TIER` | Override Tier-Auswahl (für Testing) |
| `COGNITHOR_SKIP_DETECTION` | Detection skippen, alles als "minimal" behandeln |
| `COGNITHOR_DETECTION_TIMEOUT_S` | Per-Detector-Timeout (default 10) |

---

## 13. Rollout-Plan

### Phase 0 — Code merged, Wizard hinter Flag

- Feature-Flag `first_run.enabled=False` per Default.
- Wizard ist da, lässt sich via `cognithor doctor --reconfigure` triggern.
- Kein Auto-Trigger-Verhalten.

### Phase 1 — Auto-Trigger für FRISCHE Installs

- `pip install cognithor` ohne `~/.cognithor/` → Wizard läuft auto.
- Bestehende User unverändert (siehe §10.1).
- 14-Tage-Telemetrie (opt-in via Bug-Report) auf Tier-Misclassification.

### Phase 2 — Drift-Banner

- Bestehende User sehen den Banner nach Driver-Update / Hardware-Wechsel.
- Cooldown 30d.

### Phase 3 — Default-Tier-Update

- TIERS_V1 → TIERS_V2 mit z.B. einem Qwen4.0-Tier wenn verfügbar.
- Drift-Banner triggert bei TIER_VERSION-Bump auch ohne Hardware-Change.

### Rollback an JEDEM Übergang

- Feature-Flag flippen → Wizard sofort aus.
- Detection bleibt aktiv (low-cost), Recommendation läuft nicht.

---

## 14. PR-Aufteilung

| PR | Inhalt | Est. LOC |
|---|---|---:|
| **PR-A** | `detector.py`-Erweiterungen (CUDA, sm-cap, Docker, WSL2, in-container) | ~250 |
| **PR-B** | `recommender.py` neu (HardwareTier, TIERS_V1, match_tier, Recommendation) | ~400 |
| **PR-C** | `wizard.py` neu (CLI-Flow, ANSI-rendering, prompt-helpers) | ~350 |
| **PR-D** | `cognithor doctor` CLI-Subcommands | ~200 |
| **PR-E** | Config-Merge + Atomic-Write + Backup-Rotation | ~200 |
| **PR-F** | first_boot.py-Integration (Wizard-Trigger-Logik) | ~150 |
| **PR-G** | Hardware-Hash + Drift-Banner + Cooldown | ~200 |
| **PR-H** | Flutter `hardware_wizard_screen.dart` + `OnboardingProvider` | ~600 |
| **PR-I** | Unit-Tests (~80) + Integration-Tests (~25) + Property-Tests (~10) | ~1 200 |
| **PR-J** | Hardware-Matrix-Test-Skripte (mock profiles) | ~300 |
| **PR-K** | Doku: README "Hardware-Aware First-Run", FIRST_BOOT.md update | ~150 |

Gesamt: 11 PRs, ~4 000 LOC, Sprint-Länge ~3 Wochen.

---

## 15. Open Questions

1. **Q1.** Sollen wir Modelle **automatisch** beim Wizard-Apply herunterladen,
   oder bei erster Verwendung pull-on-demand?
   *Recommendation: pull-on-demand mit klarer Progress-Anzeige im Chat-UI.
   Auto-Download bei Wizard-Apply als opt-in (`--download-models`-Flag).*

2. **Q2.** Sollte `cognithor doctor` auch **Performance-Bench** machen
   (kurzer Token-Throughput-Test) um Tier-Empfehlung zu validieren?
   *Recommendation: Sprint-2-Erweiterung. v1: nur Detection-Match.*

3. **Q3.** Wie umgehen mit User die Cognithor in einem Container betreiben,
   aber Host-GPU durchgereicht haben? Detection sieht GPU, aber WSL2-Detection ist N/A.
   *Recommendation: Container-Mode-Flag + dedizierte Tier-Suche
   `enterprise-vllm-nvfp4-container`. Sprint-2.*

4. **Q4.** GitHub-Install via `git clone … && pip install -e .` läuft
   denselben Wizard?
   *Yes. `pip install`-Path ist gleich. Marker = `.cognithor_initialized`.*

5. **Q5.** Soll Wizard auch beim **Update** (`pip install --upgrade`) laufen?
   *Recommendation: nicht automatisch. Update-Flow kann aber `cognithor doctor`
   in Release-Notes empfehlen wenn TIER_VERSION-Bump.*

---

## 16. Acceptance-Kriterien

- [ ] Owner-Approval auf §6 (Tier-Definitionen) und §7 (Wizard-Flow).
- [ ] Hardware-Matrix (§11.5) mind. 6 Konfigurationen verifiziert (mocked).
- [ ] Alle 24 Edge-Cases (§3) als Test-Case oder explizite Notiz im Code.
- [ ] Backup-Rotation getestet auf 100-Run-Loop ohne Disk-Leak.

---

## 17. References

- `src/cognithor/system/detector.py` — bestehende Detection (re-use).
- `scripts/first_boot.py` — bestehender Boot-Flow (extend).
- `src/cognithor/core/vllm_orchestrator.py` — vLLM-Container-Steuerung.
- `docs/superpowers/specs/2026-05-07-fast-path-router-spec.md` — Modell-Empfehlungen.
- `docs/FIRST_BOOT.md` — User-facing Doku (muss erweitert werden).
- `docs/CONFIG_REFERENCE.md` — Config-Schema-Doku.
