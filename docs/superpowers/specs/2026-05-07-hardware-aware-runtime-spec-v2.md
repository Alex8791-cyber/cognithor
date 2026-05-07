# Hardware-Aware Runtime — Endgültige Architektur (v2)

> **Status:** Spec v2.0 · **Date:** 2026-05-07 · **Owner:** Alexander Söllner
> **Ersetzt:** `2026-05-07-hardware-aware-first-run.md` (v1, war MVP-Spec).
> **Ziel:** *Diese Logik wird nicht mehr angefasst.* Modell-Updates,
> neue Hardware-Generationen, neue Backends — alles via externe
> versionierte Daten ohne Code-Change möglich.
> **Risk-Class:** ARCHITECTURAL — fundamentale Cognithor-Boot-Schicht.

## 0. Kern-Idee in einem Satz

**Cognithor bekommt einen 12-Layer-Stack der die Hardware kennt, die Möglichkeiten daraus ableitet, gegen ein extern-signiertes Modell-Manifest matched, dem User transparent die Pareto-Optimalen Konfigurationen zeigt, atomar schreibt, im Betrieb permanent verifiziert und sich selbst heilt.**

---

## 1. Warum v1 nicht reichte

Die v1-Spec deckte First-Boot + Tier-Empfehlung ab. **Was sie nicht löste:**

| Lücke v1 | Konsequenz | Lösung in v2 |
|---|---|---|
| Tier-Definitionen hartcodiert in Python | Jedes neue Modell → Code-PR + Release | **Layer 3:** External Model-Manifest (versioned, signed) |
| Keine Runtime-Adaptation | Nach Wizard-Apply: Cognithor merkt nicht wenn Modell schlechter performed als erwartet | **Layer 7:** Performance-Telemetry + Auto-Drift-Detection |
| Keine Capability-Abstraktion | Neue GPU-Architektur (z.B. Rubin nach Blackwell) → Code-Change in Tier-Match-Logik | **Layer 2:** Capability-Flags als stabile Abstraktion |
| Keine Model-Availability-Verifikation | HF-Repo gelöscht / Ollama-Pull-Fehler → silent broken state | **Layer 8:** Health-Checks + Fallback-Cascade |
| Keine Solver-Logik (multi-objective) | "Beste" Tier per hartcodierter Reihenfolge — nicht user-objective-aware | **Layer 4:** Constraint-Solver + Pareto-Frontier |
| Keine Concurrent-Safety | Zwei `cognithor doctor` parallel → config-corruption | **Layer 6:** File-Lock + atomic-write + schema-version |
| Keine Adversarial-Robustness | nvidia-smi liefert garbage → Wizard schreibt falsche Config | **Layer 1.5:** Sanity-Checks + Cross-Validation |
| Keine TRUST-Integration für Recommender | Recommendation = Black-Box | **Layer 10:** TRUST-1 Receipt-able, TRUST-2 explainable |
| Keine Manifest-Update-Strategie | Tiers veralten | **Layer 11:** Manifest-Refresh + Version-Pinning |

Die v2 fasst all diese Lücken in einer 12-Layer-Architektur zusammen.

---

## 2. Die 12 Layer — Übersicht

```
┌─────────────────────────────────────────────────────────────────────┐
│  L12 — TRUST-Integration (audit-trail jeder Decision)               │
├─────────────────────────────────────────────────────────────────────┤
│  L11 — Manifest-Update-Mechanism (signed, versioned, refreshable)   │
├─────────────────────────────────────────────────────────────────────┤
│  L10 — Cost-Awareness (€/MTok, Cloud-vs-Local Trade-off)            │
├─────────────────────────────────────────────────────────────────────┤
│  L9 — Health & Monitoring (per-Komponente, drift-alerts)            │
├─────────────────────────────────────────────────────────────────────┤
│  L8 — Adversarial Robustness (sanity-checks, fallback-cascade)      │
├─────────────────────────────────────────────────────────────────────┤
│  L7 — Runtime-Adaptation (perf-telemetry, auto re-recommend)        │
├─────────────────────────────────────────────────────────────────────┤
│  L6 — Apply-Engine (atomic, lock, backup, rollback)                 │
├─────────────────────────────────────────────────────────────────────┤
│  L5 — Wizard / UI (CLI + Flutter, idempotent, resumable)            │
├─────────────────────────────────────────────────────────────────────┤
│  L4 — Solver (multi-objective, Pareto, deterministisch)             │
├─────────────────────────────────────────────────────────────────────┤
│  L3 — Model-Manifest (external, signed, versioned)                  │
├─────────────────────────────────────────────────────────────────────┤
│  L2 — Capability-Mapping (raw detection → stable flags)             │
├─────────────────────────────────────────────────────────────────────┤
│  L1.5 — Cross-Validation (sanity-check Detection-Output)            │
├─────────────────────────────────────────────────────────────────────┤
│  L1 — Hardware-Detection (raw probes, multi-vendor, multi-GPU)      │
└─────────────────────────────────────────────────────────────────────┘
```

Jedes Layer ist **isoliert testbar** und hat **klare Outputs**, die das nächst-höhere konsumiert. Keine Layer-Skips. Erweiterung passiert nur per **neue Daten** (L3-Manifest) oder **neuer Capability-Flag** (L2), niemals per Logik-Änderung in L4–L10.

---

## 3. Threat Model — exhaustiv (47 Threats)

### 3.1 Detection-Threats (L1)

| # | Threat | Severity | Layer | Mitigation |
|--|--------|---------:|------|-----------|
| T1 | nvidia-smi crashed (Driver-Update kaputt) | HIGH | L1 | Timeout 10 s, exit-code prüfen, NIEMALS hängen, Fallback auf "no_gpu" + WARN |
| T2 | nvidia-smi liefert mit Komma-Locale (DE-Win) | MEDIUM | L1 | `--format=csv,noheader,nounits` + `replace(",", ".")` |
| T3 | Mehrere GPUs (Multi-GPU-Box) | MEDIUM | L1 | Enumerate alle, Pool-VRAM + Master-GPU, Multi-GPU-Setup als L3-Tier-Flag |
| T4 | NVIDIA + AMD im selben System (heterogen) | LOW | L1 | Beide enumerieren, Solver wählt eine GPU für Inferenz, andere für Display |
| T5 | Apple M3 Max Unified Memory ≠ VRAM | HIGH | L1 | `unified_memory=True`-Flag + spezielle Sizing-Logik in L4 |
| T6 | AMD ROCm vs ROCm-disabled | HIGH | L1 | `rocm-smi` + `rocminfo` cross-check |
| T7 | Intel Arc / oneAPI | LOW | L1 | erkennen + "experimental" mark, kein NVFP4-Pfad |
| T8 | Pre-Ampere NVIDIA (Pascal/Volta, sm < 8.0) | HIGH | L1+L2 | `compute_capability` ableiten, NVFP4-Capability=False, FP8-Capability=False |
| T9 | Driver < Mindestversion für Compute-Capability | HIGH | L2 | Driver-Constraint im Capability-Check, Auto-Downgrade Tier |
| T10 | Cognithor läuft im Container ohne `--gpus all` | HIGH | L1 | `/.dockerenv` + `nvidia-smi` cross-check, "host-gpu sichtbar?" capability-flag |
| T11 | nvidia-smi installed aber GPU disabled (BIOS) | MEDIUM | L1 | `nvidia-smi --query-gpu=name --format=...` exit-code 0 ABER empty output → handle |
| T12 | WSL2 mit GPU-Passthrough kaputt | HIGH | L1 | In WSL: `nvidia-smi` muss laufen UND `/dev/dxg` existieren |
| T13 | Docker installiert aber Daemon down | MEDIUM | L1 | `docker info` mit Timeout 5 s, kein hängen |
| T14 | Hardware-Profil-Cache stale (Hardware geändert) | MEDIUM | L1 | Hash-Drift-Detection bei jedem Boot (siehe L7) |
| T15 | KVM-virtualized CPU lügt über Cores/Features | LOW | L1.5 | Plausibility-Check (kein 256-Core-CPU mit 8 GB RAM erwartbar) |
| T16 | RAM-Detection unter Hyper-V meldet Host-RAM | MEDIUM | L1.5 | psutil + WMI cross-check auf Windows |

### 3.2 Capability-Mapping-Threats (L2)

| # | Threat | Severity | Mitigation |
|--|--------|---------:|------------|
| T17 | Compute-Capability `12.0` (Blackwell) interpretation in capability-graph | HIGH | Lookup-Table mit ALL bekannten sm-Versionen + Forward-Compat-Default ("unknown sm > 12.x = treat as 12.0 baseline") |
| T18 | NVFP4 braucht NICHT NUR sm120 sondern auch Driver ≥ 596 UND CUDA-Toolkit ≥ 13.0 | HIGH | Capability `can_run_nvfp4` = AND von 3 Conditions, dokumentiert |
| T19 | FP8 hardware-bedingt verfügbar aber vLLM-Build hat kein FP8 | HIGH | Capability `can_run_fp8_in_vllm` = `can_run_fp8` AND `vllm_version ≥ X.Y.Z` |
| T20 | GGUF-Q4 läuft auf jedem GPU aber Ollama hat den Spezial-Quant nicht | LOW | Per-Backend-Capability separat tracken |

### 3.3 Manifest-Threats (L3)

| # | Threat | Severity | Mitigation |
|--|--------|---------:|------------|
| T21 | Manifest-Update injiziert bösen Modell-Identifier | CRITICAL | TUF-Light-Signing (PACK-4-Pattern), Pinned Root-Key, kein blind-trust |
| T22 | Manifest-Repository down (GitHub raw nicht erreichbar) | MEDIUM | Lokales Cache von letzter funktionierender Version, fall-back |
| T23 | Manifest-Schema-Bump (v2 → v3) bricht alte Cognithor-Installation | HIGH | Manifest-Schema-Version pinned pro Cognithor-Release, Forward-Compat-Mappings |
| T24 | Modell-Eintrag im Manifest verweist auf gelöschtes HF-Repo | HIGH | L8 Health-Check verifiziert vor Apply |
| T25 | Manifest-Tier-Version != Cognithor-Version (z.B. neuer Tier braucht neue vLLM-API) | MEDIUM | Manifest-Eintrag enthält `requires_cognithor: ">=0.99.0"` |

### 3.4 Solver-Threats (L4)

| # | Threat | Severity | Mitigation |
|--|--------|---------:|------------|
| T26 | Solver findet keine Lösung (Hardware unter allen Mindestschwellen) | MEDIUM | "cloud-only"-Tier als Catch-All, niemals leeres Result |
| T27 | Mehrere Lösungen mit identischem Score → nicht-deterministisch | HIGH | Tie-Breaker-Reihenfolge (lexikographisch über Tier-Name), reproducible |
| T28 | User-Objective konfliktiert (z.B. "max quality + min disk + min latency") | LOW | Pareto-Frontier zeigen, User wählt aus den nicht-dominierten Punkten |
| T29 | Solver-Performance: bei 100+ Models × 8 Capabilities = quadratisch teuer | LOW | <50 ms bei 200 Modellen via Pre-Filter (capabilities-bitmask) |

### 3.5 Wizard-Threats (L5)

| # | Threat | Severity | Mitigation |
|--|--------|---------:|------------|
| T30 | TTY nicht verfügbar (Headless / SSH ohne -t) | MEDIUM | Auto-detect, falls non-tty → schreibe `recommendation.json`, kein Crash |
| T31 | User cancelt mid-wizard | LOW | Idempotent: `.cognithor_initialized` wird erst NACH erfolgreicher Apply gesetzt |
| T32 | User cancelt während Modell-Pull | LOW | Pull-Resume via HF-`.incomplete`, kein Daten-Verlust |
| T33 | Wizard läuft parallel in zwei Terminals | HIGH | File-Lock auf `~/.cognithor/.wizard.lock` (siehe L6) |
| T34 | Flutter-Wizard und CLI-Wizard laufen gleichzeitig | HIGH | Same File-Lock |

### 3.6 Apply-Engine-Threats (L6)

| # | Threat | Severity | Mitigation |
|--|--------|---------:|------------|
| T35 | Atomic-Write fail mid-flight (disk full / permissions) | MEDIUM | tmp-File schreiben + `os.replace()`, bei Fehler tmp löschen, alte config bleibt |
| T36 | Backup-Rotation füllt Disk | LOW | Max 5 Backups, älteste löschen |
| T37 | Schema-Migration v0.99 → v1.0 ändert Feld-Pfade | HIGH | Per-Schema-Version-Migrator, idempotent, dry-run-able |

### 3.7 Runtime-Adaptation-Threats (L7)

| # | Threat | Severity | Mitigation |
|--|--------|---------:|------------|
| T38 | Performance-Telemetry triggert ständige Re-Recommendation-Banner (annoying) | HIGH | Hysteresis: erst nach 3× consecutive p95-Verschlechterung > 2× Erwartung Banner anzeigen, Cooldown 30 d |
| T39 | Hardware-Hash drift-Detection FALSE POSITIVE (Driver-Update mit selben Specs ändert Hash) | MEDIUM | Hash über STRUCT, nicht über raw String — Driver-Patch-Version excluded |
| T40 | Modell wird zwischen Apply und nächstem Boot zurückgezogen | HIGH | L8 Health-Check beim Boot, Fallback-Cascade |

### 3.8 Cost / TRUST / Misc

| # | Threat | Severity | Mitigation |
|--|--------|---------:|------------|
| T41 | Hardware-Profil leakt PII (hostname, user) | HIGH | Profil enthält nur **Klassen** (nicht IDs), kein Hostname/Username |
| T42 | Manifest-URL überträgt unverschlüsselt (HTTPS-Downgrade) | HIGH | TLS-Pinning + HSTS |
| T43 | Recommendation als Audit-Event nicht im TRUST-1-Receipt | MEDIUM | Hard wired (L12) |
| T44 | Cost-Estimate veraltet (Cloud-API-Preise ändern sich) | LOW | Manifest enthält Preise pro Modell, mit `valid_until` |
| T45 | Crash-Loop wenn Wizard auf jedem Boot fail-und-retry | CRITICAL | Wizard läuft max 1× pro 24 h auch bei Fehler, sonst Skip + log-only |
| T46 | Update-Manifest-Refresh erfasst feindliches Manifest | CRITICAL | TUF-Light + Two-Key (Root + Targets), Recall-Mechanism |
| T47 | Hardware-Detection-Time blockiert Boot > 30 s | MEDIUM | Total-Budget 12 s, Per-Probe 5 s, Async-Parallel-Probes |

---

## 4. Layer 1 — Hardware-Detection (raw probes)

### 4.1 Datenmodell

```python
# src/cognithor/system/probes/types.py
@dataclass(frozen=True)
class GpuProbe:
    vendor: Literal["nvidia", "amd", "intel", "apple", "none"]
    model: str
    vram_total_mib: int
    vram_free_mib: int
    driver_version: str
    cuda_version: str | None
    compute_capability: str | None      # "12.0", "8.9", "8.0", …
    architecture: str | None             # "blackwell", "ada", "ampere", "turing"
    pci_id: str | None                   # für Multi-GPU Disambiguierung
    uuid: str | None                     # GPU-stabile ID

@dataclass(frozen=True)
class HostProbe:
    os_name: str                         # "Windows", "Linux", "Darwin"
    os_version: str
    arch: str                            # "x86_64", "arm64"
    is_wsl2: bool
    wsl2_distro: str | None
    is_container: bool                   # /.dockerenv check
    container_runtime: str | None        # "docker", "containerd", "podman"
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_features: tuple[str, ...]        # ["avx2","avx512","f16c","bf16"]
    ram_total_mib: int
    ram_available_mib: int
    swap_total_mib: int
    disk_free_mib: int                   # Pfad: ~/.cognithor/

@dataclass(frozen=True)
class SoftwareProbe:
    docker: ProbeResult                  # version, running
    docker_compose: ProbeResult
    nvidia_container_runtime: ProbeResult
    rocm: ProbeResult
    ollama: ProbeResult
    lmstudio: ProbeResult
    vllm_local: ProbeResult              # "pip install vllm" oder Container?
    huggingface_hub: ProbeResult
    git_lfs: ProbeResult
    python_version: ProbeResult

@dataclass(frozen=True)
class NetworkProbe:
    has_internet: bool
    can_reach_huggingface: bool
    can_reach_pypi: bool
    can_reach_dockerhub: bool
    can_reach_github: bool
    bandwidth_class: Literal["unknown", "low", "medium", "high"]  # ping + speedtest
    is_metered: bool                     # via OS-API wo verfügbar

@dataclass(frozen=True)
class HardwareProfile:
    schema_version: int = 2
    detected_at_utc: str
    duration_ms: int
    gpus: tuple[GpuProbe, ...]           # Tuple, nicht List — immutable
    host: HostProbe
    software: SoftwareProbe
    network: NetworkProbe
    profile_hash: str                    # SHA-256, exclud. timestamps
```

### 4.2 Probe-Module

Jeder Probe in eigenem Modul, einheitliches Interface:

```python
# src/cognithor/system/probes/__init__.py
class Probe(Protocol):
    name: str
    timeout_s: float

    async def run(self) -> ProbeResult: ...

# Module:
# - probes/nvidia.py        — nvidia-smi parsing
# - probes/amd.py           — rocm-smi
# - probes/apple.py         — sysctl, system_profiler SPDisplaysDataType
# - probes/intel.py         — sycl-ls, intel_gpu_top
# - probes/cpu.py           — psutil + cpuinfo
# - probes/ram.py           — psutil
# - probes/disk.py          — shutil.disk_usage
# - probes/docker.py        — docker info
# - probes/wsl.py           — wsl --status
# - probes/container.py     — /.dockerenv, /proc/1/cgroup
# - probes/network.py       — DNS + curl tests
# - probes/ollama.py        — http://localhost:11434/api/version
# - probes/vllm.py          — http://localhost:8000/v1/models + pip show vllm
# - probes/lmstudio.py      — port-probe
# - probes/huggingface.py   — token-test (falls vorhanden)
```

### 4.3 Konkurrierender Probe-Run

```python
async def run_full_scan(
    *,
    timeout_s: float = 12.0,
    per_probe_timeout_s: float = 5.0,
    use_cache_if_younger_than_s: float = 86400,
) -> HardwareProfile:
    """Führt alle Probes parallel aus, mit Total-Budget."""
    cache_path = Path.home() / ".cognithor" / ".system_profile.json"
    if cached := _load_cache_if_fresh(cache_path, use_cache_if_younger_than_s):
        return cached

    probes = _load_probe_registry()
    async with asyncio.timeout(timeout_s):
        results = await asyncio.gather(
            *[_run_with_per_probe_timeout(p, per_probe_timeout_s) for p in probes],
            return_exceptions=True,
        )
    profile = _merge_into_profile(results)
    _persist_cache(profile, cache_path)
    return profile
```

**Edge-Case T47:** Wenn Total-Timeout zuschlägt: Partial-Profile zurückgeben mit `partial=True`-Flag, niemals crashen.

### 4.4 Persistenz

Cache: `~/.cognithor/.system_profile.json` (signed mit local HMAC-Key gegen Tampering-by-User).
Für Debugging: `cognithor doctor --export-profile` schreibt anonymisierte Version.

---

## 5. Layer 1.5 — Cross-Validation

Bevor das Profil L2 erreicht, läuft eine Sanity-Check-Pipeline:

```python
SANITY_RULES: tuple[SanityRule, ...] = (
    SanityRule(
        name="vram_inside_gpu_class",
        check=lambda p: all(g.vram_total_mib <= 200_000 for g in p.gpus),
        on_fail="cap_gpu_vram_at_200gb_log_warn",
    ),
    SanityRule(
        name="ram_plausible_vs_cpu",
        check=lambda p: not (p.host.cpu_cores_physical > 64 and p.host.ram_total_mib < 16_000),
        on_fail="downgrade_cpu_class_log_warn",  # Wahrscheinlich virt-lie
    ),
    SanityRule(
        name="driver_present_if_gpu_present",
        check=lambda p: not (p.gpus and not p.gpus[0].driver_version),
        on_fail="treat_gpu_as_disabled",
    ),
    SanityRule(
        name="cuda_version_matches_compute_cap",
        check=lambda p: _cuda_supports_sm(p.gpus[0]) if p.gpus else True,
        on_fail="downgrade_cuda_class",
    ),
    # … 12 weitere
)
```

Jede Regel ist **Pure Function** (`HardwareProfile -> bool`). Bei Fail: `on_fail` wird in Audit geloggt und Profile-Mutation angewandt. Profile bleibt strukturell gültig.

---

## 6. Layer 2 — Capability-Mapping

### 6.1 Capability-Flags (stable abstraction)

```python
# src/cognithor/system/capabilities.py
@dataclass(frozen=True)
class Capabilities:
    schema_version: int = 2

    # Compute
    can_run_nvfp4: bool                 # sm120+ + drv≥596 + cuda≥13
    can_run_fp8_marlin: bool            # sm89+ + cuda≥12
    can_run_fp8_native: bool            # sm89+ + cuda≥12.4
    can_run_gptq_int4: bool             # sm70+
    can_run_awq_int4: bool              # sm70+
    can_run_bnb_int8: bool              # sm70+
    can_run_gguf_cuda: bool             # sm60+
    can_run_gguf_metal: bool            # apple silicon
    can_run_gguf_rocm: bool             # AMD ROCm
    can_run_gguf_cpu: bool              # immer

    # Backends (per Software-Verfügbarkeit)
    can_run_vllm_container: bool        # docker-running + nvidia-runtime
    can_run_vllm_inprocess: bool        # pip vllm + GPU-cap
    can_run_ollama_native: bool
    can_run_lmstudio: bool
    can_run_llama_cpp: bool

    # Speicher-Klassen (für Tier-Match)
    vram_class: Literal["none", "tiny", "small", "medium", "large", "xlarge", "xxlarge"]
    # none=0, tiny<4, small<8, medium<16, large<24, xlarge<48, xxlarge≥48
    ram_class: Literal["low", "medium", "high", "extreme"]
    # low<16, medium<32, high<64, extreme≥64

    # Multi-GPU
    has_multi_gpu_homogeneous: bool
    has_multi_gpu_heterogeneous: bool
    aggregate_vram_class: Literal["none", "tiny", "small", "medium", "large", "xlarge", "xxlarge"]

    # Network
    has_internet: bool
    has_huggingface_access: bool
    is_offline_only: bool
    is_metered_connection: bool

    # Storage
    disk_class: Literal["very_low", "low", "medium", "high"]  # <30/<80/<200/≥200 GB free

    # Container/Sandbox
    is_in_container: bool
    can_reach_host_gpu: bool
```

### 6.2 Mapping-Function

```python
def map_to_capabilities(profile: HardwareProfile) -> Capabilities:
    """Pure function. Deterministisch."""
    nvfp4 = (
        any(g.compute_capability and _sm_at_least(g.compute_capability, "12.0") for g in profile.gpus)
        and any(g.driver_version and _ver_at_least(g.driver_version, "596.0") for g in profile.gpus)
        and (profile.gpus[0].cuda_version is None
             or _ver_at_least(profile.gpus[0].cuda_version, "13.0"))
    )
    # ... weitere Berechnungen
```

**Versioned + frozen** — keine "smart" Re-Computation zur Laufzeit. Eine Capability-Berechnung pro Profile-Hash, gecacht.

---

## 7. Layer 3 — External Model-Manifest

### 7.1 Warum versioniert + signiert?

Damit ein neues Modell-Release **niemals einen Cognithor-Code-Patch** braucht — nur einen YAML-PR an die Manifest. Manifest lebt **im bestehenden cognithor-Repo** unter `manifest/v2/`, signiert (TUF-Light wie PACK-4 — re-use des bestehenden Root-Keys), via `git`/`raw.githubusercontent.com` versioniert. Kein neues Repo.

### 7.2 Repo-Struktur (in bestehender cognithor-Repo)

```
github.com/Alex8791-cyber/cognithor (bestehend)
├── manifest/                       # NEU
│   ├── v2/
│   │   ├── tiers.yaml              # Tier-Definitionen
│   │   ├── models.yaml             # Modell-Catalog
│   │   ├── backends.yaml           # Backend-Capabilities-Mapping
│   │   ├── pricing.yaml            # Cloud-Provider-Preise
│   │   └── manifest.sig            # Ed25519 sig (Targets-Key)
│   ├── v3/                         # Forward-Compat
│   └── recalls/
│       └── active.json             # Recall-Mechanism
└── src/cognithor/
    └── _pinned_keys.py             # bestehend, ergänzt um MANIFEST_TARGETS_KEY
                                     #   (re-use der PACK-4-Root als Trust-Anchor)
```

**Manifest-Updates** = YAML-PR auf `main` + Re-Sign. **Cognithor-Releases sind unabhängig** vom Manifest-Update. Beim Boot werden via Online-Refresh die jeweils neuesten signierten YAMLs gefetcht; bei Offline-Install wird die im Wheel mitgeshipte Version benutzt.

### 7.3 Manifest-Schema (`tiers.yaml`)

```yaml
schema_version: 2
manifest_version: "2026.05.07.01"  # Datum + Index
expires_utc: "2026-08-07T00:00:00Z"  # signed, force-refresh nach Ablauf

tiers:
  - id: "enterprise-vllm-nvfp4-blackwell"
    display_name: "Enterprise vLLM NVFP4 (Blackwell)"
    rationale_de: "RTX 5090/6090 mit cu13+ — NVFP4+MTP liefert ~3-4× Throughput vs Ollama."
    rationale_en: "..."
    requires_capabilities:
      - "can_run_nvfp4"
      - "can_run_vllm_container"
      - "vram_class>=xlarge"        # Vergleichs-Operatoren erlaubt
      - "ram_class>=high"
      - "disk_class>=medium"
    requires_cognithor: ">=1.0.0"
    backend: "vllm"
    backend_config:
      docker_image: "vllm/vllm-openai:cu130-nightly"
      gpu_memory_utilization: 0.94
      enforce_eager: true
      cpu_offload_gb: 4
      max_model_len: 16384
    model_set:
      planner: "qwen3.6-27b-text-nvfp4-mtp"
      executor: "qwen3.5-9b"
      coder: "qwen3.6-27b-text-nvfp4-mtp"
      embedding: "qwen3-embedding-0.6b-ollama"  # hybrid, Ollama-served
      formulate: "qwen3.5-9b"
      fast_path_validator: "qwen3.5-9b"
    estimated_setup_minutes: 15
    estimated_disk_gb: 35
    performance_estimates:
      planner_tok_s_p50: 80
      executor_tok_s_p50: 200
      formulate_tok_s_p50: 200
      first_token_ms_p50: 350

  - id: "power-vllm-fp8-ada"
    # … analog
```

### 7.4 Models-Catalog (`models.yaml`)

```yaml
schema_version: 2
models:
  - id: "qwen3.6-27b-text-nvfp4-mtp"
    display_name: "Qwen3.6 27B Text NVFP4 (MTP)"
    license: "Apache-2.0"
    backend_ids:
      vllm: "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
      huggingface: "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
    requires_capabilities:
      - "can_run_nvfp4"
    disk_gb: 14
    ram_gb_min: 16
    vram_gb_min: 18
    context_length: 262144
    role_fitness:
      planner: 0.95
      executor: 0.85       # overkill für simple tools, aber funktional
      coder: 0.92
      formulate: 0.85
      embedding: 0.0       # nicht für embeddings
      fast_path_validator: 0.5  # zu groß für validator
    quality_tier: "L"      # S/M/L/XL
    speed_tier: "M"        # auf NVFP4-fähiger HW
    deprecated_after_utc: null
    superseded_by: null
    quirks:
      - "needs --enforce-eager on cu130-nightly"

  - id: "qwen3.5-9b"
    # … analog
```

### 7.5 Manifest-Loader

```python
# src/cognithor/system/manifest_loader.py
class ManifestLoader:
    """Lädt + verifiziert das externe Manifest."""

    def __init__(
        self,
        manifest_url: str = "https://raw.githubusercontent.com/Alex8791-cyber/cognithor/main/manifest/v2/tiers.yaml",
        cache_path: Path = Path.home() / ".cognithor" / "manifest_cache" / "v2",
        max_age_s: int = 30 * 86400,
    ):
        ...

    async def load(self, *, force_refresh: bool = False) -> Manifest:
        # 1. Try cache (if fresh)
        # 2. Try network (if cache stale)
        # 3. Fall back to embedded "minimum-viable" manifest in Cognithor (für offline-install)
        ...

    def verify(self, manifest_bytes: bytes, signature_bytes: bytes) -> bool:
        # Ed25519 vs PINNED_ROOT_KEY (in cognithor/_pinned_keys.py)
        ...
```

**Embedded-Fallback:** Cognithor wird mit einem **Minimum-Viable-Manifest** gebundled
(Tier `minimal-ollama-cpu` + 3 Modelle) — damit auch ohne Internet ein Wizard
durchläuft. Klar markiert "embedded, älter als ggf. online verfügbar".

### 7.6 Manifest-Update-Cadence

Default: alle 30 Tage Auto-Refresh (silent, im Hintergrund während `cognithor doctor`).
User-Trigger: `cognithor doctor --refresh-manifest`.
Force-Use einer alten Version: `cognithor doctor --pin-manifest 2026.05.07.01`.

---

## 8. Layer 4 — Solver

### 8.1 Multi-Objective-Constraint-Solver

```python
@dataclass(frozen=True)
class UserObjective:
    weight_quality: float = 0.4         # 0..1
    weight_speed: float = 0.3
    weight_cost: float = 0.2
    weight_privacy: float = 0.1         # 1.0 = strict local-only
    max_disk_gb: float | None = None
    max_setup_minutes: int | None = None
    max_cloud_eur_per_month: float | None = None
    require_offline_capable: bool = False

@dataclass(frozen=True)
class Solution:
    tier_id: str
    score: float
    score_breakdown: dict[str, float]   # quality/speed/cost/privacy components
    blockers: tuple[str, ...]           # capabilities die fehlen, falls leer = sofort startbar
    warnings: tuple[str, ...]
    estimated_first_response_s: float
    estimated_disk_gb: float
    estimated_setup_minutes: int
    estimated_cost_eur_per_month: float

def solve(
    manifest: Manifest,
    capabilities: Capabilities,
    objective: UserObjective,
    *,
    max_solutions: int = 5,
) -> tuple[Solution, ...]:
    """Returnt Pareto-optimale Solutions, sorted by composite-score desc."""
    candidates = [t for t in manifest.tiers if _capabilities_satisfy(capabilities, t.requires_capabilities)]
    pareto = _pareto_filter(candidates, objective)
    scored = sorted([_score(t, objective) for t in pareto], key=lambda s: -s.score)
    return tuple(scored[:max_solutions])
```

**Determinismus:** Tie-Breaker = lexikographisch über `tier_id`. Property-Test: `forall (manifest, caps, obj): solve(...) == solve(...)`.

### 8.2 Pareto-Filter

Eine Lösung ist **dominiert**, wenn eine andere in JEDEM Objective ≥ und in MINDESTENS einem >. Dominierte Lösungen werden nicht angezeigt. Ergebnis ist die **Pareto-Front**.

### 8.3 Default-Objective pro User-Profil

```yaml
# Embedded defaults
default_user_objective:
  weight_quality: 0.4
  weight_speed: 0.3
  weight_cost: 0.2
  weight_privacy: 0.1
```

User kann via Wizard "Profile-Preset" wählen:
- "Geschwindigkeit": Speed=0.6, Quality=0.3, Cost=0.1
- "Privacy": Privacy=1.0, Cost=0, Quality=0
- "Cost-Optimized": Cost=0.7, Quality=0.2, Speed=0.1

Oder Slider in Flutter.

---

## 9. Layer 5 — Wizard / UI

### 9.1 CLI-Wizard-Flow

```
$ cognithor

┌─ Cognithor First-Run ─────────────────────────────────────────────┐
│                                                                    │
│ Detection läuft (≤12s)…                                            │
│                                                                    │
│ ┌─ Hardware-Profil ────────────────────────────────────────────┐  │
│ │ OS: Windows 11 Pro 26200 (WSL2 ✓)                             │  │
│ │ CPU: AMD Ryzen 9 7950X3D · 16 cores · AVX-512                 │  │
│ │ RAM: 64 GB · 41 GB frei                                       │  │
│ │ GPU: NVIDIA GeForce RTX 5090 · 32 GB · sm120 · drv 596.21    │  │
│ │ Docker: läuft (28.0.1) · WSL2-Backend                         │  │
│ │ Disk: 412 GB frei (~/.cognithor)                              │  │
│ │ Internet: ja · HF erreichbar · PyPI erreichbar                │  │
│ └───────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ Capabilities: NVFP4 ✓ · FP8 ✓ · Multi-GPU ✗ · vLLM-Container ✓   │
│                                                                    │
│ Was ist dir wichtig? (Pareto-Optimum wird gezeigt)                │
│  [1] Beste Qualität     (default)                                 │
│  [2] Schnellste Antwortzeit                                       │
│  [3] Maximale Privacy (offline-only)                              │
│  [4] Minimale Disk-Nutzung                                        │
│  [5] Custom (Slider)                                              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
> 1

┌─ Pareto-optimale Konfigurationen für dich ──────────────────────┐
│                                                                  │
│ ★ #1  enterprise-vllm-nvfp4-blackwell                           │
│       Quality 95 │ Speed 90 │ Privacy 100 │ Cost €0/Monat       │
│       Setup ~15min · Disk 35 GB                                  │
│       Planner: Qwen3.6-27B-Text-NVFP4 (200 tok/s)               │
│       Executor: Qwen3.5-9B                                       │
│                                                                  │
│   #2  power-vllm-fp8-ada                                         │
│       Quality 92 │ Speed 70 │ Privacy 100 │ Cost €0             │
│       Setup ~15min · Disk 38 GB                                  │
│                                                                  │
│   #3  cloud-only-anthropic                                       │
│       Quality 100 │ Speed 95 │ Privacy 0 │ Cost ~€20/Monat      │
│       Setup ~3min · Disk 0 GB                                    │
│                                                                  │
│ [1/2/3] Wählen  [d] Details  [m] Manuell  [s] Später entscheiden│
└──────────────────────────────────────────────────────────────────┘
> 1
```

### 9.2 Flutter-Wizard

`flutter_app/lib/screens/onboarding/`:
- `detection_screen.dart` — Spinner + Live-Detection-Steps
- `objective_screen.dart` — 4 Preset-Cards + Custom-Sliders
- `solutions_screen.dart` — Karten-Carousel mit Pareto-Lösungen
- `apply_screen.dart` — Progress (Docker pull, Modell pull) + Cancel-Button
- `done_screen.dart` — "Du bist startklar" + Quick-Tour-Vorschlag

### 9.3 "Später entscheiden"

Schreibt nichts, markiert nur `~/.cognithor/.first_run_deferred` mit Datum.
Nach 7 Tagen: bei nächstem Start sanfter Reminder. Nach 30: Auto-Apply
des "minimal-ollama-cpu"-Tiers (sicherer Default).

---

## 10. Layer 6 — Apply-Engine

### 10.1 Apply-Pipeline

```python
async def apply(
    solution: Solution,
    *,
    user_confirmed: bool,
    download_models: bool = False,
    config_path: Path = Path.home() / ".cognithor" / "config.yaml",
) -> ApplyResult:
    """Idempotent + atomic + rollback-fähig."""
    if not user_confirmed:
        raise ValueError("apply requires explicit user confirmation")

    async with _file_lock(Path.home() / ".cognithor" / ".wizard.lock", timeout_s=5):
        # 1. Backup current config
        backup_path = _rotate_backup(config_path)

        try:
            # 2. Schema-migrate if needed
            current = _load_yaml(config_path) if config_path.exists() else {}
            current = _migrate_schema_to_latest(current)

            # 3. Merge (User-Overrides bleiben!)
            merged = _merge_solution_into_config(current, solution)

            # 4. Validate via Pydantic
            CognithorConfig.model_validate(merged)

            # 5. Atomic write
            _atomic_write_yaml(config_path, merged)

            # 6. Optional: pre-pull models
            if download_models:
                await _pre_pull_models(solution)

            # 7. Mark initialized
            _write_initialized_marker(solution, hardware_profile_hash)

            # 8. Audit-log
            _write_audit_event("apply.success", solution=solution.tier_id, ...)

            return ApplyResult(success=True, ...)

        except Exception as e:
            # Rollback
            _restore_backup(backup_path)
            _write_audit_event("apply.rollback", error=str(e), ...)
            raise
```

### 10.2 File-Lock

POSIX: `fcntl.flock` exclusive.
Windows: `msvcrt.locking` exclusive.

Cross-platform-Abstraktion: `cognithor.utils.file_lock.acquire(path, timeout)`.

### 10.3 Schema-Migration

```python
SCHEMA_MIGRATIONS = {
    1: _migrate_v1_to_v2,   # vor Hardware-Aware-Spec
    2: _migrate_v2_to_v3,   # zukünftig
}

def _migrate_schema_to_latest(cfg: dict) -> dict:
    cur = cfg.get("__schema_version", 1)
    while cur < CURRENT_SCHEMA_VERSION:
        cfg = SCHEMA_MIGRATIONS[cur](cfg)
        cur = cfg["__schema_version"]
    return cfg
```

---

## 11. Layer 7 — Runtime-Adaptation

### 11.1 Performance-Telemetry (lokal, opt-in für Aggregat)

```python
# src/cognithor/system/perf_tracker.py
class PerfTracker:
    """Misst per-Modell tok/s + first-token-latency in Prod."""

    async def record(
        self,
        *,
        model_id: str,
        tier_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        first_token_ms: int,
        total_ms: int,
    ) -> None:
        # in ~/.cognithor/perf_telemetry.jsonl (rotiert)
        ...

    def rolling_p95(self, model_id: str, *, window_s: int = 86400) -> float | None:
        ...
```

### 11.2 Drift-Detection

Beim Boot:

```python
async def check_drift(profile: HardwareProfile, cfg: CognithorConfig) -> DriftReport:
    stored_hash = cfg.__system_profile_hash
    current_hash = profile.profile_hash

    if stored_hash != current_hash:
        diff = _diff_profiles(_load_stored_profile(), profile)
        return DriftReport(
            hardware_changed=True,
            changed_components=diff,  # ["gpu_added", "driver_upgraded"]
            recommendation_eligible=True,
        )
    return DriftReport(hardware_changed=False)
```

### 11.3 Performance-Drift

Zur Laufzeit:

```python
def evaluate_performance_against_estimate(tracker: PerfTracker, tier: Tier) -> PerfDriftReport:
    p95 = tracker.rolling_p95(tier.model_set.planner)
    expected = tier.performance_estimates.planner_tok_s_p50
    if p95 < 0.5 * expected:           # 2× schlechter als erwartet
        return PerfDriftReport(
            severity="warn",
            advice="Re-recommendation verfügbar — Hardware unterperformt deutlich.",
        )
```

### 11.4 Hysterese gegen Banner-Spam

Banner wird **erst** angezeigt wenn:
- 3 consecutive Boot-Cycles drift_detected ODER
- 3 consecutive 24h-Windows perf_drift > 2× Erwartung

Cooldown: nach Banner 30 Tage Ruhe (auch wenn Drift weiter da).

User kann via `cognithor doctor --silence-drift-banner` permanent stummschalten.

---

## 12. Layer 8 — Adversarial Robustness

### 12.1 Model-Verfügbarkeits-Check vor Apply

```python
async def verify_model_availability(model_id: str, backend: str) -> AvailabilityCheck:
    if backend == "vllm":
        # HF API: existiert das Repo? Hat es die Quant-Variante?
        return await _check_huggingface(model_id)
    if backend == "ollama":
        # Ollama-Library: existiert das Tag?
        return await _check_ollama_library(model_id)
    ...
```

### 12.2 Fallback-Cascade

Wenn ein Modell nicht verfügbar:
1. Try `superseded_by` aus Manifest.
2. Try same `role_fitness` mit nächst-bestem `quality_tier`.
3. Try Tier-Downgrade.
4. Last resort: `cloud-only-fallback` (klar markiert, mit Cost-Hinweis).

Jeder Schritt im TRUST-1-Receipt sichtbar.

### 12.3 Sanity-Checks für Manifest

```python
def validate_manifest(m: Manifest) -> tuple[ValidationError, ...]:
    errors = []
    for tier in m.tiers:
        # Mind. 1 Modell pro Pflicht-Rolle
        for role in REQUIRED_ROLES:
            if role not in tier.model_set:
                errors.append(...)
        # Modell-Capabilities ⊆ Tier-Capabilities
        for role, model_id in tier.model_set.items():
            model = m.models[model_id]
            if not _capabilities_subset(model.requires_capabilities, tier.requires_capabilities):
                errors.append(...)
    return tuple(errors)
```

---

## 13. Layer 9 — Health & Monitoring

### 13.1 Per-Tier-Health-Endpoint

`/api/system/health` returnt:

```json
{
  "tier_id": "enterprise-vllm-nvfp4-blackwell",
  "components": {
    "vllm_container": {"status": "ok", "uptime_s": 3742},
    "planner_model_loaded": {"status": "ok", "model": "..."},
    "ollama_for_embedding": {"status": "ok"}
  },
  "drift": {"hardware": false, "performance": "warn"},
  "last_check_utc": "2026-05-07T19:30:00Z"
}
```

### 13.2 Flutter-Banner

Wenn `drift.hardware=true` ODER `drift.performance=warn`:
Banner oben in Settings → "Hardware-Konfiguration prüfen → ".

---

## 14. Layer 10 — Cost-Awareness

### 14.1 Cost-Estimate pro Tier

Manifest hat `pricing.yaml`:

```yaml
schema_version: 2
providers:
  anthropic:
    claude-opus-4-7:
      input_eur_per_mtok: 14.0
      output_eur_per_mtok: 70.0
      valid_until_utc: "2026-08-01T00:00:00Z"
  openai: …

local_inference:
  electricity_eur_per_kwh: 0.32       # User-overridable
  rtx_5090_watt_idle: 30
  rtx_5090_watt_full: 575
```

Solver berechnet `estimated_cost_eur_per_month` aus Last-Profile (avg req/day × tokens × pricing).

### 14.2 Budget-Cap

User kann `max_cloud_eur_per_month` setzen → Solver filtert Cloud-Tiers raus die das überschreiten würden.

---

## 15. Layer 11 — Manifest-Update-Mechanism

### 15.1 Update-Pfade

| Trigger | Was passiert |
|---|---|
| Auto (alle 30 d) | Background-Refresh, signature-verified, swap-on-success |
| `cognithor doctor --refresh-manifest` | Sofort, mit User-Anzeige der Diff |
| Cognithor-Release-Bump (z.B. 1.0 → 1.1) | Pinned-Manifest-Version evtl. höher, force-refresh |
| Recall (active.json hat Eintrag für aktuelle Manifest-Version) | Hard-Fail bis Refresh, keine Boot-Möglichkeit mit recalled Manifest |

### 15.2 Pinned Root-Pubkey

`src/cognithor/_pinned_keys.py` — **re-use des bestehenden PACK-4-Root-Keys**:
```python
# Bestehend:
PACK_REGISTRY_ROOT_KEY = "ed25519:base64key..."
# NEU — gleicher Trust-Anchor, separater Targets-Key signiert die Manifests:
HARDWARE_MANIFEST_TARGETS_KEY = "ed25519:base64key..."
```
Trust-Chain: PACK-4-Root signiert den Manifest-Targets-Key, der Targets-Key signiert die einzelnen Manifest-Versionen. Key-Rotation des Targets-Keys ist möglich ohne Cognithor-Release.

---

## 16. Layer 12 — TRUST-Integration

### 16.1 TRUST-1 Receipt für jede Recommendation

```json
{
  "run_id": "...",
  "event_type": "hardware_recommendation",
  "manifest_version": "2026.05.07.01",
  "manifest_signature_verified": true,
  "hardware_profile_hash": "sha256:...",
  "capabilities_summary": {…},
  "user_objective": {…},
  "solutions": [{tier_id: "...", score: 0.92, blockers: []}, ...],
  "selected_tier": "enterprise-vllm-nvfp4-blackwell",
  "user_confirmed": true,
  "config_diff": {…},
  "applied_at_utc": "..."
}
```

### 16.2 TRUST-2 Explainability

Solver liefert für jede Solution einen **rule_id** der die Auswahl begründet:
- `solver.match.capability_subset.exact`
- `solver.score.weighted_quality_dominates`
- `solver.fallback.cloud_only_no_local_capable`

### 16.3 TRUST-3 Failure-Modes (NEU)

| Wert | Wann |
|---|---|
| `MANIFEST_SIGNATURE_INVALID` | Signature-Check fehlgeschlagen |
| `MANIFEST_RECALLED` | Aktuelle Manifest-Version in active.json |
| `MANIFEST_FETCH_FAILED` | Netzwerk down, fallback auf cache |
| `MANIFEST_CACHE_STALE_OVER_LIMIT` | Cache > 90 Tage, force refresh |
| `SOLVER_NO_FEASIBLE_SOLUTION` | Keine Tier matched (extrem rare) |
| `APPLY_LOCK_TIMEOUT` | Concurrent Wizard hat Lock |
| `APPLY_VALIDATION_FAILED` | Pydantic-Schema-Fail |
| `APPLY_ROLLBACK_TRIGGERED` | Apply-Mid-Fail, Backup wiederhergestellt |
| `MODEL_PULL_FAILED_SUPERSEDED` | Modell weg, fallback auf superseded_by |
| `DRIFT_HARDWARE_CHANGED` | Hash mismatch beim Boot |
| `DRIFT_PERFORMANCE_DEGRADED` | rolling p95 < 0.5× expected |

---

## 17. File-Map (vollständig)

```
src/cognithor/system/
├── __init__.py
├── detector.py                    # bereits da, ergänzt um cuda/sm/wsl/docker probes
├── probes/
│   ├── __init__.py
│   ├── _types.py
│   ├── nvidia.py
│   ├── amd.py
│   ├── apple.py
│   ├── intel.py
│   ├── cpu.py
│   ├── ram.py
│   ├── disk.py
│   ├── docker.py
│   ├── wsl.py
│   ├── container.py
│   ├── network.py
│   ├── ollama.py
│   ├── vllm.py
│   ├── lmstudio.py
│   └── huggingface.py
├── sanity.py                       # L1.5 Cross-Validation
├── capabilities.py                 # L2 Mapping
├── manifest_loader.py              # L3 Loader + Verifier
├── manifest_models.py              # L3 Pydantic-Schemas
├── solver.py                       # L4 Solver
├── solver_objectives.py            # L4 UserObjective Presets
├── apply_engine.py                 # L6 Apply
├── schema_migrations.py            # L6 Schema-Migrators
├── perf_tracker.py                 # L7 Telemetry
├── drift_detector.py               # L7 Drift
├── health.py                       # L9 Health-Endpoint
├── recommender.py                  # L4-Wrapper (orchestriert solve())
└── wizard/
    ├── __init__.py
    ├── cli.py                      # L5 CLI-Wizard
    ├── ansi.py                     # ANSI-Render-Helpers
    └── prompts.py                  # input-Helpers

flutter_app/lib/screens/onboarding/   (L5 Flutter)
├── detection_screen.dart
├── objective_screen.dart
├── solutions_screen.dart
├── apply_screen.dart
└── done_screen.dart

flutter_app/lib/providers/
└── onboarding_provider.dart

scripts/
├── first_boot.py                  # bestehend, ergänzt um Wizard-Trigger
└── doctor.py                      # NEU — `cognithor doctor` CLI

docs/quality/manifest-format-v2.md  # Schema-Doku
docs/runbooks/manifest-key-rotation.md

# Embedded fallback (für offline-install)
src/cognithor/system/embedded_manifest.yaml

# External tier repo
github.com/Alex8791-cyber/cognithor-tiers (NEU public)
```

---

## 18. Test-Strategie (per Layer)

### 18.1 Unit-Tests (jedes Layer isoliert)

Pro Layer eigenes Testfile, ≥80 % Coverage:

| Layer | Tests | Testfile |
|---|---:|---|
| L1 Probes | ~60 | `tests/test_system/test_probes_*.py` |
| L1.5 Sanity | ~20 | `tests/test_system/test_sanity.py` |
| L2 Capabilities | ~30 | `tests/test_system/test_capabilities.py` |
| L3 Manifest | ~25 | `tests/test_system/test_manifest_loader.py` |
| L4 Solver | ~40 | `tests/test_system/test_solver.py` |
| L5 Wizard | ~25 | `tests/test_system/test_wizard.py` |
| L6 Apply | ~30 | `tests/test_system/test_apply_engine.py` |
| L7 Runtime | ~20 | `tests/test_system/test_drift.py` |
| L8 Adversarial | ~20 | `tests/test_system/test_robustness.py` |
| L9 Health | ~10 | `tests/test_system/test_health.py` |
| L10 Cost | ~10 | `tests/test_system/test_cost.py` |
| L11 Update | ~15 | `tests/test_system/test_manifest_refresh.py` |
| L12 TRUST | ~15 | `tests/test_system/test_trust_wiring.py` |

**Total: ~320 unit-tests.**

### 18.2 Property-Tests (Hypothesis)

- `forall profile: capabilities(profile) is deterministic`
- `forall (manifest, caps, obj): solve(...) returns valid Pareto`
- `forall config_state: apply(solution) is idempotent`
- `forall hardware_change: drift_detected ⟺ hash_changed`

### 18.3 Hardware-Matrix (Mock-basiert)

Mock-Profiles für 12 Hardware-Konfigurationen (siehe v1-Spec §11.5), jede gegen Solver, jede mit erwartetem Tier-Output.

### 18.4 Integration-Tests

- Full Wizard-Flow mit Mock-Manifest, Mock-Detection, real Apply.
- Concurrent-Wizard (zwei `cognithor doctor` parallel) → Lock-Test.
- Manifest-Recall-Flow.
- Schema-Migration v0.99 → v1.0 (mit echtem alten config.yaml).

### 18.5 Adversarial-Tests

- Manifest mit gefälschter Signatur → reject.
- Manifest mit recalled-Eintrag → reject.
- nvidia-smi-Output garbage → fallback.
- Disk-full während Apply → rollback.
- Modell-Pull mid-flight cancel → resume-able.

### 18.6 Manuelle End-to-End

Vor v1.0:
- Real RTX 5090-Box: Wizard → Apply → erste Anfrage (Win11+WSL2)
- Real RTX 4090: Wizard → Apply → erste Anfrage (Linux nativ)
- Real M3 Max: Wizard → Apply → erste Anfrage
- Real ohne GPU: Wizard → Apply → erste Anfrage

---

## 19. PR-Aufteilung (vollständig)

| PR | Layer | Inhalt | Est. LOC | Reviewer |
|---|---|---|---:|---|
| PR-01 | L1 | Probe-Architektur + nvidia/cpu/ram/disk/network probes | 600 | Owner |
| PR-02 | L1 | amd / apple / intel probes | 350 | Owner |
| PR-03 | L1 | docker / wsl / container / ollama / vllm / lmstudio / hf probes | 450 | Owner |
| PR-04 | L1.5 | Sanity-Rules + 16 Regeln | 300 | Owner |
| PR-05 | L2 | Capability-Mapping + Lookup-Tabellen | 400 | Owner |
| PR-06 | L3 | Manifest-Schemas (Pydantic) + Embedded-Manifest | 500 | Owner |
| PR-07 | L3 | Manifest-Loader + Verifier (TUF-Light) + cognithor-tiers Repo-Setup | 500 | Owner |
| PR-08 | L4 | Solver + Pareto-Filter + Scoring | 600 | Owner |
| PR-09 | L4 | UserObjective + Preset-Library | 200 | Owner |
| PR-10 | L5 | CLI-Wizard + ANSI-Renderer | 700 | Owner |
| PR-11 | L5 | Flutter-Onboarding (5 Screens + Provider) | 1 200 | Owner |
| PR-12 | L6 | Apply-Engine + File-Lock + Atomic-Write | 500 | Owner |
| PR-13 | L6 | Schema-Migrators v1→v2 | 250 | Owner |
| PR-14 | L7 | Perf-Tracker + Drift-Detector + Hysterese | 450 | Owner |
| PR-15 | L8 | Model-Availability-Verifier + Fallback-Cascade | 350 | Owner |
| PR-16 | L9 | Health-Endpoint + Flutter-Drift-Banner | 300 | Owner |
| PR-17 | L10 | Cost-Awareness + Pricing-Manifest | 350 | Owner |
| PR-18 | L11 | Manifest-Refresh + Recall-Mechanism | 400 | Owner |
| PR-19 | L12 | TRUST-1/2/3-Wiring + 11 neue FailureMode | 350 | Owner |
| PR-20 | — | first_boot.py-Integration + cognithor doctor CLI | 400 | Owner |
| PR-21 | — | 320 Unit-Tests + Property-Tests | 2 500 | Owner |
| PR-22 | — | Integration-Tests + Hardware-Matrix-Mocks | 1 200 | Owner |
| PR-23 | — | Doku: README, FIRST_BOOT, CONFIG_REFERENCE, manifest-format-v2.md, runbook | 600 | Owner |
| PR-24 | — | Adversarial-Korpus-Tests | 400 | Owner |

**Gesamt: 24 PRs · ~13 200 LOC · ~6 Wochen**

---

## 20. Rollout (5 Phasen)

### Phase 0 — Code merged, Feature-Flag off
v1.x.0 release, alle PRs gemerged, `system.first_run.enabled=False`.
Owner-internes Testing auf RTX 5090.

### Phase 1 — Auto-Trigger nur für FRISCHE Installs
v1.x.1: `pip install cognithor` ohne `~/.cognithor/` → Wizard läuft auto.
Bestehende User unverändert. 30-Tage-Beta auf 5–10 fremden Maschinen.

### Phase 2 — Drift-Banner aktiviert
Bestehende User sehen Banner bei Hardware-Wechsel oder Perf-Drift.
Cooldown 30 d.

### Phase 3 — Manifest-Auto-Update
Background-Refresh aktiv. Bei neuem Tier: Banner "Neue Empfehlung verfügbar".

### Phase 4 — v2.0.0 (Default ON für alle Pfade)
Nach 90 d Drift-Daten + 0 unbeantwortete TRUST-3-Failure-Mode-Spikes →
Hardware-Aware-Runtime ist v2.0.0-Standard.

### Rollback an JEDEM Übergang

- `system.first_run.enabled=False` deaktiviert L5 sofort.
- `system.runtime_adaptation.enabled=False` deaktiviert L7.
- Manifest-Pin auf bekannte gute Version.
- Cognithor-Downgrade ändert nichts an config.yaml (forwards-compat schema).

---

## 21. Was nicht mehr angefasst werden muss (mein Versprechen)

Nach Implementierung dieser Spec werden **folgende Änderungen ohne Code-PR** möglich:

| Änderung | Wo | Wie |
|---|---|---|
| Neues Modell verfügbar (Qwen5-30B) | `cognithor/manifest/v2/models.yaml` | YAML-PR + Re-Sign |
| Neuer Hardware-Tier | `cognithor/manifest/v2/tiers.yaml` | YAML-PR + Re-Sign |
| Modell deprecated (`superseded_by`) | `cognithor/manifest/v2/models.yaml` | YAML-PR |
| Cloud-Provider-Pricing-Update | `cognithor/manifest/v2/pricing.yaml` | YAML-PR |
| Recall einer kompromittierten Manifest | `cognithor/manifest/recalls/active.json` | JSON-PR + Re-Sign |
| Neuer Capability-Flag (z.B. neue Quant-Variante) | NUR L2 capabilities.py | Code-PR (selten) |
| Neue Hardware-Architektur (z.B. Rubin nach Blackwell) | NUR L1 probe-output mapping | Code-PR (sehr selten) |
| Neuer Backend (z.B. SGLang) | NUR L1 probe + L3 backend-config | Code-PR + YAML-PR |

**Code-Änderungen reduzieren sich auf:** neue Hardware-Architektur (~1× pro 18 Monate),
neuer Backend (~1× pro 12 Monate), neuer Capability-Flag (~1× pro 6 Monate).

**Modell-Updates passieren wöchentlich, ohne dass irgendjemand Cognithor-Code anfasst.**

---

## 22. Acceptance Criteria

- [ ] Owner-Approval auf §3 (47 Threats), §6 (Capability-Flags), §7 (Manifest-Format).
- [ ] Hardware-Matrix mit ≥10 Konfigurationen mocked + mind. 4 real-validated.
- [ ] Manifest-Format-Doku (`docs/quality/manifest-format-v2.md`) public-ready.
- [ ] Ed25519-Root-Key offline minted + dokumentiert in `docs/runbooks/manifest-key-rotation.md`.
- [ ] cognithor-tiers Repo public, mit signiertem MVM-Manifest.
- [ ] 320 Unit-Tests + Property-Tests grün.
- [ ] 90 Tage Phase-1-Beta-Daten ohne kritische Drift.

---

## 23. Open Questions (alle in Spec aufgelöst, hier nur Recap)

1. ✅ **Wo lebt der Manifest?** — externes `cognithor-tiers`-Repo (§7)
2. ✅ **Wie wird er signiert?** — TUF-Light wie PACK-4 (§7.5, §15)
3. ✅ **Wer entscheidet bei Konflikten?** — User über Pareto-Frontier (§9.1)
4. ✅ **Was wenn alles fehlschlägt?** — `cloud-only`-Catch-All-Tier (§8.1)
5. ✅ **Hardware-Wechsel-Erkennung?** — Hash-Drift + Hysterese (§11)
6. ✅ **Modell verschwindet?** — `superseded_by` + Fallback-Cascade (§12)
7. ✅ **Concurrent Wizards?** — File-Lock (§10.2)
8. ✅ **Schema-Bumps?** — Migrators (§10.3)
9. ✅ **Cost-Awareness?** — Pricing-Manifest (§14)
10. ✅ **Bestehende Installationen?** — Banner + Cooldown, kein auto-overwrite (§11.4)

Keine offenen Questions mehr. **Spec ist implementation-ready.**

---

## 24. References

- v1-Spec (`docs/superpowers/specs/2026-05-07-hardware-aware-first-run.md`) — superseded.
- Fast-Path-Spec (`docs/superpowers/specs/2026-05-07-fast-path-router-spec.md`) — komplementär (Modell-Empfehlungen).
- PACK-4 (`docs/superpowers/specs/2026-05-05-pack4-registry-signing.md`) — Vorlage für TUF-Light-Signing.
- TRUST-Stack (`docs/operational_trust.md`).
- Bestehender Detector: `src/cognithor/system/detector.py`.
- Bestehender Boot: `scripts/first_boot.py`.
- vLLM-Orchestrator: `src/cognithor/core/vllm_orchestrator.py`.
