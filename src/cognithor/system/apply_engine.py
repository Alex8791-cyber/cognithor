"""Layer 6 — Apply-Engine.

Atomic, locked, validated, rollback-able write of the wizard's selected
Solution into `~/.cognithor/config.yaml`.

Invariants:
- File-Lock prevents concurrent wizards (CLI + Flutter + cognithor doctor).
- Backup-rotation keeps last 5 configs (older ones evicted).
- Pydantic-Validation runs BEFORE any disk write — schema-invalid merges
  raise without touching the file.
- `os.replace()` for atomic swap.
- On any exception during the merge: rollback from backup.
- `.cognithor_initialized` is written ONLY after successful apply.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

import yaml

from cognithor.system.capabilities import Capabilities
from cognithor.system.manifest_models import Manifest, Tier
from cognithor.system.solver import Solution, UserObjective
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "ApplyError",
    "ApplyResult",
    "apply_solution",
    "list_backups",
    "rollback_last",
]


# ── Lock ────────────────────────────────────────────────────────────────────

_LOCK_PATH = Path.home() / ".cognithor" / ".wizard.lock"


@contextlib.contextmanager
def _file_lock(timeout_s: float = 5.0) -> Iterator[None]:
    """Cross-platform exclusive file-lock around the apply pipeline.

    POSIX: fcntl.LOCK_EX | LOCK_NB.
    Windows: msvcrt.locking with retry-loop.
    """
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fp = _LOCK_PATH.open("a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            import time

            start = time.time()
            while True:
                try:
                    msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.time() - start > timeout_s:
                        raise ApplyError("wizard_lock_timeout") from None
                    time.sleep(0.1)
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ApplyError("wizard_lock_busy") from exc
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
    finally:
        fp.close()


# ── Result types ───────────────────────────────────────────────────────────


class ApplyError(Exception):
    pass


@dataclass(frozen=True)
class ApplyResult:
    success: bool
    config_path: Path
    backup_path: Path | None
    initialized_marker_path: Path
    selected_tier_id: str
    capabilities_hash: str
    audit_event_payload: dict[str, Any]


# ── Schema versioning ──────────────────────────────────────────────────────

_LATEST_SCHEMA_VERSION = 2


def _migrate_v1_to_v2(cfg: dict[str, Any]) -> dict[str, Any]:
    """v1 → v2: bookkeeping fields (`__schema_version`, `__recommended_*`)
    were originally written into config.yaml. v2 moves them to the
    `~/.cognithor/.hardware_aware.json` sidecar so config.yaml stays
    Pydantic-strict-clean."""
    cfg = dict(cfg)
    for stale_key in (
        "__schema_version",
        "__system_profile_hash",
        "__recommended_tier",
        "__recommended_at_utc",
        "__manifest_version",
    ):
        cfg.pop(stale_key, None)
    return cfg


_SCHEMA_MIGRATIONS = {1: _migrate_v1_to_v2}


def _migrate_to_latest(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply v1→v2→… migrations idempotently. Schema-version tracking moved
    to sidecar in v2 — migrations now just clean up legacy keys."""
    cur = 1 if "__schema_version" in cfg else _LATEST_SCHEMA_VERSION
    while cur < _LATEST_SCHEMA_VERSION:
        migrator = _SCHEMA_MIGRATIONS.get(cur)
        if migrator is None:
            break
        cfg = migrator(cfg)
        cur += 1
    return cfg


# ── Merge ──────────────────────────────────────────────────────────────────


def _is_default_or_missing(cfg: dict[str, Any], path: tuple[str, ...]) -> bool:
    """True iff the field at `path` is missing or matches the Cognithor
    code-default. Used to decide whether to overwrite vs respect User-Override."""
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return True
        cur = cur[key]
    # Don't try to import config defaults — heuristic: empty string, None, or default-ollama
    if cur in ("", None):
        return True
    if path == ("llm_backend_type",) and cur == "ollama":
        return True
    return False


# CognithorConfig schema-known role names. The wizard's model_set has additional
# roles (formulate, fast_path_validator) which are reserved for the Fast-Path-Spec
# and not yet in the runtime schema — those land in the sidecar instead.
_SCHEMA_KNOWN_ROLES = {"planner", "executor", "coder", "embedding"}

# Plain VLLMConfig fields (per src/cognithor/config.py:2567 VLLMConfig).
# `enable_prefix_caching` + `num_speculative_tokens` exist only as
# `vlm_*`-prefixed VLM-specific fields; for the L6 apply they go to the
# sidecar `vllm_extras` so the runtime orchestrator can pick them up
# without us mis-mapping into the wrong field name.
_SCHEMA_KNOWN_VLLM_FIELDS = {
    "docker_image",
    "gpu_memory_utilization",
    "enforce_eager",
    "cpu_offload_gb",
    "max_model_len",
    "base_url",
    "quality_default",
}


def _merge_solution(
    cfg: dict[str, Any], solution: Solution, tier: Tier, manifest: Manifest
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Idempotent merge — never overwrite User-Overrides.

    Returns (config_yaml_dict, sidecar_dict) where sidecar_dict carries
    HW-aware bookkeeping + future-only fields that aren't (yet) in the
    runtime schema.
    """
    cfg = dict(cfg)
    cfg = _migrate_to_latest(cfg)
    sidecar: dict[str, Any] = {}

    # Backend-Type
    if _is_default_or_missing(cfg, ("llm_backend_type",)):
        cfg["llm_backend_type"] = tier.backend

    # Backend-config (vLLM-specific or Ollama-base-url)
    bc = tier.backend_config
    if tier.backend == "vllm" and bc.docker_image:
        cfg.setdefault("vllm", {})
        for field, value in {
            "docker_image": bc.docker_image,
            "gpu_memory_utilization": bc.gpu_memory_utilization,
            "enforce_eager": bc.enforce_eager,
            "cpu_offload_gb": bc.cpu_offload_gb,
            "max_model_len": bc.max_model_len,
            "num_speculative_tokens": bc.num_speculative_tokens,
            "enable_prefix_caching": bc.enable_prefix_caching,
        }.items():
            if value is None:
                continue
            if field in _SCHEMA_KNOWN_VLLM_FIELDS:
                cfg["vllm"].setdefault(field, value)
            else:
                sidecar.setdefault("vllm_extras", {})[field] = value
    elif tier.backend == "ollama" and bc.base_url:
        cfg.setdefault("ollama", {})
        cfg["ollama"].setdefault("base_url", bc.base_url)

    # Models — per role, only set if not already user-defined
    cfg.setdefault("models", {})
    sidecar_models: dict[str, dict[str, Any]] = {}
    backend_id_field = tier.backend
    for role in ("planner", "executor", "coder", "embedding", "formulate", "fast_path_validator"):
        model_id = getattr(tier.model_set, role)
        model = manifest.models.get(model_id)
        if model is None:
            continue
        backend_name = (model.backend_ids or {}).get(backend_id_field)
        if backend_name is None:
            # Embedding stays Ollama-served by convention
            backend_name = (model.backend_ids or {}).get("ollama") or model_id
        if role in _SCHEMA_KNOWN_ROLES:
            cfg["models"].setdefault(role, {})
            if "name" not in cfg["models"][role]:
                cfg["models"][role]["name"] = backend_name
        else:
            sidecar_models[role] = {"name": backend_name, "model_id": model_id}

    if sidecar_models:
        sidecar["model_set_extras"] = sidecar_models

    # Hardware-aware bookkeeping → sidecar (NOT config.yaml)
    sidecar["schema_version"] = _LATEST_SCHEMA_VERSION
    sidecar["system_profile_hash"] = ""  # filled by caller before write
    sidecar["recommended_tier"] = solution.tier_id
    sidecar["recommended_at_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    sidecar["manifest_version"] = manifest.manifest_version
    sidecar["score"] = solution.score
    sidecar["score_breakdown"] = dict(solution.score_breakdown)

    return cfg, sidecar


# ── Backup-rotation ────────────────────────────────────────────────────────


def _backup_dir(config_path: Path) -> Path:
    return config_path.parent / "config_backups"


def _rotate_backup(config_path: Path, *, keep: int = 5) -> Path | None:
    if not config_path.exists():
        return None
    backup_dir = _backup_dir(config_path)
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"config.yaml.backup-{ts}"
    backup_path.write_bytes(config_path.read_bytes())

    backups = sorted(backup_dir.glob("config.yaml.backup-*"))
    while len(backups) > keep:
        with contextlib.suppress(OSError):
            backups[0].unlink()
        backups.pop(0)
    return backup_path


def list_backups(config_path: Path | None = None) -> list[Path]:
    if config_path is None:
        config_path = Path.home() / ".cognithor" / "config.yaml"
    backup_dir = _backup_dir(config_path)
    if not backup_dir.exists():
        return []
    return sorted(backup_dir.glob("config.yaml.backup-*"))


# ── Atomic write ───────────────────────────────────────────────────────────


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ── Pydantic validation pre-flight ─────────────────────────────────────────


def _validate_against_schema(cfg: dict[str, Any]) -> None:
    """Build a CognithorConfig from the merged dict to catch schema bugs.

    We don't keep the result — just verify it parses. Validation errors
    raise; the caller restores the backup.
    """
    try:
        from cognithor.config import CognithorConfig
    except ImportError as exc:
        raise ApplyError(f"config_module_import_failed: {exc}") from exc
    try:
        CognithorConfig.model_validate(cfg)
    except Exception as exc:
        # Pydantic raises ValidationError; we wrap it
        raise ApplyError(f"config_schema_invalid: {type(exc).__name__}: {exc}") from exc


# ── Public API ──────────────────────────────────────────────────────────────


def apply_solution(
    *,
    solution: Solution,
    manifest: Manifest,
    capabilities: Capabilities,
    objective: UserObjective,
    config_path: Path | None = None,
    user_confirmed: bool = False,
    download_models: bool = False,
) -> ApplyResult:
    """Apply the wizard's selected solution to ~/.cognithor/config.yaml.

    Atomic: all-or-nothing. Idempotent: applying the same solution twice
    is a no-op. Locked: concurrent apply attempts raise ApplyError.
    """
    if not user_confirmed:
        raise ApplyError("apply_requires_explicit_user_confirmation")

    if solution.blockers:
        raise ApplyError(f"cannot_apply_blocked_solution: {', '.join(solution.blockers)}")

    config_path = config_path or (Path.home() / ".cognithor" / "config.yaml")
    initialized_marker = config_path.parent / ".cognithor_initialized"

    tier = next((t for t in manifest.tiers if t.id == solution.tier_id), None)
    if tier is None:
        raise ApplyError(f"tier_not_in_manifest: {solution.tier_id}")

    audit_payload: dict[str, Any] = {
        "audit_event": "hardware_aware_apply",
        "manifest_version": manifest.manifest_version,
        "selected_tier": solution.tier_id,
        "capabilities_hash": capabilities.profile_hash,
        "objective_weights": {
            "quality": objective.weight_quality,
            "speed": objective.weight_speed,
            "cost": objective.weight_cost,
            "privacy": objective.weight_privacy,
        },
        "score": solution.score,
        "score_breakdown": solution.score_breakdown,
        "applied_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with _file_lock(timeout_s=5.0):
        # Read existing config (or {})
        existing: dict[str, Any] = {}
        if config_path.exists():
            try:
                existing = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ApplyError(f"existing_config_unparseable: {exc}") from exc

        # Backup BEFORE write
        backup_path = _rotate_backup(config_path)

        # Merge → (config_yaml, sidecar)
        merged, sidecar = _merge_solution(existing, solution, tier, manifest)
        sidecar["system_profile_hash"] = capabilities.profile_hash

        # Validate before writing — abort if schema-invalid
        try:
            _validate_against_schema(merged)
        except ApplyError as exc:
            log.error("apply_validation_failed", error=str(exc))
            audit_payload["error"] = str(exc)
            audit_payload["status"] = "validation_failed"
            raise

        # Write atomically
        try:
            _atomic_write_yaml(config_path, merged)
        except Exception as exc:
            # Restore backup
            if backup_path and backup_path.exists():
                config_path.write_bytes(backup_path.read_bytes())
            raise ApplyError(f"atomic_write_failed: {exc}") from exc

        # Sidecar (bookkeeping + future fields) next to config.yaml
        try:
            sidecar_path = config_path.parent / ".hardware_aware.json"
            sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            log.warning("sidecar_write_failed", error=str(exc))

        # Mark initialized (only on success)
        try:
            initialized_marker.write_text(
                json.dumps(
                    {
                        "tier_id": solution.tier_id,
                        "manifest_version": manifest.manifest_version,
                        "applied_at_utc": audit_payload["applied_at_utc"],
                        "capabilities_hash": capabilities.profile_hash,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("initialized_marker_write_failed", error=str(exc))

        audit_payload["status"] = "success"
        audit_payload["backup_path"] = str(backup_path) if backup_path else None

    log.info(
        "hardware_aware_apply_success",
        **{k: v for k, v in audit_payload.items() if k != "score_breakdown"},
    )

    return ApplyResult(
        success=True,
        config_path=config_path,
        backup_path=backup_path,
        initialized_marker_path=initialized_marker,
        selected_tier_id=solution.tier_id,
        capabilities_hash=capabilities.profile_hash,
        audit_event_payload=audit_payload,
    )


def rollback_last(config_path: Path | None = None) -> Path | None:
    """Restore the most recent backup. Returns the restored backup path."""
    config_path = config_path or (Path.home() / ".cognithor" / "config.yaml")
    backups = list_backups(config_path)
    if not backups:
        return None
    last = backups[-1]
    with _file_lock(timeout_s=5.0):
        config_path.write_bytes(last.read_bytes())
    log.info("config_rolled_back", from_=str(last), to=str(config_path))
    return last
