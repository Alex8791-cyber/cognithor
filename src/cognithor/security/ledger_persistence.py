"""Boot-time wiring of the six TRUST disk-persistence stores.

This module is the production entry point for the disk-persistence
sprint shipped in PRs #515-#520. It exists so that:

1. Every operational-trust ledger has a canonical SQLite file under
   ``~/.cognithor/audit/`` after first boot.
2. Future writes through the canonical in-memory ledgers are
   transparently mirrored to disk via best-effort write-through —
   the in-memory ledger remains the authoritative read path during
   the session, the disk stores are the cross-restart authority.
3. Registry-style ledgers (fingerprint, migration, scope) seed
   their in-memory caches from disk at boot so the Trace-UI shows
   the historic state, not just what happened in this process.
4. Append-style ledgers (backend-dispatch, escalation, cost) start
   each session with an empty in-memory cache — disk is the long
   tail, in-memory is the live tail.

Failure model: every operation here is best-effort. A corrupt
SQLite file, missing audit dir, or schema-version mismatch logs a
warning and skips the binding for that ledger — the in-memory
ledger continues to function so the gateway still boots. We do
NOT raise; persistence is value-add, not a boot pre-requisite.

Lifecycle::

    from cognithor.security.ledger_persistence import (
        open_canonical_stores_and_bind,
    )

    # Once during gateway boot, before register_runtime_binaries():
    bound = open_canonical_stores_and_bind()
    if bound.errors:
        for ledger_name, exc in bound.errors.items():
            log.warning("ledger_persistence_skipped", ledger=ledger_name, ...)

The module is idempotent — calling ``open_canonical_stores_and_bind``
twice is safe; the second call detects the already-bound flag and
returns the existing handles.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.security.backend_dispatch_store import BackendDispatchStore
    from cognithor.security.cloud_escalation_store import CloudEscalationStore
    from cognithor.security.cost_ledger_store import CostLedgerStore
    from cognithor.security.fingerprint_store import FingerprintStore
    from cognithor.security.migration_ledger_store import MigrationLedgerStore
    from cognithor.security.permission_scope_store import PermissionScopeStore

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------


_DEFAULT_AUDIT_SUBDIR = "audit"


def default_audit_dir() -> Path:
    """Return ``~/.cognithor/audit/``, creating it if missing.

    The directory is the canonical home for every TRUST disk store.
    Inno Setup, the bootstrap script, and the Flutter UI all assume
    this layout.
    """
    home = Path.home() / ".cognithor"
    audit_dir = home / _DEFAULT_AUDIT_SUBDIR
    audit_dir.mkdir(parents=True, exist_ok=True)
    return audit_dir


def _store_path(audit_dir: Path, name: str) -> Path:
    return audit_dir / f"{name}.sqlite"


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@dataclass
class CanonicalStores:
    """Handles to every opened TRUST store + binding diagnostics.

    ``errors`` is keyed by ledger short name (``"dispatch"``,
    ``"escalation"``, ``"fingerprint"``, ``"cost"``, ``"migration"``,
    ``"scope"``) when an open or bind step failed for that ledger.
    Empty when everything came up cleanly.
    """

    audit_dir: Path
    dispatch: BackendDispatchStore | None = None
    escalation: CloudEscalationStore | None = None
    fingerprint: FingerprintStore | None = None
    cost: CostLedgerStore | None = None
    migration: MigrationLedgerStore | None = None
    scope: PermissionScopeStore | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def close_all(self) -> None:
        """Close every successfully opened store. Safe to call
        multiple times. Used by tests + graceful gateway shutdown."""
        for s in (
            self.dispatch,
            self.escalation,
            self.fingerprint,
            self.cost,
            self.migration,
            self.scope,
        ):
            if s is not None:
                with contextlib.suppress(Exception):
                    s.close()


# Module-global flag so a double-call is a no-op (defensive — gateway
# boot phases run once, but tests + hot-reload paths may call again).
_BOUND: CanonicalStores | None = None


def open_canonical_stores_and_bind(
    *,
    audit_dir: Path | None = None,
) -> CanonicalStores:
    """Open every canonical TRUST store and wire write-through.

    Steps (each independent — failure of one doesn't poison the rest):

    1. Resolve ``audit_dir`` (default: ``~/.cognithor/audit/``).
    2. Open each of the six SQLite stores in a fixed order.
    3. Seed registry-style in-memory ledgers (fingerprint, migration,
       scope) from disk so reads reflect historic state.
    4. Patch the canonical in-memory ledgers' mutation methods to
       write through to the corresponding store.

    Args:
        audit_dir: Override for tests. Production passes ``None``.

    Returns:
        A :class:`CanonicalStores` instance. Inspect ``errors`` to
        see which ledgers (if any) couldn't be bound.
    """
    global _BOUND
    if _BOUND is not None:
        return _BOUND

    target_dir = audit_dir if audit_dir is not None else default_audit_dir()
    bundle = CanonicalStores(audit_dir=target_dir)

    _open_dispatch_store(bundle)
    _open_escalation_store(bundle)
    _open_fingerprint_store(bundle)
    _open_cost_store(bundle)
    _open_migration_store(bundle)
    _open_scope_store(bundle)

    _seed_registry_ledgers_from_disk(bundle)
    _bind_write_through(bundle)
    _persist_pre_bind_migration_steps(bundle)

    _BOUND = bundle
    if bundle.errors:
        log.warning(
            "ledger_persistence_partial",
            audit_dir=str(target_dir),
            failed_ledgers=sorted(bundle.errors.keys()),
        )
    else:
        log.info(
            "ledger_persistence_ready",
            audit_dir=str(target_dir),
        )
    return bundle


def reset_for_tests() -> None:
    """Drop the global handle so a fresh ``open_canonical_stores_and_bind``
    can run. Test helper only — production never calls this."""
    global _BOUND
    if _BOUND is not None:
        _BOUND.close_all()
    _BOUND = None
    _restore_canonical_methods()


# ---------------------------------------------------------------------------
# Per-ledger open helpers
# ---------------------------------------------------------------------------


def _open_dispatch_store(bundle: CanonicalStores) -> None:
    try:
        from cognithor.security.backend_dispatch_store import (
            BackendDispatchStore,
        )

        bundle.dispatch = BackendDispatchStore(_store_path(bundle.audit_dir, "backend_dispatch"))
    except Exception as exc:
        bundle.errors["dispatch"] = repr(exc)
        log.warning("ledger_persistence_open_failed", ledger="dispatch", error=str(exc))


def _open_escalation_store(bundle: CanonicalStores) -> None:
    try:
        from cognithor.security.cloud_escalation_store import (
            CloudEscalationStore,
        )

        bundle.escalation = CloudEscalationStore(_store_path(bundle.audit_dir, "cloud_escalation"))
    except Exception as exc:
        bundle.errors["escalation"] = repr(exc)
        log.warning("ledger_persistence_open_failed", ledger="escalation", error=str(exc))


def _open_fingerprint_store(bundle: CanonicalStores) -> None:
    try:
        from cognithor.security.fingerprint_store import FingerprintStore

        bundle.fingerprint = FingerprintStore(_store_path(bundle.audit_dir, "fingerprints"))
    except Exception as exc:
        bundle.errors["fingerprint"] = repr(exc)
        log.warning(
            "ledger_persistence_open_failed",
            ledger="fingerprint",
            error=str(exc),
        )


def _open_cost_store(bundle: CanonicalStores) -> None:
    try:
        from cognithor.security.cost_ledger_store import CostLedgerStore

        bundle.cost = CostLedgerStore(_store_path(bundle.audit_dir, "cost_ledger"))
    except Exception as exc:
        bundle.errors["cost"] = repr(exc)
        log.warning("ledger_persistence_open_failed", ledger="cost", error=str(exc))


def _open_migration_store(bundle: CanonicalStores) -> None:
    try:
        from cognithor.security.migration_ledger_store import (
            MigrationLedgerStore,
        )

        bundle.migration = MigrationLedgerStore(_store_path(bundle.audit_dir, "migrations"))
    except Exception as exc:
        bundle.errors["migration"] = repr(exc)
        log.warning(
            "ledger_persistence_open_failed",
            ledger="migration",
            error=str(exc),
        )


def _open_scope_store(bundle: CanonicalStores) -> None:
    try:
        from cognithor.security.permission_scope_store import (
            PermissionScopeStore,
        )

        bundle.scope = PermissionScopeStore(_store_path(bundle.audit_dir, "scope_registry"))
    except Exception as exc:
        bundle.errors["scope"] = repr(exc)
        log.warning("ledger_persistence_open_failed", ledger="scope", error=str(exc))


# ---------------------------------------------------------------------------
# Registry-style seeding (fingerprint, migration, scope)
# ---------------------------------------------------------------------------


def _seed_registry_ledgers_from_disk(bundle: CanonicalStores) -> None:
    """Pull existing rows from registry-style stores into their
    in-memory ledgers so reads reflect historic state.

    Append-style ledgers (dispatch, escalation, cost) are NOT seeded
    — those are write-only logs whose useful read shape is "what
    happened during this session". The Trace-UI queries the disk
    directly when it needs the long tail.
    """
    if bundle.fingerprint is not None:
        try:
            from cognithor.security.fingerprint import FINGERPRINT_LEDGER

            for name in bundle.fingerprint.names():
                for fp in bundle.fingerprint.history(name):
                    # ``register`` is idempotent — re-registering an
                    # already-known hash is a no-op.
                    FINGERPRINT_LEDGER.register(fp)
        except Exception as exc:
            bundle.errors["fingerprint_seed"] = repr(exc)
            log.warning(
                "ledger_persistence_seed_failed",
                ledger="fingerprint",
                error=str(exc),
            )

    if bundle.migration is not None:
        try:
            from cognithor.security.migration_ledger import (
                MIGRATION_LEDGER,
                MigrationChainError,
            )

            for step in bundle.migration.steps():
                with contextlib.suppress(MigrationChainError, ValueError):
                    MIGRATION_LEDGER.record(step)
        except Exception as exc:
            bundle.errors["migration_seed"] = repr(exc)
            log.warning(
                "ledger_persistence_seed_failed",
                ledger="migration",
                error=str(exc),
            )

    if bundle.scope is not None:
        try:
            from cognithor.security.permission_scope import SCOPE_REGISTRY

            for scope in bundle.scope.list_scopes():
                SCOPE_REGISTRY.register(scope)
        except Exception as exc:
            bundle.errors["scope_seed"] = repr(exc)
            log.warning(
                "ledger_persistence_seed_failed",
                ledger="scope",
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Pre-bind migration replay (TRUST-10 self-audit emitter wiring)
# ---------------------------------------------------------------------------


def _persist_pre_bind_migration_steps(bundle: CanonicalStores) -> None:
    """Replay every in-memory migration step into the disk store.

    Several `_record_*_migration()` helpers across the security
    package run at import time and append to the canonical
    in-memory ``MIGRATION_LEDGER`` before this module's bind step
    has a chance to wire write-through. Without this replay, the
    TRUST-10 self-audit emitters would never reach disk.

    Idempotent: ``MigrationLedgerStore.record`` rejects duplicate
    ``migration_id`` via :class:`MigrationChainError`. We suppress
    that — re-runs after the first boot are no-ops.
    """
    if bundle.migration is None:
        return
    try:
        from cognithor.security.migration_ledger import (
            MIGRATION_LEDGER,
            MigrationChainError,
        )

        for step in MIGRATION_LEDGER.steps():
            with contextlib.suppress(MigrationChainError, ValueError):
                bundle.migration.record(step)
    except Exception as exc:
        bundle.errors["migration_replay"] = repr(exc)
        log.warning(
            "ledger_persistence_replay_failed",
            ledger="migration",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Method-patching write-through
# ---------------------------------------------------------------------------

# Originals are saved here so ``reset_for_tests`` can restore them.
_ORIGINAL_METHODS: dict[str, Any] = {}


def _bind_write_through(bundle: CanonicalStores) -> None:
    """Wrap each canonical in-memory ledger's mutation method so that
    every successful in-memory write also fires the corresponding
    store call. Disk failures are logged + swallowed so an unhealthy
    SQLite file can't take down the live agent loop."""

    if bundle.dispatch is not None:
        _wrap_in_memory_record(
            module_name="cognithor.security.backend_dispatch",
            attr_name="BACKEND_DISPATCH_LEDGER",
            method_name="record",
            store_callable=bundle.dispatch.append,
            ledger_label="dispatch",
        )

    if bundle.escalation is not None:
        _wrap_in_memory_record(
            module_name="cognithor.security.cloud_escalation",
            attr_name="ESCALATION_LEDGER",
            method_name="record",
            store_callable=bundle.escalation.append,
            ledger_label="escalation",
        )

    if bundle.fingerprint is not None:
        _wrap_in_memory_record(
            module_name="cognithor.security.fingerprint",
            attr_name="FINGERPRINT_LEDGER",
            method_name="register",
            store_callable=bundle.fingerprint.register,
            ledger_label="fingerprint_register",
        )
        _wrap_in_memory_record(
            module_name="cognithor.security.fingerprint",
            attr_name="FINGERPRINT_LEDGER",
            method_name="remove",
            store_callable=bundle.fingerprint.remove,
            ledger_label="fingerprint_remove",
        )

    if bundle.cost is not None:
        _wrap_in_memory_record(
            module_name="cognithor.security.cost_ledger",
            attr_name="COST_LEDGER",
            method_name="record",
            store_callable=bundle.cost.record,
            ledger_label="cost",
        )

    if bundle.migration is not None:
        _wrap_in_memory_record(
            module_name="cognithor.security.migration_ledger",
            attr_name="MIGRATION_LEDGER",
            method_name="record",
            store_callable=bundle.migration.record,
            ledger_label="migration",
        )

    if bundle.scope is not None:
        _wrap_in_memory_record(
            module_name="cognithor.security.permission_scope",
            attr_name="SCOPE_REGISTRY",
            method_name="register",
            store_callable=bundle.scope.register,
            ledger_label="scope_register",
        )
        _wrap_scope_remove(bundle)
        _wrap_scope_clear(bundle)


def _wrap_in_memory_record(
    *,
    module_name: str,
    attr_name: str,
    method_name: str,
    store_callable: Any,
    ledger_label: str,
) -> None:
    """Replace ``<module>.<attr>.<method>`` with a wrapper that calls
    the original AND then ``store_callable`` with the same args.

    Idempotent: a previously-wrapped method is detected via the
    ``__cognithor_disk_wrapped__`` marker and re-wrapping is skipped.
    """
    import importlib

    module = importlib.import_module(module_name)
    instance = getattr(module, attr_name)
    original = getattr(instance, method_name)

    # Already wrapped? Don't double-wrap (defensive against a
    # resurrected boot path).
    if getattr(original, "__cognithor_disk_wrapped__", False):
        return

    key = f"{module_name}.{attr_name}.{method_name}"
    _ORIGINAL_METHODS[key] = original

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        try:
            store_callable(*args, **kwargs)
        except Exception as exc:
            log.warning(
                "ledger_disk_write_failed",
                ledger=ledger_label,
                error=type(exc).__name__,
                error_msg=str(exc),
            )
        return result

    wrapper.__cognithor_disk_wrapped__ = True  # type: ignore[attr-defined]
    setattr(instance, method_name, wrapper)


def _wrap_scope_remove(bundle: CanonicalStores) -> None:
    """``ScopeRegistry.remove`` takes ``(axis, identity)`` and returns
    bool; mirror that into the disk store."""
    if bundle.scope is None:
        return

    import importlib

    module = importlib.import_module("cognithor.security.permission_scope")
    instance = module.SCOPE_REGISTRY
    original = instance.remove
    if getattr(original, "__cognithor_disk_wrapped__", False):
        return

    _ORIGINAL_METHODS["cognithor.security.permission_scope.SCOPE_REGISTRY.remove"] = original

    def wrapper(*args: Any, **kwargs: Any) -> bool:
        result = original(*args, **kwargs)
        try:
            assert bundle.scope is not None
            bundle.scope.remove(*args, **kwargs)
        except Exception as exc:
            log.warning(
                "ledger_disk_write_failed",
                ledger="scope_remove",
                error=type(exc).__name__,
                error_msg=str(exc),
            )
        return bool(result)

    wrapper.__cognithor_disk_wrapped__ = True  # type: ignore[attr-defined]
    instance.remove = wrapper


def _wrap_scope_clear(bundle: CanonicalStores) -> None:
    if bundle.scope is None:
        return

    import importlib

    module = importlib.import_module("cognithor.security.permission_scope")
    instance = module.SCOPE_REGISTRY
    original = instance.clear
    if getattr(original, "__cognithor_disk_wrapped__", False):
        return

    _ORIGINAL_METHODS["cognithor.security.permission_scope.SCOPE_REGISTRY.clear"] = original

    def wrapper() -> None:
        original()
        try:
            assert bundle.scope is not None
            bundle.scope.clear()
        except Exception as exc:
            log.warning(
                "ledger_disk_write_failed",
                ledger="scope_clear",
                error=type(exc).__name__,
                error_msg=str(exc),
            )

    wrapper.__cognithor_disk_wrapped__ = True  # type: ignore[attr-defined]
    instance.clear = wrapper


def _restore_canonical_methods() -> None:
    """Reverse :func:`_bind_write_through` — used by tests so a
    teardown leaves the canonical singletons untouched for the next
    test that imports them."""
    import importlib

    for key, original in list(_ORIGINAL_METHODS.items()):
        module_name, _, rest = key.partition(".")
        # rebuild full module name (module path may contain dots)
        # The keys all start with ``cognithor.security.<file>.``…
        parts = key.rsplit(".", 2)
        if len(parts) != 3:
            continue
        full_module, attr_name, method_name = parts
        try:
            module = importlib.import_module(full_module)
        except Exception:
            continue
        instance = getattr(module, attr_name, None)
        if instance is None:
            continue
        with contextlib.suppress(AttributeError):
            setattr(instance, method_name, original)
    _ORIGINAL_METHODS.clear()


__all__ = [
    "CanonicalStores",
    "default_audit_dir",
    "open_canonical_stores_and_bind",
    "reset_for_tests",
]
