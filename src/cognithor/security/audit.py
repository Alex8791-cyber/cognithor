"""Audit trail: Immutable log of all actions.

Every gatekeeper decision, every tool execution, every
sub-agent spawn is logged. Append-only, tamper-evident
via SHA-256 chain.

Security guarantees:
  - Entries can only be added, never deleted
  - Each entry contains the hash of the previous one -> chain
  - JSONL format for easy analysis
  - Credential values are masked BEFORE logging

Bible reference: §3.2 (Audit-Log), §11.5 (Audit Trail)
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.models import AuditEntry, GateStatus

log = get_logger(__name__)

# Standard patterns for credential masking in the audit log
_CREDENTIAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(sk-[a-zA-Z0-9]{4})[a-zA-Z0-9]{16,}"),
    re.compile(r"(token_[a-zA-Z0-9]{4})[a-zA-Z0-9]+"),
    re.compile(r"(password\s*[:=]\s*)\S+", re.IGNORECASE),
    re.compile(r"(secret\s*[:=]\s*)\S+", re.IGNORECASE),
    re.compile(r"(api_key\s*[:=]\s*)\S+", re.IGNORECASE),
    re.compile(r"(Bearer\s+)[a-zA-Z0-9._\-]{8,}"),
    re.compile(r"(ghp_)[a-zA-Z0-9]{30,}"),
    re.compile(r"(xox[baprs]-)[a-zA-Z0-9\-]+"),
    # AWS Access Key IDs
    re.compile(r"(AKIA)[A-Z0-9]{12,}"),
    # Private keys (PEM)
    re.compile(
        r"(-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----)[\s\S]*?(-----END)", re.DOTALL
    ),
    # Generic long hex/base64 secrets (>32 chars after key-like prefix)
    re.compile(
        r"((?:key|token|secret|credential)\s*[:=]\s*['\"]?)[a-zA-Z0-9+/=_\-]{32,}", re.IGNORECASE
    ),
]


def mask_credentials(text: str) -> str:
    """Masks credentials in a text.

    Replaces recognized patterns with partially masked versions.
    e.g. 'sk-abc123456789' -> 'sk-abc1***'
    """
    if not text:
        return text
    result = text
    for pattern in _CREDENTIAL_PATTERNS:
        result = pattern.sub(lambda m: m.group(1) + "***", result)
    return result


def mask_dict(data: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Masks credentials in a nested dict recursively.

    Args:
        data: Dict with potential credentials.
        depth: Current nesting depth (max 10).

    Returns:
        Copy of the dict with masked values.
    """
    if depth > 10:
        return data
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = mask_credentials(value)
        elif isinstance(value, dict):
            result[key] = mask_dict(value, depth + 1)
        elif isinstance(value, list):
            result[key] = [
                mask_credentials(v)
                if isinstance(v, str)
                else mask_dict(v, depth + 1)
                if isinstance(v, dict)
                else v
                for v in value
            ]
        else:
            result[key] = value
    return result


def _compute_hash(data: str, prev_hash: str) -> str:
    """Computes SHA-256 hash for chain integrity."""
    return hashlib.sha256(f"{prev_hash}|{data}".encode()).hexdigest()


class AuditTrail:
    """Immutable, append-only audit log. [B§11.5]

    Writes JSONL files with hash chain for tamper evidence.
    Credentials are automatically masked.
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        log_path: Path | str | None = None,
        hmac_key: bytes | None = None,
        ed25519_key: bytes | None = None,
    ) -> None:
        if log_path is not None:
            self._log_path = Path(log_path)
            self._log_dir = self._log_path.parent
        else:
            self._log_dir = log_dir or Path.home() / ".cognithor" / "logs"
            self._log_path = self._log_dir / "audit.jsonl"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._last_hash = "genesis"
        self._entry_count = 0
        self._hmac_key = hmac_key
        self._ed25519_key = ed25519_key

        # Resume chain from the last entry
        self._restore_chain()

    def _restore_chain(self) -> None:
        """Restores the last hash from the log."""
        if not self._log_path.exists():
            return
        try:
            last_line = ""
            with open(self._log_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last_line = line.strip()
                        self._entry_count += 1
            if last_line:
                entry = json.loads(last_line)
                self._last_hash = entry.get("hash", "genesis")
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("audit_chain_restore_failed", error=str(exc))

    def record(
        self,
        entry: AuditEntry,
        *,
        mask: bool = True,
    ) -> str:
        """Records an audit entry.

        Args:
            entry: The AuditEntry (flat structure).
            mask: Mask credentials in execution_result (default: True).

        Returns:
            Hash of the entry.
        """
        record = self._entry_to_dict(entry, mask=mask)
        data_str = json.dumps(record, ensure_ascii=False, sort_keys=True)
        entry_hash = _compute_hash(data_str, self._last_hash)
        record["prev_hash"] = self._last_hash
        record["hash"] = entry_hash

        # HMAC signature (cryptographically binding, not just tamper-evident)
        if self._hmac_key:
            record["hmac"] = hmac_mod.new(
                self._hmac_key, record["hash"].encode(), hashlib.sha256
            ).hexdigest()

        # Ed25519 asymmetric signature (verify without the secret)
        if self._ed25519_key:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                    Ed25519PrivateKey,
                )

                private_key = Ed25519PrivateKey.from_private_bytes(self._ed25519_key[:32])
                signature = private_key.sign(record["hash"].encode())
                record["ed25519_sig"] = signature.hex()
            except ImportError:
                log.warning("ed25519_requires_cryptography_package")

        line = json.dumps(record, ensure_ascii=False)
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            log.error("audit_write_failed", error=str(exc), path=str(self._log_path))
            raise

        self._last_hash = entry_hash
        self._entry_count += 1

        log.debug(
            "audit_recorded",
            session=entry.session_id,
            tool=entry.action_tool,
            status=entry.decision_status.value,
        )
        return entry_hash

    def record_event(
        self,
        session_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> str:
        """Records a free-form event (e.g. agent spawn, login).

        Args:
            session_id: Session ID.
            event_type: Type of event (e.g. 'agent_spawn', 'auth_success').
            details: Additional details.

        Returns:
            Hash of the entry.
        """
        now = datetime.now(UTC)
        safe_details = mask_dict(details or {})

        record = {
            "timestamp": now.isoformat(),
            "session_id": session_id,
            "event_type": event_type,
            "details": safe_details,
        }
        data_str = json.dumps(record, ensure_ascii=False, sort_keys=True)
        entry_hash = _compute_hash(data_str, self._last_hash)
        record["prev_hash"] = self._last_hash
        record["hash"] = entry_hash

        # HMAC signature (cryptographically binding, not just tamper-evident)
        if self._hmac_key:
            record["hmac"] = hmac_mod.new(
                self._hmac_key, record["hash"].encode(), hashlib.sha256
            ).hexdigest()

        # Ed25519 asymmetric signature (verify without the secret)
        if hasattr(self, "_ed25519_key") and self._ed25519_key:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                    Ed25519PrivateKey,
                )

                private_key = Ed25519PrivateKey.from_private_bytes(self._ed25519_key[:32])
                signature = private_key.sign(record["hash"].encode())
                record["ed25519_sig"] = signature.hex()
            except ImportError:
                log.warning("ed25519_requires_cryptography_package")

        line = json.dumps(record, ensure_ascii=False)
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            log.error("audit_event_write_failed", error=str(exc), path=str(self._log_path))
            raise

        self._last_hash = entry_hash
        self._entry_count += 1
        return entry_hash

    def verify_chain(self) -> tuple[bool, int, int]:
        """Verifies the integrity of the hash chain.

        Returns:
            Tuple of (valid, total_entries, broken_at).
            broken_at is -1 if the chain is intact.
        """
        if not self._log_path.exists():
            return (True, 0, -1)

        prev_hash = "genesis"
        count = 0
        try:
            with open(self._log_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    stored_prev = entry.get("prev_hash", "")
                    stored_hash = entry.get("hash", "")

                    if stored_prev != prev_hash:
                        return (False, i + 1, i)

                    # Verify hash
                    verify_entry = {
                        k: v
                        for k, v in entry.items()
                        if k not in ("prev_hash", "hash", "hmac", "ed25519_sig")
                    }
                    data_str = json.dumps(verify_entry, ensure_ascii=False, sort_keys=True)
                    expected = _compute_hash(data_str, prev_hash)
                    if stored_hash != expected:
                        return (False, i + 1, i)

                    prev_hash = stored_hash
                    count += 1
        except (json.JSONDecodeError, OSError):
            return (False, count, count)

        return (True, count, -1)

    @staticmethod
    def verify_signature(entry: dict[str, Any], public_key_bytes: bytes) -> bool:
        """Verify an Ed25519 signature on an audit entry.

        Args:
            entry: Audit entry dict with 'ed25519_sig' and 'hash' fields.
            public_key_bytes: 32-byte Ed25519 public key.

        Returns:
            True if signature is valid, False otherwise.
        """
        sig_hex = entry.get("ed25519_sig", "")
        hash_value = entry.get("hash", "")
        if not sig_hex or not hash_value:
            return False
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )

            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            public_key.verify(bytes.fromhex(sig_hex), hash_value.encode())
            return True
        except Exception:
            return False

    def query(
        self,
        session_id: str | None = None,
        tool: str | None = None,
        status: GateStatus | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Searches the audit log with filters.

        Args:
            session_id: Filter by session ID.
            tool: Filter by tool name.
            status: Filter by gatekeeper status.
            since: Only entries after this point in time.
            limit: Maximum number of results.

        Returns:
            List of matching audit entries.
        """
        if not self._log_path.exists():
            return []

        results: list[dict[str, Any]] = []
        try:
            with open(self._log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)

                    # Events do not have action_tool
                    if entry.get("event_type"):
                        if tool or status:
                            continue
                        if session_id and entry.get("session_id") != session_id:
                            continue
                        if since:
                            ts = entry.get("timestamp", "")
                            try:
                                entry_time = datetime.fromisoformat(ts)
                                if entry_time < since:
                                    continue
                            except ValueError:
                                continue
                        results.append(entry)
                        if len(results) >= limit:
                            break
                        continue

                    if session_id and entry.get("session_id") != session_id:
                        continue
                    if tool and entry.get("action_tool") != tool:
                        continue
                    if status and entry.get("decision_status") != status.value:
                        continue
                    if since:
                        ts = entry.get("timestamp", "")
                        try:
                            entry_time = datetime.fromisoformat(ts)
                            if entry_time < since:
                                continue
                        except ValueError:
                            continue

                    results.append(entry)
                    if len(results) >= limit:
                        break
        except (json.JSONDecodeError, OSError) as exc:
            log.debug("audit_query_read_error", error=str(exc), exc_info=True)

        return results

    def get_anchor(self) -> dict[str, Any]:
        """Get current chain state for blockchain anchoring.

        Returns a dict suitable for writing to a blockchain or external store
        to prove the audit log existed in this exact state at this time.
        """
        return {
            "hash": self._last_hash,
            "entry_count": self._entry_count,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @property
    def entry_count(self) -> int:
        """Number of recorded entries."""
        return self._entry_count

    @property
    def last_hash(self) -> str:
        """Last hash in the chain."""
        return self._last_hash

    @property
    def log_path(self) -> Path:
        """Path to the audit log file."""
        return self._log_path

    def _entry_to_dict(self, entry: AuditEntry, *, mask: bool = True) -> dict[str, Any]:
        """Converts a flat AuditEntry into a serializable dict."""
        result: dict[str, Any] = {
            "timestamp": entry.timestamp.isoformat(),
            "session_id": entry.session_id,
            "action_tool": entry.action_tool,
            "action_params_hash": entry.action_params_hash,
            "decision_status": entry.decision_status.value,
            "decision_reason": entry.decision_reason,
            "risk_level": entry.risk_level.value,
            "policy_name": entry.policy_name,
            "user_override": entry.user_override,
        }
        if entry.execution_result:
            result["execution_result"] = (
                mask_credentials(entry.execution_result) if mask else entry.execution_result
            )
        if entry.error:
            result["error"] = entry.error
        return result
