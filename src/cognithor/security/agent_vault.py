"""Cognithor · Per-Agent Vault & Session-Isolation.

Complete data and session separation per agent:

  - AgentSecret:            A secret with rotation and expiration
  - AgentVault:             Isolated vault per agent
  - VaultRotator:           Automatic credential rotation
  - IsolatedSessionStore:   Separate session stores per agent
  - SessionFirewall:        Cross-session access control
  - AgentVaultManager:      Central management of all agent vaults

Architecture Bible: §14.5 (Secrets-Management), §17.3 (Multi-Tenant-Isolation)
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import secrets
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ============================================================================
# Agent Secret
# ============================================================================


class SecretType(Enum):
    API_KEY = "api_key"
    TOKEN = "token"
    PASSWORD = "password"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    ENCRYPTION_KEY = "encryption_key"
    WEBHOOK_SECRET = "webhook_secret"
    DATABASE_URL = "database_url"


class SecretStatus(Enum):
    ACTIVE = "active"
    ROTATED = "rotated"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class AgentSecret:
    """A secret with lifecycle management."""

    secret_id: str
    agent_id: str
    name: str
    secret_type: SecretType
    status: SecretStatus = SecretStatus.ACTIVE
    created_at: str = ""
    expires_at: str = ""
    last_rotated: str = ""
    rotation_count: int = 0
    # The actual value is only stored encrypted
    _encrypted_value: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "secret_id": self.secret_id,
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.secret_type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "rotation_count": self.rotation_count,
        }

    @property
    def is_active(self) -> bool:
        return self.status == SecretStatus.ACTIVE

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return self.expires_at < time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ============================================================================
# Agent Vault (isolated vault per agent)
# ============================================================================


class AgentVault:
    """Isolated secret vault for a single agent.

    Each agent has its own vault with its own namespace.
    Cross-agent access is not possible.

    The encryption key is derived deterministically from ``agent_id`` **plus**
    an external ``master_secret``.  Identical inputs produce the same key,
    so encrypted secrets survive process restarts.  Without knowledge of
    the ``master_secret`` the key cannot be reconstructed.
    """

    def __init__(self, agent_id: str, *, master_secret: bytes = b"") -> None:
        self._agent_id = agent_id
        self._secrets: dict[str, AgentSecret] = {}
        self._counter = 0
        self._access_log: list[dict[str, Any]] = []
        # Deterministic key derivation: agent_id + master_secret.
        # The master_secret is managed by AgentVaultManager (generated
        # once, persisted to ~/.cognithor/vault_master.key).  Without
        # master_secret the key depends only on agent_id (legacy compat).
        self._salt = hashlib.sha256(f"vault-salt:{agent_id}".encode()).digest()[:16]
        raw_key_material = f"vault:{agent_id}".encode() + master_secret
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=600_000,
        )
        fernet_key = base64.urlsafe_b64encode(kdf.derive(raw_key_material))
        self._fernet = Fernet(fernet_key)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def store(
        self,
        name: str,
        value: str,
        secret_type: SecretType = SecretType.API_KEY,
        *,
        ttl_hours: int = 0,
    ) -> AgentSecret:
        """Stores a new secret in the vault."""
        self._counter += 1
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        expires = ""
        if ttl_hours > 0:
            expires_ts = time.time() + ttl_hours * 3600
            expires = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires_ts))

        secret = AgentSecret(
            secret_id=f"SEC-{self._agent_id[:6]}-{self._counter:04d}",
            agent_id=self._agent_id,
            name=name,
            secret_type=secret_type,
            created_at=now,
            expires_at=expires,
            _encrypted_value=self._encrypt(value),
        )
        self._secrets[secret.secret_id] = secret
        self._log("store", secret.secret_id)
        return secret

    def retrieve(self, secret_id: str) -> str | None:
        """Retrieves a secret (only for the owning agent)."""
        secret = self._secrets.get(secret_id)
        if not secret or not secret.is_active:
            self._log("retrieve_failed", secret_id)
            return None
        self._log("retrieve", secret_id)
        return self._decrypt(secret._encrypted_value)

    def rotate(self, secret_id: str, new_value: str) -> AgentSecret | None:
        """Rotates a secret (new value, same ID)."""
        secret = self._secrets.get(secret_id)
        if not secret:
            return None
        secret._encrypted_value = self._encrypt(new_value)
        secret.last_rotated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        secret.rotation_count += 1
        self._log("rotate", secret_id)
        return secret

    def revoke(self, secret_id: str) -> bool:
        """Revokes a secret and removes it from the vault."""
        secret = self._secrets.get(secret_id)
        if not secret:
            return False
        secret.status = SecretStatus.REVOKED
        secret._encrypted_value = ""
        self._log("revoke", secret_id)
        # Remove revoked secrets from the vault (no reason to keep them)
        del self._secrets[secret_id]
        return True

    def expire_check(self) -> list[AgentSecret]:
        """Checks and marks expired secrets."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        expired = []
        for secret in self._secrets.values():
            if secret.is_active and secret.expires_at and secret.expires_at < now:
                secret.status = SecretStatus.EXPIRED
                expired.append(secret)
        return expired

    def active_secrets(self) -> list[AgentSecret]:
        return [s for s in self._secrets.values() if s.is_active]

    def all_secrets(self) -> list[AgentSecret]:
        return list(self._secrets.values())

    @property
    def secret_count(self) -> int:
        return len(self._secrets)

    def access_log(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(reversed(self._access_log[-limit:]))

    def _encrypt(self, value: str) -> str:
        """Fernet-based authenticated encryption."""
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def _decrypt(self, encrypted: str) -> str:
        """Fernet-based decryption with fallback for legacy XOR-encrypted data."""
        try:
            return self._fernet.decrypt(encrypted.encode("ascii")).decode("utf-8")
        except InvalidToken:
            # Fallback: attempt to detect legacy XOR-encrypted hex data.
            # This path is only reached for data encrypted before the Fernet
            # migration and will be removed in a future release.
            with contextlib.suppress(ValueError):
                bytes.fromhex(encrypted)
                # Legacy XOR cannot be decrypted without the old key material,
                # so we surface the failure clearly.
            raise ValueError(
                "Decryption failed: token is neither valid Fernet nor "
                "recoverable legacy data. Re-encrypt the secret."
            ) from None

    def _log(self, action: str, target: str) -> None:
        self._access_log.append(
            {
                "action": action,
                "target": target,
                "agent_id": self._agent_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    def stats(self) -> dict[str, Any]:
        all_s = list(self._secrets.values())
        return {
            "agent_id": self._agent_id,
            "total_secrets": len(all_s),
            "active": sum(1 for s in all_s if s.status == SecretStatus.ACTIVE),
            "expired": sum(1 for s in all_s if s.status == SecretStatus.EXPIRED),
            "revoked": sum(1 for s in all_s if s.status == SecretStatus.REVOKED),
            "access_events": len(self._access_log),
        }


# ============================================================================
# Vault Rotator
# ============================================================================


@dataclass
class RotationPolicy:
    """Rotation policy for credentials."""

    policy_id: str
    secret_type: SecretType
    rotation_interval_hours: int = 720  # 30 days
    max_age_hours: int = 2160  # 90 days
    auto_rotate: bool = True
    notify_before_hours: int = 168  # 7 days before expiration

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "type": self.secret_type.value,
            "interval_h": self.rotation_interval_hours,
            "max_age_h": self.max_age_hours,
            "auto": self.auto_rotate,
        }


class VaultRotator:
    """Automatic credential rotation across all agent vaults."""

    DEFAULT_POLICIES = [
        RotationPolicy("ROT-API", SecretType.API_KEY, 720, 2160),
        RotationPolicy("ROT-TOK", SecretType.TOKEN, 24, 168, True, 4),
        RotationPolicy("ROT-PWD", SecretType.PASSWORD, 2160, 8760),
        RotationPolicy("ROT-CERT", SecretType.CERTIFICATE, 8760, 26280),
    ]

    def __init__(self, load_defaults: bool = True) -> None:
        self._policies: dict[str, RotationPolicy] = {}
        self._rotation_log: list[dict[str, Any]] = []
        if load_defaults:
            for p in self.DEFAULT_POLICIES:
                self._policies[p.policy_id] = p

    def add_policy(self, policy: RotationPolicy) -> None:
        self._policies[policy.policy_id] = policy

    def get_policy(self, secret_type: SecretType) -> RotationPolicy | None:
        return next(
            (p for p in self._policies.values() if p.secret_type == secret_type),
            None,
        )

    def check_rotation_needed(self, vault: AgentVault) -> list[AgentSecret]:
        """Checks which secrets need rotation."""
        needs_rotation = []
        now_ts = time.time()
        for secret in vault.active_secrets():
            policy = self.get_policy(secret.secret_type)
            if not policy or not policy.auto_rotate:
                continue
            # Check age
            last_change = secret.last_rotated or secret.created_at
            if last_change:
                try:
                    import calendar

                    change_ts = calendar.timegm(time.strptime(last_change, "%Y-%m-%dT%H:%M:%SZ"))
                except (ValueError, OverflowError):
                    change_ts = now_ts
                age_hours = (now_ts - change_ts) / 3600
                if age_hours > policy.rotation_interval_hours:
                    needs_rotation.append(secret)
        return needs_rotation

    def auto_rotate(self, vault: AgentVault) -> list[str]:
        """Performs automatic rotation."""
        rotated_ids = []
        for secret in vault.active_secrets():
            policy = self.get_policy(secret.secret_type)
            if not policy or not policy.auto_rotate:
                continue
            # Generate new value
            new_value = secrets.token_urlsafe(32)
            vault.rotate(secret.secret_id, new_value)
            rotated_ids.append(secret.secret_id)
            self._rotation_log.append(
                {
                    "agent_id": vault.agent_id,
                    "secret_id": secret.secret_id,
                    "type": secret.secret_type.value,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
        return rotated_ids

    @property
    def policy_count(self) -> int:
        return len(self._policies)

    def stats(self) -> dict[str, Any]:
        return {
            "policies": len(self._policies),
            "total_rotations": len(self._rotation_log),
            "policies_list": [p.to_dict() for p in self._policies.values()],
        }


# ============================================================================
# Isolated Session Store
# ============================================================================


@dataclass
class AgentSession:
    """An isolated session for an agent."""

    session_id: str
    agent_id: str
    tenant_id: str = ""
    created_at: str = ""
    last_active: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "active": self.is_active,
            "created_at": self.created_at,
            "data_keys": list(self.data.keys()),
        }


class IsolatedSessionStore:
    """Fully separated session stores per agent.

    Each agent has its own namespace. Cross-agent access
    is prevented by the SessionFirewall.
    """

    def __init__(self) -> None:
        self._stores: dict[str, dict[str, AgentSession]] = {}
        self._counter = 0

    def create_session(
        self, agent_id: str, tenant_id: str = "", data: dict[str, Any] | None = None
    ) -> AgentSession:
        """Creates a new session for an agent."""
        self._counter += 1
        session = AgentSession(
            session_id=f"SESS-{agent_id[:6]}-{self._counter:04d}",
            agent_id=agent_id,
            tenant_id=tenant_id,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_active=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            data=data or {},
        )
        if agent_id not in self._stores:
            self._stores[agent_id] = {}
        self._stores[agent_id][session.session_id] = session
        return session

    def get_session(self, agent_id: str, session_id: str) -> AgentSession | None:
        """Gets a session -- only from the owning store."""
        store = self._stores.get(agent_id)
        if not store:
            return None
        return store.get(session_id)

    def close_session(self, agent_id: str, session_id: str) -> bool:
        store = self._stores.get(agent_id)
        if not store:
            return False
        session = store.get(session_id)
        if not session:
            return False
        session.is_active = False
        return True

    def destroy_session(self, agent_id: str, session_id: str) -> bool:
        store = self._stores.get(agent_id)
        if not store:
            return False
        if session_id in store:
            del store[session_id]
            return True
        return False

    def agent_sessions(self, agent_id: str) -> list[AgentSession]:
        store = self._stores.get(agent_id, {})
        return list(store.values())

    def active_sessions(self, agent_id: str) -> list[AgentSession]:
        return [s for s in self.agent_sessions(agent_id) if s.is_active]

    def purge_agent(self, agent_id: str) -> int:
        """Deletes all sessions for an agent (on compromise)."""
        store = self._stores.pop(agent_id, {})
        return len(store)

    @property
    def store_count(self) -> int:
        return len(self._stores)

    @property
    def total_sessions(self) -> int:
        return sum(len(s) for s in self._stores.values())

    def stats(self) -> dict[str, Any]:
        total = 0
        active = 0
        for store in self._stores.values():
            for sess in store.values():
                total += 1
                if sess.is_active:
                    active += 1
        return {
            "agent_stores": len(self._stores),
            "total_sessions": total,
            "active_sessions": active,
        }


# ============================================================================
# Session Firewall
# ============================================================================


class SessionFirewall:
    """Prevents cross-agent session access."""

    def __init__(self, session_store: IsolatedSessionStore) -> None:
        self._store = session_store
        self._violations: list[dict[str, Any]] = []

    def authorize(self, requesting_agent: str, target_agent: str, session_id: str) -> bool:
        """Checks if access is allowed."""
        if requesting_agent != target_agent:
            self._violations.append(
                {
                    "requester": requesting_agent,
                    "target": target_agent,
                    "session_id": session_id,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "action": "BLOCKED",
                }
            )
            return False
        return True

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def violations(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(reversed(self._violations[-limit:]))

    def stats(self) -> dict[str, Any]:
        return {
            "total_violations": len(self._violations),
            "unique_attackers": len(set(v["requester"] for v in self._violations)),
        }


# ============================================================================
# Agent Vault Manager (Central Management)
# ============================================================================


def _load_or_create_master_secret(path: str | None = None) -> bytes:
    """Loads or generates the vault master secret.

    The secret is stored in ``~/.cognithor/vault_master.key``
    (or in the provided ``path``).  It has 32 bytes of entropy and
    is generated only once.
    """
    from pathlib import Path

    if path is None:
        home = Path(os.environ.get("COGNITHOR_HOME", Path.home() / ".cognithor"))
        key_file = home / "vault_master.key"
    else:
        key_file = Path(path)

    if key_file.exists():
        raw = key_file.read_bytes()
        if len(raw) >= 32:
            return raw[:32]
        # Corrupt/truncated file — regenerate.

    # Generate 32 bytes of cryptographic randomness.
    master = os.urandom(32)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_bytes(master)
    # Restrict permissions: owner-only read/write.
    with contextlib.suppress(OSError):
        key_file.chmod(0o600)
    # On Windows, restrict ACL to current user only.
    if sys.platform == "win32":
        try:
            import subprocess

            username = os.environ.get("USERNAME", "")
            if username:
                subprocess.run(
                    ["icacls", str(key_file), "/inheritance:r", "/grant:r", f"{username}:(R,W)"],
                    capture_output=True,
                    timeout=10,
                )
        except Exception:
            pass  # Best-effort ACL restriction
    return master


class AgentVaultManager:
    """Central management of all agent vaults + sessions.

    On startup a master secret is loaded (or generated),
    which flows into the key derivation of each ``AgentVault``.
    """

    def __init__(self, *, master_secret_path: str | None = None) -> None:
        self._master_secret = _load_or_create_master_secret(master_secret_path)
        self._vaults: dict[str, AgentVault] = {}
        self._sessions = IsolatedSessionStore()
        self._firewall = SessionFirewall(self._sessions)
        self._rotator = VaultRotator()

    @property
    def sessions(self) -> IsolatedSessionStore:
        return self._sessions

    @property
    def firewall(self) -> SessionFirewall:
        return self._firewall

    @property
    def rotator(self) -> VaultRotator:
        return self._rotator

    def create_vault(self, agent_id: str) -> AgentVault:
        vault = AgentVault(agent_id, master_secret=self._master_secret)
        self._vaults[agent_id] = vault
        return vault

    def get_vault(self, agent_id: str) -> AgentVault | None:
        return self._vaults.get(agent_id)

    def destroy_vault(self, agent_id: str) -> bool:
        """Destroys vault + all sessions for an agent."""
        if agent_id in self._vaults:
            # Revoke all secrets
            vault = self._vaults[agent_id]
            for secret in vault.all_secrets():
                vault.revoke(secret.secret_id)
            del self._vaults[agent_id]
            # Purge sessions
            self._sessions.purge_agent(agent_id)
            return True
        return False

    def rotate_all(self) -> dict[str, list[str]]:
        """Rotates credentials in all vaults."""
        results = {}
        for agent_id, vault in self._vaults.items():
            rotated = self._rotator.auto_rotate(vault)
            if rotated:
                results[agent_id] = rotated
        return results

    @property
    def vault_count(self) -> int:
        return len(self._vaults)

    def stats(self) -> dict[str, Any]:
        vaults = list(self._vaults.values())
        return {
            "total_vaults": len(vaults),
            "total_secrets": sum(v.secret_count for v in vaults),
            "sessions": self._sessions.stats(),
            "firewall": self._firewall.stats(),
            "rotation": self._rotator.stats(),
        }
