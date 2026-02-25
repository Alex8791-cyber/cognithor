"""Auth-Gateway: Token-basierte Authentifizierung mit Per-Agent-Sessions.

Stellt bereit:
  - AuthGateway: Zentraler Authentifizierungs-Service
  - GatewayToken: JWT-ähnliche Token für Session-Identifizierung
  - AgentSession: Isolierte Login-Sessions pro Agent
  - TokenValidator: Validiert und erneuert Token

In Multi-Tenant-Szenarien erstellt das SSO-Gateway pro Agent
separate Login-Sessions, die über Gateway-Token unterschieden werden.

Bibel-Referenz: §14 (Security), §8 (Agent-Separation)
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class AuthMethod(Enum):
    """Unterstützte Authentifizierungs-Methoden."""

    TOKEN = "token"          # Gateway-Token
    API_KEY = "api_key"      # API-Schlüssel
    SSO = "sso"              # Single-Sign-On
    LOCAL = "local"          # Lokale Authentifizierung


@dataclass
class GatewayToken:
    """Token für Session-Identifizierung.

    Enthält User-ID, Agent-ID und Ablaufzeit.
    Jeder Token ist eindeutig einem User+Agent zugeordnet.
    """

    token_id: str
    user_id: str
    agent_id: str
    token_hash: str  # SHA-256 Hash des Tokens (nicht das Token selbst)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    auth_method: AuthMethod = AuthMethod.TOKEN
    scopes: list[str] = field(default_factory=list)
    revoked: bool = False
    last_used: datetime | None = None
    use_count: int = 0

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

    @property
    def is_valid(self) -> bool:
        return not self.revoked and not self.is_expired

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "auth_method": self.auth_method.value,
            "scopes": self.scopes,
            "revoked": self.revoked,
            "is_valid": self.is_valid,
            "use_count": self.use_count,
        }


@dataclass
class AgentSession:
    """Isolierte Login-Session für einen Agenten.

    Jeder Agent bekommt eine eigene Session mit:
      - Eigenem Token
      - Eigenen Scopes/Permissions
      - Eigener Ablaufzeit
      - Getrenntem Audit-Trail
    """

    session_id: str
    user_id: str
    agent_id: str
    token_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_key(self) -> str:
        return f"{self.user_id}:{self.agent_id}:{self.session_id}"

    def touch(self) -> None:
        """Aktualisiert den Zeitstempel der letzten Aktivität."""
        self.last_activity = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }


class AuthGateway:
    """Zentraler Authentifizierungs-Service.

    Verwaltet:
      - Token-Erstellung und -Validierung
      - Per-Agent-Sessions
      - Berechtigungs-Prüfung
      - Token-Revokation
      - Audit-Trail

    SSO-Flow:
      1. User authentifiziert sich (lokal oder SSO)
      2. Gateway erstellt pro Agent ein Token
      3. Token wird bei jedem Request mitgesendet
      4. Gateway validiert Token und gibt Kontext zurück
    """

    DEFAULT_TOKEN_TTL = 86400  # 24 Stunden in Sekunden

    def __init__(self, *, token_ttl: int = DEFAULT_TOKEN_TTL) -> None:
        self._tokens: dict[str, GatewayToken] = {}  # token_id → Token
        self._token_hashes: dict[str, str] = {}      # hash → token_id
        self._sessions: dict[str, AgentSession] = {}  # session_key → Session
        self._user_sessions: dict[str, list[str]] = {}  # user_id → [session_keys]
        self._token_ttl = token_ttl
        self._audit: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Token-Erstellung
    # ------------------------------------------------------------------

    def create_token(
        self,
        user_id: str,
        agent_id: str,
        *,
        auth_method: AuthMethod = AuthMethod.TOKEN,
        scopes: list[str] | None = None,
        ttl_seconds: int | None = None,
    ) -> tuple[str, GatewayToken]:
        """Erstellt ein neues Gateway-Token.

        Returns:
            (raw_token, token_object) - Raw-Token nur einmal zurückgegeben!
        """
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        token_id = secrets.token_hex(8)

        ttl = ttl_seconds if ttl_seconds is not None else self._token_ttl
        from datetime import timedelta
        expires = datetime.now(UTC) + timedelta(seconds=ttl) if ttl > 0 else None

        token = GatewayToken(
            token_id=token_id,
            user_id=user_id,
            agent_id=agent_id,
            token_hash=token_hash,
            auth_method=auth_method,
            scopes=scopes or [],
            expires_at=expires,
        )

        self._tokens[token_id] = token
        self._token_hashes[token_hash] = token_id

        self._audit_log("token_created", user_id, agent_id, token_id=token_id)
        log.info("auth_token_created", user_id=user_id, agent_id=agent_id)

        return raw_token, token

    def validate_token(self, raw_token: str) -> GatewayToken | None:
        """Validiert ein Token und gibt das Token-Objekt zurück."""
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        token_id = self._token_hashes.get(token_hash)
        if not token_id:
            return None

        token = self._tokens.get(token_id)
        if not token or not token.is_valid:
            return None

        token.last_used = datetime.now(UTC)
        token.use_count += 1
        return token

    def revoke_token(self, token_id: str) -> bool:
        """Widerruft ein Token und entfernt es aus dem Hash-Index."""
        token = self._tokens.get(token_id)
        if not token:
            return False
        token.revoked = True
        # Aus Hash-Index entfernen um Memory Leak zu verhindern
        self._token_hashes = {
            h: tid for h, tid in self._token_hashes.items() if tid != token_id
        }
        self._audit_log("token_revoked", token.user_id, token.agent_id, token_id=token_id)
        return True

    def revoke_all_for_user(self, user_id: str) -> int:
        """Widerruft alle Tokens eines Users."""
        count = 0
        revoked_ids: list[str] = []
        for tid, token in self._tokens.items():
            if token.user_id == user_id and not token.revoked:
                token.revoked = True
                revoked_ids.append(tid)
                count += 1
        # Hash-Index aufräumen um Memory Leak zu verhindern
        if revoked_ids:
            revoked_set = set(revoked_ids)
            self._token_hashes = {
                h: tid for h, tid in self._token_hashes.items()
                if tid not in revoked_set
            }
            self._audit_log("all_tokens_revoked", user_id, "", count=count)
        return count

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(
        self,
        user_id: str,
        agent_id: str,
        token_id: str,
    ) -> AgentSession:
        """Erstellt eine isolierte Agent-Session."""
        session_id = secrets.token_hex(8)
        session = AgentSession(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            token_id=token_id,
        )

        self._sessions[session.session_key] = session
        self._user_sessions.setdefault(user_id, []).append(session.session_key)

        self._audit_log("session_created", user_id, agent_id, session_id=session_id)
        return session

    def get_session(self, user_id: str, agent_id: str) -> AgentSession | None:
        """Gibt die aktive Session für User+Agent zurück."""
        for key in self._user_sessions.get(user_id, []):
            session = self._sessions.get(key)
            if session and session.agent_id == agent_id and session.active:
                return session
        return None

    def end_session(self, session_key: str) -> bool:
        """Beendet eine Session und entfernt sie aus dem Speicher."""
        session = self._sessions.get(session_key)
        if not session:
            return False
        session.active = False
        # Session aus _sessions entfernen um Memory Leak zu verhindern
        del self._sessions[session_key]
        # Clean up _user_sessions
        user_keys = self._user_sessions.get(session.user_id)
        if user_keys and session_key in user_keys:
            user_keys.remove(session_key)
        self._audit_log("session_ended", session.user_id, session.agent_id)
        return True

    def cleanup_expired(self) -> dict[str, int]:
        """Entfernt abgelaufene/widerrufene Tokens und inaktive Sessions.

        Sollte periodisch aufgerufen werden um Memory Leaks zu verhindern.

        Returns:
            Dict mit Anzahl entfernter Tokens und Sessions.
        """
        from datetime import timedelta

        now = datetime.now(UTC)
        one_hour_ago = now - timedelta(hours=1)

        # Revoked Tokens entfernen die älter als 1h sind
        expired_token_ids = [
            tid for tid, token in self._tokens.items()
            if (token.revoked and token.created_at < one_hour_ago)
            or (token.is_expired and token.expires_at and token.expires_at < one_hour_ago)
        ]
        for tid in expired_token_ids:
            token = self._tokens.pop(tid, None)
            if token:
                self._token_hashes = {
                    h: t for h, t in self._token_hashes.items() if t != tid
                }

        # Inaktive Sessions entfernen
        inactive_keys = [
            key for key, session in self._sessions.items()
            if not session.active
        ]
        for key in inactive_keys:
            session = self._sessions.pop(key, None)
            if session:
                user_keys = self._user_sessions.get(session.user_id)
                if user_keys and key in user_keys:
                    user_keys.remove(key)

        return {
            "tokens_removed": len(expired_token_ids),
            "sessions_removed": len(inactive_keys),
        }

    def user_sessions(self, user_id: str) -> list[AgentSession]:
        """Alle Sessions eines Users."""
        return [
            self._sessions[key]
            for key in self._user_sessions.get(user_id, [])
            if key in self._sessions
        ]

    def active_sessions(self, user_id: str) -> list[AgentSession]:
        """Nur aktive Sessions eines Users."""
        return [s for s in self.user_sessions(user_id) if s.active]

    # ------------------------------------------------------------------
    # SSO-Login-Flow
    # ------------------------------------------------------------------

    def login(
        self,
        user_id: str,
        agent_ids: list[str],
        *,
        auth_method: AuthMethod = AuthMethod.SSO,
    ) -> dict[str, tuple[str, AgentSession]]:
        """SSO-Login: Erstellt pro Agent ein Token + Session.

        Returns:
            Dict von agent_id → (raw_token, session)
        """
        result: dict[str, tuple[str, AgentSession]] = {}
        for agent_id in agent_ids:
            raw_token, token = self.create_token(
                user_id, agent_id, auth_method=auth_method,
            )
            session = self.create_session(user_id, agent_id, token.token_id)
            result[agent_id] = (raw_token, session)

        self._audit_log("sso_login", user_id, "", agents=agent_ids)
        return result

    def logout(self, user_id: str) -> int:
        """Logout: Beendet alle Sessions und widerruft alle Tokens."""
        revoked = self.revoke_all_for_user(user_id)
        # Sessions aus _sessions entfernen um Memory Leak zu verhindern
        for key in list(self._user_sessions.get(user_id, [])):
            session = self._sessions.pop(key, None)
            if session:
                session.active = False
        # _user_sessions-Eintrag komplett entfernen
        self._user_sessions.pop(user_id, None)
        self._audit_log("logout", user_id, "")
        return revoked

    # ------------------------------------------------------------------
    # Berechtigungen
    # ------------------------------------------------------------------

    def check_scope(self, token: GatewayToken, required_scope: str) -> bool:
        """Prüft ob ein Token den erforderlichen Scope hat."""
        if not token.scopes:
            return False  # Keine Scopes = kein Zugriff (deny-by-default)
        return required_scope in token.scopes or "*" in token.scopes

    # ------------------------------------------------------------------
    # Statistiken
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Auth-Gateway-Statistiken."""
        tokens = list(self._tokens.values())
        sessions = list(self._sessions.values())
        return {
            "total_tokens": len(tokens),
            "active_tokens": sum(1 for t in tokens if t.is_valid),
            "revoked_tokens": sum(1 for t in tokens if t.revoked),
            "expired_tokens": sum(1 for t in tokens if t.is_expired),
            "total_sessions": len(sessions),
            "active_sessions": sum(1 for s in sessions if s.active),
            "unique_users": len(self._user_sessions),
            "audit_entries": len(self._audit),
        }

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        return list(self._audit)

    _AUDIT_MAX_SIZE: int = 10_000

    def _audit_log(self, action: str, user_id: str, agent_id: str, **extra: Any) -> None:
        if len(self._audit) >= self._AUDIT_MAX_SIZE:
            self._audit = self._audit[-self._AUDIT_MAX_SIZE // 2 :]
        self._audit.append({
            "action": action,
            "user_id": user_id,
            "agent_id": agent_id,
            "timestamp": datetime.now(UTC).isoformat(),
            **extra,
        })
