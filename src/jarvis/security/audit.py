"""Audit-Trail: Unveränderliches Protokoll aller Aktionen.

Jede Gatekeeper-Entscheidung, jede Tool-Ausführung, jeder
Sub-Agent-Spawn wird protokolliert. Append-only, tamper-evident
via SHA-256-Chain.

Sicherheitsgarantien:
  - Einträge können nur hinzugefügt, nie gelöscht werden
  - Jeder Eintrag enthält den Hash des vorherigen → Kette
  - JSONL-Format für einfache Analyse
  - Credential-Werte werden VOR dem Logging maskiert

Bibel-Referenz: §3.2 (Audit-Log), §11.5 (Audit Trail)
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.models import AuditEntry, GateStatus

log = get_logger(__name__)

# Standard-Patterns für Credential-Maskierung im Audit-Log
_CREDENTIAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(sk-[a-zA-Z0-9]{4})[a-zA-Z0-9]{16,}"),
    re.compile(r"(token_[a-zA-Z0-9]{4})[a-zA-Z0-9]+"),
    re.compile(r"(password\s*[:=]\s*)\S+", re.IGNORECASE),
    re.compile(r"(secret\s*[:=]\s*)\S+", re.IGNORECASE),
    re.compile(r"(api_key\s*[:=]\s*)\S+", re.IGNORECASE),
    re.compile(r"(Bearer\s+)[a-zA-Z0-9._\-]{8,}"),
    re.compile(r"(ghp_)[a-zA-Z0-9]{30,}"),
    re.compile(r"(xox[baprs]-)[a-zA-Z0-9\-]+"),
]


def mask_credentials(text: str) -> str:
    """Maskiert Credentials in einem Text.

    Ersetzt erkannte Patterns durch teilmaskierte Versionen.
    z.B. 'sk-abc123456789' → 'sk-abc1***'
    """
    if not text:
        return text
    result = text
    for pattern in _CREDENTIAL_PATTERNS:
        result = pattern.sub(lambda m: m.group(1) + "***", result)
    return result


def mask_dict(data: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Maskiert Credentials in einem verschachtelten Dict rekursiv.

    Args:
        data: Dict mit potentiellen Credentials.
        depth: Aktuelle Verschachtelungstiefe (max 10).

    Returns:
        Kopie des Dicts mit maskierten Werten.
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
    """Berechnet SHA-256-Hash für Chain-Integrität."""
    return hashlib.sha256(f"{prev_hash}|{data}".encode()).hexdigest()


class AuditTrail:
    """Unveränderliches, append-only Audit-Protokoll. [B§11.5]

    Schreibt JSONL-Dateien mit Hash-Chain für Tamper-Evidence.
    Credentials werden automatisch maskiert.
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self._log_dir = log_dir or Path.home() / ".jarvis" / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / "audit.jsonl"
        self._last_hash = "genesis"
        self._entry_count = 0

        # Chain vom letzten Eintrag fortsetzen
        self._restore_chain()

    def _restore_chain(self) -> None:
        """Stellt den letzten Hash aus dem Log wieder her."""
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
        """Protokolliert einen Audit-Eintrag.

        Args:
            entry: Der AuditEntry (flache Struktur).
            mask: Credentials im execution_result maskieren (default: True).

        Returns:
            Hash des Eintrags.
        """
        record = self._entry_to_dict(entry, mask=mask)
        data_str = json.dumps(record, ensure_ascii=False, sort_keys=True)
        entry_hash = _compute_hash(data_str, self._last_hash)
        record["prev_hash"] = self._last_hash
        record["hash"] = entry_hash

        line = json.dumps(record, ensure_ascii=False)
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            log.error("audit_write_failed", error=str(exc))
            return ""

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
        """Protokolliert ein freies Event (z.B. Agent-Spawn, Login).

        Args:
            session_id: Session-ID.
            event_type: Typ des Events (z.B. 'agent_spawn', 'auth_success').
            details: Zusätzliche Details.

        Returns:
            Hash des Eintrags.
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

        line = json.dumps(record, ensure_ascii=False)
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            log.error("audit_event_write_failed", error=str(exc))
            return ""

        self._last_hash = entry_hash
        self._entry_count += 1
        return entry_hash

    def verify_chain(self) -> tuple[bool, int, int]:
        """Prüft die Integrität der Hash-Chain.

        Returns:
            Tuple von (valid, total_entries, broken_at).
            broken_at ist -1 wenn die Chain intakt ist.
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
                        k: v for k, v in entry.items() if k not in ("prev_hash", "hash")
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

    def query(
        self,
        session_id: str | None = None,
        tool: str | None = None,
        status: GateStatus | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Durchsucht das Audit-Log mit Filtern.

        Args:
            session_id: Filter nach Session-ID.
            tool: Filter nach Tool-Name.
            status: Filter nach Gatekeeper-Status.
            since: Nur Einträge nach diesem Zeitpunkt.
            limit: Maximale Anzahl Ergebnisse.

        Returns:
            Liste passender Audit-Einträge.
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

                    # Events haben kein action_tool
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
        except (json.JSONDecodeError, OSError):
            pass

        return results

    @property
    def entry_count(self) -> int:
        """Anzahl der protokollierten Einträge."""
        return self._entry_count

    @property
    def last_hash(self) -> str:
        """Letzter Hash in der Chain."""
        return self._last_hash

    @property
    def log_path(self) -> Path:
        """Pfad zur Audit-Log-Datei."""
        return self._log_path

    def _entry_to_dict(self, entry: AuditEntry, *, mask: bool = True) -> dict[str, Any]:
        """Konvertiert flache AuditEntry in ein serialisierbares Dict."""
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
