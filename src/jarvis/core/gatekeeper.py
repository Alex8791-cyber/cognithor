"""Gatekeeper: Deterministic policy engine.

NO LLM. NO EXCEPTIONS. Purely rule-based.
Checks every single PlannedAction against the policy.

Security guarantees:
  - Destructive shell commands are ALWAYS blocked
  - Path access outside allowed directories is ALWAYS blocked
  - Credentials are ALWAYS masked
  - Every decision is immutably logged

Bible reference: §3.2 (Gatekeeper), §11 (Security)
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from jarvis.models import (
    AuditEntry,
    GateDecision,
    GateStatus,
    OperationMode,
    PlannedAction,
    PolicyMatch,
    PolicyParamMatch,
    PolicyRule,
    RiskLevel,
    SessionContext,
    ToolCapability,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.audit import AuditLogger
    from jarvis.config import JarvisConfig

log = get_logger(__name__)


class Gatekeeper:
    """Deterministic policy enforcer. No LLM. No exceptions. [B§3.2]

    Every action is checked against the policy before the Executor
    is allowed to execute it. The decision is based on:
      1. Policy rules (loaded from YAML)
      2. Risk classification (tool type + parameters)
      3. Path validation (only allowed directories)
      4. Credential detection (patterns in parameters)

    The four risk levels [B§3.2]:
      GREEN  -> ALLOW   (execute automatically)
      YELLOW -> INFORM  (execute + inform user)
      ORANGE -> APPROVE (user must confirm)
      RED    -> BLOCK   (blocked, manual release required)
    """

    # Tools die auch im OFFLINE-Modus Netzwerkzugriff haben duerfen (Recherche)
    _OFFLINE_ALLOWED_NETWORK_TOOLS: frozenset[str] = frozenset({
        "web_search", "web_fetch", "fetch_url",
    })

    def __init__(
        self,
        config: JarvisConfig,
        audit_logger: AuditLogger | None = None,
        operation_mode: OperationMode | None = None,
    ) -> None:
        """Initialisiert den Gatekeeper mit Security-Konfiguration und Policy-Regeln."""
        self._config = config
        self._audit_logger = audit_logger
        self._operation_mode = operation_mode
        self._policies: list[PolicyRule] = []
        self._credential_patterns: list[re.Pattern[str]] = []
        self._blocked_command_patterns: list[re.Pattern[str]] = []
        self._allowed_paths: list[Path] = []
        self._audit_path = config.logs_dir / "gatekeeper.jsonl"
        self._initialized = False

        # Optional: Capability Matrix (F8)
        self._capability_matrix: Any = None
        try:
            from jarvis.security.capabilities import CapabilityMatrix
            self._capability_matrix = CapabilityMatrix()
        except Exception:
            pass

    def initialize(self) -> None:
        """Lädt Policies und kompiliert Regex-Patterns.

        Wird einmal beim Start aufgerufen. Kompiliert alle Regex
        beim Laden, nicht bei jedem evaluate()-Aufruf.
        """
        # Policies laden
        self._policies = self._load_policies()
        self._policies.sort(key=lambda r: r.priority, reverse=True)

        # Credential-Patterns kompilieren
        self._credential_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self._config.security.credential_patterns
        ]

        # Blockierte Befehle kompilieren
        self._blocked_command_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self._config.security.blocked_commands
        ]

        # Erlaubte Pfade normalisieren
        self._allowed_paths = [
            Path(p).expanduser().resolve() for p in self._config.security.allowed_paths
        ]

        self._initialized = True

        log.info(
            "gatekeeper_initialized",
            policy_count=len(self._policies),
            credential_patterns=len(self._credential_patterns),
            blocked_commands=len(self._blocked_command_patterns),
            allowed_paths=[str(p) for p in self._allowed_paths],
        )

    # --- Public Policy API (fuer ReplayEngine u.a.) ---

    def get_policies(self) -> list[PolicyRule]:
        """Gibt eine Kopie der aktuellen Policy-Regeln zurueck."""
        return list(self._policies)

    def set_policies(self, policies: list[PolicyRule]) -> None:
        """Ersetzt die Policy-Regeln (fuer Replay/Testing)."""
        self._policies = sorted(policies, key=lambda r: r.priority, reverse=True)

    def evaluate(
        self,
        action: PlannedAction,
        context: SessionContext,
    ) -> GateDecision:
        """Prüft eine einzelne PlannedAction gegen alle Policies. [B§3.2]

        Reihenfolge der Prüfungen (first-match wins bei BLOCK):
          1. Credential-Scan → MASK wenn gefunden
          2. Explizite Policy-Regeln → Ergebnis der Regel
          3. Pfad-Validierung → BLOCK wenn außerhalb
          4. Destruktive Befehls-Patterns → BLOCK
          5. Default-Risiko-Klassifizierung → GREEN/YELLOW/ORANGE/RED

        Args:
            action: Die zu prüfende Aktion
            context: Session-Kontext (für kontextabhängige Regeln)

        Returns:
            GateDecision mit Status, Grund, und ggf. maskierten Params
        """
        if not self._initialized:
            self.initialize()

        # --- Schritt 0: OperationMode-Enforcement ---
        if self._operation_mode == OperationMode.OFFLINE:
            tool_lower = action.tool.lower()
            if tool_lower not in self._OFFLINE_ALLOWED_NETWORK_TOOLS:
                tool_caps = (
                    self._capability_matrix.get_spec(action.tool)
                    if self._capability_matrix
                    else None
                )
                if tool_caps and (
                    ToolCapability.NETWORK_HTTP in tool_caps.capabilities
                    or ToolCapability.NETWORK_WS in tool_caps.capabilities
                ):
                    decision = GateDecision(
                        status=GateStatus.BLOCK,
                        reason=f"Tool '{action.tool}' benoetigt Netzwerk, aber System ist im OFFLINE-Modus",
                        risk_level=RiskLevel.RED,
                        original_action=action,
                        policy_name="operation_mode_offline",
                    )
                    self._write_audit(action, decision, context)
                    return decision

        # --- Schritt 1: Credential-Scan ---
        masked_params, has_credentials = self._scan_credentials(action.params)
        if has_credentials:
            decision = GateDecision(
                status=GateStatus.MASK,
                reason="Credential in Parametern erkannt -- maskiert",
                risk_level=RiskLevel.YELLOW,
                original_action=action,
                masked_params=masked_params,
                policy_name="credential_masking",
            )
            self._write_audit(action, decision, context)
            return decision

        # --- Schritt 2: Explizite Policy-Regeln (höchste Priorität zuerst) ---
        for rule in self._policies:
            if self._matches_rule(action, rule):
                risk = self._status_to_risk(rule.action)
                decision = GateDecision(
                    status=rule.action,
                    reason=rule.reason,
                    risk_level=risk,
                    original_action=action,
                    policy_name=rule.name,
                )
                self._write_audit(action, decision, context)
                log.debug(
                    "gatekeeper_policy_match",
                    tool=action.tool,
                    policy=rule.name,
                    status=rule.action.value,
                )
                return decision

        # --- Schritt 3: Pfad-Validierung ---
        path_check = self._validate_paths(action)
        if path_check is not None:
            self._write_audit(action, path_check, context)
            return path_check

        # --- Schritt 4: Destruktive Shell-Befehle ---
        if action.tool in ("exec_command", "shell_exec", "shell"):
            cmd = str(action.params.get("command", ""))
            cmd_check = self._check_command(cmd, action)
            if cmd_check is not None:
                self._write_audit(action, cmd_check, context)
                return cmd_check

        # --- Schritt 5: Capability-Matrix-Check (optional) ---
        if self._capability_matrix is not None:
            try:
                # Only check known tools -- unknown tools fall through to default
                spec = self._capability_matrix.get_spec(action.tool)
                if spec is not None:
                    from jarvis.security.capabilities import STANDARD as _std_profile
                    violations = self._capability_matrix.get_violations(action.tool, _std_profile)
                    if violations:
                        decision = GateDecision(
                            status=GateStatus.BLOCK,
                            reason=f"Capability-Verletzung: {', '.join(violations)}",
                            risk_level=RiskLevel.RED,
                            original_action=action,
                            policy_name="capability_matrix",
                        )
                        self._write_audit(action, decision, context)
                        return decision
            except Exception:
                pass  # Matrix-Fehler ignorieren, Fallback auf Default

        # --- Schritt 6: Default-Risiko-Klassifizierung ---
        risk = self._classify_risk(action)
        status = self._risk_to_status(risk)
        decision = GateDecision(
            status=status,
            reason=f"Default-Klassifizierung: {risk.name}",
            risk_level=risk,
            original_action=action,
            policy_name="default_classification",
        )
        self._write_audit(action, decision, context)
        return decision

    def evaluate_plan(
        self,
        steps: list[PlannedAction],
        context: SessionContext,
    ) -> list[GateDecision]:
        """Prüft alle Schritte eines Plans.

        Returns:
            Liste von GateDecisions, eine pro Step.
        """
        return [self.evaluate(step, context) for step in steps]

    # =========================================================================
    # Private Methoden
    # =========================================================================

    def _classify_risk(self, action: PlannedAction) -> RiskLevel:
        """Klassifiziert das Risiko einer Aktion nach Tool-Typ. [B§3.2]"""
        tool = action.tool.lower()

        # GREEN: Read-Only Operationen
        green_tools = {
            "read_file",
            "list_directory",
            "search_memory",
            "get_entity",
            "search",
            "list_jobs",
        }
        if tool in green_tools:
            return RiskLevel.GREEN

        # YELLOW: Schreibende aber ungefährliche Operationen
        yellow_tools = {
            "write_file",
            "edit_file",
            "save_to_memory",
            "schedule_job",
        }
        if tool in yellow_tools:
            return RiskLevel.YELLOW

        # ORANGE: Operationen die User-Bestätigung brauchen
        orange_tools = {
            "email_send",
            "delete_file",
            "fetch_url",
        }
        if tool in orange_tools:
            return RiskLevel.ORANGE

        # RED: Shell-Befehle und unbekannte Tools
        red_tools = {
            "exec_command",
            "shell_exec",
            "shell",
        }
        if tool in red_tools:
            return RiskLevel.RED

        # Unbekannte Tools → ORANGE (Fail-Safe: lieber nachfragen)
        return RiskLevel.ORANGE

    def _risk_to_status(self, risk: RiskLevel) -> GateStatus:
        """Konvertiert RiskLevel in GateStatus."""
        mapping = {
            RiskLevel.GREEN: GateStatus.ALLOW,
            RiskLevel.YELLOW: GateStatus.INFORM,
            RiskLevel.ORANGE: GateStatus.APPROVE,
            RiskLevel.RED: GateStatus.BLOCK,
        }
        return mapping.get(risk, GateStatus.BLOCK)

    def _status_to_risk(self, status: GateStatus) -> RiskLevel:
        """Konvertiert GateStatus in RiskLevel (für Audit)."""
        mapping = {
            GateStatus.ALLOW: RiskLevel.GREEN,
            GateStatus.INFORM: RiskLevel.YELLOW,
            GateStatus.APPROVE: RiskLevel.ORANGE,
            GateStatus.BLOCK: RiskLevel.RED,
            GateStatus.MASK: RiskLevel.YELLOW,
        }
        return mapping.get(status, RiskLevel.RED)

    def _matches_rule(self, action: PlannedAction, rule: PolicyRule) -> bool:
        """Prüft ob eine Aktion zu einer Policy-Regel passt."""
        match = rule.match

        # Tool-Match
        if match.tool != "*" and match.tool.lower() != action.tool.lower():
            return False

        # Parameter-Match
        for param_name, param_match in match.params.items():
            if param_name == "*":
                # Alle Parameter scannen
                if not self._any_param_matches(action.params, param_match):
                    return False
            else:
                param_value = action.params.get(param_name)
                if param_value is None:
                    return False
                if not self._param_matches(str(param_value), param_match):
                    return False

        return True

    def _param_matches(self, value: str, match: PolicyParamMatch) -> bool:
        """Prüft ob ein einzelner Parameter-Wert zu einem Match passt."""
        # Regex
        if match.regex is not None:
            try:
                if not re.search(match.regex, value, re.IGNORECASE):
                    return False
            except re.error:
                log.warning("invalid_policy_regex", regex=match.regex)
                return False

        # startswith
        if match.startswith is not None:
            prefixes = (
                match.startswith if isinstance(match.startswith, list) else [match.startswith]
            )
            if not any(value.startswith(p) for p in prefixes):
                return False

        # not_startswith
        if match.not_startswith is not None:
            prefixes = (
                match.not_startswith
                if isinstance(match.not_startswith, list)
                else [match.not_startswith]
            )
            if any(value.startswith(p) for p in prefixes):
                return False

        # contains
        if match.contains is not None:
            patterns = match.contains if isinstance(match.contains, list) else [match.contains]
            if not any(p in value for p in patterns):
                return False

        # contains_pattern (Regex in Wert)
        if match.contains_pattern is not None:
            try:
                if not re.search(match.contains_pattern, value, re.IGNORECASE):
                    return False
            except re.error:
                return False

        # equals
        return not (match.equals is not None and value != match.equals)

    def _any_param_matches(
        self,
        params: dict[str, Any],
        match: PolicyParamMatch,
    ) -> bool:
        """Prüft ob irgendein Parameter zu einem Match passt (Wildcard *)."""
        return any(self._param_matches(str(value), match) for value in params.values())

    def _validate_paths(self, action: PlannedAction) -> GateDecision | None:
        """Prüft ob Dateipfade in den Parametern erlaubt sind. [B§3.2]

        Returns:
            GateDecision(BLOCK) wenn ein Pfad ungültig ist, sonst None.
        """
        # Nur für Datei-Operationen relevant
        file_tools = {"read_file", "write_file", "edit_file", "list_directory", "delete_file"}
        if action.tool.lower() not in file_tools:
            return None

        path_str = action.params.get("path", "")
        if not path_str:
            return None

        try:
            target = Path(path_str).expanduser().resolve()
        except (ValueError, OSError):
            return GateDecision(
                status=GateStatus.BLOCK,
                reason=f"Ungültiger Pfad: {path_str}",
                risk_level=RiskLevel.RED,
                original_action=action,
                policy_name="path_validation",
            )

        # Prüfe ob Pfad in einem erlaubten Verzeichnis liegt
        for allowed in self._allowed_paths:
            try:
                # resolve() verhindert Symlink-Tricks und ../../ Traversals
                target.relative_to(allowed)
                return None  # Pfad ist erlaubt
            except ValueError:
                continue

        return GateDecision(
            status=GateStatus.BLOCK,
            reason=f"Pfad außerhalb erlaubter Verzeichnisse: {path_str}",
            risk_level=RiskLevel.RED,
            original_action=action,
            policy_name="path_validation",
        )

    def _check_command(
        self,
        command: str,
        action: PlannedAction,
    ) -> GateDecision | None:
        """Prüft einen Shell-Befehl gegen destruktive Patterns.

        Returns:
            GateDecision(BLOCK) wenn destruktiv, sonst None.
        """
        if not command.strip():
            return None

        for pattern in self._blocked_command_patterns:
            if pattern.search(command):
                return GateDecision(
                    status=GateStatus.BLOCK,
                    reason=f"Destruktiver Shell-Befehl erkannt: Pattern '{pattern.pattern}'",
                    risk_level=RiskLevel.RED,
                    original_action=action,
                    policy_name="blocked_command",
                )

        return None

    def _scan_credentials(
        self,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Scannt Parameter auf Credentials und maskiert sie.

        Returns:
            Tuple von (maskierte_params, hat_credentials).
            Credentials werden durch '***MASKED***' ersetzt.
        """
        if not params:
            return params, False

        has_credentials = False
        masked = {}

        for key, value in params.items():
            str_value = str(value)
            masked_value = str_value
            key_has_credential = False

            for pattern in self._credential_patterns:
                if pattern.search(str_value):
                    key_has_credential = True
                    masked_value = pattern.sub("***MASKED***", masked_value)

            if key_has_credential:
                has_credentials = True
                masked[key] = masked_value
            else:
                masked[key] = value  # Original-Typ beibehalten

        return masked, has_credentials

    def _load_policies(self) -> list[PolicyRule]:
        """Lädt Policy-Regeln aus YAML-Dateien."""
        rules: list[PolicyRule] = []

        for policy_file in [
            self._config.policies_dir / "default.yaml",
            self._config.policies_dir / "custom.yaml",
        ]:
            if not policy_file.exists():
                continue

            try:
                with open(policy_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not isinstance(data, dict) or "rules" not in data:
                    continue

                for rule_data in data["rules"]:
                    try:
                        rule = self._parse_rule(rule_data)
                        rules.append(rule)
                    except Exception as exc:
                        log.warning(
                            "invalid_policy_rule",
                            file=str(policy_file),
                            rule=rule_data.get("name", "?"),
                            error=str(exc),
                        )
            except Exception as exc:
                log.error(
                    "policy_load_failed",
                    file=str(policy_file),
                    error=str(exc),
                )

        log.info("policies_loaded", count=len(rules))
        return rules

    def _parse_rule(self, data: dict[str, Any]) -> PolicyRule:
        """Parst eine einzelne Policy-Regel aus YAML-Daten."""
        match_data = data.get("match", {})
        params_match = {}

        if "params" in match_data:
            for param_name, param_criteria in match_data["params"].items():
                if isinstance(param_criteria, dict):
                    params_match[param_name] = PolicyParamMatch(**param_criteria)
                elif isinstance(param_criteria, str):
                    params_match[param_name] = PolicyParamMatch(equals=param_criteria)

        policy_match = PolicyMatch(
            tool=match_data.get("tool", "*"),
            params=params_match,
        )

        return PolicyRule(
            name=data["name"],
            match=policy_match,
            action=GateStatus(data["action"]),
            reason=data.get("reason", ""),
            priority=data.get("priority", 0),
        )

    def _write_audit(
        self,
        action: PlannedAction,
        decision: GateDecision,
        context: SessionContext,
    ) -> None:
        """Schreibt einen Audit-Eintrag ins Log. [B§3.2]

        WICHTIG: Credentials werden IMMER maskiert, auch im Audit-Log.
        """
        # Params hashen statt im Klartext loggen
        params_str = json.dumps(action.params, sort_keys=True, default=str)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]

        entry = AuditEntry(
            timestamp=datetime.now(UTC),
            session_id=context.session_id,
            action_tool=action.tool,
            action_params_hash=params_hash,
            decision_status=decision.status,
            decision_reason=decision.reason,
            risk_level=decision.risk_level,
            policy_name=decision.policy_name,
        )

        # JSONL schreiben (append)
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        except OSError as exc:
            log.error("audit_write_failed", error=str(exc))

        # Auch ins structlog
        log.info(
            "gatekeeper_decision",
            tool=action.tool,
            status=decision.status.value,
            risk=decision.risk_level.name,
            reason=decision.reason[:100],
            policy=decision.policy_name,
            session=context.session_id[:8],
        )

        # Zentrales Audit-Logging (AuditLogger-Integration)
        if self._audit_logger:
            self._audit_logger.log_gatekeeper(
                decision.status.value,
                decision.reason,
                tool_name=action.tool,
            )
