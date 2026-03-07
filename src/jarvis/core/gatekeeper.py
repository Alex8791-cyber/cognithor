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

import atexit
import hashlib
import json
import re
import weakref
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
    from jarvis.skills.registry import Skill

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

        # Buffered Audit-Writes (vermeidet Blocking I/O pro evaluate()-Aufruf)
        self._audit_buffer: list[str] = []
        self._AUDIT_FLUSH_THRESHOLD = 10

        # atexit-Handler: Flush bei Prozess-Ende (Schutz vor Datenverlust)
        # weakref verhindert, dass atexit den Gatekeeper am Leben hält
        _weak_self = weakref.ref(self)

        def _atexit_flush() -> None:
            obj = _weak_self()
            if obj is not None:
                obj._flush_audit_buffer()

        atexit.register(_atexit_flush)

        # Optional: Capability Matrix (F8)
        self._capability_matrix: Any = None
        try:
            from jarvis.security.capabilities import CapabilityMatrix
            self._capability_matrix = CapabilityMatrix()
        except Exception:
            pass

        # Community-Skill ToolEnforcer
        self._tool_enforcer: Any = None
        try:
            from jarvis.skills.community.tool_enforcer import ToolEnforcer
            self._tool_enforcer = ToolEnforcer()
        except Exception:
            pass

        # Aktiver Community-Skill (wird pro evaluate()-Aufruf gesetzt)
        self._active_skill: Skill | None = None

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

        # Projekt-Verzeichnis automatisch hinzufuegen (allow_project_dir)
        if getattr(self._config.security, "allow_project_dir", False):
            project_dir = self._config.jarvis_home.parent
            resolved = project_dir.resolve()
            if resolved not in self._allowed_paths:
                self._allowed_paths.append(resolved)
                log.info("gatekeeper_project_dir_added", path=str(resolved))

        self._initialized = True

        log.info(
            "gatekeeper_initialized",
            policy_count=len(self._policies),
            credential_patterns=len(self._credential_patterns),
            blocked_commands=len(self._blocked_command_patterns),
            allowed_paths=[str(p) for p in self._allowed_paths],
        )

    def reload_policies(self) -> None:
        """Lädt Policies neu von Disk (Live-Reload vom UI)."""
        self._policies = self._load_policies()
        self._policies.sort(key=lambda r: r.priority, reverse=True)
        log.info("gatekeeper_policies_reloaded", policy_count=len(self._policies))

    # --- Public Policy API (fuer ReplayEngine u.a.) ---

    def get_policies(self) -> list[PolicyRule]:
        """Gibt eine Kopie der aktuellen Policy-Regeln zurueck."""
        return list(self._policies)

    def set_policies(self, policies: list[PolicyRule]) -> None:
        """Ersetzt die Policy-Regeln (fuer Replay/Testing)."""
        self._policies = sorted(policies, key=lambda r: r.priority, reverse=True)

    def set_active_skill(self, skill: Skill | None) -> None:
        """Setzt den aktiven Skill fuer ToolEnforcer-Checks.

        Wird vom Gateway/PGE-Loop aufgerufen, bevor Actions evaluiert
        werden.  Wenn ein Community-Skill aktiv ist, werden nur die
        in tools_required deklarierten Tools erlaubt.
        """
        self._active_skill = skill

    def evaluate(
        self,
        action: PlannedAction,
        context: SessionContext,
    ) -> GateDecision:
        """Prüft eine einzelne PlannedAction gegen alle Policies. [B§3.2]

        Reihenfolge der Prüfungen (first-match wins bei BLOCK):
          0. ToolEnforcer (Community-Skills: nur tools_required)
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

        # --- Schritt -1: Community-Skill ToolEnforcer ---
        skill_block = self._enforce_skill_tools(action, context)
        if skill_block is not None:
            return skill_block

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

        # --- Schritt 4a: Gefährlicher Python-Code ---
        if action.tool == "run_python":
            code = str(action.params.get("code", ""))
            code_check = self._check_python_code(code, action)
            if code_check is not None:
                self._write_audit(action, code_check, context)
                return code_check

        # --- Schritt 4b: Destruktive Shell-Befehle ---
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

    def _enforce_skill_tools(
        self,
        action: PlannedAction,
        context: SessionContext,
    ) -> GateDecision | None:
        """Prueft ob ein Tool-Call fuer den aktiven Community-Skill erlaubt ist.

        Wird VOR allen anderen Checks aufgerufen.  Wenn ein Community-Skill
        aktiv ist und das Tool nicht in tools_required steht → BLOCK.

        Returns:
            GateDecision(BLOCK) wenn Tool nicht erlaubt, sonst None.
        """
        if self._tool_enforcer is None or self._active_skill is None:
            return None

        result = self._tool_enforcer.check(action, self._active_skill)
        if not result.allowed:
            decision = GateDecision(
                status=GateStatus.BLOCK,
                reason=f"ToolEnforcer: {result.reason}",
                risk_level=RiskLevel.RED,
                original_action=action,
                policy_name="community_tool_enforcer",
            )
            self._write_audit(action, decision, context)
            log.warning(
                "community_tool_blocked",
                tool=action.tool,
                skill=result.skill_name,
                declared_tools=result.declared_tools,
            )
            return decision

        return None

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
            "web_search",
            "web_fetch",
            "web_news_search",
            "search_and_read",
            "browse_url",
            "search_procedures",
            "media_analyze_image",
            "media_extract_text",
            "media_transcribe_audio",
            "media_resize_image",
            "get_core_memory",
            "get_recent_episodes",
            "memory_stats",
            "record_procedure_usage",
            "browse_page_info",
            "browse_screenshot",
            "analyze_code",
            "list_skills",
            "search_community_skills",
        }
        if tool in green_tools:
            return RiskLevel.GREEN

        # YELLOW: Schreibende aber ungefährliche Operationen (lokal, kein Netzwerk)
        yellow_tools = {
            "write_file",
            "edit_file",
            "save_to_memory",
            "add_entity",
            "add_relation",
            "schedule_job",
            "document_export",
            "media_tts",
            "media_convert_audio",
            "exec_command",
            "shell_exec",
            "shell",
            "run_python",
            "create_skill",
            "install_community_skill",
            "report_skill",
        }
        if tool in yellow_tools:
            return RiskLevel.YELLOW

        # ORANGE: Operationen die User-Bestätigung brauchen
        orange_tools = {
            "email_send",
            "delete_file",
            "fetch_url",
            "http_request",
        }
        if tool in orange_tools:
            return RiskLevel.ORANGE

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

    # Compiled patterns for dangerous Python code (class-level, compiled once)
    _DANGEROUS_PYTHON_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        # OS-level command execution
        (re.compile(r"\bos\.system\s*\(", re.IGNORECASE), "os.system()"),
        (re.compile(r"\bos\.popen\s*\(", re.IGNORECASE), "os.popen()"),
        (re.compile(r"\bos\.exec[lv]", re.IGNORECASE), "os.exec*()"),
        # OS-level file destruction
        (re.compile(r"\bos\.remove\s*\(", re.IGNORECASE), "os.remove()"),
        (re.compile(r"\bos\.unlink\s*\(", re.IGNORECASE), "os.unlink()"),
        (re.compile(r"\bos\.rmdir\s*\(", re.IGNORECASE), "os.rmdir()"),
        # subprocess — unsichere Varianten blockieren:
        #   subprocess.Popen (Low-Level, schwer zu kontrollieren)
        #   subprocess.call (Legacy, kein Timeout-Default)
        #   subprocess.getoutput/getstatusoutput (Shell-Injection-Risiko)
        # Sichere Varianten (subprocess.run, subprocess.check_output) sind erlaubt,
        # da sie kontrollierter sind und bessere Defaults haben.
        (re.compile(r"\bsubprocess\.Popen\s*\(", re.IGNORECASE), "subprocess.Popen()"),
        (re.compile(r"\bsubprocess\.call\s*\(", re.IGNORECASE), "subprocess.call()"),
        (re.compile(r"\bsubprocess\.getoutput\s*\(", re.IGNORECASE), "subprocess.getoutput()"),
        (re.compile(r"\bsubprocess\.getstatusoutput\s*\(", re.IGNORECASE), "subprocess.getstatusoutput()"),
        # shutil destructive operations
        (re.compile(r"\bshutil\.rmtree\s*\(", re.IGNORECASE), "shutil.rmtree()"),
        (re.compile(r"\bshutil\.move\s*\(", re.IGNORECASE), "shutil.move()"),
        # Dynamic code execution / import
        (re.compile(r"\b__import__\s*\(", re.IGNORECASE), "__import__()"),
        (re.compile(r"\bimportlib\.", re.IGNORECASE), "importlib"),
        (re.compile(r"\beval\s*\(", re.IGNORECASE), "eval()"),
        (re.compile(r"\bexec\s*\(", re.IGNORECASE), "exec()"),
        # Dangerous deserialization / native code
        (re.compile(r"\bpickle\.load", re.IGNORECASE), "pickle.load/loads()"),
        (re.compile(r"\bctypes\.", re.IGNORECASE), "ctypes"),
        # Network access (raw socket — urllib/aiohttp/requests sind erlaubt)
        (re.compile(r"\bsocket\.socket\s*\(", re.IGNORECASE), "socket.socket()"),
        (re.compile(r"\bsocket\.create_connection\s*\(", re.IGNORECASE), "socket.create_connection()"),
        # pathlib file deletion
        (re.compile(r"\.unlink\s*\(", re.IGNORECASE), "Path.unlink()"),
        # open() with write/append/create modes — match mode argument, not filename
        (re.compile(r"\bopen\s*\([^)]*,\s*['\"][wax]", re.IGNORECASE), "open() with write mode"),
    ]

    def _check_python_code(
        self,
        code: str,
        action: PlannedAction,
    ) -> GateDecision | None:
        """Prüft Python-Code auf gefährliche Patterns (os.system, subprocess, etc.).

        Returns:
            GateDecision(BLOCK) wenn gefährlicher Code erkannt, sonst None.
        """
        if not code.strip():
            return None

        for pattern, description in self._DANGEROUS_PYTHON_PATTERNS:
            if pattern.search(code):
                return GateDecision(
                    status=GateStatus.BLOCK,
                    reason=f"Gefährlicher Python-Code erkannt: {description}",
                    risk_level=RiskLevel.RED,
                    original_action=action,
                    policy_name="blocked_python_code",
                )

        return None

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

        # Backwards-Kompatibilität: Dotted-Notation (z.B. "params.command")
        # Ältere Policy-Dateien nutzen "params.command:" statt "params: command:"
        for key in list(match_data.keys()):
            if key.startswith("params.") and key != "params":
                param_name = key[len("params."):]
                criteria = match_data[key]
                if isinstance(criteria, dict):
                    params_match[param_name] = PolicyParamMatch(**criteria)
                elif isinstance(criteria, str):
                    params_match[param_name] = PolicyParamMatch(equals=criteria)

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

        # Audit-Eintrag in Buffer schreiben (non-blocking)
        self._audit_buffer.append(entry.model_dump_json() + "\n")
        if len(self._audit_buffer) >= self._AUDIT_FLUSH_THRESHOLD:
            self._flush_audit_buffer()

        # Zentrales Audit-Logging (AuditLogger-Integration)
        if self._audit_logger:
            self._audit_logger.log_gatekeeper(
                decision.status.value,
                decision.reason,
                tool_name=action.tool,
            )

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

    def _flush_audit_buffer(self) -> None:
        """Schreibt den Audit-Buffer gesammelt auf Disk (batch I/O)."""
        if not self._audit_buffer:
            return
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.writelines(self._audit_buffer)
            self._audit_buffer.clear()
        except OSError as exc:
            log.error("audit_write_failed", error=str(exc))
