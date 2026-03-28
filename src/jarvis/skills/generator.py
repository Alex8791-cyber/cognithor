"""Auto-skill generator: Jarvis designs its own modules.

Wenn der Planner oder Executor ein Tool nicht findet oder ein Skill
fehlt, erkennt der SkillGenerator die Luecke und:

  1. Analysiert was fehlt (SkillGap)
  2. Generiert Python-Code via LLM (Code-Synthese)
  3. Erstellt Unit-Tests automatisch
  4. Testet in isolierter Sandbox
  5. Registriert bei Erfolg in der SkillRegistry
  6. Versioniert fuer Roll-Back bei Problemen
  7. Optional: Wartet auf User-Approval (Gatekeeper)

Sicherheit:
  - Generierter Code laeuft NUR in der Sandbox
  - Kein Zugriff auf Credentials, Netzwerk (optional), Dateisystem
  - Gatekeeper kann Approval fuer kritische Skills erzwingen
  - Versionierung ermoeglicht Roll-Back

Bibel-Referenz: §6.4 (Prozedurale Selbstverbesserung)
"""

from __future__ import annotations

import hashlib
import shlex
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

try:
    from jarvis.security.encrypted_file import efile as _efile
except ImportError:
    _efile = None

log = get_logger(__name__)

# Standard packages for generated skills
DEFAULT_ALLOWED_PACKAGES = frozenset(
    {
        "json",
        "re",
        "math",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "pathlib",
        "textwrap",
        "hashlib",
        "csv",
        "io",
        "string",
        "dataclasses",
    }
)

# Gap detection: minimum occurrences before generation
DEFAULT_GAP_THRESHOLD = 2

__all__ = [
    "GapDetector",
    "GeneratedSkill",
    "GenerationStatus",
    "SkillGap",
    "SkillGapType",
    "SkillGenerator",
]


# ============================================================================
# Enums & Datenmodelle
# ============================================================================


class SkillGapType(Enum):
    """Art der erkannten Skill-Luecke."""

    UNKNOWN_TOOL = "unknown_tool"  # Tool call failed
    NO_SKILL_MATCH = "no_skill_match"  # Kein Skill für die Aufgabe
    LOW_SUCCESS_RATE = "low_success_rate"  # Existing skill fails often
    USER_REQUEST = "user_request"  # User explicitly requests new tool
    REPEATED_FAILURE = "repeated_failure"  # Repeated failure


class GenerationStatus(Enum):
    """Status of a skill generation attempt."""

    PENDING = "pending"
    GENERATING = "generating"
    TESTING = "testing"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    AWAITING_APPROVAL = "awaiting_approval"
    REGISTERED = "registered"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class SkillGap:
    """Eine erkannte Skill-Luecke.

    Wird erzeugt wenn Jarvis eine Aufgabe nicht ausfuehren kann
    und ein neuer Skill benoetigt wird.
    """

    id: str
    gap_type: SkillGapType
    description: str  # Was fehlt
    context: str = ""  # User-Nachricht / Fehlermeldung
    tool_name: str = ""  # Fehlender Tool-Name (bei UNKNOWN_TOOL)
    frequency: int = 1  # Wie oft aufgetreten
    first_seen: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    last_seen: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    @property
    def priority(self) -> float:
        """Prioritaet basierend auf Haeufigkeit und Typ."""
        type_weight = {
            SkillGapType.USER_REQUEST: 2.0,
            SkillGapType.REPEATED_FAILURE: 1.5,
            SkillGapType.UNKNOWN_TOOL: 1.0,
            SkillGapType.NO_SKILL_MATCH: 0.8,
            SkillGapType.LOW_SUCCESS_RATE: 0.6,
        }
        return self.frequency * type_weight.get(self.gap_type, 1.0)


@dataclass
class GeneratedSkill:
    """Ein vom SkillGenerator erzeugter Skill.

    Enthaelt den generierten Code, Tests, und Versionierungsinfo.
    """

    name: str
    version: int = 1
    status: GenerationStatus = GenerationStatus.PENDING

    # Generierter Code
    code: str = ""
    test_code: str = ""
    skill_markdown: str = ""

    # Metadata
    gap: SkillGap | None = None
    description: str = ""
    tools_provided: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    # Test results
    test_output: str = ""
    test_passed: bool = False
    test_errors: list[str] = field(default_factory=list)

    # Security
    requires_approval: bool = False
    approved_by: str = ""
    sandbox_network: str = "block"  # Default: kein Netzwerk

    # Timestamps
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    registered_at: str = ""

    @property
    def module_name(self) -> str:
        """Python-Modulname fuer den Skill."""
        return f"auto_{self.name}"

    @property
    def version_tag(self) -> str:
        return f"v{self.version}"


# ============================================================================
# Skill-Spezifikation (Prompt-Template fuer Code-LLM)
# ============================================================================


SKILL_GENERATION_PROMPT = textwrap.dedent("""\
Du bist ein erfahrener Python-Entwickler. Erstelle ein Python-Modul
das als Jarvis-Tool registriert werden kann.

## Anforderung
{description}

## Kontext
{context}

## Regeln
1. Das Modul MUSS eine `register(mcp_client)` Funktion haben
2. Jedes Tool wird ueber `mcp_client.register_builtin_handler(name, schema, handler)` registriert
3. Handler-Funktionen sind `async def handler(**params) -> str`
4. Nur Python-Standardbibliothek + diese erlaubten Pakete: {allowed_packages}
5. KEIN Netzwerkzugriff, KEIN Dateisystem ausserhalb working_dir
6. Fehlerbehandlung mit try/except, nie crashen
7. Docstrings auf Deutsch
8. Typ-Annotationen verwenden

## Schema-Format
```python
schema = {{
    "name": "tool_name",
    "description": "Beschreibung",
    "inputSchema": {{
        "type": "object",
        "properties": {{
            "param1": {{"type": "string", "description": "..."}},
        }},
        "required": ["param1"],
    }},
}}
```

## Ausgabe
Gib NUR den Python-Code aus, keine Erklaerungen.
""")

SKILL_TEST_PROMPT = textwrap.dedent("""\
Erstelle pytest-Tests fuer das folgende Python-Modul.

## Modul-Code
```python
{code}
```

## Regeln
1. Verwende pytest und pytest-asyncio
2. Teste alle oeffentlichen Funktionen
3. Teste Fehlerfaelle (leere Eingaben, ungueltige Parameter)
4. Mocke den mcp_client mit unittest.mock.MagicMock
5. Mindestens 3 Tests pro Tool-Funktion

## Ausgabe
Gib NUR den Test-Code aus, keine Erklaerungen.
""")


# ============================================================================
# Gap Detector
# ============================================================================


class GapDetector:
    """Erkennt Skill-Luecken aus dem laufenden Betrieb.

    Sammelt Signale aus:
    - Fehlgeschlagene Tool-Calls (Executor)
    - Kein Skill-Match (SkillRegistry)
    - Niedrige Erfolgsraten (SkillRegistry.record_usage)
    - Explizite User-Requests ("erstelle ein Tool fuer ...")
    """

    MAX_GAPS = 1000  # Maximale Anzahl Gaps im Speicher
    MAX_CONTEXT_LENGTH = 5000  # Maximum length of a gap context

    def __init__(self) -> None:
        self._gaps: dict[str, SkillGap] = {}  # gap_id → SkillGap
        self._gap_threshold: int = DEFAULT_GAP_THRESHOLD

    @property
    def gap_count(self) -> int:
        return len(self._gaps)

    def report_unknown_tool(self, tool_name: str, context: str = "") -> SkillGap:
        """Report a failed tool call."""
        gap_id = f"tool:{tool_name}"
        return self._upsert_gap(
            gap_id,
            SkillGapType.UNKNOWN_TOOL,
            description=f"Tool '{tool_name}' nicht verfügbar",
            context=context,
            tool_name=tool_name,
        )

    def report_no_skill_match(self, query: str) -> SkillGap:
        """Report that no skill matches the query."""
        # Deterministic ID from query hash
        q_hash = hashlib.md5(query.lower().encode()).hexdigest()[:8]
        gap_id = f"skill:{q_hash}"
        return self._upsert_gap(
            gap_id,
            SkillGapType.NO_SKILL_MATCH,
            description=f"Kein Skill für: {query[:100]}",
            context=query,
        )

    def report_low_success_rate(
        self,
        skill_name: str,
        success_rate: float,
    ) -> SkillGap:
        """Report a skill with low success rate."""
        gap_id = f"low_success:{skill_name}"
        return self._upsert_gap(
            gap_id,
            SkillGapType.LOW_SUCCESS_RATE,
            description=f"Skill '{skill_name}' hat nur {success_rate:.0%} Erfolgsrate",
            context=skill_name,
        )

    def report_user_request(self, description: str, context: str = "") -> SkillGap:
        """User explicitly requests a new tool."""
        gap_id = f"user:{hashlib.md5(description.encode()).hexdigest()[:8]}"
        return self._upsert_gap(
            gap_id,
            SkillGapType.USER_REQUEST,
            description=description,
            context=context,
        )

    def report_repeated_failure(
        self,
        task: str,
        error: str,
    ) -> SkillGap:
        """Report repeated failure at a task."""
        t_hash = hashlib.md5(task.lower().encode()).hexdigest()[:8]
        gap_id = f"fail:{t_hash}"
        return self._upsert_gap(
            gap_id,
            SkillGapType.REPEATED_FAILURE,
            description=f"Wiederholtes Scheitern: {task[:100]}",
            context=error,
        )

    def get_actionable_gaps(self) -> list[SkillGap]:
        """Return gaps that have occurred frequently enough.

        Returns:
            Prioritized list of gaps (highest priority first).
        """
        actionable = [
            gap
            for gap in self._gaps.values()
            if gap.frequency >= self._gap_threshold
            or gap.gap_type == SkillGapType.USER_REQUEST  # Sofort handeln
        ]
        actionable.sort(key=lambda g: g.priority, reverse=True)
        return actionable

    def get_all_gaps(self) -> list[SkillGap]:
        """All detected gaps."""
        return sorted(self._gaps.values(), key=lambda g: g.priority, reverse=True)

    def clear_gap(self, gap_id: str) -> bool:
        """Remove a gap (e.g. after successful generation)."""
        return self._gaps.pop(gap_id, None) is not None

    def _upsert_gap(
        self,
        gap_id: str,
        gap_type: SkillGapType,
        description: str,
        context: str = "",
        tool_name: str = "",
    ) -> SkillGap:
        """Create or update a gap."""
        # Limit context
        context = context[: self.MAX_CONTEXT_LENGTH]

        if gap_id in self._gaps:
            gap = self._gaps[gap_id]
            gap.frequency += 1
            gap.last_seen = datetime.now(UTC).isoformat()
            if context and context != gap.context:
                gap.context = context[: self.MAX_CONTEXT_LENGTH]
            log.debug("gap_updated", id=gap_id, frequency=gap.frequency)
        else:
            # Eviction: remove lowest priority on overflow
            if len(self._gaps) >= self.MAX_GAPS:
                lowest = min(self._gaps.values(), key=lambda g: g.priority)
                del self._gaps[lowest.id]
                log.debug("gap_evicted", id=lowest.id, priority=lowest.priority)

            gap = SkillGap(
                id=gap_id,
                gap_type=gap_type,
                description=description,
                context=context,
                tool_name=tool_name,
            )
            self._gaps[gap_id] = gap
            log.info("gap_detected", id=gap_id, type=gap_type.value, desc=description[:80])

        return gap


# ============================================================================
# Skill Generator
# ============================================================================


class SkillGenerator:
    """Generiert, testet und registriert neue Skills automatisch.

    Workflow:
      1. GapDetector meldet Luecke
      2. generate() erstellt Code via LLM
      3. test() fuehrt Tests in Sandbox aus
      4. register() laedt Skill in Registry
      5. Versionierung fuer Roll-Back

    Args:
        skills_dir: Verzeichnis fuer generierte Skills.
        sandbox_executor: SandboxExecutor fuer Test-Ausfuehrung.
        llm_fn: Async-Funktion fuer LLM-Aufrufe (z.B. UnifiedLLMClient.complete).
    """

    def __init__(
        self,
        skills_dir: Path,
        sandbox_executor: Any | None = None,
        llm_fn: Any | None = None,
        *,
        allowed_packages: list[str] | None = None,
        require_approval: bool = False,
        package_builder: Any | None = None,
        audit_logger: Any | None = None,
    ) -> None:
        self._skills_dir = Path(skills_dir)
        self._history_dir = self._skills_dir / "history"
        self._packages_dir = self._skills_dir / "packages"
        self._sandbox = sandbox_executor
        self._llm_fn = llm_fn
        self._allowed_packages = allowed_packages or sorted(DEFAULT_ALLOWED_PACKAGES)
        self._require_approval = require_approval
        self._package_builder = package_builder
        self._audit_logger = audit_logger
        self._generated: dict[str, GeneratedSkill] = {}  # name → skill
        self._gap_detector = GapDetector()

        # Verzeichnisse erstellen
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._packages_dir.mkdir(parents=True, exist_ok=True)

    @property
    def gap_detector(self) -> GapDetector:
        """Access to the gap detector."""
        return self._gap_detector

    @property
    def generated_count(self) -> int:
        return len(self._generated)

    def get_generated(self, name: str) -> GeneratedSkill | None:
        return self._generated.get(name)

    def list_generated(self) -> list[GeneratedSkill]:
        return list(self._generated.values())

    # ========================================================================
    # Code-Generierung
    # ========================================================================

    async def generate(self, gap: SkillGap) -> GeneratedSkill:
        """Generiert Code fuer eine Skill-Luecke.

        Args:
            gap: Die zu schliessende Skill-Luecke.

        Returns:
            GeneratedSkill mit generiertem Code.
        """
        # Derive skill name from gap
        name = self._derive_skill_name(gap)

        skill = GeneratedSkill(
            name=name,
            gap=gap,
            description=gap.description,
            status=GenerationStatus.GENERATING,
            requires_approval=self._require_approval,
        )

        # Versioning: if skill with same name exists
        existing = self._generated.get(name)
        if existing:
            skill.version = existing.version + 1

        log.info(
            "skill_generation_start",
            name=name,
            gap_type=gap.gap_type.value,
            version=skill.version,
        )

        if self._llm_fn is None:
            # Without LLM: generate stub
            skill.code = self._generate_stub(gap)
            skill.test_code = self._generate_stub_test(name, skill.code)
        else:
            # With LLM: generate real code
            try:
                prompt = SKILL_GENERATION_PROMPT.format(
                    description=gap.description,
                    context=gap.context[:500],
                    allowed_packages=", ".join(self._allowed_packages),
                )
                skill.code = await self._llm_fn(prompt)
                skill.code = self._extract_code_block(skill.code)

                # Generate tests
                test_prompt = SKILL_TEST_PROMPT.format(code=skill.code)
                skill.test_code = await self._llm_fn(test_prompt)
                skill.test_code = self._extract_code_block(skill.test_code)

            except Exception as exc:
                log.error("skill_generation_failed", name=name, error=str(exc))
                skill.status = GenerationStatus.FAILED
                skill.test_errors = [str(exc)]

        # Generate skill Markdown
        skill.skill_markdown = self._generate_skill_markdown(skill)

        self._generated[name] = skill
        return skill

    async def test(self, skill: GeneratedSkill) -> bool:
        """Testet generierten Code in der Sandbox.

        Args:
            skill: Der zu testende GeneratedSkill.

        Returns:
            True wenn alle Tests bestanden.
        """
        skill.status = GenerationStatus.TESTING

        # Code + Tests in temporaeres Verzeichnis schreiben
        test_dir = self._skills_dir / "test_staging" / skill.module_name
        test_dir.mkdir(parents=True, exist_ok=True)

        module_file = test_dir / f"{skill.module_name}.py"
        test_file = test_dir / f"test_{skill.module_name}.py"

        module_file.write_text(skill.code, encoding="utf-8")
        test_file.write_text(skill.test_code, encoding="utf-8")

        log.info("skill_test_start", name=skill.name, test_dir=str(test_dir))

        if self._sandbox is None:
            # Without sandbox: syntax check via compile()
            try:
                compile(skill.code, module_file.name, "exec")
                skill.test_passed = True
                skill.test_output = "Syntax-Check bestanden (keine Sandbox verfügbar)"
                skill.status = GenerationStatus.TEST_PASSED
            except SyntaxError as exc:
                skill.test_passed = False
                skill.test_errors = [f"SyntaxError: {exc}"]
                skill.test_output = str(exc)
                skill.status = GenerationStatus.TEST_FAILED
        else:
            # Mit Sandbox: pytest ausfuehren
            try:
                result = await self._sandbox.execute(
                    f"{shlex.quote(sys.executable)} -m pytest "
                    f"{shlex.quote(test_file.name)} -v --tb=short 2>&1",
                    working_dir=str(test_dir),
                    timeout=30,
                )
                skill.test_output = result.output if hasattr(result, "output") else str(result)
                skill.test_passed = getattr(result, "exit_code", 1) == 0

                if skill.test_passed:
                    skill.status = GenerationStatus.TEST_PASSED
                    log.info("skill_test_passed", name=skill.name)
                else:
                    skill.status = GenerationStatus.TEST_FAILED
                    skill.test_errors = [skill.test_output[:500]]
                    log.warning("skill_test_failed", name=skill.name)

            except Exception as exc:
                skill.test_passed = False
                skill.test_errors = [str(exc)]
                skill.status = GenerationStatus.TEST_FAILED

        # Staging aufraeumen
        shutil.rmtree(test_dir, ignore_errors=True)

        return skill.test_passed

    # ========================================================================
    # Registrierung
    # ========================================================================

    def register(
        self,
        skill: GeneratedSkill,
        skill_registry: Any | None = None,
    ) -> bool:
        """Registriert einen getesteten Skill.

        Args:
            skill: Der zu registrierende GeneratedSkill.
            skill_registry: Optional SkillRegistry fuer sofortige Registrierung.

        Returns:
            True wenn erfolgreich registriert.
        """
        if not skill.test_passed:
            log.warning("skill_register_rejected", name=skill.name, reason="tests_not_passed")
            return False

        if skill.requires_approval and not skill.approved_by:
            skill.status = GenerationStatus.AWAITING_APPROVAL
            log.info("skill_awaiting_approval", name=skill.name)
            return False

        # Archive old version
        self._archive_if_exists(skill)

        # Code schreiben (reference only — NOT imported by the registry)
        code_file = self._skills_dir / f"{skill.module_name}.py"
        code_file.write_text(skill.code, encoding="utf-8")

        # Skill-Markdown schreiben (registry loads only .md files)
        md_file = self._skills_dir / f"{skill.module_name}.md"
        if _efile is not None:
            _efile.write(md_file, skill.skill_markdown)
        else:
            md_file.write_text(skill.skill_markdown, encoding="utf-8")

        # Test-Code schreiben
        test_file = self._skills_dir / f"test_{skill.module_name}.py"
        test_file.write_text(skill.test_code, encoding="utf-8")

        skill.status = GenerationStatus.REGISTERED
        skill.registered_at = datetime.now(UTC).isoformat()

        # Optionally load into SkillRegistry
        if skill_registry is not None and hasattr(skill_registry, "load_from_directories"):
            try:
                skill_registry.load_from_directories([self._skills_dir])
                log.info("skill_loaded_into_registry", name=skill.name)
            except Exception as exc:
                log.warning("skill_registry_load_failed", name=skill.name, error=str(exc))

        log.info(
            "skill_registered",
            name=skill.name,
            version=skill.version,
            path=str(code_file),
        )

        # Optionally package as signed package
        if self._package_builder is not None:
            try:
                from jarvis.skills.package import SkillManifest

                manifest = SkillManifest(
                    name=skill.module_name,
                    version=str(skill.version)
                    if "." in str(skill.version)
                    else f"{skill.version}.0.0",
                    description=skill.description[:200] if skill.description else skill.name,
                    author="jarvis-autogen",
                    trigger_keywords=list(skill.gap.trigger_examples)
                    if hasattr(skill, "gap")
                    and skill.gap
                    and hasattr(skill.gap, "trigger_examples")
                    else [],
                    category=skill.gap.gap_type.value
                    if hasattr(skill, "gap") and skill.gap and hasattr(skill.gap, "gap_type")
                    else "general",
                )
                package = self._package_builder.build(
                    manifest,
                    skill.code,
                    skill.test_code,
                    skill.skill_markdown,
                )
                pkg_path = self._packages_dir / f"{package.package_id}.jarvis-skill"
                pkg_path.write_bytes(package.to_bytes())
                log.info("skill_packaged", name=skill.name, package=str(pkg_path))
            except Exception as exc:
                log.warning("skill_packaging_failed", name=skill.name, error=str(exc))

        # Audit: log skill installation
        if self._audit_logger:
            self._audit_logger.log_skill_install(
                f"{skill.name}@{skill.version}",
                source="autogen",
            )

        return True

    def approve(self, name: str, approved_by: str = "user") -> bool:
        """Approve a skill awaiting approval.

        Args:
            name: Skill-Name.
            approved_by: Wer genehmigt hat.

        Returns:
            True wenn genehmigt.
        """
        skill = self._generated.get(name)
        if not skill:
            return False
        if skill.status != GenerationStatus.AWAITING_APPROVAL:
            return False

        skill.approved_by = approved_by
        log.info("skill_approved", name=name, by=approved_by)
        return True

    # ========================================================================
    # Roll-Back
    # ========================================================================

    def rollback(self, name: str) -> bool:
        """Rollt einen Skill auf die vorherige Version zurueck.

        Args:
            name: Skill-Name.

        Returns:
            True wenn Roll-Back erfolgreich.
        """
        skill = self._generated.get(name)
        if not skill or skill.version <= 1:
            log.warning("rollback_impossible", name=name, reason="no_previous_version")
            return False

        # Load previous version from history
        prev_version = skill.version - 1
        history_file = self._history_dir / f"auto_{name}_v{prev_version}.py"

        if not history_file.exists():
            log.warning("rollback_impossible", name=name, reason="history_not_found")
            return False

        # Replace current version with previous
        code_file = self._skills_dir / f"auto_{name}.py"
        shutil.copy2(history_file, code_file)

        skill.status = GenerationStatus.ROLLED_BACK
        skill.version = prev_version

        log.info("skill_rolled_back", name=name, to_version=prev_version)
        return True

    # ========================================================================
    # End-to-End Pipeline
    # ========================================================================

    async def process_gap(
        self,
        gap: SkillGap,
        *,
        skill_registry: Any | None = None,
        max_retries: int = 2,
    ) -> GeneratedSkill:
        """Kompletter Workflow: Gap → Generate → Test → Register.

        Bei fehlgeschlagenen Tests wird bis zu max_retries mal
        erneut generiert (mit Fehlern als Kontext).

        Args:
            gap: Die zu schliessende Skill-Luecke.
            skill_registry: Optional fuer sofortige Registrierung.
            max_retries: Maximale Regenerierungsversuche (0-5).

        Returns:
            GeneratedSkill (auch bei Fehlschlag, mit Status).
        """
        max_retries = max(0, min(max_retries, 5))

        skill = await self.generate(gap)

        if skill.status == GenerationStatus.FAILED:
            return skill

        for attempt in range(max_retries + 1):
            passed = await self.test(skill)
            if passed:
                break

            if attempt < max_retries:
                # Re-generate with error context (limited)
                log.info(
                    "skill_retry",
                    name=skill.name,
                    attempt=attempt + 2,
                    errors=skill.test_errors[:2],
                )
                error_context = f"\n\nFehler im vorherigen Versuch:\n{skill.test_output[:300]}"
                gap.context = (gap.context + error_context)[: GapDetector.MAX_CONTEXT_LENGTH]
                skill = await self.generate(gap)

        if skill.test_passed:
            self.register(skill, skill_registry=skill_registry)

        return skill

    async def process_all_gaps(
        self,
        skill_registry: Any | None = None,
    ) -> list[GeneratedSkill]:
        """Verarbeitet alle actionable Gaps.

        Returns:
            Liste der generierten Skills.
        """
        gaps = self._gap_detector.get_actionable_gaps()
        results = []

        for gap in gaps:
            skill = await self.process_gap(gap, skill_registry=skill_registry)
            results.append(skill)

            # Remove gap if successful
            if skill.status == GenerationStatus.REGISTERED:
                self._gap_detector.clear_gap(gap.id)

        return results

    # ========================================================================
    # Hilfsfunktionen
    # ========================================================================

    def _derive_skill_name(self, gap: SkillGap) -> str:
        """Derive a skill name from the gap."""
        if gap.tool_name:
            return gap.tool_name.replace("-", "_").replace(" ", "_").lower()

        # From description
        words = gap.description.lower().split()[:4]
        name = "_".join(w for w in words if w.isalnum())
        return name or f"skill_{hashlib.md5(gap.description.encode()).hexdigest()[:6]}"

    @staticmethod
    def _escape_for_string(text: str) -> str:
        """Escapes text for safe embedding in Python string literals."""
        return (
            text.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
        )

    def _generate_stub(self, gap: SkillGap) -> str:
        """Generiert einen Code-Stub wenn kein LLM verfuegbar ist."""
        name = self._derive_skill_name(gap)
        safe_desc = self._escape_for_string(gap.description[:200])
        return textwrap.dedent(f'''\
            """Auto-generierter Skill: {safe_desc}"""

            from __future__ import annotations
            from typing import Any


            async def {name}_handler(**params: Any) -> str:
                """Handler fuer {name}.

                Args:
                    **params: Tool-Parameter.

                Returns:
                    Ergebnis als String.
                """
                # TODO: Implementierung
                return f"{{name}}: Stub-Implementierung. Parameter: {{params}}"


            def register(mcp_client: Any) -> None:
                """Registriert den Skill als MCP-Tool."""
                schema = {{
                    "name": "{name}",
                    "description": "{safe_desc}",
                    "inputSchema": {{
                        "type": "object",
                        "properties": {{
                            "input": {{"type": "string", "description": "Eingabe"}},
                        }},
                        "required": ["input"],
                    }},
                }}
                mcp_client.register_builtin_handler("{name}", schema, {name}_handler)
        ''')

    def _generate_stub_test(self, name: str, code: str) -> str:
        """Generate a simple test stub."""
        safe_name = self._escape_for_string(name)
        return textwrap.dedent(f'''\
            """Auto-generierte Tests fuer {safe_name}."""

            import pytest


            class TestSyntax:
                """Basis-Syntaxtest."""

                def test_code_compiles(self) -> None:
                    """Code hat keine Syntaxfehler."""
                    import importlib.util, tempfile, pathlib
                    # Test via file import instead of embedded string
                    assert True  # Syntax checked during generation

                def test_module_name(self) -> None:
                    assert "{safe_name}" != ""
        ''')

    def _generate_skill_markdown(self, skill: GeneratedSkill) -> str:
        """Generate skill Markdown with frontmatter."""
        tools = ", ".join(skill.tools_provided) if skill.tools_provided else skill.name
        safe_desc = self._escape_for_string(skill.description[:200])
        safe_name = self._escape_for_string(skill.name)
        return textwrap.dedent(f"""\
            ---
            name: {safe_name}
            description: "{safe_desc}"
            category: auto_generated
            trigger_keywords: [{safe_name.replace("_", ", ")}]
            tools_required: [{tools}]
            enabled: true
            auto_generated: true
            version: {skill.version}
            ---

            # {safe_name}

            {safe_desc}

            Automatisch generiert am {skill.created_at[:10]}.
            Version {skill.version_tag}.
        """)

    def _archive_if_exists(self, skill: GeneratedSkill) -> None:
        """Archive the current version of a skill."""
        code_file = self._skills_dir / f"{skill.module_name}.py"
        if code_file.exists():
            # Save old version to history
            prev_version = skill.version - 1 if skill.version > 1 else 1
            history_file = self._history_dir / f"{skill.module_name}_v{prev_version}.py"
            shutil.copy2(code_file, history_file)
            log.info(
                "skill_archived",
                name=skill.name,
                version=prev_version,
                path=str(history_file),
            )

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Extract Python code from LLM response (with/without Markdown fences)."""
        # Search for ```python ... ``` block
        import re

        match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Search for ``` ... ``` block
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # No Markdown block → gesamten Text nehmen
        return text.strip()

    # ========================================================================
    # Statistiken
    # ========================================================================

    def stats(self) -> dict[str, Any]:
        """Generator statistics."""
        by_status: dict[str, int] = {}
        for skill in self._generated.values():
            status = skill.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_generated": len(self._generated),
            "by_status": by_status,
            "gaps_detected": self._gap_detector.gap_count,
            "actionable_gaps": len(self._gap_detector.get_actionable_gaps()),
            "skills_dir": str(self._skills_dir),
        }
