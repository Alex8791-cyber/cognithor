"""Deterministic Message Bindings: Regelbasiertes Agent-Routing.

Im Gegensatz zum probabilistischen Keyword-Matching liefert das
Bindings-System deterministische Routing-Entscheidungen. Bindings
werden in Prioritätsreihenfolge ausgewertet -- die erste passende
Regel gewinnt. Erst wenn kein Binding greift, fällt das System
auf das bestehende Keyword-/Pattern-Matching zurück.

Inspiriert von OpenClaws Bindings-System, das Nachrichten über
Channel-Filter, Regex-Patterns und User-Zuordnungen verteilt.

Architektur:
  IncomingMessage → MessageContext extrahieren
                  → BindingEngine.evaluate(context)
                  → Erste passende Regel → Agent
                  → Kein Match → Keyword-Routing (Fallback)

Binding-Typen (alle AND-verknüpft innerhalb einer Regel):
  - Channel-Filter:    Nur bestimmte Kanäle (telegram, cli, api, ...)
  - User-Filter:       Nur bestimmte User-IDs
  - Command-Prefixes:  Slash-Commands (/code, /research, /hilfe)
  - Regex-Patterns:    Beliebige Muster auf den Nachrichtentext
  - Metadata-Match:    Key-Value-Bedingungen auf msg.metadata
  - Zeitfenster:       Nur in bestimmten Tageszeiten/Wochentagen
  - Negation:          Invertierte Bedingungen (NOT-Logik)

Bibel-Referenz: §9.2 (Multi-Agent-Routing -- Bindings-Erweiterung)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Enums
# ============================================================================


class BindingMatchResult(Enum):
    """Ergebnis der Binding-Auswertung."""

    MATCH = "match"
    NO_MATCH = "no_match"
    DISABLED = "disabled"
    ERROR = "error"


class Weekday(Enum):
    """ISO-Wochentage (Montag=1 ... Sonntag=7)."""

    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7


# Abkürzungen für YAML-Konfiguration
WEEKDAY_ALIASES: dict[str, Weekday] = {
    "mo": Weekday.MONDAY, "mon": Weekday.MONDAY, "montag": Weekday.MONDAY,
    "di": Weekday.TUESDAY, "tue": Weekday.TUESDAY, "dienstag": Weekday.TUESDAY,
    "mi": Weekday.WEDNESDAY, "wed": Weekday.WEDNESDAY, "mittwoch": Weekday.WEDNESDAY,
    "do": Weekday.THURSDAY, "thu": Weekday.THURSDAY, "donnerstag": Weekday.THURSDAY,
    "fr": Weekday.FRIDAY, "fri": Weekday.FRIDAY, "freitag": Weekday.FRIDAY,
    "sa": Weekday.SATURDAY, "sat": Weekday.SATURDAY, "samstag": Weekday.SATURDAY,
    "so": Weekday.SUNDAY, "sun": Weekday.SUNDAY, "sonntag": Weekday.SUNDAY,
}


# ============================================================================
# Datenmodelle
# ============================================================================


@dataclass
class TimeWindow:
    """Zeitfenster-Bedingung.

    Definiert wann ein Binding aktiv ist:
      - Tageszeit (start_time bis end_time)
      - Wochentage (z.B. nur Mo-Fr)
      - Timezone (Default: Europe/Berlin)

    Beispiele:
      Geschäftszeiten: TimeWindow(start="08:00", end="18:00", weekdays=[mo-fr])
      Wochenende:      TimeWindow(weekdays=[sa, so])
      Nachts:          TimeWindow(start="22:00", end="06:00")
    """

    start_time: time | None = None  # None = 00:00
    end_time: time | None = None  # None = 23:59
    weekdays: list[Weekday] = field(default_factory=list)  # Leer = alle Tage
    timezone: str = "Europe/Berlin"

    def matches(self, now: datetime | None = None) -> bool:
        """Prüft ob der aktuelle Zeitpunkt im Fenster liegt.

        Args:
            now: Aktueller Zeitpunkt (für Tests). Default: jetzt.

        Returns:
            True wenn im Zeitfenster.
        """
        if now is None:
            try:
                from zoneinfo import ZoneInfo
                now = datetime.now(ZoneInfo(self.timezone))
            except Exception:
                now = datetime.now()

        # Wochentag prüfen
        if self.weekdays:
            # Python: Monday=0 ... Sunday=6 → ISO: Monday=1 ... Sunday=7
            iso_weekday = now.isoweekday()
            if not any(wd.value == iso_weekday for wd in self.weekdays):
                return False

        # Tageszeit prüfen
        current_time = now.time()
        start = self.start_time or time(0, 0)
        end = self.end_time or time(23, 59, 59)

        if start <= end:
            # Normales Fenster (z.B. 08:00 - 18:00)
            return start <= current_time <= end
        else:
            # Über Mitternacht (z.B. 22:00 - 06:00)
            return current_time >= start or current_time <= end


@dataclass
class MessageContext:
    """Vollständiger Kontext einer eingehenden Nachricht für Binding-Auswertung.

    Wird aus IncomingMessage extrahiert und enthält alle routing-relevanten
    Informationen. Trennt die Binding-Logik von der Message-Struktur.
    """

    text: str
    channel: str = ""
    user_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None

    # Abgeleitete Felder (werden beim Erstellen berechnet)
    command: str = ""  # Erster Slash-Command (z.B. "/code")
    text_without_command: str = ""  # Text nach dem Command

    def __post_init__(self) -> None:
        """Extrahiert Command-Prefix aus dem Text."""
        stripped = self.text.strip()
        if stripped.startswith("/"):
            parts = stripped.split(maxsplit=1)
            self.command = parts[0].lower()
            self.text_without_command = parts[1] if len(parts) > 1 else ""
        else:
            self.command = ""
            self.text_without_command = stripped

    @classmethod
    def from_incoming(cls, msg: Any) -> MessageContext:
        """Erstellt MessageContext aus einer IncomingMessage.

        Args:
            msg: IncomingMessage-Instanz.

        Returns:
            MessageContext mit allen extrahierten Feldern.
        """
        return cls(
            text=getattr(msg, "text", ""),
            channel=getattr(msg, "channel", ""),
            user_id=getattr(msg, "user_id", ""),
            metadata=dict(getattr(msg, "metadata", {})),
            timestamp=getattr(msg, "timestamp", None),
        )


@dataclass
class MessageBinding:
    """Deterministische Routing-Regel.

    Alle Bedingungen sind AND-verknüpft: Nur wenn ALLE gesetzten
    Bedingungen zutreffen, matcht das Binding. Nicht gesetzte
    Bedingungen (None/leer) werden ignoriert.

    Attributes:
        name: Eindeutiger Name des Bindings.
        target_agent: Name des Ziel-Agenten.
        priority: Höher = wird zuerst ausgewertet (Default: 100).
        description: Menschenlesbare Beschreibung.

        channels: Erlaubte Kanäle (None = alle).
        user_ids: Erlaubte User-IDs (None = alle).
        command_prefixes: Slash-Commands die matchen (z.B. ["/code"]).
        message_patterns: Regex-Patterns auf den Text.
        metadata_conditions: Key-Value-Bedingungen auf Metadata.
        time_windows: Zeitfenster in denen das Binding aktiv ist.

        negate: Invertiert das Ergebnis (NOT-Logik).
        stop_processing: Bei Match keine weiteren Bindings prüfen (Default: True).
        enabled: Binding aktiv/inaktiv.
    """

    name: str
    target_agent: str
    priority: int = 100

    # Beschreibung
    description: str = ""

    # --- Bedingungen (alle AND-verknüpft) ---

    # Channel-Filter
    channels: list[str] | None = None

    # User-Filter
    user_ids: list[str] | None = None

    # Command-Prefix-Filter
    command_prefixes: list[str] | None = None

    # Regex-Pattern-Filter (auf Nachrichtentext)
    message_patterns: list[str] | None = None

    # Metadata-Bedingungen (Key muss existieren und Wert matchen)
    metadata_conditions: dict[str, str] | None = None

    # Zeitfenster
    time_windows: list[TimeWindow] | None = None

    # --- Verhalten ---

    negate: bool = False  # Invertiert Ergebnis
    stop_processing: bool = True  # Bei Match keine weiteren Bindings
    enabled: bool = True

    # --- Kompilierte Patterns (intern) ---
    _compiled_patterns: list[re.Pattern] = field(
        default_factory=list, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        """Kompiliert Regex-Patterns beim Erstellen."""
        self._compile()

    def _compile(self) -> None:
        """Kompiliert die Message-Patterns zu Regex-Objekten."""
        self._compiled_patterns = []
        if self.message_patterns:
            for pattern_str in self.message_patterns:
                try:
                    self._compiled_patterns.append(
                        re.compile(pattern_str, re.IGNORECASE),
                    )
                except re.error as exc:
                    log.warning(
                        "binding_pattern_compile_error",
                        binding=self.name,
                        pattern=pattern_str,
                        error=str(exc),
                    )

    def evaluate(self, ctx: MessageContext) -> BindingMatchResult:
        """Wertet das Binding gegen einen MessageContext aus.

        Alle gesetzten Bedingungen müssen zutreffen (AND).
        Nicht gesetzte Bedingungen werden ignoriert.

        Args:
            ctx: Nachrichtenkontext.

        Returns:
            BindingMatchResult.MATCH oder NO_MATCH.
        """
        if not self.enabled:
            return BindingMatchResult.DISABLED

        try:
            raw_match = self._evaluate_conditions(ctx)
        except Exception as exc:
            log.warning(
                "binding_evaluation_error",
                binding=self.name,
                error=str(exc),
            )
            return BindingMatchResult.ERROR

        # Negation anwenden
        if self.negate:
            raw_match = not raw_match

        return BindingMatchResult.MATCH if raw_match else BindingMatchResult.NO_MATCH

    def _evaluate_conditions(self, ctx: MessageContext) -> bool:
        """Interne Auswertung aller Bedingungen.

        Returns:
            True wenn alle gesetzten Bedingungen matchen.
        """
        # 1. Channel-Filter
        if self.channels is not None:
            if ctx.channel not in self.channels:
                return False

        # 2. User-Filter
        if self.user_ids is not None:
            if ctx.user_id not in self.user_ids:
                return False

        # 3. Command-Prefix-Filter
        if self.command_prefixes is not None:
            if not ctx.command:
                return False
            if ctx.command not in self.command_prefixes:
                return False

        # 4. Regex-Pattern-Filter (mindestens ein Pattern muss matchen)
        if self.message_patterns is not None:
            if not self._compiled_patterns:
                return False
            text = ctx.text
            if not any(p.search(text) for p in self._compiled_patterns):
                return False

        # 5. Metadata-Bedingungen (alle müssen matchen)
        if self.metadata_conditions is not None:
            for key, expected_value in self.metadata_conditions.items():
                actual = ctx.metadata.get(key)
                if actual is None:
                    return False
                if str(actual) != expected_value:
                    return False

        # 6. Zeitfenster (mindestens ein Fenster muss passen)
        if self.time_windows is not None:
            if not any(tw.matches(ctx.timestamp) for tw in self.time_windows):
                return False

        # Alle Bedingungen erfüllt
        return True


@dataclass
class BindingMatchInfo:
    """Detailliertes Ergebnis einer Binding-Auswertung."""

    binding: MessageBinding
    result: BindingMatchResult
    target_agent: str

    @property
    def matched(self) -> bool:
        return self.result == BindingMatchResult.MATCH


# ============================================================================
# Binding Engine
# ============================================================================


class BindingEngine:
    """Wertet Bindings in Prioritätsreihenfolge aus.

    Die Engine ist das Herzstück des deterministischen Routings.
    Sie evaluiert alle aktiven Bindings gegen den MessageContext
    und gibt die erste passende Regel zurück.

    Reihenfolge:
      1. Bindings nach Priorität sortiert (höchste zuerst)
      2. Bei gleicher Priorität: alphabetisch nach Name
      3. Erste passende Regel gewinnt (deterministic)
      4. Kein Match → None (Fallback auf Keyword-Routing)

    Usage:
        engine = BindingEngine()
        engine.add_binding(MessageBinding(
            name="telegram_to_organizer",
            target_agent="organizer",
            channels=["telegram"],
        ))

        ctx = MessageContext(text="Was ist heute?", channel="telegram")
        match = engine.evaluate(ctx)
        # → BindingMatchInfo(binding=..., target_agent="organizer")
    """

    def __init__(self) -> None:
        self._bindings: dict[str, MessageBinding] = {}
        self._sorted_bindings: list[MessageBinding] = []

    @property
    def binding_count(self) -> int:
        """Anzahl registrierter Bindings."""
        return len(self._bindings)

    @property
    def active_count(self) -> int:
        """Anzahl aktiver Bindings."""
        return sum(1 for b in self._bindings.values() if b.enabled)

    # ========================================================================
    # Binding-Verwaltung
    # ========================================================================

    def add_binding(self, binding: MessageBinding) -> None:
        """Registriert ein neues Binding (oder überschreibt bestehendes).

        Args:
            binding: Das zu registrierende Binding.
        """
        self._bindings[binding.name] = binding
        self._resort()
        log.info(
            "binding_added",
            name=binding.name,
            target=binding.target_agent,
            priority=binding.priority,
        )

    def add_bindings(self, bindings: list[MessageBinding]) -> None:
        """Registriert mehrere Bindings auf einmal.

        Args:
            bindings: Liste von Bindings.
        """
        for binding in bindings:
            self._bindings[binding.name] = binding
        self._resort()
        log.info("bindings_bulk_added", count=len(bindings))

    def remove_binding(self, name: str) -> bool:
        """Entfernt ein Binding.

        Args:
            name: Name des zu entfernenden Bindings.

        Returns:
            True wenn das Binding existierte und entfernt wurde.
        """
        if name in self._bindings:
            del self._bindings[name]
            self._resort()
            log.info("binding_removed", name=name)
            return True
        return False

    def get_binding(self, name: str) -> MessageBinding | None:
        """Gibt ein Binding per Name zurück."""
        return self._bindings.get(name)

    def list_bindings(self) -> list[MessageBinding]:
        """Alle Bindings in Prioritätsreihenfolge."""
        return list(self._sorted_bindings)

    def enable_binding(self, name: str) -> bool:
        """Aktiviert ein Binding."""
        binding = self._bindings.get(name)
        if binding:
            binding.enabled = True
            return True
        return False

    def disable_binding(self, name: str) -> bool:
        """Deaktiviert ein Binding."""
        binding = self._bindings.get(name)
        if binding:
            binding.enabled = False
            return True
        return False

    def clear(self) -> None:
        """Entfernt alle Bindings."""
        self._bindings.clear()
        self._sorted_bindings.clear()

    def _resort(self) -> None:
        """Sortiert Bindings nach Priorität (absteigend), dann Name."""
        self._sorted_bindings = sorted(
            self._bindings.values(),
            key=lambda b: (-b.priority, b.name),
        )

    # ========================================================================
    # Auswertung
    # ========================================================================

    def evaluate(self, ctx: MessageContext) -> BindingMatchInfo | None:
        """Wertet alle Bindings gegen den MessageContext aus.

        First-Match-Wins: Die erste passende Regel (nach Priorität)
        bestimmt den Ziel-Agenten. Deterministic -- gleiches Input
        ergibt immer gleiches Output.

        Args:
            ctx: Nachrichtenkontext.

        Returns:
            BindingMatchInfo bei Match, None wenn kein Binding passt.
        """
        for binding in self._sorted_bindings:
            result = binding.evaluate(ctx)

            if result == BindingMatchResult.MATCH:
                log.info(
                    "binding_matched",
                    binding=binding.name,
                    target=binding.target_agent,
                    channel=ctx.channel,
                    user=ctx.user_id,
                )
                return BindingMatchInfo(
                    binding=binding,
                    result=result,
                    target_agent=binding.target_agent,
                )

            if result == BindingMatchResult.ERROR:
                log.warning(
                    "binding_evaluation_skipped",
                    binding=binding.name,
                    reason="evaluation_error",
                )

        return None

    def evaluate_all(self, ctx: MessageContext) -> list[BindingMatchInfo]:
        """Wertet ALLE Bindings aus (für Debugging/Monitoring).

        Im Gegensatz zu evaluate() stoppt diese Methode nicht beim
        ersten Match, sondern gibt alle Ergebnisse zurück.

        Args:
            ctx: Nachrichtenkontext.

        Returns:
            Liste aller Binding-Ergebnisse.
        """
        results = []
        for binding in self._sorted_bindings:
            result = binding.evaluate(ctx)
            results.append(BindingMatchInfo(
                binding=binding,
                result=result,
                target_agent=binding.target_agent,
            ))
        return results

    # ========================================================================
    # YAML Persistenz
    # ========================================================================

    def save_yaml(self, path: Path) -> None:
        """Speichert alle Bindings als YAML.

        Args:
            path: Pfad zur YAML-Datei.
        """
        import yaml

        bindings_data = []
        for binding in self._sorted_bindings:
            data: dict[str, Any] = {
                "name": binding.name,
                "target_agent": binding.target_agent,
                "priority": binding.priority,
            }

            if binding.description:
                data["description"] = binding.description
            if binding.channels is not None:
                data["channels"] = binding.channels
            if binding.user_ids is not None:
                data["user_ids"] = binding.user_ids
            if binding.command_prefixes is not None:
                data["command_prefixes"] = binding.command_prefixes
            if binding.message_patterns is not None:
                data["message_patterns"] = binding.message_patterns
            if binding.metadata_conditions is not None:
                data["metadata_conditions"] = binding.metadata_conditions
            if binding.time_windows is not None:
                windows = []
                for tw in binding.time_windows:
                    w: dict[str, Any] = {}
                    if tw.start_time:
                        w["start"] = tw.start_time.strftime("%H:%M")
                    if tw.end_time:
                        w["end"] = tw.end_time.strftime("%H:%M")
                    if tw.weekdays:
                        w["weekdays"] = [wd.name.lower() for wd in tw.weekdays]
                    if tw.timezone != "Europe/Berlin":
                        w["timezone"] = tw.timezone
                    windows.append(w)
                data["time_windows"] = windows
            if binding.negate:
                data["negate"] = True
            if not binding.stop_processing:
                data["stop_processing"] = False
            if not binding.enabled:
                data["enabled"] = False

            bindings_data.append(data)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.dump(
                {"bindings": bindings_data},
                default_flow_style=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        log.info("bindings_yaml_saved", path=str(path), count=len(bindings_data))

    @classmethod
    def from_yaml(cls, path: Path) -> BindingEngine:
        """Lädt Bindings aus einer YAML-Datei.

        Erwartetes Format:
            bindings:
              - name: telegram_coding
                target_agent: coder
                priority: 200
                channels: [telegram]
                command_prefixes: ["/code", "/shell"]

              - name: business_hours_support
                target_agent: support_agent
                priority: 150
                time_windows:
                  - start: "08:00"
                    end: "18:00"
                    weekdays: [mo, di, mi, do, fr]

        Args:
            path: Pfad zur YAML-Datei.

        Returns:
            Konfigurierte BindingEngine.
        """
        import yaml

        engine = cls()

        if not path.exists():
            log.info("bindings_yaml_not_found", path=str(path))
            return engine

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("bindings_yaml_parse_error", path=str(path), error=str(exc))
            return engine

        if not data or "bindings" not in data:
            return engine

        for entry in data["bindings"]:
            try:
                binding = _parse_binding_entry(entry)
                engine.add_binding(binding)
            except Exception as exc:
                log.warning(
                    "binding_parse_error",
                    entry=str(entry)[:200],
                    error=str(exc),
                )

        return engine

    # ========================================================================
    # Statistiken
    # ========================================================================

    def stats(self) -> dict[str, Any]:
        """Engine-Statistiken."""
        agent_distribution: dict[str, int] = {}
        for b in self._bindings.values():
            agent_distribution[b.target_agent] = (
                agent_distribution.get(b.target_agent, 0) + 1
            )

        return {
            "total_bindings": len(self._bindings),
            "active_bindings": self.active_count,
            "agent_distribution": agent_distribution,
            "bindings": [
                {
                    "name": b.name,
                    "target": b.target_agent,
                    "priority": b.priority,
                    "enabled": b.enabled,
                }
                for b in self._sorted_bindings
            ],
        }


# ============================================================================
# YAML Parsing Helpers
# ============================================================================


def _parse_time(s: str) -> time:
    """Parst einen Zeit-String (HH:MM oder HH:MM:SS)."""
    parts = s.strip().split(":")
    if len(parts) == 2:
        return time(int(parts[0]), int(parts[1]))
    if len(parts) == 3:
        return time(int(parts[0]), int(parts[1]), int(parts[2]))
    msg = f"Ungültiges Zeitformat: '{s}' (erwartet HH:MM oder HH:MM:SS)"
    raise ValueError(msg)


def _parse_weekday(s: str) -> Weekday:
    """Parst einen Wochentag-String."""
    key = s.strip().lower()
    if key in WEEKDAY_ALIASES:
        return WEEKDAY_ALIASES[key]
    # Versuche direkt als Enum-Name
    try:
        return Weekday[key.upper()]
    except KeyError:
        msg = f"Unbekannter Wochentag: '{s}'"
        raise ValueError(msg) from None


def _parse_time_window(data: dict[str, Any]) -> TimeWindow:
    """Parst ein TimeWindow aus einem YAML-Dict."""
    start = _parse_time(data["start"]) if "start" in data else None
    end = _parse_time(data["end"]) if "end" in data else None
    weekdays = [_parse_weekday(d) for d in data.get("weekdays", [])]
    tz = data.get("timezone", "Europe/Berlin")

    return TimeWindow(
        start_time=start,
        end_time=end,
        weekdays=weekdays,
        timezone=tz,
    )


def _parse_binding_entry(entry: dict[str, Any]) -> MessageBinding:
    """Parst ein einzelnes Binding aus einem YAML-Dict."""
    name = entry["name"]
    target = entry["target_agent"]
    priority = entry.get("priority", 100)

    # Zeitfenster parsen
    time_windows = None
    if "time_windows" in entry:
        time_windows = [_parse_time_window(tw) for tw in entry["time_windows"]]

    return MessageBinding(
        name=name,
        target_agent=target,
        priority=priority,
        description=entry.get("description", ""),
        channels=entry.get("channels"),
        user_ids=entry.get("user_ids"),
        command_prefixes=entry.get("command_prefixes"),
        message_patterns=entry.get("message_patterns"),
        metadata_conditions=entry.get("metadata_conditions"),
        time_windows=time_windows,
        negate=entry.get("negate", False),
        stop_processing=entry.get("stop_processing", True),
        enabled=entry.get("enabled", True),
    )


# ============================================================================
# Factory-Funktionen für häufige Binding-Patterns
# ============================================================================


def channel_binding(
    name: str,
    target_agent: str,
    channels: list[str],
    *,
    priority: int = 100,
) -> MessageBinding:
    """Erstellt ein Channel-basiertes Binding.

    Beispiel: Alle Telegram-Nachrichten an den Organizer.

    Args:
        name: Binding-Name.
        target_agent: Ziel-Agent.
        channels: Liste der Kanäle.
        priority: Priorität.
    """
    return MessageBinding(
        name=name,
        target_agent=target_agent,
        channels=channels,
        priority=priority,
        description=f"Channel-Binding: {channels} → {target_agent}",
    )


def command_binding(
    name: str,
    target_agent: str,
    commands: list[str],
    *,
    priority: int = 200,
) -> MessageBinding:
    """Erstellt ein Command-basiertes Binding.

    Beispiel: /code → Coding-Agent

    Args:
        name: Binding-Name.
        target_agent: Ziel-Agent.
        commands: Slash-Commands (z.B. ["/code", "/shell"]).
        priority: Priorität (default 200, höher als Channel-Bindings).
    """
    # Normalisiere Commands
    normalized = [c.lower() if c.startswith("/") else f"/{c.lower()}" for c in commands]

    return MessageBinding(
        name=name,
        target_agent=target_agent,
        command_prefixes=normalized,
        priority=priority,
        description=f"Command-Binding: {normalized} → {target_agent}",
    )


def user_binding(
    name: str,
    target_agent: str,
    user_ids: list[str],
    *,
    priority: int = 150,
    channels: list[str] | None = None,
) -> MessageBinding:
    """Erstellt ein User-basiertes Binding.

    Beispiel: Bestimmte User immer an einen Premium-Agenten.

    Args:
        name: Binding-Name.
        target_agent: Ziel-Agent.
        user_ids: Liste der User-IDs.
        priority: Priorität.
        channels: Optional Channel-Filter.
    """
    return MessageBinding(
        name=name,
        target_agent=target_agent,
        user_ids=user_ids,
        channels=channels,
        priority=priority,
        description=f"User-Binding: {user_ids} → {target_agent}",
    )


def regex_binding(
    name: str,
    target_agent: str,
    patterns: list[str],
    *,
    priority: int = 180,
) -> MessageBinding:
    """Erstellt ein Regex-basiertes Binding.

    Beispiel: Alle Nachrichten mit "BU-Tarif" oder "Berufsunfähigkeit" → Versicherungs-Agent.

    Args:
        name: Binding-Name.
        target_agent: Ziel-Agent.
        patterns: Regex-Patterns.
        priority: Priorität.
    """
    return MessageBinding(
        name=name,
        target_agent=target_agent,
        message_patterns=patterns,
        priority=priority,
        description=f"Regex-Binding: {patterns} → {target_agent}",
    )


def schedule_binding(
    name: str,
    target_agent: str,
    *,
    start: str = "08:00",
    end: str = "18:00",
    weekdays: list[str] | None = None,
    priority: int = 50,
) -> MessageBinding:
    """Erstellt ein zeitgesteuertes Binding.

    Beispiel: Während Geschäftszeiten → Support-Agent.

    Args:
        name: Binding-Name.
        target_agent: Ziel-Agent.
        start: Startzeit (HH:MM).
        end: Endzeit (HH:MM).
        weekdays: Wochentage (z.B. ["mo", "di", "mi", "do", "fr"]).
        priority: Priorität (default 50, niedrig da meist als Fallback).
    """
    wds = [_parse_weekday(d) for d in weekdays] if weekdays else []

    return MessageBinding(
        name=name,
        target_agent=target_agent,
        time_windows=[TimeWindow(
            start_time=_parse_time(start),
            end_time=_parse_time(end),
            weekdays=wds,
        )],
        priority=priority,
        description=f"Schedule-Binding: {start}-{end} → {target_agent}",
    )
