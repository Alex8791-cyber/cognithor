"""Deterministic Message Bindings: Rule-based agent routing.

Unlike probabilistic keyword matching, the bindings system provides
deterministic routing decisions. Bindings are evaluated in priority
order -- the first matching rule wins. Only when no binding matches
does the system fall back to keyword/pattern matching.

Inspired by OpenClaw's bindings system, which distributes messages
via channel filters, regex patterns and user assignments.

Architecture:
  IncomingMessage -> extract MessageContext
                  -> BindingEngine.evaluate(context)
                  -> First matching rule -> Agent
                  -> No match -> Keyword routing (fallback)

Binding types (all AND-linked within a rule):
  - Channel filter:    Only specific channels (telegram, cli, api, ...)
  - User filter:       Only specific user IDs
  - Command prefixes:  Slash commands (/code, /research, /hilfe)
  - Regex patterns:    Arbitrary patterns on the message text
  - Metadata match:    Key-value conditions on msg.metadata
  - Time windows:      Only during certain times/weekdays
  - Negation:          Inverted conditions (NOT logic)

Reference: §9.2 (Multi-Agent Routing -- Bindings Extension)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)


# ============================================================================
# Enums
# ============================================================================


class BindingMatchResult(Enum):
    """Result of binding evaluation."""

    MATCH = "match"
    NO_MATCH = "no_match"
    DISABLED = "disabled"
    ERROR = "error"


class Weekday(Enum):
    """ISO weekdays (Monday=1 ... Sunday=7)."""

    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7


# Abbreviations for YAML configuration
WEEKDAY_ALIASES: dict[str, Weekday] = {
    "mo": Weekday.MONDAY,
    "mon": Weekday.MONDAY,
    "montag": Weekday.MONDAY,
    "di": Weekday.TUESDAY,
    "tue": Weekday.TUESDAY,
    "dienstag": Weekday.TUESDAY,
    "mi": Weekday.WEDNESDAY,
    "wed": Weekday.WEDNESDAY,
    "mittwoch": Weekday.WEDNESDAY,
    "do": Weekday.THURSDAY,
    "thu": Weekday.THURSDAY,
    "donnerstag": Weekday.THURSDAY,
    "fr": Weekday.FRIDAY,
    "fri": Weekday.FRIDAY,
    "freitag": Weekday.FRIDAY,
    "sa": Weekday.SATURDAY,
    "sat": Weekday.SATURDAY,
    "samstag": Weekday.SATURDAY,
    "so": Weekday.SUNDAY,
    "sun": Weekday.SUNDAY,
    "sonntag": Weekday.SUNDAY,
}


# ============================================================================
# Data models
# ============================================================================


@dataclass
class TimeWindow:
    """Time window condition.

    Defines when a binding is active:
      - Time of day (start_time to end_time)
      - Weekdays (e.g. Mon-Fri only)
      - Timezone (default: Europe/Berlin)

    Examples:
      Business hours: TimeWindow(start="08:00", end="18:00", weekdays=[mo-fr])
      Weekend:        TimeWindow(weekdays=[sa, so])
      Night:          TimeWindow(start="22:00", end="06:00")
    """

    start_time: time | None = None  # None = 00:00
    end_time: time | None = None  # None = 23:59
    weekdays: list[Weekday] = field(default_factory=list)  # Empty = all days
    timezone: str = "Europe/Berlin"

    def matches(self, now: datetime | None = None) -> bool:
        """Check whether the current time falls within the window.

        Args:
            now: Current time (for tests). Default: now.

        Returns:
            True if within the time window.
        """
        if now is None:
            try:
                from zoneinfo import ZoneInfo

                now = datetime.now(ZoneInfo(self.timezone))
            except Exception:
                now = datetime.now()

        # Check weekday
        if self.weekdays:
            # Python: Monday=0 ... Sunday=6 -> ISO: Monday=1 ... Sunday=7
            iso_weekday = now.isoweekday()
            if not any(wd.value == iso_weekday for wd in self.weekdays):
                return False

        # Check time of day
        current_time = now.time()
        start = self.start_time or time(0, 0)
        end = self.end_time or time(23, 59, 59)

        if start <= end:
            # Normal window (e.g. 08:00 - 18:00)
            return start <= current_time <= end
        else:
            # Across midnight (e.g. 22:00 - 06:00)
            return current_time >= start or current_time <= end


@dataclass
class MessageContext:
    """Full context of an incoming message for binding evaluation.

    Extracted from IncomingMessage and contains all routing-relevant
    information. Separates binding logic from the message structure.
    """

    text: str
    channel: str = ""
    user_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None

    # Derived fields (computed on creation)
    command: str = ""  # First slash command (e.g. "/code")
    text_without_command: str = ""  # Text after the command

    def __post_init__(self) -> None:
        """Extract command prefix from the text."""
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
        """Create MessageContext from an IncomingMessage.

        Args:
            msg: IncomingMessage instance.

        Returns:
            MessageContext with all extracted fields.
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
    """Deterministic routing rule.

    All conditions are AND-linked: only when ALL set conditions
    are met does the binding match. Unset conditions (None/empty)
    are ignored.

    Attributes:
        name: Unique name of the binding.
        target_agent: Name of the target agent.
        priority: Higher = evaluated first (default: 100).
        description: Human-readable description.

        channels: Allowed channels (None = all).
        user_ids: Allowed user IDs (None = all).
        command_prefixes: Slash commands that match (e.g. ["/code"]).
        message_patterns: Regex patterns on the text.
        metadata_conditions: Key-value conditions on metadata.
        time_windows: Time windows during which the binding is active.

        negate: Invert the result (NOT logic).
        stop_processing: On match, skip remaining bindings (default: True).
        enabled: Binding active/inactive.
    """

    name: str
    target_agent: str
    priority: int = 100

    # Description
    description: str = ""

    # --- Conditions (all AND-linked) ---

    # Channel filter
    channels: list[str] | None = None

    # User filter
    user_ids: list[str] | None = None

    # Command prefix filter
    command_prefixes: list[str] | None = None

    # Regex pattern filter (on message text)
    message_patterns: list[str] | None = None

    # Metadata conditions (key must exist and value must match)
    metadata_conditions: dict[str, str] | None = None

    # Time windows
    time_windows: list[TimeWindow] | None = None

    # --- Behavior ---

    negate: bool = False  # Invert result
    stop_processing: bool = True  # On match, skip remaining bindings
    enabled: bool = True

    # --- Compiled patterns (internal) ---
    _compiled_patterns: list[re.Pattern[str]] = field(
        default_factory=list,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Compile regex patterns on creation."""
        self._compile()

    def _compile(self) -> None:
        """Compile message patterns to regex objects."""
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
        """Evaluate the binding against a MessageContext.

        All set conditions must be met (AND).
        Unset conditions are ignored.

        Args:
            ctx: Message context.

        Returns:
            BindingMatchResult.MATCH or NO_MATCH.
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

        # Apply negation
        if self.negate:
            raw_match = not raw_match

        return BindingMatchResult.MATCH if raw_match else BindingMatchResult.NO_MATCH

    def _evaluate_conditions(self, ctx: MessageContext) -> bool:
        """Internal evaluation of all conditions.

        Returns:
            True if all set conditions match.
        """
        # 1. Channel-Filter
        if self.channels is not None and ctx.channel not in self.channels:
            return False

        # 2. User-Filter
        if self.user_ids is not None and ctx.user_id not in self.user_ids:
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

        # 5. Metadata conditions (all must match)
        if self.metadata_conditions is not None:
            for key, expected_value in self.metadata_conditions.items():
                actual = ctx.metadata.get(key)
                if actual is None:
                    return False
                if str(actual) != expected_value:
                    return False

        # 6. Zeitfenster (mindestens ein Fenster muss passen)
        if self.time_windows is not None and not any(
            tw.matches(ctx.timestamp) for tw in self.time_windows
        ):
            return False

        # All conditions met
        return True


@dataclass
class BindingMatchInfo:
    """Detailed result of a binding evaluation."""

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
    """Evaluate bindings in priority order.

    The engine is the core of deterministic routing.
    It evaluates all active bindings against the MessageContext
    and returns the first matching rule.

    Order:
      1. Bindings sorted by priority (highest first)
      2. On equal priority: alphabetically by name
      3. First matching rule wins (deterministic)
      4. No match -> None (fallback to keyword routing)

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
        """Number of registered bindings."""
        return len(self._bindings)

    @property
    def active_count(self) -> int:
        """Number of active bindings."""
        return sum(1 for b in self._bindings.values() if b.enabled)

    # ========================================================================
    # Binding management
    # ========================================================================

    def add_binding(self, binding: MessageBinding) -> None:
        """Register a new binding (or overwrite an existing one).

        Args:
            binding: The binding to register.
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
        """Register multiple bindings at once.

        Args:
            bindings: List of bindings.
        """
        for binding in bindings:
            self._bindings[binding.name] = binding
        self._resort()
        log.info("bindings_bulk_added", count=len(bindings))

    def remove_binding(self, name: str) -> bool:
        """Remove a binding.

        Args:
            name: Name of the binding to remove.

        Returns:
            True if the binding existed and was removed.
        """
        if name in self._bindings:
            del self._bindings[name]
            self._resort()
            log.info("binding_removed", name=name)
            return True
        return False

    def get_binding(self, name: str) -> MessageBinding | None:
        """Return a binding by name."""
        return self._bindings.get(name)

    def list_bindings(self) -> list[MessageBinding]:
        """All bindings in priority order."""
        return list(self._sorted_bindings)

    def enable_binding(self, name: str) -> bool:
        """Enable a binding."""
        binding = self._bindings.get(name)
        if binding:
            binding.enabled = True
            return True
        return False

    def disable_binding(self, name: str) -> bool:
        """Disable a binding."""
        binding = self._bindings.get(name)
        if binding:
            binding.enabled = False
            return True
        return False

    def clear(self) -> None:
        """Remove all bindings."""
        self._bindings.clear()
        self._sorted_bindings.clear()

    def _resort(self) -> None:
        """Sort bindings by priority (descending), then name."""
        self._sorted_bindings = sorted(
            self._bindings.values(),
            key=lambda b: (-b.priority, b.name),
        )

    # ========================================================================
    # Evaluation
    # ========================================================================

    def evaluate(self, ctx: MessageContext) -> BindingMatchInfo | None:
        """Evaluate all bindings against the MessageContext.

        First-match-wins: the first matching rule (by priority)
        determines the target agent. Deterministic -- same input
        always yields same output.

        Args:
            ctx: Message context.

        Returns:
            BindingMatchInfo on match, None if no binding matches.
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
        """Evaluate ALL bindings (for debugging/monitoring).

        Unlike evaluate(), this method does not stop at the
        first match but returns all results.

        Args:
            ctx: Message context.

        Returns:
            List of all binding results.
        """
        results = []
        for binding in self._sorted_bindings:
            result = binding.evaluate(ctx)
            results.append(
                BindingMatchInfo(
                    binding=binding,
                    result=result,
                    target_agent=binding.target_agent,
                )
            )
        return results

    # ========================================================================
    # YAML Persistenz
    # ========================================================================

    def save_yaml(self, path: Path) -> None:
        """Save all bindings as YAML.

        Args:
            path: Path to the YAML file.
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
        """Load bindings from a YAML file.

        Expected format:
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
            path: Path to the YAML file.

        Returns:
            Configured BindingEngine.
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
    # Statistics
    # ========================================================================

    def stats(self) -> dict[str, Any]:
        """Engine statistics."""
        agent_distribution: dict[str, int] = {}
        for b in self._bindings.values():
            agent_distribution[b.target_agent] = agent_distribution.get(b.target_agent, 0) + 1

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
    """Parse a time string (HH:MM or HH:MM:SS)."""
    parts = s.strip().split(":")
    if len(parts) == 2:
        return time(int(parts[0]), int(parts[1]))
    if len(parts) == 3:
        return time(int(parts[0]), int(parts[1]), int(parts[2]))
    msg = f"Invalid time format: '{s}' (expected HH:MM or HH:MM:SS)"
    raise ValueError(msg)


def _parse_weekday(s: str) -> Weekday:
    """Parse a weekday string."""
    key = s.strip().lower()
    if key in WEEKDAY_ALIASES:
        return WEEKDAY_ALIASES[key]
    # Try directly as enum name
    try:
        return Weekday[key.upper()]
    except KeyError:
        msg = f"Unknown weekday: '{s}'"
        raise ValueError(msg) from None


def _parse_time_window(data: dict[str, Any]) -> TimeWindow:
    """Parse a TimeWindow from a YAML dict."""
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
    """Parse a single binding from a YAML dict."""
    name = entry["name"]
    target = entry["target_agent"]
    priority = entry.get("priority", 100)

    # Parse time windows
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
# Factory functions for common binding patterns
# ============================================================================


def channel_binding(
    name: str,
    target_agent: str,
    channels: list[str],
    *,
    priority: int = 100,
) -> MessageBinding:
    """Create a channel-based binding.

    Example: All Telegram messages to the organizer.

    Args:
        name: Binding name.
        target_agent: Target agent.
        channels: List of channels.
        priority: Priority.
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
    """Create a command-based binding.

    Example: /code -> Coding agent

    Args:
        name: Binding name.
        target_agent: Target agent.
        commands: Slash commands (e.g. ["/code", "/shell"]).
        priority: Priority (default 200, higher than channel bindings).
    """
    # Normalize commands
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
    """Create a user-based binding.

    Example: Certain users always to a premium agent.

    Args:
        name: Binding name.
        target_agent: Target agent.
        user_ids: List of user IDs.
        priority: Priority.
        channels: Optional channel filter.
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
    """Create a regex-based binding.

    Example: All messages with "BU-Tarif" or "Berufsunfaehigkeit" -> insurance agent.

    Args:
        name: Binding name.
        target_agent: Target agent.
        patterns: Regex patterns.
        priority: Priority.
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
    """Create a time-based binding.

    Example: During business hours -> support agent.

    Args:
        name: Binding name.
        target_agent: Target agent.
        start: Start time (HH:MM).
        end: End time (HH:MM).
        weekdays: Weekdays (e.g. ["mo", "di", "mi", "do", "fr"]).
        priority: Priority (default 50, low as usually a fallback).
    """
    wds = [_parse_weekday(d) for d in weekdays] if weekdays else []

    return MessageBinding(
        name=name,
        target_agent=target_agent,
        time_windows=[
            TimeWindow(
                start_time=_parse_time(start),
                end_time=_parse_time(end),
                weekdays=wds,
            )
        ],
        priority=priority,
        description=f"Schedule-Binding: {start}-{end} → {target_agent}",
    )
