"""
cognitio/temporal.py

Husserl's phenomenological time perception + session history.

TemporalDensityTracker:
    - Tracks the last 200 interaction timestamps
    - Computes temporal density (measures interaction frequency)
    - Detects and summarizes sleep/idle periods
    - Stamps created memories with temporal_density
    - Permanently records the last 30 sessions (when we talked, message count, duration)
    - Produces English temporal context for the LLM
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

# ─────────────────────────────────────────────
# DATE / DURATION HELPERS
# ─────────────────────────────────────────────

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _fmt_dt(dt: datetime) -> str:
    """
    Convert a UTC datetime to a natural-language English format.

    Example: 'Monday, 3 February 2025, 14:27 UTC'
    """
    day_name = _DAYS[dt.weekday()]
    month_name = _MONTHS[dt.month]
    return f"{day_name}, {dt.day} {month_name} {dt.year}, {dt.hour:02d}:{dt.minute:02d} UTC"


def _fmt_duration(td: timedelta) -> str:
    """
    Convert a timedelta to a human-readable English duration string.

    Examples: '3 minutes', '2 hours 15 minutes', '1 day 4 hours', '3 days'
    """
    total_secs = int(td.total_seconds())
    if total_secs < 0:
        return "unknown"

    days = total_secs // 86400
    hours = (total_secs % 86400) // 3600
    minutes = (total_secs % 3600) // 60

    parts: list[str] = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0 or not parts:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return " ".join(parts)


def _fmt_relative(td: timedelta) -> str:
    """
    Convert a timedelta to a relative English past-tense expression.

    Examples: 'just now', '5 minutes ago', '3 hours ago', 'yesterday',
              '4 days ago', '2 weeks ago'
    """
    total_secs = td.total_seconds()
    total_min = total_secs / 60
    total_hours = total_min / 60
    total_days = total_hours / 24

    if total_min < 2:
        return "just now"
    elif total_min < 60:
        return f"{int(total_min)} minutes ago"
    elif total_hours < 24:
        return f"{int(total_hours)} hours ago"
    elif total_days < 2:
        return "yesterday"
    elif total_days < 7:
        return f"{int(total_days)} days ago"
    elif total_days < 14:
        return "last week"
    else:
        return f"{int(total_days // 7)} weeks ago"


# ─────────────────────────────────────────────
# SESSION RECORD
# ─────────────────────────────────────────────


class SessionRecord:
    """
    Summary of a single conversation session.

    Fields:
        started_at:    Session start time (UTC)
        ended_at:      Session end time (UTC), None = still active
        message_count: Total messages in this session (user + assistant)
    """

    __slots__ = ("ended_at", "message_count", "started_at")

    def __init__(
        self,
        started_at: datetime,
        ended_at: datetime | None = None,
        message_count: int = 0,
    ) -> None:
        self.started_at = started_at
        self.ended_at = ended_at
        self.message_count = message_count

    def duration_seconds(self) -> float | None:
        """Session duration in seconds. None if still active."""
        if self.ended_at is None:
            return None
        return max(0.0, (self.ended_at - self.started_at).total_seconds())

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionRecord:
        return cls(
            started_at=datetime.fromisoformat(d["started_at"]),
            ended_at=(datetime.fromisoformat(d["ended_at"]) if d.get("ended_at") else None),
            message_count=d.get("message_count", 0),
        )


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────


class TemporalDensityTracker:
    """
    Tracker modelling Husserl's phenomenological time perception.

    Measures interaction density (number of interactions in the last 60 minutes / max).
    Detects sleep periods: gaps longer than 1 hour since last activity.
    Permanently records the last 30 sessions.

    Parameters:
        max_history: Maximum number of interaction timestamps to retain
        sleep_threshold_minutes: Minimum gap duration to count as sleep (minutes)
    """

    MAX_HISTORY = 200
    MAX_SESSION_LOG = 30
    SLEEP_THRESHOLD_MINUTES = 60  # gaps longer than 1 hour = sleep

    def __init__(self, sleep_threshold_minutes: int = 60) -> None:
        self._timestamps: deque[datetime] = deque(maxlen=self.MAX_HISTORY)
        self.last_active: datetime | None = None
        self.session_start: datetime = datetime.now(UTC)
        self.sleep_threshold_minutes = sleep_threshold_minutes
        self._sleep_reported: bool = False  # sleep summary delivered only once

        # Session history
        self._session_log: list[SessionRecord] = []
        self._current_session: SessionRecord | None = None

    # ─────────────────────────────────────────────
    # SESSION MANAGEMENT
    # ─────────────────────────────────────────────

    def start_session(self) -> None:
        """
        Start a new session.

        If a previous session is still open it is closed now (ended_at = now).
        Called by the engine after _load_state() during init.
        """
        now = datetime.now(UTC)

        # Close previous session if still open
        if self._current_session is not None and self._current_session.ended_at is None:
            self._current_session.ended_at = now
            self._session_log.append(self._current_session)
            if len(self._session_log) > self.MAX_SESSION_LOG:
                self._session_log = self._session_log[-self.MAX_SESSION_LOG :]

        # Open new session
        self._current_session = SessionRecord(started_at=now, message_count=0)
        self.session_start = now

    def _increment_session_messages(self) -> None:
        """Increment the message counter for the current session."""
        if self._current_session is not None:
            self._current_session.message_count += 1

    # ─────────────────────────────────────────────
    # INTERACTION RECORDING
    # ─────────────────────────────────────────────

    def record_interaction(self) -> None:
        """Append the current time as an interaction record and update last_active."""
        now = datetime.now(UTC)
        self._timestamps.append(now)
        self.last_active = now
        self._increment_session_messages()

    # ─────────────────────────────────────────────
    # DENSITY COMPUTATION
    # ─────────────────────────────────────────────

    def compute_density(self, window_minutes: int = 60) -> float:
        """
        Compute interaction density over the last window_minutes.

        Formula: (interactions in window) / (window duration in minutes)
        Normalized: 0.0–1.0 (2 interactions/minute = 1.0 saturation)

        Parameters:
            window_minutes: Density window (minutes)

        Returns:
            float: Normalized density (0.0–1.0)
        """
        if not self._timestamps:
            return 0.0

        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=window_minutes)

        count = sum(1 for ts in self._timestamps if ts >= cutoff)

        # Normalize: 2 interactions/minute = full saturation (1.0)
        max_interactions = window_minutes * 2
        density = min(1.0, count / max_interactions)
        return round(density, 4)

    def classify_period(self) -> str:
        """
        Classify the current period density.

        Returns:
            str: 'intense' | 'normal' | 'idle'
        """
        density = self.compute_density()
        if density >= 0.4:
            return "intense"
        elif density >= 0.1:
            return "normal"
        else:
            return "idle"

    # ─────────────────────────────────────────────
    # SLEEP DETECTION
    # ─────────────────────────────────────────────

    def get_sleep_duration(self) -> timedelta | None:
        """
        Sleep duration since last activity.

        Returns:
            timedelta: Sleep duration (if threshold exceeded)
            None: No sleep detected
        """
        if self.last_active is None:
            return None

        now = datetime.now(UTC)
        elapsed = now - self.last_active

        if elapsed.total_seconds() >= self.sleep_threshold_minutes * 60:
            return elapsed

        return None

    def get_sleep_summary(self) -> str | None:
        """
        Produce a human-readable sleep period summary.

        Reported only once (_sleep_reported flag).
        Returns None on subsequent calls.

        Returns:
            str: Sleep summary ("I was in sleep mode for X hours Y minutes")
            None: No sleep or already reported
        """
        if self._sleep_reported:
            return None

        duration = self.get_sleep_duration()
        if duration is None:
            return None

        self._sleep_reported = True
        return f"I was in sleep mode for {_fmt_duration(duration)}. Now active again."

    def reset_sleep_flag(self) -> None:
        """Reset the sleep report flag (for testing)."""
        self._sleep_reported = False

    def finalize_session(self) -> None:
        """
        Finalize the current session — set ended_at = now and append to session_log.

        Called before shutdown (chat.py exit, frontend close).
        This enables correct answers to 'when did our last conversation end?'.

        No-op if called multiple times or if _current_session is None.
        """
        if self._current_session is not None and self._current_session.ended_at is None:
            self._current_session.ended_at = datetime.now(UTC)
            self._session_log.append(self._current_session)
            if len(self._session_log) > self.MAX_SESSION_LOG:
                self._session_log = self._session_log[-self.MAX_SESSION_LOG :]
            self._current_session = None

    # ─────────────────────────────────────────────
    # LLM TEMPORAL CONTEXT
    # ─────────────────────────────────────────────

    def get_temporal_context_for_llm(self) -> str:
        """
        Produce an English temporal awareness context for the LLM.

        Added to every build_context_for_llm() call so the model can answer
        questions like "When did we last talk?", "How long have you been asleep?",
        "How long have we been talking?".

        Returns:
            str: English natural-language summary
        """
        now = datetime.now(UTC)
        parts: list[str] = []

        # Current time
        parts.append(f"Now: {_fmt_dt(now)}.")

        # Completed sessions (excluding current)
        closed = [s for s in self._session_log if s.ended_at is not None]

        if closed:
            last = closed[-1]
            ago = _fmt_relative(now - last.started_at)
            start_str = _fmt_dt(last.started_at)
            dur_secs = last.duration_seconds() or 0
            dur_str = _fmt_duration(timedelta(seconds=dur_secs))
            msg_count = last.message_count

            parts.append(
                f"Our last conversation started {ago} ({start_str}), "
                f"lasted {dur_str}, and included {msg_count} messages."
            )

            # Idle time: from end of last session to start of current session
            if self._current_session is not None:
                sleep_td = self._current_session.started_at - last.ended_at  # type: ignore[operator]
            else:
                sleep_td = now - last.ended_at  # type: ignore[operator]

            if sleep_td.total_seconds() > 60:
                parts.append(
                    f"Idle time between sessions: "
                    f"{_fmt_duration(sleep_td)} ({_fmt_relative(sleep_td)})."
                )
        else:
            parts.append("I have no record of a previous completed conversation.")

        # Current session
        if self._current_session is not None:
            session_age = now - self._current_session.started_at
            msg_count = self._current_session.message_count
            parts.append(
                f"This session started {_fmt_relative(session_age)} "
                f"({_fmt_dt(self._current_session.started_at)}), "
                f"with {msg_count} messages so far."
            )

        # Total recorded sessions
        total = len(self._session_log)
        if total > 0:
            parts.append(
                f"I have {total} completed session record{'s' if total != 1 else ''} in total."
            )

        return " ".join(parts)

    # ─────────────────────────────────────────────
    # SERIALIZATION
    # ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a dict."""
        return {
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "session_start": self.session_start.isoformat(),
            "sleep_threshold_minutes": self.sleep_threshold_minutes,
            "sleep_reported": self._sleep_reported,
            # Session history
            "session_log": [s.to_dict() for s in self._session_log],
            "current_session": (self._current_session.to_dict() if self._current_session else None),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalDensityTracker:
        """Construct a TemporalDensityTracker from a dict."""
        tracker = cls(sleep_threshold_minutes=data.get("sleep_threshold_minutes", 60))

        if data.get("last_active"):
            tracker.last_active = datetime.fromisoformat(data["last_active"])

        if data.get("session_start"):
            tracker.session_start = datetime.fromisoformat(data["session_start"])

        tracker._sleep_reported = data.get("sleep_reported", False)

        # Load session history
        tracker._session_log = [SessionRecord.from_dict(d) for d in data.get("session_log", [])]
        raw_cur = data.get("current_session")
        if raw_cur:
            tracker._current_session = SessionRecord.from_dict(raw_cur)

        return tracker
