"""
tests/test_identity/cognitio/test_temporal.py

Pure-unit tests for cognithor.identity.cognitio.temporal.
Covers helper functions (_fmt_dt, _fmt_duration, _fmt_relative),
SessionRecord, and TemporalDensityTracker — including density computation,
sleep detection, session management, serialization, and the LLM context string.
No external services or freezegun needed: timestamps are injected via the
public API or by directly setting internal state.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cognithor.identity.cognitio.temporal import (
    SessionRecord,
    TemporalDensityTracker,
    _fmt_dt,
    _fmt_duration,
    _fmt_relative,
)

# ---------------------------------------------------------------------------
# _fmt_dt
# ---------------------------------------------------------------------------


class TestFmtDt:
    def test_monday_3_february_2025(self):
        dt = datetime(2025, 2, 3, 14, 27, tzinfo=UTC)
        assert _fmt_dt(dt) == "Monday, 3 February 2025, 14:27 UTC"

    def test_saturday_1_january_2000(self):
        dt = datetime(2000, 1, 1, 0, 0, tzinfo=UTC)
        assert _fmt_dt(dt) == "Saturday, 1 January 2000, 00:00 UTC"

    def test_hour_minute_zero_padded(self):
        dt = datetime(2024, 6, 5, 9, 3, tzinfo=UTC)
        result = _fmt_dt(dt)
        assert result.endswith("09:03 UTC")


# ---------------------------------------------------------------------------
# _fmt_duration
# ---------------------------------------------------------------------------


class TestFmtDuration:
    @pytest.mark.parametrize(
        "td, expected",
        [
            (timedelta(seconds=0), "0 minutes"),
            (timedelta(seconds=59), "0 minutes"),
            (timedelta(minutes=1), "1 minute"),
            (timedelta(minutes=3), "3 minutes"),
            (timedelta(hours=2, minutes=15), "2 hours 15 minutes"),
            (timedelta(hours=1), "1 hour"),
            (timedelta(days=1, hours=4), "1 day 4 hours"),
            (timedelta(days=3), "3 days"),
            (timedelta(days=1, hours=0, minutes=0), "1 day"),
        ],
    )
    def test_parametrized(self, td, expected):
        assert _fmt_duration(td) == expected

    def test_negative_returns_unknown(self):
        assert _fmt_duration(timedelta(seconds=-1)) == "unknown"


# ---------------------------------------------------------------------------
# _fmt_relative
# ---------------------------------------------------------------------------


class TestFmtRelative:
    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (30, "just now"),
            (119, "just now"),  # < 2 minutes
            (120, "2 minutes ago"),
            (300, "5 minutes ago"),
            (3599, "59 minutes ago"),
            (3600, "1 hours ago"),
            (7200, "2 hours ago"),
            (86399, "23 hours ago"),
            (86400, "yesterday"),  # 24 h exactly
            (86400 * 1.5, "yesterday"),
            (86400 * 2, "2 days ago"),
            (86400 * 6, "6 days ago"),
            (86400 * 7, "last week"),
            (86400 * 10, "last week"),
            (86400 * 14, "2 weeks ago"),
            (86400 * 21, "3 weeks ago"),
        ],
    )
    def test_parametrized(self, seconds, expected):
        td = timedelta(seconds=seconds)
        assert _fmt_relative(td) == expected


# ---------------------------------------------------------------------------
# SessionRecord
# ---------------------------------------------------------------------------


class TestSessionRecord:
    def test_duration_none_when_open(self):
        s = SessionRecord(started_at=datetime(2025, 1, 1, 10, 0, tzinfo=UTC))
        assert s.duration_seconds() is None

    def test_duration_closed(self):
        s = SessionRecord(
            started_at=datetime(2025, 1, 1, 10, 0, tzinfo=UTC),
            ended_at=datetime(2025, 1, 1, 10, 30, tzinfo=UTC),
        )
        assert s.duration_seconds() == 1800.0

    def test_duration_never_negative(self):
        s = SessionRecord(
            started_at=datetime(2025, 1, 1, 10, 0, tzinfo=UTC),
            ended_at=datetime(2025, 1, 1, 9, 59, tzinfo=UTC),  # ended before started
        )
        assert s.duration_seconds() == 0.0

    def test_round_trip_to_from_dict(self):
        s = SessionRecord(
            started_at=datetime(2025, 3, 15, 8, 30, tzinfo=UTC),
            ended_at=datetime(2025, 3, 15, 9, 0, tzinfo=UTC),
            message_count=12,
        )
        d = s.to_dict()
        s2 = SessionRecord.from_dict(d)
        assert s2.started_at == s.started_at
        assert s2.ended_at == s.ended_at
        assert s2.message_count == s.message_count

    def test_from_dict_no_ended_at(self):
        d = {"started_at": "2025-01-01T10:00:00+00:00", "ended_at": None, "message_count": 0}
        s = SessionRecord.from_dict(d)
        assert s.ended_at is None


# ---------------------------------------------------------------------------
# TemporalDensityTracker — density
# ---------------------------------------------------------------------------


class TestTemporalDensityEmpty:
    def test_zero_when_no_interactions(self):
        t = TemporalDensityTracker()
        assert t.compute_density() == 0.0


class TestTemporalDensityComputation:
    def _inject_timestamps(self, tracker: TemporalDensityTracker, count: int) -> None:
        """Inject recent timestamps directly into the internal deque."""
        now = datetime.now(UTC)
        for _ in range(count):
            tracker._timestamps.append(now - timedelta(seconds=1))

    def test_density_saturates_at_one(self):
        t = TemporalDensityTracker()
        # 60 min window * 2 interactions/min = 120 interactions needed for saturation
        self._inject_timestamps(t, 120)
        assert t.compute_density(window_minutes=60) == 1.0

    def test_density_proportional(self):
        t = TemporalDensityTracker()
        self._inject_timestamps(t, 60)  # half of 120
        density = t.compute_density(window_minutes=60)
        assert abs(density - 0.5) < 1e-4

    def test_density_never_exceeds_one(self):
        t = TemporalDensityTracker()
        self._inject_timestamps(t, 500)
        assert t.compute_density() <= 1.0

    def test_density_rounded_to_four_decimal_places(self):
        t = TemporalDensityTracker()
        self._inject_timestamps(t, 1)
        d = t.compute_density(window_minutes=60)
        assert d == round(d, 4)


# ---------------------------------------------------------------------------
# TemporalDensityTracker — classify_period
# ---------------------------------------------------------------------------


class TestClassifyPeriod:
    def _tracker_with_density(self, density_target: float) -> TemporalDensityTracker:
        """Build a tracker whose compute_density() returns density_target."""
        t = TemporalDensityTracker()
        # Inject n interactions into a 60-min window: density = n / 120
        n = int(density_target * 120)
        for _ in range(n):
            t._timestamps.append(datetime.now(UTC) - timedelta(seconds=5))
        return t

    def test_intense_when_density_ge_04(self):
        t = self._tracker_with_density(0.5)
        assert t.classify_period() == "intense"

    def test_normal_when_density_between_01_and_04(self):
        t = self._tracker_with_density(0.15)
        assert t.classify_period() == "normal"

    def test_idle_when_density_below_01(self):
        t = TemporalDensityTracker()  # empty → density 0.0
        assert t.classify_period() == "idle"


# ---------------------------------------------------------------------------
# TemporalDensityTracker — sleep detection
# ---------------------------------------------------------------------------


class TestSleepDetection:
    def test_no_last_active_returns_none(self):
        t = TemporalDensityTracker(sleep_threshold_minutes=60)
        assert t.get_sleep_duration() is None

    def test_recent_activity_no_sleep(self):
        t = TemporalDensityTracker(sleep_threshold_minutes=60)
        t.last_active = datetime.now(UTC) - timedelta(minutes=30)
        assert t.get_sleep_duration() is None

    def test_old_activity_detected_as_sleep(self):
        t = TemporalDensityTracker(sleep_threshold_minutes=60)
        t.last_active = datetime.now(UTC) - timedelta(hours=3)
        result = t.get_sleep_duration()
        assert result is not None
        assert result.total_seconds() >= 3 * 3600

    def test_sleep_summary_reported_once(self):
        t = TemporalDensityTracker(sleep_threshold_minutes=60)
        t.last_active = datetime.now(UTC) - timedelta(hours=2)
        first = t.get_sleep_summary()
        second = t.get_sleep_summary()
        assert first is not None
        assert second is None

    def test_sleep_summary_contains_duration(self):
        t = TemporalDensityTracker(sleep_threshold_minutes=60)
        t.last_active = datetime.now(UTC) - timedelta(hours=2)
        summary = t.get_sleep_summary()
        assert "sleep mode" in summary
        assert "hour" in summary

    def test_reset_sleep_flag_allows_re_report(self):
        t = TemporalDensityTracker(sleep_threshold_minutes=60)
        t.last_active = datetime.now(UTC) - timedelta(hours=2)
        t.get_sleep_summary()
        t.reset_sleep_flag()
        assert t.get_sleep_summary() is not None


# ---------------------------------------------------------------------------
# TemporalDensityTracker — session management
# ---------------------------------------------------------------------------


class TestSessionManagement:
    def test_start_session_opens_current_session(self):
        t = TemporalDensityTracker()
        t.start_session()
        assert t._current_session is not None
        assert t._current_session.ended_at is None

    def test_start_session_closes_previous(self):
        t = TemporalDensityTracker()
        t.start_session()
        t.start_session()  # second call should close the first
        assert len(t._session_log) == 1
        assert t._session_log[0].ended_at is not None

    def test_finalize_session_closes_current(self):
        t = TemporalDensityTracker()
        t.start_session()
        t.finalize_session()
        assert t._current_session is None
        assert len(t._session_log) == 1
        assert t._session_log[0].ended_at is not None

    def test_finalize_session_idempotent(self):
        t = TemporalDensityTracker()
        t.start_session()
        t.finalize_session()
        t.finalize_session()  # no-op
        assert len(t._session_log) == 1

    def test_record_interaction_increments_message_count(self):
        t = TemporalDensityTracker()
        t.start_session()
        t.record_interaction()
        t.record_interaction()
        assert t._current_session.message_count == 2

    def test_session_log_capped_at_max(self):
        t = TemporalDensityTracker()
        for _ in range(TemporalDensityTracker.MAX_SESSION_LOG + 5):
            t.start_session()
        # +1 final session still open, log holds at most MAX_SESSION_LOG
        assert len(t._session_log) <= TemporalDensityTracker.MAX_SESSION_LOG


# ---------------------------------------------------------------------------
# TemporalDensityTracker — serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip_empty(self):
        t = TemporalDensityTracker()
        d = t.to_dict()
        t2 = TemporalDensityTracker.from_dict(d)
        assert t2.sleep_threshold_minutes == t.sleep_threshold_minutes
        assert t2.last_active == t.last_active

    def test_round_trip_with_session(self):
        t = TemporalDensityTracker()
        t.start_session()
        t.record_interaction()
        t.finalize_session()
        d = t.to_dict()
        t2 = TemporalDensityTracker.from_dict(d)
        assert len(t2._session_log) == 1
        assert t2._session_log[0].message_count == 1

    def test_sleep_reported_flag_preserved(self):
        t = TemporalDensityTracker()
        t._sleep_reported = True
        d = t.to_dict()
        t2 = TemporalDensityTracker.from_dict(d)
        assert t2._sleep_reported is True

    def test_last_active_preserved(self):
        t = TemporalDensityTracker()
        ts = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
        t.last_active = ts
        d = t.to_dict()
        t2 = TemporalDensityTracker.from_dict(d)
        assert t2.last_active == ts


# ---------------------------------------------------------------------------
# TemporalDensityTracker — get_temporal_context_for_llm
# ---------------------------------------------------------------------------


class TestTemporalContextForLlm:
    def test_contains_now_prefix(self):
        t = TemporalDensityTracker()
        ctx = t.get_temporal_context_for_llm()
        assert ctx.startswith("Now:")

    def test_no_prior_session_message(self):
        t = TemporalDensityTracker()
        ctx = t.get_temporal_context_for_llm()
        assert "no record" in ctx

    def test_with_closed_session_mentions_last_conversation(self):
        t = TemporalDensityTracker()
        past = datetime.now(UTC) - timedelta(hours=3)
        sr = SessionRecord(
            started_at=past,
            ended_at=past + timedelta(minutes=30),
            message_count=5,
        )
        t._session_log = [sr]
        ctx = t.get_temporal_context_for_llm()
        assert "last conversation" in ctx
        assert "5 messages" in ctx

    def test_current_session_info_included(self):
        t = TemporalDensityTracker()
        t.start_session()
        t.record_interaction()
        ctx = t.get_temporal_context_for_llm()
        assert "This session" in ctx
        assert "1 messages" in ctx
