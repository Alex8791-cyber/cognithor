"""Datetime primitive catalog (Sprint-26.3).

25 primitives spanning parse/format/arithmetic/compare/timezone/
calendar. All operations are tz-aware: naive inputs default to UTC,
DST-correct calendar-arithmetic is preferred over naive duration
arithmetic where applicable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Catalog entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatetimePrimitive:
    """One Datetime transformer primitive."""

    name: str
    fn: Callable[..., Any]
    cost: float
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            msg = f"Invalid Datetime primitive name: {self.name!r}"
            raise ValueError(msg)
        if self.cost < 0:
            msg = f"Datetime primitive cost must be >= 0, got {self.cost}"
            raise ValueError(msg)


class DatetimeCatalog:
    """Append-only catalog of :class:`DatetimePrimitive` entries."""

    def __init__(self) -> None:
        self._entries: dict[str, DatetimePrimitive] = {}

    def add(self, primitive: DatetimePrimitive) -> None:
        if primitive.name in self._entries:
            msg = f"Datetime primitive {primitive.name!r} already registered"
            raise ValueError(msg)
        self._entries[primitive.name] = primitive

    def get(self, name: str) -> DatetimePrimitive:
        if name not in self._entries:
            msg = f"Unknown Datetime primitive {name!r}"
            raise KeyError(msg)
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


DATETIME_PRIMITIVE_NAMES: tuple[str, ...] = (
    # Parse
    "parse_iso8601",
    "parse_format",
    "parse_epoch",
    # Format
    "format_iso",
    "format_strftime",
    "format_human_de",
    # Arithmetic
    "add_duration",
    "sub_duration",
    "diff_seconds",
    "diff_days",
    "truncate_to",
    # Compare
    "is_before",
    "is_after",
    "same_day",
    "weekday",
    "is_weekend",
    "business_days_between",
    # Timezone
    "to_utc",
    "to_zone",
    "convert_zone",
    # Calendar
    "start_of_day",
    "start_of_month",
    "end_of_month",
    "next_business_day",
    "last_weekday",
)


# ---------------------------------------------------------------------------
# Primitive implementations
# ---------------------------------------------------------------------------


def _ensure_tz_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _parse_iso8601(value: str) -> datetime:
    cleaned = value.strip().replace("Z", "+00:00")
    return _ensure_tz_aware(datetime.fromisoformat(cleaned))


def _parse_format(value: str, fmt: str) -> datetime:
    return _ensure_tz_aware(datetime.strptime(value, fmt))


def _parse_epoch(value: float) -> datetime:
    return datetime.fromtimestamp(float(value), tz=UTC)


def _format_iso(dt: datetime) -> str:
    return _ensure_tz_aware(dt).isoformat()


def _format_strftime(dt: datetime, fmt: str) -> str:
    return _ensure_tz_aware(dt).strftime(fmt)


_DE_DAYS = (
    "Montag",
    "Dienstag",
    "Mittwoch",
    "Donnerstag",
    "Freitag",
    "Samstag",
    "Sonntag",
)
_DE_MONTHS = (
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
)


def _format_human_de(dt: datetime) -> str:
    """German human-friendly format: 'Montag, 04. Mai 2026, 13:42 Uhr'."""
    aware = _ensure_tz_aware(dt)
    return (
        f"{_DE_DAYS[aware.weekday()]}, "
        f"{aware.day:02d}. {_DE_MONTHS[aware.month - 1]} {aware.year}, "
        f"{aware.hour:02d}:{aware.minute:02d} Uhr"
    )


def _add_duration(
    dt: datetime,
    *,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
) -> datetime:
    return _ensure_tz_aware(dt) + timedelta(
        days=days, hours=hours, minutes=minutes, seconds=seconds
    )


def _sub_duration(
    dt: datetime,
    *,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
) -> datetime:
    return _ensure_tz_aware(dt) - timedelta(
        days=days, hours=hours, minutes=minutes, seconds=seconds
    )


def _diff_seconds(a: datetime, b: datetime) -> float:
    return (_ensure_tz_aware(a) - _ensure_tz_aware(b)).total_seconds()


def _diff_days(a: datetime, b: datetime) -> int:
    return (_ensure_tz_aware(a).date() - _ensure_tz_aware(b).date()).days


_TRUNC_UNITS = {"second", "minute", "hour", "day", "month", "year"}


def _truncate_to(dt: datetime, unit: str) -> datetime:
    if unit not in _TRUNC_UNITS:
        msg = f"truncate_to: unknown unit {unit!r}, expected one of {sorted(_TRUNC_UNITS)}"
        raise ValueError(msg)
    aware = _ensure_tz_aware(dt)
    if unit == "second":
        return aware.replace(microsecond=0)
    if unit == "minute":
        return aware.replace(second=0, microsecond=0)
    if unit == "hour":
        return aware.replace(minute=0, second=0, microsecond=0)
    if unit == "day":
        return aware.replace(hour=0, minute=0, second=0, microsecond=0)
    if unit == "month":
        return aware.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return aware.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _is_before(a: datetime, b: datetime) -> bool:
    return _ensure_tz_aware(a) < _ensure_tz_aware(b)


def _is_after(a: datetime, b: datetime) -> bool:
    return _ensure_tz_aware(a) > _ensure_tz_aware(b)


def _same_day(a: datetime, b: datetime) -> bool:
    return _ensure_tz_aware(a).date() == _ensure_tz_aware(b).date()


def _weekday(dt: datetime) -> int:
    """Return weekday number — Monday=0 ... Sunday=6."""
    return _ensure_tz_aware(dt).weekday()


def _is_weekend(dt: datetime) -> bool:
    return _ensure_tz_aware(dt).weekday() >= 5


def _business_days_between(a: datetime, b: datetime) -> int:
    """Inclusive business-day count between two dates (a <= b)."""
    lo = _ensure_tz_aware(a).date()
    hi = _ensure_tz_aware(b).date()
    if lo > hi:
        lo, hi = hi, lo
    count = 0
    cur = lo
    while cur <= hi:
        if cur.weekday() < 5:
            count += 1
        cur = cur + timedelta(days=1)
    return count


def _to_utc(dt: datetime) -> datetime:
    return _ensure_tz_aware(dt).astimezone(UTC)


def _to_zone(dt: datetime, tz: str) -> datetime:
    return _ensure_tz_aware(dt).astimezone(ZoneInfo(tz))


def _convert_zone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    """Re-interpret a naive datetime in ``from_tz`` then convert to ``to_tz``."""
    if dt.tzinfo is None:
        anchored = dt.replace(tzinfo=ZoneInfo(from_tz))
    else:
        anchored = dt.astimezone(ZoneInfo(from_tz))
    return anchored.astimezone(ZoneInfo(to_tz))


def _start_of_day(dt: datetime) -> datetime:
    return _truncate_to(dt, "day")


def _start_of_month(dt: datetime) -> datetime:
    return _truncate_to(dt, "month")


def _end_of_month(dt: datetime) -> datetime:
    aware = _ensure_tz_aware(dt)
    if aware.month == 12:
        next_month_first = aware.replace(year=aware.year + 1, month=1, day=1)
    else:
        next_month_first = aware.replace(month=aware.month + 1, day=1)
    last_day = (next_month_first - timedelta(days=1)).day
    return aware.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999999)


def _next_business_day(dt: datetime) -> datetime:
    aware = _start_of_day(dt) + timedelta(days=1)
    while aware.weekday() >= 5:
        aware = aware + timedelta(days=1)
    return aware


_WEEKDAY_NAMES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _last_weekday(dt: datetime, name: str) -> datetime:
    """Most recent occurrence of ``name`` weekday at or before ``dt``."""
    target = _WEEKDAY_NAMES.get(name.lower())
    if target is None:
        msg = f"last_weekday: unknown weekday {name!r}"
        raise ValueError(msg)
    aware = _start_of_day(dt)
    delta = (aware.weekday() - target) % 7
    return aware - timedelta(days=delta)


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------


def build_datetime_catalog() -> DatetimeCatalog:
    cat = DatetimeCatalog()

    def add(name: str, fn: Callable[..., Any], cost: float, desc: str) -> None:
        cat.add(DatetimePrimitive(name=name, fn=fn, cost=cost, description=desc))

    add("parse_iso8601", _parse_iso8601, 0.2, "Parse ISO-8601 string (UTC default)")
    add("parse_format", _parse_format, 0.3, "Parse with strptime format spec")
    add("parse_epoch", _parse_epoch, 0.2, "Parse Unix epoch seconds (float)")
    add("format_iso", _format_iso, 0.1, "Format as ISO-8601 string")
    add("format_strftime", _format_strftime, 0.2, "Format with strftime spec")
    add("format_human_de", _format_human_de, 0.3, "German 'Montag, 04. Mai 2026' format")
    add("add_duration", _add_duration, 0.3, "Add days/hours/minutes/seconds")
    add("sub_duration", _sub_duration, 0.3, "Subtract days/hours/minutes/seconds")
    add("diff_seconds", _diff_seconds, 0.2, "a - b in seconds (float)")
    add("diff_days", _diff_days, 0.2, "a.date - b.date in days")
    add("truncate_to", _truncate_to, 0.3, "Truncate to second/minute/hour/day/month/year")
    add("is_before", _is_before, 0.1, "a < b")
    add("is_after", _is_after, 0.1, "a > b")
    add("same_day", _same_day, 0.2, "a.date == b.date")
    add("weekday", _weekday, 0.1, "Monday=0 ... Sunday=6")
    add("is_weekend", _is_weekend, 0.2, "Sat/Sun → True")
    add("business_days_between", _business_days_between, 0.4, "Inclusive Mon-Fri count")
    add("to_utc", _to_utc, 0.2, "Convert to UTC")
    add("to_zone", _to_zone, 0.3, "Convert to named timezone")
    add("convert_zone", _convert_zone, 0.4, "Re-anchor naive datetime then convert")
    add("start_of_day", _start_of_day, 0.2, "Truncate to day")
    add("start_of_month", _start_of_month, 0.2, "Truncate to month")
    add("end_of_month", _end_of_month, 0.3, "Last microsecond of the month")
    add("next_business_day", _next_business_day, 0.4, "Next Mon-Fri after dt")
    add("last_weekday", _last_weekday, 0.4, "Last occurrence of named weekday at/before dt")

    return cat
