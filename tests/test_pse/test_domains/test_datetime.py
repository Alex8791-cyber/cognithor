"""Tests for the Datetime domain (Sprint-26.3)."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from cognithor.channels.program_synthesis.domains.datetime_dsl import (
    DATETIME_PRIMITIVE_NAMES,
    DatetimeCatalog,
    DatetimeDomain,
    DatetimePrimitive,
    DatetimeVerifierError,
    build_datetime_catalog,
    register_datetime_domain,
)
from cognithor.channels.program_synthesis.domains.registry import DomainRegistry


class TestDatetimeCatalog:
    def test_builds(self) -> None:
        cat = build_datetime_catalog()
        assert isinstance(cat, DatetimeCatalog)
        assert len(cat) == len(DATETIME_PRIMITIVE_NAMES)

    def test_at_least_25_primitives(self) -> None:
        assert len(DATETIME_PRIMITIVE_NAMES) >= 25

    def test_all_canonical_names_registered(self) -> None:
        cat = build_datetime_catalog()
        for name in DATETIME_PRIMITIVE_NAMES:
            assert name in cat

    def test_invalid_primitive_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid Datetime primitive name"):
            DatetimePrimitive(name="bad-!", fn=lambda x: x, cost=0.1)

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            DatetimePrimitive(name="p", fn=lambda x: x, cost=-1.0)


class TestParsing:
    def test_parse_iso8601_naive(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("parse_iso8601").fn("2026-05-04T13:42:00")
        assert out == datetime(2026, 5, 4, 13, 42, tzinfo=UTC)

    def test_parse_iso8601_with_z(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("parse_iso8601").fn("2026-05-04T13:42:00Z")
        assert out.tzinfo == UTC

    def test_parse_format(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("parse_format").fn("04.05.2026", fmt="%d.%m.%Y")
        assert out.year == 2026 and out.month == 5

    def test_parse_epoch(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("parse_epoch").fn(0)
        assert out == datetime(1970, 1, 1, tzinfo=UTC)


class TestFormatting:
    def test_format_iso(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("format_iso").fn(datetime(2026, 5, 4, tzinfo=UTC))
        assert out.startswith("2026-05-04")

    def test_format_strftime(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("format_strftime").fn(datetime(2026, 5, 4, tzinfo=UTC), fmt="%Y/%m/%d")
        assert out == "2026/05/04"

    def test_format_human_de(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("format_human_de").fn(datetime(2026, 5, 4, 13, 42, tzinfo=UTC))
        assert "Mai 2026" in out
        assert "13:42 Uhr" in out


class TestArithmetic:
    def test_add_duration(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("add_duration").fn(datetime(2026, 5, 4, tzinfo=UTC), days=7)
        assert out == datetime(2026, 5, 11, tzinfo=UTC)

    def test_diff_seconds(self) -> None:
        cat = build_datetime_catalog()
        a = datetime(2026, 5, 4, 12, 0, 30, tzinfo=UTC)
        b = datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC)
        assert cat.get("diff_seconds").fn(a, b) == 30.0

    def test_diff_days(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("diff_days").fn(
            datetime(2026, 5, 11, tzinfo=UTC),
            datetime(2026, 5, 4, tzinfo=UTC),
        )
        assert out == 7

    def test_truncate_to_day(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("truncate_to").fn(datetime(2026, 5, 4, 13, 42, 18, tzinfo=UTC), unit="day")
        assert out == datetime(2026, 5, 4, tzinfo=UTC)

    def test_truncate_to_invalid(self) -> None:
        cat = build_datetime_catalog()
        with pytest.raises(ValueError, match="unknown unit"):
            cat.get("truncate_to").fn(datetime.now(UTC), unit="century")


class TestCompare:
    def test_is_before(self) -> None:
        cat = build_datetime_catalog()
        a = datetime(2026, 5, 4, tzinfo=UTC)
        b = datetime(2026, 5, 5, tzinfo=UTC)
        assert cat.get("is_before").fn(a, b) is True
        assert cat.get("is_after").fn(a, b) is False

    def test_same_day(self) -> None:
        cat = build_datetime_catalog()
        a = datetime(2026, 5, 4, 0, 0, tzinfo=UTC)
        b = datetime(2026, 5, 4, 23, 59, tzinfo=UTC)
        assert cat.get("same_day").fn(a, b) is True

    def test_weekday_monday_is_zero(self) -> None:
        cat = build_datetime_catalog()
        # 2026-05-04 is a Monday
        assert cat.get("weekday").fn(datetime(2026, 5, 4, tzinfo=UTC)) == 0

    def test_is_weekend(self) -> None:
        cat = build_datetime_catalog()
        # 2026-05-09 is a Saturday
        assert cat.get("is_weekend").fn(datetime(2026, 5, 9, tzinfo=UTC)) is True
        # 2026-05-04 is a Monday
        assert cat.get("is_weekend").fn(datetime(2026, 5, 4, tzinfo=UTC)) is False

    def test_business_days_between(self) -> None:
        cat = build_datetime_catalog()
        # 2026-05-04 (Mon) to 2026-05-08 (Fri) = 5 business days inclusive
        out = cat.get("business_days_between").fn(
            datetime(2026, 5, 4, tzinfo=UTC),
            datetime(2026, 5, 8, tzinfo=UTC),
        )
        assert out == 5


class TestTimezone:
    def test_to_utc(self) -> None:
        cat = build_datetime_catalog()
        berlin = datetime(2026, 5, 4, 14, 0, tzinfo=ZoneInfo("Europe/Berlin"))
        out = cat.get("to_utc").fn(berlin)
        assert out.tzinfo == UTC
        assert out.hour == 12  # CEST is UTC+2 in May

    def test_to_zone(self) -> None:
        cat = build_datetime_catalog()
        utc = datetime(2026, 5, 4, 12, 0, tzinfo=UTC)
        out = cat.get("to_zone").fn(utc, tz="Europe/Berlin")
        assert out.hour == 14

    def test_convert_zone_naive_input(self) -> None:
        cat = build_datetime_catalog()
        # naive 12:00 in Europe/Berlin → 11:00 UTC (CEST is +2 → really 10:00 UTC)
        # Actually 12:00 Europe/Berlin in May = 10:00 UTC
        naive = datetime(2026, 5, 4, 12, 0)
        out = cat.get("convert_zone").fn(naive, from_tz="Europe/Berlin", to_tz="UTC")
        assert out.hour == 10


class TestCalendar:
    def test_start_of_month(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("start_of_month").fn(datetime(2026, 5, 23, 13, 42, tzinfo=UTC))
        assert out == datetime(2026, 5, 1, tzinfo=UTC)

    def test_end_of_month(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("end_of_month").fn(datetime(2026, 5, 4, tzinfo=UTC))
        assert out.day == 31
        assert out.hour == 23 and out.minute == 59

    def test_end_of_february_non_leap(self) -> None:
        cat = build_datetime_catalog()
        out = cat.get("end_of_month").fn(datetime(2026, 2, 4, tzinfo=UTC))
        assert out.day == 28

    def test_next_business_day_from_friday(self) -> None:
        cat = build_datetime_catalog()
        # 2026-05-08 is a Friday; next business day is Monday 11
        out = cat.get("next_business_day").fn(datetime(2026, 5, 8, tzinfo=UTC))
        assert out.weekday() == 0  # Monday
        assert out.day == 11

    def test_last_weekday(self) -> None:
        cat = build_datetime_catalog()
        # From Monday 2026-05-04, last Friday = 2026-05-01
        out = cat.get("last_weekday").fn(datetime(2026, 5, 4, tzinfo=UTC), name="friday")
        assert out == datetime(2026, 5, 1, tzinfo=UTC)

    def test_last_weekday_invalid(self) -> None:
        cat = build_datetime_catalog()
        with pytest.raises(ValueError, match="unknown weekday"):
            cat.get("last_weekday").fn(datetime.now(UTC), name="funday")


class TestDatetimeDomain:
    def test_metadata(self) -> None:
        d = DatetimeDomain()
        m = d.metadata
        assert m.name == "datetime"
        assert m.benchmark_name == "tempeval-3"
        assert m.benchmark_target == 0.80

    def test_register(self) -> None:
        reg = DomainRegistry()
        register_datetime_domain(reg)
        assert isinstance(reg.get("datetime"), DatetimeDomain)

    def test_verify_pipeline(self) -> None:
        d = DatetimeDomain()
        program = [
            {"primitive": "parse_iso8601", "args": {}},
            {"primitive": "add_duration", "args": {"days": 7}},
            {"primitive": "format_iso", "args": {}},
        ]
        ok = d.verify(
            program,
            [
                {
                    "input": "2026-05-04T00:00:00Z",
                    "output": "2026-05-11T00:00:00+00:00",
                }
            ],
        )
        assert ok

    def test_verify_dict_program(self) -> None:
        d = DatetimeDomain()
        ok = d.verify(
            {"program": [{"primitive": "weekday", "args": {}}]},
            [
                {
                    "input": datetime(2026, 5, 4, tzinfo=UTC),
                    "output": 0,
                }
            ],
        )
        assert ok

    def test_verify_mismatch_raises(self) -> None:
        d = DatetimeDomain()
        with pytest.raises(DatetimeVerifierError, match="!= expected"):
            d.verify(
                [{"primitive": "weekday", "args": {}}],
                [{"input": datetime(2026, 5, 4, tzinfo=UTC), "output": 99}],
            )

    def test_verify_value_error_caught(self) -> None:
        d = DatetimeDomain()
        with pytest.raises(DatetimeVerifierError, match="ValueError"):
            d.verify(
                [{"primitive": "truncate_to", "args": {"unit": "century"}}],
                [{"input": datetime.now(UTC), "output": "x"}],
            )

    def test_verify_unknown_primitive_raises(self) -> None:
        d = DatetimeDomain()
        with pytest.raises(DatetimeVerifierError, match="Unknown Datetime"):
            d.verify(
                [{"primitive": "unknown_x", "args": {}}],
                [{"input": "x", "output": "x"}],
            )

    def test_program_must_be_list_or_dict(self) -> None:
        d = DatetimeDomain()
        with pytest.raises(DatetimeVerifierError, match="must be"):
            d.verify("not-a-list", [])

    def test_step_must_be_mapping(self) -> None:
        d = DatetimeDomain()
        with pytest.raises(DatetimeVerifierError, match="must be a mapping"):
            d.verify(["not-a-step"], [])

    def test_step_missing_primitive(self) -> None:
        d = DatetimeDomain()
        with pytest.raises(DatetimeVerifierError, match="missing 'primitive'"):
            d.verify([{"args": {}}], [])
