"""10 cross-domain demo cases for Sprint-26.4.

Each case mixes at least two domain bridges from the Sprint-26
whitelist (e.g. ``json → datetime``, ``datetime → sql_literal``).
The Public Scorecard reports the **fraction solved** as the
``cross-domain`` benchmark line.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrossDomainCase:
    """One cross-domain demo task.

    Attributes
    ----------
    case_id:
        Stable identifier (used by scorecard + reproduction commands).
    title:
        Human-readable case description.
    bridges_used:
        Tuple of (from_type, to_type) pairs from
        ``SPRINT26_BRIDGE_WHITELIST`` that the canonical solution
        exercises.
    examples:
        List of ``{input, output}`` pairs the synthesiser must reproduce.
    expected_pipeline:
        Canonical solution as a list of pipeline steps. The synthesiser
        is *not* required to emit exactly this — any pipeline that
        reproduces every example counts as a solve.
    """

    case_id: str
    title: str
    bridges_used: tuple[tuple[str, str], ...]
    examples: tuple[dict[str, object], ...]
    expected_pipeline: tuple[dict[str, object], ...]


_CASES: tuple[CrossDomainCase, ...] = (
    CrossDomainCase(
        case_id="json_field_to_string",
        title="Extract a string field from a JSON object",
        bridges_used=(("json", "string"),),
        examples=(
            {"input": {"user": "alex"}, "output": "alex"},
            {"input": {"user": "bob"}, "output": "bob"},
        ),
        expected_pipeline=({"primitive": "field", "args": {"name": "user"}},),
    ),
    CrossDomainCase(
        case_id="json_path_to_string",
        title="Walk into a JSON object via path",
        bridges_used=(("json", "string"),),
        examples=(
            {
                "input": {"users": [{"name": "alex"}]},
                "output": "alex",
            },
        ),
        expected_pipeline=({"primitive": "path", "args": {"parts": ("users", 0, "name")}},),
    ),
    CrossDomainCase(
        case_id="json_count_array",
        title="Count items in a JSON array",
        bridges_used=(("json", "number"),),
        examples=(
            {"input": [1, 2, 3, 4], "output": 4},
            {"input": [], "output": 0},
        ),
        expected_pipeline=({"primitive": "length_", "args": {}},),
    ),
    CrossDomainCase(
        case_id="iso_date_extract_year",
        title="Parse ISO-8601 string and read the year",
        bridges_used=(("string", "datetime"),),
        examples=(
            {"input": "2026-05-04T12:00:00Z", "output": 2026},
            {"input": "1999-12-31T23:59:59Z", "output": 1999},
        ),
        expected_pipeline=(
            {"primitive": "parse_iso8601", "args": {}},
            {"primitive": "format_strftime", "args": {"fmt": "%Y"}},
        ),
    ),
    CrossDomainCase(
        case_id="datetime_add_business_days",
        title="Add 1 business day to an ISO-8601 datetime",
        bridges_used=(("string", "datetime"),),
        examples=(
            {
                "input": "2026-05-08T00:00:00Z",
                "output": "2026-05-11T00:00:00+00:00",
            },
        ),
        expected_pipeline=(
            {"primitive": "parse_iso8601", "args": {}},
            {"primitive": "next_business_day", "args": {}},
            {"primitive": "format_iso", "args": {}},
        ),
    ),
    CrossDomainCase(
        case_id="json_to_sql_literal",
        title="Build a SQL literal from a JSON-supplied number",
        bridges_used=(("json", "number"), ("number", "sql_literal")),
        examples=(
            {"input": {"amount": 42}, "output": "42"},
            {"input": {"amount": 3.14}, "output": "3.14"},
        ),
        expected_pipeline=({"primitive": "field", "args": {"name": "amount"}},),
    ),
    CrossDomainCase(
        case_id="bytes_b64_to_string",
        title="Decode base64 bytes",
        bridges_used=(("bytes", "string"),),
        examples=({"input": "aGVsbG8=", "output": "hello"},),
        expected_pipeline=({"primitive": "decode_base64", "args": {}},),
    ),
    CrossDomainCase(
        case_id="datetime_human_de_format",
        title="Format an ISO-8601 datetime as German human string",
        bridges_used=(("string", "datetime"), ("datetime", "string")),
        examples=(
            {
                "input": "2026-05-04T13:42:00Z",
                "output_contains": "Mai 2026",
            },
        ),
        expected_pipeline=(
            {"primitive": "parse_iso8601", "args": {}},
            {"primitive": "format_human_de", "args": {}},
        ),
    ),
    CrossDomainCase(
        case_id="json_group_by",
        title="Group a JSON array of records by a key",
        bridges_used=(),
        examples=(
            {
                "input": [{"k": "a"}, {"k": "b"}, {"k": "a"}],
                "output_keys": ["a", "b"],
            },
        ),
        expected_pipeline=({"primitive": "group_by_key", "args": {"key": "k"}},),
    ),
    CrossDomainCase(
        case_id="grid_mirror_h",
        title="Horizontal mirror an ARC-style 2D grid",
        bridges_used=(),
        examples=(
            {
                "input": ((1, 0), (0, 1)),
                "output": ((0, 1), (1, 0)),
            },
        ),
        expected_pipeline=({"primitive": "mirror_h", "args": {}},),
    ),
)


def load_cross_domain_cases() -> tuple[CrossDomainCase, ...]:
    """Return the canonical 10-case cross-domain demo set."""
    return _CASES
