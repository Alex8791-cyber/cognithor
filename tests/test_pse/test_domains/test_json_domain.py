"""Tests for the JSON domain (Sprint-26.2)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.domains.json_dsl import (
    JSON_PRIMITIVE_NAMES,
    JsonCatalog,
    JsonDomain,
    JsonPrimitive,
    JsonVerifierError,
    build_json_catalog,
    register_json_domain,
)
from cognithor.channels.program_synthesis.domains.registry import DomainRegistry


class TestJsonCatalog:
    def test_builds(self) -> None:
        cat = build_json_catalog()
        assert isinstance(cat, JsonCatalog)
        assert len(cat) == len(JSON_PRIMITIVE_NAMES)

    def test_all_canonical_names_registered(self) -> None:
        cat = build_json_catalog()
        for name in JSON_PRIMITIVE_NAMES:
            assert name in cat

    def test_field_returns_value(self) -> None:
        cat = build_json_catalog()
        assert cat.get("field").fn({"x": 1}, name="x") == 1

    def test_field_missing_returns_none(self) -> None:
        cat = build_json_catalog()
        assert cat.get("field").fn({"x": 1}, name="y") is None

    def test_index_negative(self) -> None:
        cat = build_json_catalog()
        assert cat.get("index").fn([1, 2, 3], i=-1) == 3

    def test_index_out_of_range_returns_none(self) -> None:
        cat = build_json_catalog()
        assert cat.get("index").fn([1, 2, 3], i=10) is None

    def test_path_walks_mixed(self) -> None:
        cat = build_json_catalog()
        out = cat.get("path").fn(
            {"users": [{"name": "alex"}]},
            parts=("users", 0, "name"),
        )
        assert out == "alex"

    def test_length_handles_null(self) -> None:
        cat = build_json_catalog()
        assert cat.get("length_").fn(None) == 0

    def test_type_of(self) -> None:
        cat = build_json_catalog()
        assert cat.get("type_of").fn(None) == "null"
        assert cat.get("type_of").fn(True) == "bool"
        assert cat.get("type_of").fn(1) == "int"
        assert cat.get("type_of").fn("s") == "string"
        assert cat.get("type_of").fn([]) == "array"
        assert cat.get("type_of").fn({}) == "object"

    def test_to_entries_round_trip(self) -> None:
        cat = build_json_catalog()
        original = {"a": 1, "b": 2}
        entries = cat.get("to_entries").fn(original)
        rebuilt = cat.get("from_entries").fn(entries)
        assert rebuilt == original

    def test_group_by_key(self) -> None:
        cat = build_json_catalog()
        out = cat.get("group_by_key").fn(
            [{"k": "a"}, {"k": "b"}, {"k": "a"}],
            key="k",
        )
        assert set(out.keys()) == {"a", "b"}
        assert len(out["a"]) == 2

    def test_unique_by_key(self) -> None:
        cat = build_json_catalog()
        out = cat.get("unique_by_key").fn(
            [{"k": "a"}, {"k": "a"}, {"k": "b"}],
            key="k",
        )
        assert len(out) == 2

    def test_flatten_default_depth_one(self) -> None:
        cat = build_json_catalog()
        out = cat.get("flatten_").fn([[1, 2], [3, [4]]])
        assert out == [1, 2, 3, [4]]

    def test_invalid_primitive_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON primitive name"):
            JsonPrimitive(name="bad-name!", fn=lambda x: x, cost=0.1)

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            JsonPrimitive(name="p", fn=lambda x: x, cost=-1.0)


class TestJsonDomain:
    def test_metadata(self) -> None:
        d = JsonDomain()
        m = d.metadata
        assert m.name == "json"
        assert m.benchmark_target == 0.65

    def test_register(self) -> None:
        reg = DomainRegistry()
        register_json_domain(reg)
        assert isinstance(reg.get("json"), JsonDomain)

    def test_verify_simple_pipeline(self) -> None:
        d = JsonDomain()
        program = [
            {"primitive": "field", "args": {"name": "user"}},
            {"primitive": "field", "args": {"name": "name"}},
        ]
        ok = d.verify(
            program,
            [
                {"input": {"user": {"name": "alex"}}, "output": "alex"},
                {"input": {"user": {"name": "bob"}}, "output": "bob"},
            ],
        )
        assert ok

    def test_verify_mismatch_raises(self) -> None:
        d = JsonDomain()
        with pytest.raises(JsonVerifierError, match="!= expected"):
            d.verify(
                [{"primitive": "field", "args": {"name": "x"}}],
                [{"input": {"x": 1}, "output": 99}],
            )

    def test_verify_dict_program_shape(self) -> None:
        d = JsonDomain()
        program = {
            "program": [
                {"primitive": "length_", "args": {}},
            ]
        }
        ok = d.verify(
            program,
            [{"input": [1, 2, 3], "output": 3}],
        )
        assert ok

    def test_verify_bad_primitive_name(self) -> None:
        d = JsonDomain()
        with pytest.raises(JsonVerifierError, match="Unknown JSON"):
            d.verify(
                [{"primitive": "no_such_prim", "args": {}}],
                [{"input": 1, "output": 1}],
            )

    def test_verify_missing_primitive_key(self) -> None:
        d = JsonDomain()
        with pytest.raises(JsonVerifierError, match="missing 'primitive'"):
            d.verify(
                [{"args": {}}],
                [{"input": 1, "output": 1}],
            )

    def test_verify_args_not_mapping(self) -> None:
        d = JsonDomain()
        with pytest.raises(JsonVerifierError, match="must be a mapping"):
            d.verify(
                [{"primitive": "length_", "args": []}],
                [{"input": [], "output": 0}],
            )

    def test_verify_pipeline_with_typeerror(self) -> None:
        d = JsonDomain()
        with pytest.raises(JsonVerifierError, match="TypeError"):
            d.verify(
                [{"primitive": "field", "args": {"unknown_arg": "x"}}],
                [{"input": {"x": 1}, "output": 1}],
            )

    def test_program_is_neither_list_nor_dict(self) -> None:
        d = JsonDomain()
        with pytest.raises(JsonVerifierError, match="must be list"):
            d.verify("nope", [])

    def test_program_dict_missing_program_key(self) -> None:
        d = JsonDomain()
        with pytest.raises(JsonVerifierError, match="must be a list"):
            d.verify({"program": "nope"}, [])

    def test_step_not_a_mapping(self) -> None:
        d = JsonDomain()
        with pytest.raises(JsonVerifierError, match="must be a mapping"):
            d.verify(["just-a-string"], [])

    def test_at_least_20_primitives(self) -> None:
        assert len(JSON_PRIMITIVE_NAMES) >= 20

    def test_no_duplicate_primitive_names(self) -> None:
        names = JSON_PRIMITIVE_NAMES
        assert len(names) == len(set(names))
