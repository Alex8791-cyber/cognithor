"""JSON primitive catalog (Sprint-26.2).

Each primitive is a pure transformer ``(JsonValue) -> JsonValue``.
Composition happens through chaining; the verifier runs the final
pipeline against the example inputs and compares the outputs.

Sprint-26.2 ships ~20 primitives covering the path-extraction and
basic transformation surface. The full jq-cookbook subset (set ops,
recursion, aggregation across arrays) lands incrementally in
Sprint-26.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

JsonValue = Any  # JSON-serialisable Python object


# ---------------------------------------------------------------------------
# Catalog entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JsonPrimitive:
    """One JSON transformer primitive."""

    name: str
    fn: Callable[..., JsonValue]
    cost: float
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            msg = f"Invalid JSON primitive name: {self.name!r}"
            raise ValueError(msg)
        if self.cost < 0:
            msg = f"JSON primitive cost must be >= 0, got {self.cost}"
            raise ValueError(msg)


class JsonCatalog:
    """Append-only catalog of :class:`JsonPrimitive` entries."""

    def __init__(self) -> None:
        self._entries: dict[str, JsonPrimitive] = {}

    def add(self, primitive: JsonPrimitive) -> None:
        if primitive.name in self._entries:
            msg = f"JSON primitive {primitive.name!r} already registered"
            raise ValueError(msg)
        self._entries[primitive.name] = primitive

    def get(self, name: str) -> JsonPrimitive:
        if name not in self._entries:
            msg = f"Unknown JSON primitive {name!r}"
            raise KeyError(msg)
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


# ---------------------------------------------------------------------------
# Primitive name list
# ---------------------------------------------------------------------------

JSON_PRIMITIVE_NAMES: tuple[str, ...] = (
    "field",
    "index",
    "path",
    "select_where",
    "has_key",
    "contains_value",
    "length_",
    "type_of",
    "map_",
    "filter_",
    "to_entries",
    "from_entries",
    "group_by_key",
    "sort_by_key",
    "merge_",
    "flatten_",
    "unique_by_key",
    "make_object",
    "make_array",
    "if_then_else",
)


# ---------------------------------------------------------------------------
# Primitive implementations
# ---------------------------------------------------------------------------


def _field(value: JsonValue, name: str) -> JsonValue:
    if not isinstance(value, dict):
        return None
    return value.get(name)


def _index(value: JsonValue, i: int) -> JsonValue:
    if not isinstance(value, list):
        return None
    if -len(value) <= i < len(value):
        return value[i]
    return None


def _path(value: JsonValue, parts: tuple[Any, ...]) -> JsonValue:
    cur: Any = value
    for part in parts:
        if isinstance(part, str):
            cur = _field(cur, part)
        elif isinstance(part, int):
            cur = _index(cur, part)
        else:
            return None
        if cur is None:
            return None
    return cur


def _select_where(value: JsonValue, predicate: Callable[[JsonValue], bool]) -> JsonValue:
    if not isinstance(value, list):
        return value if predicate(value) else None
    return [v for v in value if predicate(v)]


def _has_key(value: JsonValue, name: str) -> bool:
    return isinstance(value, dict) and name in value


def _contains_value(value: JsonValue, needle: Any) -> bool:
    if isinstance(value, list):
        return needle in value
    if isinstance(value, str):
        return isinstance(needle, str) and needle in value
    if isinstance(value, dict):
        return needle in value.values()
    return False


def _length(value: JsonValue) -> int:
    if value is None:
        return 0
    if isinstance(value, str | list | dict):
        return len(value)
    return 1


def _type_of(value: JsonValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _map_(value: JsonValue, fn: Callable[[JsonValue], JsonValue]) -> list[JsonValue]:
    if not isinstance(value, list):
        return []
    return [fn(v) for v in value]


def _filter_(value: JsonValue, predicate: Callable[[JsonValue], bool]) -> list[JsonValue]:
    if not isinstance(value, list):
        return []
    return [v for v in value if predicate(v)]


def _to_entries(value: JsonValue) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    return [{"key": k, "value": v} for k, v in value.items()]


def _from_entries(value: JsonValue) -> dict[str, Any]:
    if not isinstance(value, list):
        return {}
    out: dict[str, Any] = {}
    for entry in value:
        if isinstance(entry, dict) and "key" in entry:
            out[entry["key"]] = entry.get("value")
    return out


def _group_by_key(value: JsonValue, key: str) -> dict[str, list[JsonValue]]:
    if not isinstance(value, list):
        return {}
    groups: dict[str, list[JsonValue]] = {}
    for v in value:
        k = _field(v, key)
        if k is None:
            continue
        groups.setdefault(str(k), []).append(v)
    return groups


def _sort_by_key(value: JsonValue, key: str) -> list[JsonValue]:
    if not isinstance(value, list):
        return []
    return sorted(value, key=lambda v: _field(v, key))


def _merge_(left: JsonValue, right: JsonValue) -> JsonValue:
    if isinstance(left, dict) and isinstance(right, dict):
        return {**left, **right}
    if isinstance(left, list) and isinstance(right, list):
        return [*left, *right]
    return right if right is not None else left


def _flatten(value: JsonValue, depth: int = 1) -> list[JsonValue]:
    if not isinstance(value, list):
        return []
    if depth <= 0:
        return list(value)
    out: list[JsonValue] = []
    for v in value:
        if isinstance(v, list):
            out.extend(_flatten(v, depth - 1))
        else:
            out.append(v)
    return out


def _unique_by_key(value: JsonValue, key: str) -> list[JsonValue]:
    if not isinstance(value, list):
        return []
    seen: set[Any] = set()
    out: list[JsonValue] = []
    for v in value:
        k = _field(v, key)
        marker = repr(k)  # fall back when k isn't hashable
        if marker in seen:
            continue
        seen.add(marker)
        out.append(v)
    return out


def _make_object(pairs: tuple[tuple[str, JsonValue], ...]) -> dict[str, Any]:
    return {k: v for k, v in pairs}


def _make_array(items: tuple[JsonValue, ...]) -> list[JsonValue]:
    return list(items)


def _if_then_else(
    predicate: bool,
    then_value: JsonValue,
    else_value: JsonValue,
) -> JsonValue:
    return then_value if predicate else else_value


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------


def build_json_catalog() -> JsonCatalog:
    """Return a fresh :class:`JsonCatalog` with all 20 primitives."""
    cat = JsonCatalog()

    def add(name: str, fn: Callable[..., JsonValue], cost: float, desc: str) -> None:
        cat.add(JsonPrimitive(name=name, fn=fn, cost=cost, description=desc))

    add("field", _field, 0.1, "obj.<name>")
    add("index", _index, 0.1, "arr[<i>] (negative-aware)")
    add("path", _path, 0.3, "Compound (string | int) path walk")
    add("select_where", _select_where, 0.4, "Filter by callable predicate")
    add("has_key", _has_key, 0.1, "object has key")
    add("contains_value", _contains_value, 0.1, "value in array/string/object")
    add("length_", _length, 0.1, "len of string/array/object (null → 0)")
    add("type_of", _type_of, 0.1, "JSON type name as string")
    add("map_", _map_, 0.4, "Map fn over array (non-array → [])")
    add("filter_", _filter_, 0.4, "Filter array by callable predicate")
    add("to_entries", _to_entries, 0.3, "object → [{key, value}, ...]")
    add("from_entries", _from_entries, 0.3, "[{key, value}, ...] → object")
    add("group_by_key", _group_by_key, 0.5, "Group array by key")
    add("sort_by_key", _sort_by_key, 0.4, "Sort array by key")
    add("merge_", _merge_, 0.3, "Shallow merge two values")
    add("flatten_", _flatten, 0.3, "Flatten array up to depth")
    add("unique_by_key", _unique_by_key, 0.4, "Remove duplicates by key")
    add("make_object", _make_object, 0.2, "Build object from (k, v) pairs")
    add("make_array", _make_array, 0.2, "Build array from items")
    add("if_then_else", _if_then_else, 0.2, "Conditional value")

    return cat
