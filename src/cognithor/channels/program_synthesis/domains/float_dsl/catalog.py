"""Float-precision primitive catalog (Sprint-26.3).

Floating-point-aware primitives for synthesis tasks where naive
arithmetic produces wrong answers due to representation error,
catastrophic cancellation, or NaN/Inf propagation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class FloatPrimitive:
    name: str
    fn: Callable[..., Any]
    cost: float
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            msg = f"Invalid Float primitive name: {self.name!r}"
            raise ValueError(msg)
        if self.cost < 0:
            msg = f"Float primitive cost must be >= 0, got {self.cost}"
            raise ValueError(msg)


class FloatCatalog:
    def __init__(self) -> None:
        self._entries: dict[str, FloatPrimitive] = {}

    def add(self, primitive: FloatPrimitive) -> None:
        if primitive.name in self._entries:
            msg = f"Float primitive {primitive.name!r} already registered"
            raise ValueError(msg)
        self._entries[primitive.name] = primitive

    def get(self, name: str) -> FloatPrimitive:
        if name not in self._entries:
            msg = f"Unknown Float primitive {name!r}"
            raise KeyError(msg)
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


FLOAT_PRIMITIVE_NAMES: tuple[str, ...] = (
    "naive_sum",
    "kahan_sum",
    "neumaier_sum",
    "safe_div",
    "safe_log",
    "safe_sqrt",
    "nearly_equal",
    "relative_error",
    "absolute_error",
    "clamp_finite",
    "replace_nan",
    "replace_inf",
    "is_subnormal",
    "is_nan",
    "is_inf",
    "next_after",
    "round_half_to_even",
)


# ---------------------------------------------------------------------------
# Primitive implementations
# ---------------------------------------------------------------------------


def _naive_sum(values: Sequence[float]) -> float:
    """Naive left-to-right sum — provided as a baseline / counter-example."""
    total = 0.0
    for v in values:
        total += float(v)
    return total


def _kahan_sum(values: Sequence[float]) -> float:
    """Kahan compensated summation — reduces accumulator drift."""
    total = 0.0
    c = 0.0
    for v in values:
        y = float(v) - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


def _neumaier_sum(values: Sequence[float]) -> float:
    """Neumaier improved compensated summation — handles cases where
    a partial sum is *smaller* than the next addend."""
    total = 0.0
    c = 0.0
    for raw in values:
        v = float(raw)
        t = total + v
        if abs(total) >= abs(v):
            c += (total - t) + v
        else:
            c += (v - t) + total
        total = t
    return total + c


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0.0 or math.isnan(b):
        return default
    return float(a) / float(b)


def _safe_log(value: float, default: float = float("-inf")) -> float:
    v = float(value)
    if v <= 0.0 or math.isnan(v):
        return default
    return math.log(v)


def _safe_sqrt(value: float, default: float = 0.0) -> float:
    v = float(value)
    if v < 0.0 or math.isnan(v):
        return default
    return math.sqrt(v)


def _nearly_equal(a: float, b: float, *, eps: float = 1e-9) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    diff = abs(float(a) - float(b))
    if diff <= eps:
        return True
    largest = max(abs(float(a)), abs(float(b)))
    return diff <= largest * eps


def _relative_error(actual: float, expected: float) -> float:
    if expected == 0.0:
        return abs(float(actual))
    return abs((float(actual) - float(expected)) / float(expected))


def _absolute_error(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected))


def _clamp_finite(value: float, *, lo: float = -1e308, hi: float = 1e308) -> float:
    v = float(value)
    if math.isnan(v):
        return 0.0
    if math.isinf(v):
        return hi if v > 0 else lo
    return max(lo, min(hi, v))


def _replace_nan(value: float, default: float = 0.0) -> float:
    v = float(value)
    if math.isnan(v):
        return float(default)
    return v


def _replace_inf(value: float, default: float = 0.0) -> float:
    v = float(value)
    if math.isinf(v):
        return float(default)
    return v


def _is_subnormal(value: float) -> bool:
    v = float(value)
    if v == 0.0 or not math.isfinite(v):
        return False
    return abs(v) < 2.2250738585072014e-308  # smallest normal float64


def _is_nan(value: float) -> bool:
    return math.isnan(float(value))


def _is_inf(value: float) -> bool:
    return math.isinf(float(value))


def _next_after(value: float, direction: float) -> float:
    return math.nextafter(float(value), float(direction))


def _round_half_to_even(value: float, *, ndigits: int = 0) -> float:
    return round(float(value), ndigits)


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------


def build_float_catalog() -> FloatCatalog:
    cat = FloatCatalog()

    def add(name: str, fn: Callable[..., Any], cost: float, desc: str) -> None:
        cat.add(FloatPrimitive(name=name, fn=fn, cost=cost, description=desc))

    add("naive_sum", _naive_sum, 0.2, "Naive left-to-right sum (drift-prone)")
    add("kahan_sum", _kahan_sum, 0.4, "Kahan compensated sum")
    add("neumaier_sum", _neumaier_sum, 0.5, "Neumaier improved compensated sum")
    add("safe_div", _safe_div, 0.2, "a/b with default on b=0/NaN")
    add("safe_log", _safe_log, 0.3, "log(x) with default on x<=0/NaN")
    add("safe_sqrt", _safe_sqrt, 0.3, "sqrt(x) with default on x<0/NaN")
    add("nearly_equal", _nearly_equal, 0.3, "Relative/absolute eps comparison")
    add("relative_error", _relative_error, 0.2, "|actual-expected|/|expected|")
    add("absolute_error", _absolute_error, 0.1, "|actual-expected|")
    add("clamp_finite", _clamp_finite, 0.3, "Clamp NaN/Inf to finite range")
    add("replace_nan", _replace_nan, 0.2, "Replace NaN with default")
    add("replace_inf", _replace_inf, 0.2, "Replace Inf with default")
    add("is_subnormal", _is_subnormal, 0.2, "True for denormalised float64")
    add("is_nan", _is_nan, 0.1, "math.isnan")
    add("is_inf", _is_inf, 0.1, "math.isinf")
    add("next_after", _next_after, 0.3, "math.nextafter")
    add("round_half_to_even", _round_half_to_even, 0.2, "Banker's rounding")

    return cat
