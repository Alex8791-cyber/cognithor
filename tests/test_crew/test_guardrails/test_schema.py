"""Task 26 — schema built-in guardrail tests (Pydantic validation)."""

from pydantic import BaseModel

from cognithor.crew.guardrails.builtin import schema
from cognithor.crew.output import TaskOutput


class Product(BaseModel):
    name: str
    price: float


def _out(raw: str) -> TaskOutput:
    return TaskOutput(task_id="t", agent_role="w", raw=raw)


def test_schema_passes_on_valid_json():
    g = schema(Product)
    r = g(_out('{"name": "Widget", "price": 9.99}'))
    assert r.passed


def test_schema_fails_on_missing_field():
    g = schema(Product)
    r = g(_out('{"name": "Widget"}'))
    assert not r.passed
    assert "price" in (r.feedback or "").lower()


def test_schema_fails_on_invalid_json():
    g = schema(Product)
    r = g(_out("not json"))
    assert not r.passed
    assert "json" in (r.feedback or "").lower()


def test_schema_fails_on_type_mismatch():
    g = schema(Product)
    r = g(_out('{"name": "x", "price": "not a number"}'))
    assert not r.passed
