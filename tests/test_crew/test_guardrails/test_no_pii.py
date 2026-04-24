"""Task 25 — no_pii built-in guardrail tests (DE-focused)."""

from cognithor.crew.guardrails.builtin import no_pii
from cognithor.crew.output import TaskOutput


def _out(raw: str) -> TaskOutput:
    return TaskOutput(task_id="t", agent_role="w", raw=raw)


def test_clean_text_passes():
    g = no_pii()
    r = g(_out("Dies ist ein völlig harmloser Satz ohne persönliche Daten."))
    assert r.passed
    assert r.pii_detected is False


def test_email_detected():
    g = no_pii()
    r = g(_out("Kontakt: max.mustermann@example.com"))
    assert not r.passed
    assert r.pii_detected is True
    assert "email" in (r.feedback or "").lower() or "e-mail" in (r.feedback or "").lower()


def test_german_iban_detected():
    g = no_pii()
    r = g(_out("Konto: DE89 3704 0044 0532 0130 00"))
    assert not r.passed
    assert r.pii_detected is True


def test_german_phone_detected():
    g = no_pii()
    for ph in ["+49 30 12345678", "030 123 456 78", "0171-1234567", "0049 30 12345"]:
        r = g(_out(f"Telefon: {ph}"))
        assert not r.passed, f"Phone '{ph}' was not detected"


def test_german_steuer_id_11_digit_detected():
    g = no_pii()
    r = g(_out("Steuer-ID 12 345 678 901"))
    assert not r.passed


def test_multiple_pii_listed_in_feedback():
    g = no_pii()
    r = g(_out("Max: max@example.com, IBAN DE89 3704 0044 0532 0130 00"))
    assert not r.passed
    fb = (r.feedback or "").lower()
    assert "email" in fb or "e-mail" in fb
    assert "iban" in fb
