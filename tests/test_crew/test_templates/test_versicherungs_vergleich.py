"""Task 43 — versicherungs-vergleich template E2E test.

The DACH insurance-comparison template is spec-mandated (§4.5 AC 5) to be
offline-capable AND guarded by both ``no_pii()`` and a ``StringGuardrail``
wired through ``chain(...)``. These tests assert the rendered ``crew.py``
actually ships those exact patterns — a regression here means the
compliance profile drifted.
"""

from pathlib import Path

from cognithor.crew.cli.init_cmd import run_init


def test_versicherungs_template_ships_all_required_files(tmp_path: Path):
    project = tmp_path / "vv"
    rc = run_init(name="vv", template="versicherungs-vergleich", directory=project, lang="de")
    assert rc == 0

    # 8 user-editable files from the template tree
    assert (project / "pyproject.toml").exists()
    assert (project / "src" / "vv" / "crew.py").exists()
    assert (project / "src" / "vv" / "main.py").exists()
    assert (project / "src" / "vv" / "__init__.py").exists()
    assert (project / "tests" / "test_crew.py").exists()
    assert (project / "README.md").exists()
    assert (project / ".env.example").exists()
    assert (project / "config" / "agents.yaml").exists()
    assert (project / "config" / "tasks.yaml").exists()
    # Auto-injected gitignore
    assert (project / ".gitignore").exists()


def test_versicherungs_template_is_offline_capable(tmp_path: Path):
    """Rendered crew.py must show the offline + compliance contract.

    Specifically:
      * ``tools=[]`` — no network access
      * ``no_pii`` — PII-leak protection
      * ``StringGuardrail`` — LLM-checked §34d-neutrality rule
      * ``chain(`` — both guardrails composed
    """
    project = tmp_path / "vv"
    rc = run_init(name="vv", template="versicherungs-vergleich", directory=project, lang="de")
    assert rc == 0

    crew_file = (project / "src" / "vv" / "crew.py").read_text(encoding="utf-8")
    assert "tools=[]" in crew_file, "Agents must have tools=[] (offline-capable)"
    assert "no_pii" in crew_file, "Must wire no_pii() for DSGVO compliance"
    assert "StringGuardrail" in crew_file, "Must wire StringGuardrail for §34d neutrality"
    assert "chain(" in crew_file, (
        "Must compose guardrails via chain(no_pii(), StringGuardrail(...))"
    )
