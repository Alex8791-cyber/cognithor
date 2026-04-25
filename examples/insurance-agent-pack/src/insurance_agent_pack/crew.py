"""Insurance pre-advisory crew — sequential PGE-Trinity demo.

Process:
1. NeedsAssessor — turns interview answers into a structured profile.
2. PolicyAnalyst — extracts facts from any uploaded policy PDFs.
3. ComplianceGatekeeper — verifies the user's intent is pre-advisory.
4. ReportGenerator — composes the final markdown report.

Sequential because the steps are causally ordered. PGE-Trinity is enforced
inside each agent's CrewTask through Cognithor's Planner/Gatekeeper/Executor;
the ComplianceGatekeeper here is an *additional* visible check on top.
"""

from __future__ import annotations

from cognithor.crew import Crew, CrewProcess, CrewTask
from insurance_agent_pack.agents.compliance_gatekeeper import build_compliance_gatekeeper
from insurance_agent_pack.agents.needs_assessor import build_needs_assessor
from insurance_agent_pack.agents.policy_analyst import build_policy_analyst
from insurance_agent_pack.agents.report_generator import build_report_generator


def build_team(*, model: str = "ollama/qwen3:8b") -> Crew:
    """Construct the 4-agent insurance pre-advisory Crew."""
    needs = build_needs_assessor(model=model)
    policy = build_policy_analyst(model=model)
    compliance = build_compliance_gatekeeper(model=model)
    reporter = build_report_generator(model=model)

    tasks = [
        CrewTask(
            description=(
                "Führe ein strukturiertes Interview durch und produziere ein "
                "Bedarfsprofil als JSON."
            ),
            expected_output="JSON-Bedarfsprofil mit den Feldern aus dem System-Prompt.",
            agent=needs,
        ),
        CrewTask(
            description=(
                "Analysiere alle übergebenen Versicherungspolicen-PDFs mit "
                "`pdf_extract_text`. Erstelle die Übersichts-Tabelle."
            ),
            expected_output="Markdown-Tabelle der bestehenden Policen.",
            agent=policy,
        ),
        CrewTask(
            description=(
                "Prüfe das Anliegen des Nutzers gegen die §34d-Pre-Beratungs-Regeln. "
                "Gib ein JSON-Verdict (`allowed`, `category`, `reason`) zurück."
            ),
            expected_output="JSON-Compliance-Verdict.",
            agent=compliance,
        ),
        CrewTask(
            description=(
                "Erstelle den finalen Pre-Beratungs-Report im Markdown-Format. "
                "Beachte das Compliance-Verdict und brich ab, wenn `allowed=false`."
            ),
            expected_output="Markdown-Report mit Beobachtungen + Lücken-Themen.",
            agent=reporter,
        ),
    ]

    return Crew(
        agents=[needs, policy, compliance, reporter],
        tasks=tasks,
        process=CrewProcess.SEQUENTIAL,
        verbose=False,
    )
