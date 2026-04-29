"""Cognithor · Gateway message utilities — extracted from `gateway.py`.

Stateless and near-stateless helpers that the `Gateway`-class delegates to.
Each function takes the `Gateway` instance as its first argument (`gw`)
when it needs gateway-internal state, or is a free function when it
doesn't (`resolve_relative_dates`).

This module is part of the staged `gateway.py` split documented in
`project_v0960_refactor_backlog.md`. The pattern matches the successful
`channels/config_routes/` split: thin orchestrator `gateway.py` keeps
public API, sub-modules host the bulk logic. Public surface stays at
`from cognithor.gateway.gateway import Gateway`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from cognithor.models import ActionPlan, PlannedAction, RiskLevel, ToolResult
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.gateway.gateway import Gateway
    from cognithor.models import IncomingMessage, WorkingMemory

log = get_logger(__name__)

# Sentinel substrings emitted by the search backend when nothing useful
# came back. Used by `maybe_presearch` to decide whether to skip presearch
# enrichment. Same literals as in `gateway.py`.
_PRESEARCH_NO_RESULTS = "Keine Ergebnisse"
_PRESEARCH_NO_ENGINE = "Keine Suchengine"


def extract_attachments(gw: Gateway, results: list[ToolResult]) -> list[str]:
    """Extrahiert Dateipfade aus Tool-Ergebnissen fuer den Anhang-Versand.

    Prueft ob das Tool-Ergebnis einen gueltigen Dateipfad enthaelt und ob
    die Datei existiert.
    """
    from pathlib import Path

    attachments: list[str] = []
    for result in results:
        if not result.success:
            continue
        if result.tool_name not in gw._ATTACHMENT_TOOLS:
            continue
        # content contains the file path
        candidate = result.content.strip()
        if not candidate:
            continue
        try:
            path = Path(candidate)
            if (
                path.exists()
                and path.is_file()
                and path.suffix.lower() in gw._ATTACHMENT_EXTENSIONS
            ):
                attachments.append(str(path))
                log.info("attachment_detected", tool=result.tool_name, path=str(path))
        except (ValueError, OSError):
            continue
    return attachments


# ── Automatic pre-search for factual questions ────────────────────
#
# `gw._FACT_QUESTION_PATTERNS` and `gw._SKIP_PRESEARCH_PATTERNS` are
# class-level constants on `Gateway` (defined in `gateway.py`). They stay
# the single source of truth — these helpers only read them via the `gw`
# parameter, never duplicate them here.


def is_fact_question(gw: Gateway, text: str) -> bool:
    """Prueft ob eine Nachricht eine Faktenfrage ist, die Web-Recherche braucht."""
    # Zu kurz → wahrscheinlich kein Fakten-Query
    if len(text) < 15:
        return False

    # Check skip patterns (commands, opinions, explanations)
    for skip_pat in gw._SKIP_PRESEARCH_PATTERNS:
        if skip_pat.search(text):
            return False

    # Check factual question patterns
    return any(fact_pat.search(text) for fact_pat in gw._FACT_QUESTION_PATTERNS)


async def classify_coding_task(gw: Gateway, user_message: str) -> tuple[bool, str]:
    """Klassifiziert ob eine Nachricht eine Coding-Aufgabe ist und deren Komplexitaet.

    Nutzt einen schnellen LLM-Call mit dem Executor-Modell.

    Returns:
        (is_coding, complexity) -- complexity ist "simple" oder "complex"
    """
    if not gw._model_router or not gw._llm:
        return False, "simple"

    # Fast-path: skip LLM call if message has no code signals at all.
    # Saves 10-30s of GPU time for conversational messages.
    _lower = user_message.lower()
    _code_signals = (
        "code",
        "bug",
        "fix",
        "fehler",
        "function",
        "funktion",
        "class",
        "import",
        "refactor",
        "refaktor",
        "test",
        "debug",
        "compile",
        "build",
        "deploy",
        "api",
        "endpoint",
        "database",
        "sql",
        "script",
        "programm",
        "implement",
        "variable",
        "return",
        "async",
        "await",
        "exception",
        "error",
        "python",
        "javascript",
        "typescript",
        "java",
        "rust",
        "golang",
        "git",
        "commit",
        "merge",
        "branch",
        "pull request",
        "pr ",
        "datei",
        "file",
        "ordner",
        "verzeichnis",
        "pfad",
        "```",
        "def ",
        "class ",
        "const ",
        "let ",
        "var ",
    )
    if not any(sig in _lower for sig in _code_signals):
        return False, "simple"

    classify_prompt = (
        "Klassifiziere die folgende Nachricht:\n"
        "1. Ist es eine Coding/Programmier-Aufgabe? (ja/nein)\n"
        "2. Wenn ja: Ist es einfach (einzelne Funktion, kleines Fix, Snippet)\n"
        "   oder komplex (Multi-File, Architektur, Refactoring, neues Feature)?\n\n"
        'Antworte NUR mit einem JSON: {"coding": true/false, "complexity": "simple"/"complex"}'
    )

    model = gw._model_router.select_model("simple_tool_call", "low")

    try:
        response = await gw._llm.chat(
            model=model,
            messages=[
                {"role": "system", "content": classify_prompt},
                {"role": "user", "content": user_message[:500]},
            ],
            temperature=0.1,
            format_json=True,
        )

        text = response.get("message", {}).get("content", "")
        # JSON aus Antwort extrahieren
        import json as _json_mod

        # <think>...</think> Bloecke entfernen (qwen3)
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        data = _json_mod.loads(text)
        is_coding = bool(data.get("coding", False))
        complexity = data.get("complexity", "simple")
        if complexity not in ("simple", "complex"):
            complexity = "simple"
        return is_coding, complexity

    except Exception as exc:
        log.debug("coding_classify_failed", error=str(exc)[:200])
        return False, "simple"


def resolve_relative_dates(text: str) -> str:
    """Ersetzt relative Zeitangaben durch konkrete Datumsangaben.

    'morgen' → '01.03.2026', 'heute' → '28.02.2026', etc.
    """
    from datetime import datetime, timedelta

    now = datetime.now()
    today = now.date()

    # Mapping: (regex_pattern, Datum-Offset oder Callback)
    replacements: list[tuple[str, str]] = [
        (r"\bheute\b", today.strftime("%d.%m.%Y")),
        (r"\bmorgen\b", (today + timedelta(days=1)).strftime("%d.%m.%Y")),
        (r"\bübermorgen\b", (today + timedelta(days=2)).strftime("%d.%m.%Y")),
        (r"\bgestern\b", (today - timedelta(days=1)).strftime("%d.%m.%Y")),
        (r"\bvorgestern\b", (today - timedelta(days=2)).strftime("%d.%m.%Y")),
    ]

    # Weekday-based resolution: "naechsten Montag", "am Freitag", etc.
    _wochentage = {
        "montag": 0,
        "dienstag": 1,
        "mittwoch": 2,
        "donnerstag": 3,
        "freitag": 4,
        "samstag": 5,
        "sonntag": 6,
    }
    for tag_name, weekday_num in _wochentage.items():
        days_ahead = (weekday_num - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # "am Montag" = nächster Montag
        target = today + timedelta(days=days_ahead)
        replacements.append(
            (
                rf"\b(?:nächsten?\s+|am\s+|kommenden?\s+)?{tag_name}\b",
                target.strftime("%d.%m.%Y"),
            ),
        )

    # "naechste Woche" / "diese Woche" / "dieses Wochenende"
    replacements.append(
        (
            r"\bnächste(?:r|s|n)?\s+woche\b",
            f"Woche ab {(today + timedelta(days=7 - today.weekday())).strftime('%d.%m.%Y')}",
        ),
    )
    replacements.append(
        (
            r"\bdiese(?:s|r|n)?\s+wochenende\b",
            f"{(today + timedelta(days=5 - today.weekday())).strftime('%d.%m.%Y')}",
        ),
    )

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def build_reddit_forced_plan(gw: Gateway, user_text: str) -> ActionPlan | None:
    """Build a forced ActionPlan that calls reddit_scan directly.

    Extracts product and subreddit names from the user message.
    Returns None if insufficient info to build a plan.
    """
    import re as _re

    text_lower = user_text.lower()

    # Extract subreddits from message (r/Name or just Name after "in")
    subs = _re.findall(r"r/(\w+)", user_text)
    if not subs:
        # Try "in XYZ" pattern
        m = _re.search(r"\bin\s+(\w+)", text_lower)
        if m and m.group(1) not in ("einem", "einem", "der", "die", "das", "den"):
            subs = [m.group(1)]

    # Extract product name — use config default or try to find in text
    product = ""
    _social_cfg_route = getattr(gw._config, "social", None)
    if _social_cfg_route:
        product = getattr(_social_cfg_route, "reddit_product_name", "") or ""

    # Try to find product in text (after "für/for")
    m = _re.search(r"(?:fuer|für|for)\s+(\w+)", user_text, _re.IGNORECASE)
    if m:
        candidate = m.group(1)
        # Don't use common stop words as product
        if candidate.lower() not in ("mich", "uns", "dich", "das", "die", "den"):
            product = candidate

    if not product:
        return None

    params: dict[str, Any] = {"product": product}
    if subs:
        params["subreddits"] = ",".join(subs)

    return ActionPlan(
        goal=f"Reddit-Leads fuer {product} scannen" + (f" in r/{',r/'.join(subs)}" if subs else ""),
        reasoning=("Skill reddit_lead_hunter matched — direkter reddit_scan Aufruf (hard-routed)"),
        steps=[
            PlannedAction(
                tool="reddit_scan",
                params=params,
                rationale=f"Reddit nach {product}-relevanten Posts scannen",
                risk_estimate=RiskLevel.GREEN,
            )
        ],
        confidence=0.95,
    )


async def maybe_presearch(gw: Gateway, msg: IncomingMessage, wm: WorkingMemory) -> str | None:
    """Fuehrt automatisch eine Web-Suche durch wenn die Nachricht eine Faktenfrage ist.

    Returns:
        Suchergebnis-Text wenn Ergebnisse gefunden wurden, sonst None.
    """
    if not gw._is_fact_question(msg.text):
        return None

    # WebTools-Instanz finden
    web_tools = None
    if gw._mcp_client:
        web_tools = getattr(gw._mcp_client, "_web_tools", None)
        if web_tools is None:
            # Fallback: WebTools aus registrierten Handlern extrahieren
            handler = gw._mcp_client.get_handler("web_search")
            if handler is not None:
                # Handler ist eine gebundene Methode von WebTools
                web_tools = getattr(handler, "__self__", None)

    if web_tools is None:
        log.debug("presearch_skip_no_webtools")
        return None

    # Formulate search query as keywords (not as a question)
    query = msg.text.strip()
    # Strip command suffixes ("Recherchiere das online", etc.)
    # Longer phrases first so "bitte such" matches before "such"
    for splitter in (
        "recherchiere das",
        "recherchiere",
        "recherchier",
        "bitte such",
        "such das",
        "such online",
        "finde heraus",
        "schau nach",
        "google",
    ):
        idx = query.lower().find(splitter)
        if idx > 10:  # Nur abschneiden wenn genug Fragetext davor steht
            query = query[:idx].strip()
            break
    query = query.rstrip("?!.").strip()
    # Remove question words for better search results
    for prefix in (
        "wann hat",
        "wann haben",
        "wann wurde",
        "wann war",
        "wo hat",
        "wo haben",
        "wo wurde",
        "wo war",
        "wer hat",
        "wer ist",
        "was ist mit",
        "was hat",
        "stimmt es dass",
        "ist es wahr dass",
    ):
        if query.lower().startswith(prefix):
            query = query[len(prefix) :].strip()
            break

    # Resolve relative time references to concrete dates
    query = gw._resolve_relative_dates(query)

    try:
        log.info("presearch_start", query=query[:80])
        result_text = await web_tools.web_search(
            query=query,
            num_results=5,
            language="de",
            timelimit="m",
        )

        if (
            result_text
            and _PRESEARCH_NO_RESULTS not in result_text
            and _PRESEARCH_NO_ENGINE not in result_text
        ):
            log.info("presearch_found", chars=len(result_text))
            return result_text[:8000]
        else:
            log.info("presearch_no_results", query=query[:80])
            return None

    except Exception as exc:
        log.warning("presearch_failed", error=str(exc)[:200])
        return None


async def answer_from_presearch(gw: Gateway, user_message: str, search_results: str) -> str:
    """Generiert eine Antwort AUSSCHLIEẞLICH basierend auf Suchergebnissen.

    Umgeht den Planner komplett — das LLM bekommt NUR die Suchergebnisse
    und die Frage des Users, ohne Moeglichkeit auf Trainingswissen zurueckzugreifen.

    Nutzt den unified LLM-Client (funktioniert mit jedem Backend).
    """
    if not gw._llm:
        return ""
    system = (
        "You are a fact assistant. You answer questions EXCLUSIVELY "
        "based on the provided search results.\n\n"
        "ABSOLUTE RULES:\n"
        "1. Use ONLY information from the search results below.\n"
        "2. The search results are CURRENT FACTS from the internet.\n"
        "3. If the search results describe an event, it DID happen.\n"
        "4. NEVER say 'there is no evidence' or 'that did not happen'.\n"
        "5. NEVER say 'according to my knowledge' or 'as far as I know'.\n"
        "6. Cite dates, names, and facts DIRECTLY from the results.\n"
        "7. Answer in the same language as the user's question, concisely and informatively.\n"
        "8. You have NO knowledge of your own. You know ONLY the search results.\n"
        "9. Answer DIRECTLY without reasoning process. Brief and factual."
    )

    # /no_think disables qwen3's internal reasoning mode for fast response
    user_prompt = (
        f"SEARCH RESULTS:\n\n{search_results}\n\n"
        f"---\n\n"
        f"QUESTION: {user_message}\n\n"
        f"Answer the question ONLY based on the search results above. /no_think"
    )

    # Select model via ModelRouter (backend-agnostic)
    if gw._model_router:
        model = gw._model_router.select_model("planning", "high")
    else:
        model = gw._config.models.planner.name

    try:
        response = await gw._llm.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            top_p=0.9,
        )

        answer = response.get("message", {}).get("content", "")
        # qwen3 may include <think>...</think> blocks — remove them
        answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
        if answer.strip():
            log.info("presearch_answer_generated", chars=len(answer))
            return answer.strip()

    except Exception as exc:
        log.error("presearch_answer_failed", error=str(exc)[:200])

    # Fallback: use regular PGE loop
    return ""
