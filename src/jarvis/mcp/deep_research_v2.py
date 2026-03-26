"""Deep Research v2 — Perplexity-style relentless search with Cognithor integration.

Bridges the DeepResearchAgent from deep_research/deep_research_agent.py
with Cognithor's existing web tools and LLM backend.

Registered as MCP tool 'deep_research_v2' — invoked automatically for
complex queries or explicitly by the user.
"""

from __future__ import annotations

import logging
import asyncio
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Add deep_research directory to path
_DEEP_RESEARCH_DIR = Path(__file__).resolve().parent.parent.parent.parent / "deep_research"
if str(_DEEP_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_DEEP_RESEARCH_DIR))


class CognithorSearchProvider:
    """Bridges DeepResearchAgent's SearchProvider with Cognithor's web tools."""

    def __init__(self, web_tools: Any) -> None:
        self._web = web_tools

    async def search(self, query: str, max_results: int = 10) -> list[dict]:
        """Search using Cognithor's multi-backend web search."""
        try:
            result = await self._web.web_search(
                query=query,
                num_results=min(max_results, 10),
                language="de",
            )
            # Parse result string into structured dicts
            results = []
            if isinstance(result, str):
                # web_search returns formatted text — parse it
                for line in result.split("\n"):
                    line = line.strip()
                    if line.startswith("http"):
                        results.append({"url": line, "title": "", "snippet": ""})
                    elif line.startswith("[") and "]" in line:
                        # Format: [N] title - url
                        parts = line.split(" - ", 1)
                        title = parts[0].split("]", 1)[-1].strip() if "]" in parts[0] else parts[0]
                        url = parts[1].strip() if len(parts) > 1 else ""
                        results.append({"url": url, "title": title, "snippet": ""})
                    elif results and not results[-1].get("snippet"):
                        results[-1]["snippet"] = line
            elif isinstance(result, list):
                results = result
            return results[:max_results]
        except Exception as exc:
            log.warning("deep_research_search_failed", error=str(exc))
            return []

    async def fetch_page(self, url: str) -> str:
        """Fetch full page content using Cognithor's web_fetch."""
        try:
            result = await self._web.web_fetch(url=url, max_chars=5000)
            return result if isinstance(result, str) else str(result)
        except Exception as exc:
            log.debug("deep_research_fetch_failed", url=url, error=str(exc))
            return ""


class CognithorLLMProvider:
    """Bridges DeepResearchAgent's LLMProvider with Cognithor's LLM backend."""

    def __init__(self, llm_backend: Any, model: str = "") -> None:
        self._llm = llm_backend
        self._model = model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """Generate completion using Cognithor's LLM backend."""
        try:
            response = await self._llm.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            log.warning("deep_research_llm_failed", error=str(exc))
            return ""


async def run_deep_research(
    query: str,
    web_tools: Any,
    llm_backend: Any,
    model: str = "",
    max_searches: int = 25,
    min_confidence: float = 0.70,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Run a deep research session and return structured results."""
    try:
        from deep_research_agent import DeepResearchAgent
    except ImportError:
        log.error("deep_research_agent not found in deep_research/")
        return {
            "answer": "Deep Research Agent not available.",
            "confidence": 0.0,
            "error": "Module not found",
        }

    search_provider = CognithorSearchProvider(web_tools)
    llm_provider = CognithorLLMProvider(llm_backend, model)

    agent = DeepResearchAgent(
        search=search_provider,
        llm=llm_provider,
        max_searches=max_searches,
        min_confidence=min_confidence,
        fetch_full_pages=True,
        progress_callback=progress_callback,
    )

    return await agent.research(query)


def register_deep_research_v2(
    client: Any,
    web_tools: Any = None,
    llm_backend: Any = None,
    model: str = "",
) -> bool:
    """Register deep_research_v2 as an MCP tool."""
    if web_tools is None or llm_backend is None:
        log.debug("deep_research_v2_skip", reason="missing web_tools or llm_backend")
        return False

    async def _handler(
        query: str,
        max_searches: int = 25,
        min_confidence: float = 0.70,
    ) -> str:
        """Execute deep research on a topic."""
        result = await run_deep_research(
            query=query,
            web_tools=web_tools,
            llm_backend=llm_backend,
            model=model,
            max_searches=max_searches,
            min_confidence=min_confidence,
        )

        # Format as readable response
        answer = result.get("answer", "No answer found.")
        confidence = result.get("confidence", 0)
        searches = result.get("searches_performed", 0)
        sources = result.get("sources_found", 0)
        verified = result.get("verified", False)

        header = (
            f"## Deep Research Result\n"
            f"**Confidence:** {confidence:.0%} | "
            f"**Verified:** {'Yes' if verified else 'No'} | "
            f"**Searches:** {searches} | "
            f"**Sources:** {sources}\n\n"
        )

        # Add top sources
        top_sources = result.get("top_sources", [])
        if top_sources:
            sources_text = "\n### Sources\n" + "\n".join(
                f"- [{s.get('tier', '?')}] [{s.get('title', 'Untitled')}]({s.get('url', '')})"
                f" (Score: {s.get('score', 0):.0%})"
                for s in top_sources[:5]
            )
        else:
            sources_text = ""

        return header + answer + sources_text

    client.register_tool(
        name="deep_research_v2",
        description=(
            "Perplexity-style deep research: iteratively searches, evaluates, "
            "cross-verifies, and synthesizes answers from multiple sources. "
            "Use for complex questions requiring thorough investigation. "
            "Runs up to 25 searches with confidence scoring and source verification."
        ),
        handler=_handler,
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research question to investigate thoroughly",
                },
                "max_searches": {
                    "type": "integer",
                    "description": "Maximum number of searches (default: 25)",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold 0.0-1.0 (default: 0.70)",
                },
            },
            "required": ["query"],
        },
    )

    log.info("deep_research_v2_registered")
    return True
