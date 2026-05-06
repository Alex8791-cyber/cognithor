"""Tests for the Operational-Trust PR-B ``channel_contributions``
upstream fix in :class:`~cognithor.core.reflector.Reflector`.

Pre-fix bug: ``Reflector.reflect`` fed a HARDCODED phantom
``{"vector": 0.5, "bm25": 0.3, "graph": 0.2}`` to the
:class:`~cognithor.memory.weight_optimizer.SearchWeightOptimizer` —
every call learned from the same constants regardless of which channels
actually contributed.

PR-B fix: the reflector now reads ``working_memory.injected_memories``
and computes the per-channel hit share via:

    contribution = (number of hits from channel) / (total hits)

A channel "hits" a result when its per-channel score > 0. When no
search results were injected (empty mix), the EMA update is skipped
entirely instead of feeding zeros.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.config import CognithorConfig
from cognithor.core.model_router import ModelRouter, OllamaClient
from cognithor.core.reflector import Reflector, _compute_channel_contributions
from cognithor.models import (
    ActionPlan,
    AgentResult,
    Chunk,
    MemorySearchResult,
    MemoryTier,
    Message,
    MessageRole,
    PlannedAction,
    SessionContext,
    ToolResult,
    WorkingMemory,
)

# ============================================================================
# _compute_channel_contributions: pure function
# ============================================================================


def _make_result(
    *,
    bm25: float = 0.0,
    vector: float = 0.0,
    graph: float = 0.0,
    chunk_id: str = "chunk-x",
) -> MemorySearchResult:
    chunk = Chunk(
        id=chunk_id,
        text="some text",
        source_path="test.md",
        memory_tier=MemoryTier.EPISODIC,
        timestamp=datetime.now(UTC),
        content_hash="hash-x",
    )
    return MemorySearchResult(
        chunk=chunk,
        score=max(bm25, vector, graph),
        bm25_score=bm25,
        vector_score=vector,
        graph_score=graph,
        recency_factor=1.0,
    )


class TestComputeChannelContributions:
    def test_empty_input_returns_empty_dict(self) -> None:
        assert _compute_channel_contributions([]) == {}

    def test_all_zero_scores_returns_empty(self) -> None:
        results = [_make_result(bm25=0.0, vector=0.0, graph=0.0)]
        assert _compute_channel_contributions(results) == {}

    def test_single_channel_dominates(self) -> None:
        results = [
            _make_result(vector=0.9, chunk_id="a"),
            _make_result(vector=0.7, chunk_id="b"),
            _make_result(vector=0.5, chunk_id="c"),
        ]
        contrib = _compute_channel_contributions(results)
        assert contrib == {"vector": 1.0, "bm25": 0.0, "graph": 0.0}

    def test_mixed_channels_proportional(self) -> None:
        # 2 vector hits, 1 bm25 hit, 1 graph hit -> total 4 hits.
        results = [
            _make_result(vector=0.8, chunk_id="a"),
            _make_result(vector=0.6, chunk_id="b"),
            _make_result(bm25=0.5, chunk_id="c"),
            _make_result(graph=0.4, chunk_id="d"),
        ]
        contrib = _compute_channel_contributions(results)
        assert contrib["vector"] == pytest.approx(0.5)
        assert contrib["bm25"] == pytest.approx(0.25)
        assert contrib["graph"] == pytest.approx(0.25)
        assert sum(contrib.values()) == pytest.approx(1.0)

    def test_overlapping_channels_count_per_channel(self) -> None:
        """A single result that scored on multiple channels counts once
        for each contributing channel — the formula is "number of hits"
        per channel, independent of per-channel intensity."""
        results = [
            _make_result(bm25=0.5, vector=0.5, chunk_id="a"),  # hits 2 channels
            _make_result(graph=0.4, chunk_id="b"),  # hits 1 channel
        ]
        contrib = _compute_channel_contributions(results)
        # 1 vector hit + 1 bm25 hit + 1 graph hit = 3 total hits.
        assert contrib["vector"] == pytest.approx(1 / 3)
        assert contrib["bm25"] == pytest.approx(1 / 3)
        assert contrib["graph"] == pytest.approx(1 / 3)

    def test_NOT_the_old_phantom_constants(self) -> None:
        """Regression: the value MUST NEVER be the pre-fix phantom mix."""
        results = [_make_result(vector=0.9)]
        contrib = _compute_channel_contributions(results)
        assert contrib != {"vector": 0.5, "bm25": 0.3, "graph": 0.2}


# ============================================================================
# Reflector.reflect: integration with WeightOptimizer
# ============================================================================


@pytest.fixture()
def config(tmp_path: Any) -> CognithorConfig:
    return CognithorConfig(cognithor_home=tmp_path)


@pytest.fixture()
def mock_ollama() -> AsyncMock:
    return AsyncMock(spec=OllamaClient)


@pytest.fixture()
def mock_router() -> MagicMock:
    router = MagicMock(spec=ModelRouter)
    router.select_model.return_value = "qwen3:32b"
    router.get_model_config.return_value = {
        "temperature": 0.3,
        "top_p": 0.9,
        "context_window": 32768,
    }
    return router


@pytest.fixture()
def session() -> SessionContext:
    return SessionContext(session_id="sess-pr-b", channel="cli", user_id="alex")


@pytest.fixture()
def working_memory_with_search() -> WorkingMemory:
    """Working memory whose injected memories reflect a real channel mix.

    3 results: 2 vector-only hits, 1 bm25-only hit. Expected share:
    vector=2/3, bm25=1/3, graph=0.
    """
    wm = WorkingMemory(session_id="sess-pr-b")
    wm.add_message(Message(role=MessageRole.USER, content="recherche-bericht"))
    wm.injected_memories = [
        _make_result(vector=0.9, chunk_id="r1"),
        _make_result(vector=0.7, chunk_id="r2"),
        _make_result(bm25=0.5, chunk_id="r3"),
    ]
    return wm


def _make_agent_result_with_tools() -> AgentResult:
    plans = [
        ActionPlan(
            goal="Recherche-Bericht",
            reasoning="x",
            steps=[
                PlannedAction(
                    tool="memory_search",
                    params={"query": "x"},
                    rationale="r",
                )
            ],
        )
    ]
    tool_results = [
        ToolResult(tool_name="memory_search", content="ok", is_error=False),
    ]
    return AgentResult(
        response="ok",
        plans=plans,
        tool_results=tool_results,
        total_iterations=2,
        total_duration_ms=100,
        model_used="qwen3:32b",
        success=True,
    )


GOOD_JSON = """{
  "success_score": 0.85,
  "evaluation": "Ok",
  "extracted_facts": [],
  "session_summary": {
    "goal": "x",
    "outcome": "ok",
    "key_decisions": [],
    "open_items": [],
    "tools_used": ["memory_search"],
    "duration_ms": 100
  }
}"""


class TestReflectorPopulatesRealContributions:
    """The reflector must feed REAL channel mix to the optimizer."""

    async def test_real_channel_mix_passed_to_optimizer(
        self,
        config: CognithorConfig,
        mock_ollama: AsyncMock,
        mock_router: MagicMock,
        session: SessionContext,
        working_memory_with_search: WorkingMemory,
    ) -> None:
        mock_ollama.chat.return_value = {
            "message": {"content": GOOD_JSON},
            "prompt_eval_count": 10,
            "eval_count": 10,
        }

        weight_optimizer = MagicMock()
        reflector = Reflector(
            config,
            mock_ollama,
            mock_router,
            weight_optimizer=weight_optimizer,
        )

        agent_result = _make_agent_result_with_tools()
        await reflector.reflect(session, working_memory_with_search, agent_result)

        # The optimizer was called with the COMPUTED real share, NOT
        # the pre-fix phantom constants.
        assert weight_optimizer.record_outcome.called
        call_kwargs = weight_optimizer.record_outcome.call_args.kwargs
        contrib = call_kwargs["channel_contributions"]
        assert contrib["vector"] == pytest.approx(2 / 3)
        assert contrib["bm25"] == pytest.approx(1 / 3)
        assert contrib["graph"] == pytest.approx(0.0)
        # Critically NOT the phantom.
        assert contrib != {"vector": 0.5, "bm25": 0.3, "graph": 0.2}
        # session_id is forwarded for snapshot meta linkage.
        assert call_kwargs["session_id"] == session.session_id

    async def test_skips_optimizer_when_no_search_hits(
        self,
        config: CognithorConfig,
        mock_ollama: AsyncMock,
        mock_router: MagicMock,
        session: SessionContext,
    ) -> None:
        """Empty injected_memories → record_outcome is NOT called.

        Pre-fix this fed the optimizer with the phantom constants; PR-B
        skips the call so the EMA isn't biased by sessions that did no
        search at all.
        """
        mock_ollama.chat.return_value = {
            "message": {"content": GOOD_JSON},
            "prompt_eval_count": 10,
            "eval_count": 10,
        }

        wm = WorkingMemory(session_id=session.session_id)
        wm.add_message(Message(role=MessageRole.USER, content="hello"))
        # NO injected_memories — simulates a session without search.

        weight_optimizer = MagicMock()
        reflector = Reflector(
            config,
            mock_ollama,
            mock_router,
            weight_optimizer=weight_optimizer,
        )

        agent_result = _make_agent_result_with_tools()
        await reflector.reflect(session, wm, agent_result)

        # The skip path: no record_outcome call (empty contributions).
        assert not weight_optimizer.record_outcome.called

    async def test_audit_callback_wired_to_optimizer(
        self,
        config: CognithorConfig,
        mock_ollama: AsyncMock,
        mock_router: MagicMock,
    ) -> None:
        """Reflector wires its audit-emit helper into the optimizer at
        construction time (mirrors PR-A's CausalAnalyzer pattern)."""
        weight_optimizer = MagicMock()

        Reflector(
            config,
            mock_ollama,
            mock_router,
            weight_optimizer=weight_optimizer,
        )

        # The setter should have been called once with a callable.
        assert weight_optimizer.set_audit_emit_callback.called
        bound_cb = weight_optimizer.set_audit_emit_callback.call_args.args[0]
        assert callable(bound_cb)
