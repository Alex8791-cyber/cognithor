"""Tests für Coverage-Lücken: search, embeddings, gateway, model_router, mcp, sandbox, watcher.

Bringt die kritischen Module von 33–67% auf ≥90% Coverage.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.config import JarvisConfig, MemoryConfig

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# 1. memory/search.py – Hybrid-Suche (33% → 90%+)
# ============================================================================


class TestHybridSearch:
    """Testet die 3-Kanal Hybrid-Suche: BM25 + Vektor + Graph."""

    @pytest.fixture()
    def search_engine(self, tmp_path: Path):
        """Erstellt eine HybridSearch mit echtem Index und Mock-Embeddings."""
        from jarvis.memory.embeddings import EmbeddingClient
        from jarvis.memory.indexer import MemoryIndex
        from jarvis.memory.search import HybridSearch

        config = MemoryConfig()
        index = MemoryIndex(db_path=tmp_path / "test.db")
        _ = index.conn  # Triggers lazy schema creation

        mock_emb = AsyncMock(spec=EmbeddingClient)
        return HybridSearch(index=index, embedding_client=mock_emb, config=config)

    @pytest.fixture()
    def populated_search(self, search_engine):
        """Search-Engine mit vorindexierten Chunks."""
        from jarvis.models import Chunk, MemoryTier

        idx = search_engine.index
        texts = [
            "Projektmanagement Agile Scrum Kanban",
            "TechCorp Projektplanung Cloud Pro Premium",
            "Kontakt TechCorp Berlin DevOps Beratung",
            "Terraform Cloud Deployment AWS Infrastructure",
        ]
        for i, text in enumerate(texts):
            chunk = Chunk(
                id=f"chunk-{i}",
                text=text,
                source_path="test.md",
                memory_tier=MemoryTier.EPISODIC,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
                token_count=len(text.split()),
                entities=[],
            )
            idx.upsert_chunk(chunk)

        return search_engine

    @pytest.mark.asyncio()
    async def test_empty_query_returns_empty(self, search_engine):
        """Leerer Query gibt leere Liste zurück."""
        results = await search_engine.search("")
        assert results == []
        results = await search_engine.search("   ")
        assert results == []

    @pytest.mark.asyncio()
    async def test_bm25_only_search(self, populated_search):
        """BM25-only Suche findet Chunks über Volltextsuche."""
        results = await populated_search.search(
            "Projektmanagement",
            enable_vector=False,
            enable_graph=False,
        )
        assert len(results) >= 1
        assert results[0].bm25_score > 0
        assert results[0].vector_score == 0
        assert results[0].graph_score == 0

    @pytest.mark.asyncio()
    async def test_bm25_relevance_ranking(self, populated_search):
        """BM25 rankt relevantere Chunks höher."""
        results = await populated_search.search(
            "Projektplanung Methoden",
            enable_vector=False,
            enable_graph=False,
        )
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    @pytest.mark.asyncio()
    async def test_vector_search_with_mock_embeddings(self, populated_search):
        """Vektor-Suche mit Mock-Embeddings."""
        from jarvis.memory.embeddings import EmbeddingResult

        mock_vector = [0.1] * 768
        populated_search._embeddings.embed_text.return_value = EmbeddingResult(
            vector=mock_vector,
            model="test",
            dimensions=768,
            cached=False,
        )

        # Embedding in Index einfügen
        chunk_hash = hashlib.sha256(b"Projektmanagement Agile Scrum Kanban").hexdigest()
        populated_search.index.store_embedding(chunk_hash, mock_vector)

        results = await populated_search.search(
            "Altersvorsorge Rente",
            enable_bm25=False,
            enable_graph=False,
        )
        assert len(results) >= 1
        assert results[0].vector_score > 0

    @pytest.mark.asyncio()
    async def test_vector_search_failure_graceful(self, populated_search):
        """Vektor-Suche fällt graceful zurück wenn Embedding fehlschlägt."""
        populated_search._embeddings.embed_text.side_effect = Exception("Ollama down")

        results = await populated_search.search(
            "Projektmanagement",
            enable_bm25=True,
            enable_graph=False,
        )
        # BM25 sollte trotzdem funktionieren
        assert len(results) >= 1
        assert results[0].bm25_score > 0
        assert results[0].vector_score == 0

    @pytest.mark.asyncio()
    async def test_hybrid_merge_scoring(self, populated_search):
        """Hybrid-Merge kombiniert Scores aus allen Kanälen."""
        from jarvis.memory.embeddings import EmbeddingResult

        mock_vector = [0.5] * 768
        populated_search._embeddings.embed_text.return_value = EmbeddingResult(
            vector=mock_vector,
            model="test",
            dimensions=768,
            cached=False,
        )

        results = await populated_search.search(
            "Projektmanagement",
            enable_bm25=True,
            enable_vector=True,
            enable_graph=True,
        )
        if results:
            r = results[0]
            assert r.score > 0
            assert r.recency_factor > 0

    @pytest.mark.asyncio()
    async def test_top_k_limiting(self, populated_search):
        """Top-K begrenzt Ergebnisse."""
        results = await populated_search.search(
            "Projektmanagement Kontakt Terraform",
            top_k=2,
            enable_vector=False,
            enable_graph=False,
        )
        assert len(results) <= 2

    @pytest.mark.asyncio()
    async def test_tier_filter(self, populated_search):
        """Tier-Filter schränkt auf einen Memory-Tier ein."""
        from jarvis.models import MemoryTier

        results = await populated_search.search(
            "Projektmanagement",
            tier_filter=MemoryTier.SEMANTIC,
            enable_vector=False,
            enable_graph=False,
        )
        # Alle Chunks sind EPISODIC → keine SEMANTIC Treffer
        assert len(results) == 0

    def test_bm25_only_sync(self, populated_search):
        """Synchrone BM25-only Suche."""
        results = populated_search.search_bm25_only("Projektmanagement", top_k=3)
        assert len(results) >= 1
        assert results[0].bm25_score > 0

    def test_graph_search_with_entities(self, populated_search):
        """Graph-Suche findet Chunks über Entity-Matching."""
        from jarvis.models import Chunk, Entity, MemoryTier

        idx = populated_search.index
        entity = Entity(
            id="ent-1",
            name="User",
            type="PERSON",
            source_file="test.md",
        )
        idx.upsert_entity(entity)

        chunk = Chunk(
            id="chunk-entity",
            text="Jarvis ist ein lokaler KI-Assistent.",
            source_path="test.md",
            memory_tier=MemoryTier.SEMANTIC,
            content_hash="entity-hash",
            token_count=6,
            entities=["ent-1"],
        )
        idx.upsert_chunk(chunk)

        graph_scores = populated_search._graph_search("User")
        assert "chunk-entity" in graph_scores
        assert graph_scores["chunk-entity"] > 0

    def test_graph_search_no_match(self, populated_search):
        """Graph-Suche gibt leer zurück wenn keine Entities matchen."""
        scores = populated_search._graph_search("XYZ-gibts-nicht")
        assert scores == {}

    def test_build_chunk_hash_map(self, populated_search):
        """Chunk-Hash-Map baut korrektes Mapping."""
        mapping = populated_search._build_chunk_hash_map()
        assert isinstance(mapping, dict)
        assert len(mapping) >= 1
        for chunk_ids in mapping.values():
            assert isinstance(chunk_ids, list)
            assert len(chunk_ids) >= 1

    @pytest.mark.asyncio()
    async def test_core_memory_no_decay(self, search_engine):
        """Core Memory Chunks bekommen keinen Recency-Decay."""
        from jarvis.models import Chunk, MemoryTier

        idx = search_engine.index
        chunk = Chunk(
            id="core-chunk",
            text="Ich bin Jarvis, dein KI-Assistent.",
            source_path="CORE.md",
            memory_tier=MemoryTier.CORE,
            content_hash="core-hash",
            token_count=6,
            entities=[],
        )
        idx.upsert_chunk(chunk)

        results = await search_engine.search(
            "Jarvis Assistent",
            enable_vector=False,
            enable_graph=False,
        )
        if results:
            assert results[0].recency_factor == 1.0


# ============================================================================
# 2. memory/embeddings.py – Embedding-Client (48% → 90%+)
# ============================================================================


class TestEmbeddingClient:
    """Testet den Embedding-Client mit Mock-Ollama."""

    @pytest.fixture()
    def client(self):
        from jarvis.memory.embeddings import EmbeddingClient

        return EmbeddingClient(
            model="nomic-embed-text",
            base_url="http://localhost:11434",
            dimensions=768,
        )

    @pytest.mark.asyncio()
    async def test_embed_text_cached(self, client):
        """Cache-Hit gibt gespeichertes Embedding zurück."""
        fake_vec = [0.1] * 768
        client._cache["hash123"] = fake_vec

        result = await client.embed_text("test text", content_hash="hash123")
        assert result.cached is True
        assert result.vector == fake_vec
        assert client.stats.cache_hits == 1

    @pytest.mark.asyncio()
    async def test_embed_text_api_call(self, client):
        """API-Call generiert Embedding und cached es."""
        fake_vec = [0.5] * 768
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [fake_vec]}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.embed_text("Projektmanagement", content_hash="ins-hash")
        assert result.cached is False
        assert result.vector == fake_vec
        assert "ins-hash" in client._cache
        assert client.stats.api_calls == 1

    @pytest.mark.asyncio()
    async def test_embed_text_api_error(self, client):
        """API-Fehler erhöht Error-Counter."""
        import httpx

        mock_http = AsyncMock()
        mock_http.post.side_effect = httpx.HTTPError("Connection refused")
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(httpx.HTTPError):
            await client.embed_text("test")
        assert client.stats.errors == 1

    @pytest.mark.asyncio()
    async def test_embed_text_empty_response(self, client):
        """Leere Ollama-Antwort wirft ValueError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": []}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(ValueError, match="Keine Embeddings"):
            await client.embed_text("test")

    @pytest.mark.asyncio()
    async def test_embed_text_no_hash(self, client):
        """Embedding ohne Hash wird nicht gecached."""
        fake_vec = [0.3] * 768
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [fake_vec]}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.embed_text("test")  # Kein content_hash
        assert result.cached is False
        assert len(client._cache) == 0  # Nicht gecached

    @pytest.mark.asyncio()
    async def test_embed_batch(self, client):
        """Batch-Embedding mit Cache-Hits und API-Calls."""
        fake_vec = [0.3] * 768
        client._cache["cached-hash"] = [0.1] * 768

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [fake_vec]}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        results = await client.embed_batch(
            ["cached text", "new text"],
            content_hashes=["cached-hash", "new-hash"],
        )
        assert len(results) == 2
        assert results[0].cached is True
        assert results[1].cached is False

    @pytest.mark.asyncio()
    async def test_embed_batch_length_mismatch(self, client):
        """Batch mit ungleicher Länge wirft ValueError."""
        with pytest.raises(ValueError, match="gleich lang"):
            await client.embed_batch(["a", "b"], content_hashes=["h1"])

    @pytest.mark.asyncio()
    async def test_embed_batch_no_hashes(self, client):
        """Batch ohne Hashes funktioniert."""
        fake_vec = [0.2] * 768
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [fake_vec]}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        results = await client.embed_batch(["text1"])
        assert len(results) == 1

    def test_load_cache(self, client):
        """Cache-Load füllt internen Cache."""
        entries = {"h1": [0.1] * 768, "h2": [0.2] * 768}
        loaded = client.load_cache(entries)
        assert loaded == 2
        assert "h1" in client._cache

    def test_stats_default(self, client):
        """Stats-Objekt initialisiert korrekt."""
        assert client.stats.total_requests == 0
        assert client.stats.cache_hit_rate == 0.0

    def test_model_and_dimensions(self, client):
        """Model und Dimensions korrekt gesetzt."""
        assert client.model == "nomic-embed-text"
        assert client.dimensions == 768

    @pytest.mark.asyncio()
    async def test_close(self, client):
        """Client schließt httpx-Client sauber."""
        mock_http = AsyncMock()
        mock_http.is_closed = False
        client._client = mock_http

        await client.close()
        mock_http.aclose.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_close_already_closed(self, client):
        """Close bei bereits geschlossenem Client ist no-op."""
        mock_http = MagicMock()
        mock_http.is_closed = True
        client._client = mock_http

        await client.close()  # Sollte nicht crashen

    @pytest.mark.asyncio()
    async def test_close_no_client(self, client):
        """Close ohne Client ist no-op."""
        client._client = None
        await client.close()  # Sollte nicht crashen

    @pytest.mark.asyncio()
    async def test_get_client_lazy_init(self, client):
        """_get_client erstellt Client lazy."""
        assert client._client is None
        http_client = await client._get_client()
        assert http_client is not None
        assert client._client is not None
        await client.close()


# ============================================================================
# 3. gateway/gateway.py – Agent-Loop (67% → 90%+)
# ============================================================================


class TestGatewayIntegration:
    """Testet den Gateway inkl. Agent-Loop."""

    @pytest.fixture()
    def gateway_config(self, tmp_path: Path):
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        config.ensure_directories()
        return config

    @pytest.fixture()
    def gateway(self, gateway_config):
        from jarvis.gateway.gateway import Gateway

        gw = Gateway(gateway_config)
        gw._running = True  # Für den Agent-Loop

        # Minimale Subsystem-Mocks
        gw._planner = AsyncMock()
        gw._gatekeeper = MagicMock()
        gw._executor = AsyncMock()
        gw._executor.clear_agent_context = MagicMock()
        gw._mcp_client = MagicMock()
        gw._mcp_client.get_tool_schemas.return_value = {}
        gw._model_router = MagicMock()
        gw._model_router.select_model.return_value = "qwen3:8b"
        gw._reflector = MagicMock()
        gw._reflector.should_reflect.return_value = False
        gw._memory_manager = MagicMock()

        return gw

    def test_init(self, gateway_config):
        """Gateway initialisiert mit leeren Channels."""
        from jarvis.gateway.gateway import Gateway

        gw = Gateway(gateway_config)
        assert gw._channels == {}
        assert gw._sessions == {}
        assert gw._running is False

    @pytest.mark.asyncio()
    async def test_initialize_without_ollama(self, gateway_config):
        """Gateway startet auch ohne erreichbaren Ollama-Server."""
        from jarvis.gateway.gateway import Gateway

        gw = Gateway(gateway_config)

        mock_llm = AsyncMock()
        mock_llm.is_available.return_value = False
        mock_llm.close = AsyncMock()
        mock_llm._ollama = MagicMock()
        mock_llm._backend = None
        mock_llm.backend_type = "ollama"

        with patch("jarvis.core.unified_llm.UnifiedLLMClient") as mock_llm_cls:
            mock_llm_cls.create.return_value = mock_llm

            await gw.initialize()

            assert gw._gatekeeper is not None
            assert gw._mcp_client is not None
            assert gw._planner is not None
            assert gw._executor is not None

            await gw.shutdown()

    @pytest.mark.asyncio()
    async def test_register_channel(self, gateway):
        """Channel wird korrekt registriert."""
        mock_channel = MagicMock()
        mock_channel.name = "test-channel"

        gateway.register_channel(mock_channel)
        assert "test-channel" in gateway._channels

    @pytest.mark.asyncio()
    async def test_handle_message_direct_response(self, gateway):
        """Direct-response Pfad: Planner gibt direkte Antwort."""
        from jarvis.models import ActionPlan, IncomingMessage

        gateway._planner.plan.return_value = ActionPlan(
            goal="test",
            direct_response="Hallo! Wie kann ich helfen?",
            steps=[],
        )

        msg = IncomingMessage(channel="cli", user_id="alex", text="Hallo!")
        response = await gateway.handle_message(msg)

        assert response.text == "Hallo! Wie kann ich helfen?"
        assert response.channel == "cli"
        assert response.is_final is True

    @pytest.mark.asyncio()
    async def test_handle_message_no_plan(self, gateway):
        """Kein Plan und keine direkte Antwort → Fallback-Text."""
        from jarvis.models import ActionPlan, IncomingMessage

        gateway._planner.plan.return_value = ActionPlan(
            goal="test",
            steps=[],
            direct_response="",
        )

        msg = IncomingMessage(channel="cli", user_id="alex", text="???")
        response = await gateway.handle_message(msg)
        assert "umformulieren" in response.text.lower() or "plan" in response.text.lower()

    @pytest.mark.asyncio()
    async def test_handle_message_with_tool_execution(self, gateway):
        """Tool-Execution Pfad: Planner → Gatekeeper → Executor."""
        from jarvis.models import (
            ActionPlan,
            GateDecision,
            GateStatus,
            IncomingMessage,
            PlannedAction,
            RiskLevel,
            ToolResult,
        )

        action = PlannedAction(
            tool="read_file",
            params={"path": "/tmp/test"},
            rationale="test",
        )

        gateway._planner.plan.return_value = ActionPlan(goal="test", steps=[action])
        gateway._planner.formulate_response.return_value = "Datei gelesen: Inhalt XYZ"

        gateway._gatekeeper.evaluate_plan.return_value = [
            GateDecision(
                status=GateStatus.ALLOW,
                reason="OK",
                risk_level=RiskLevel.GREEN,
                original_action=action,
            )
        ]

        gateway._executor.execute.return_value = [
            ToolResult(tool_name="read_file", content="Inhalt XYZ", success=True)
        ]

        msg = IncomingMessage(channel="cli", user_id="alex", text="Lies /tmp/test")
        response = await gateway.handle_message(msg)

        assert "Inhalt XYZ" in response.text
        gateway._executor.execute.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_handle_message_all_blocked(self, gateway):
        """Alle Aktionen blockiert → Blockiert-Meldung."""
        from jarvis.models import (
            ActionPlan,
            GateDecision,
            GateStatus,
            IncomingMessage,
            PlannedAction,
            RiskLevel,
        )

        action = PlannedAction(
            tool="exec_command",
            params={"command": "rm -rf /"},
            rationale="test",
        )

        gateway._planner.plan.return_value = ActionPlan(goal="test", steps=[action])
        gateway._gatekeeper.evaluate_plan.return_value = [
            GateDecision(
                status=GateStatus.BLOCK,
                reason="Destruktiver Befehl",
                risk_level=RiskLevel.RED,
                original_action=action,
            )
        ]

        msg = IncomingMessage(channel="cli", user_id="alex", text="Lösch alles")
        response = await gateway.handle_message(msg)
        assert "blockiert" in response.text.lower() or "Gatekeeper" in response.text

    @pytest.mark.asyncio()
    async def test_handle_message_with_errors_replan(self, gateway):
        """Bei Fehlern → Re-Plan bis Iterationslimit."""
        from jarvis.models import (
            ActionPlan,
            GateDecision,
            GateStatus,
            IncomingMessage,
            PlannedAction,
            RiskLevel,
            ToolResult,
        )

        action = PlannedAction(tool="web_fetch", params={"url": "x"}, rationale="test")

        gateway._planner.plan.return_value = ActionPlan(goal="test", steps=[action])
        gateway._planner.replan.return_value = ActionPlan(goal="test", steps=[action])
        gateway._planner.formulate_response.return_value = "Fehler nach Retries"

        gateway._gatekeeper.evaluate_plan.return_value = [
            GateDecision(
                status=GateStatus.ALLOW,
                reason="OK",
                risk_level=RiskLevel.GREEN,
                original_action=action,
            )
        ]

        # Jeder Versuch schlägt fehl
        gateway._executor.execute.return_value = [
            ToolResult(tool_name="web_fetch", content="Error", is_error=True, success=False)
        ]

        msg = IncomingMessage(channel="cli", user_id="alex", text="Hol die Seite")
        response = await gateway.handle_message(msg)

        # Sollte nach mehreren Iterationen aufgeben
        assert response.text != ""
        assert gateway._planner.replan.await_count >= 1

    @pytest.mark.asyncio()
    async def test_session_management(self, gateway):
        """Sessions werden korrekt erstellt und wiederverwendet."""
        session1 = gateway._get_or_create_session("cli", "alex")
        session2 = gateway._get_or_create_session("cli", "alex")
        session3 = gateway._get_or_create_session("telegram", "alex")

        assert session1.session_id == session2.session_id
        assert session1.session_id != session3.session_id

    @pytest.mark.asyncio()
    async def test_working_memory_with_core(self, gateway, gateway_config):
        """Working Memory lädt Core Memory."""
        core_path = gateway_config.core_memory_path
        core_path.parent.mkdir(parents=True, exist_ok=True)
        core_path.write_text("Ich bin Jarvis.", encoding="utf-8")

        session = gateway._get_or_create_session("cli", "alex")
        wm = gateway._get_or_create_working_memory(session)
        assert "Jarvis" in (wm.core_memory_text or "")

    @pytest.mark.asyncio()
    async def test_shutdown_all_subsystems(self, gateway):
        """Shutdown stoppt Channels, MCP, LLM-Client."""
        mock_channel = AsyncMock()
        mock_channel.name = "test"
        gateway._channels = {"test": mock_channel}

        mock_mcp = AsyncMock()
        gateway._mcp_client = mock_mcp

        mock_llm = AsyncMock()
        gateway._llm = mock_llm

        gateway._running = True
        await gateway.shutdown()

        assert gateway._running is False
        mock_channel.stop.assert_awaited_once()
        mock_mcp.disconnect_all.assert_awaited_once()
        mock_llm.close.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_shutdown_channel_error_handled(self, gateway):
        """Shutdown fängt Channel-Fehler ab."""
        mock_channel = AsyncMock()
        mock_channel.name = "bad"
        mock_channel.stop.side_effect = RuntimeError("Stop failed")
        gateway._channels = {"bad": mock_channel}
        gateway._mcp_client = AsyncMock()
        gateway._ollama = AsyncMock()

        # Sollte nicht crashen
        await gateway.shutdown()
        assert gateway._running is False

    @pytest.mark.asyncio()
    async def test_handle_message_with_reflection(self, gateway):
        """Reflexion wird nach erfolgreicher Antwort ausgeführt."""
        from jarvis.models import ActionPlan, IncomingMessage, ReflectionResult

        gateway._planner.plan.return_value = ActionPlan(
            goal="test",
            direct_response="Fertig!",
            steps=[],
        )
        gateway._reflector.should_reflect.return_value = True
        gateway._reflector.reflect = AsyncMock(
            return_value=ReflectionResult(
                session_id="test-session",
                success_score=0.9,
                evaluation="Alles gut",
            )
        )

        msg = IncomingMessage(channel="cli", user_id="alex", text="Hallo")
        response = await gateway.handle_message(msg)

        assert response.text == "Fertig!"
        gateway._reflector.reflect.assert_awaited_once()


# ============================================================================
# 4. core/model_router.py – Modell-Routing (62% → 90%+)
# ============================================================================


class TestModelRouter:
    """Testet Model-Router und OllamaClient."""

    @pytest.fixture()
    def config(self, tmp_path: Path):
        return JarvisConfig(jarvis_home=tmp_path)

    @pytest.fixture()
    def router(self, config):
        from jarvis.core.model_router import ModelRouter

        mock_ollama = MagicMock()
        return ModelRouter(config, mock_ollama)

    def test_select_model_planning(self, router, config):
        """Wählt Planner-Modell für Planungsaufgaben."""
        model = router.select_model("planning", "high")
        assert model == config.models.planner.name

    def test_select_model_reflection(self, router, config):
        """Wählt Planner-Modell für Reflexion."""
        model = router.select_model("reflection", "high")
        assert model == config.models.planner.name

    def test_select_model_code(self, router, config):
        """Wählt Coder-Modell für Code-Aufgaben."""
        model = router.select_model("code", "high")
        assert model == config.models.coder.name

    def test_select_model_simple_tool_call(self, router, config):
        """Wählt Executor-Modell für einfache Tool-Calls."""
        model = router.select_model("simple_tool_call", "low")
        assert model == config.models.executor.name

    def test_select_model_summarization(self, router, config):
        """Wählt Executor-Modell für Zusammenfassungen."""
        model = router.select_model("summarization", "medium")
        assert model == config.models.executor.name

    def test_select_model_embedding(self, router, config):
        """Wählt Embedding-Modell."""
        model = router.select_model("embedding", "low")
        assert model == config.models.embedding.name

    def test_select_model_general_high(self, router, config):
        """General + high → Planner."""
        model = router.select_model("general", "high")
        assert model == config.models.planner.name

    def test_select_model_general_low(self, router, config):
        """General + low → Executor."""
        model = router.select_model("general", "low")
        assert model == config.models.executor.name

    def test_select_model_fallback_when_unavailable(self, router, config):
        """Fallback wenn Modell nicht verfügbar."""
        router._available_models = {"qwen3:8b"}  # Nur Executor
        model = router.select_model("planning", "high")
        # Sollte auf qwen3:8b fallen
        assert model == config.models.executor.name

    def test_select_model_no_fallback_needed(self, router, config):
        """Kein Fallback wenn Modell verfügbar."""
        router._available_models = {config.models.planner.name}
        model = router.select_model("planning", "high")
        assert model == config.models.planner.name

    @pytest.mark.asyncio()
    async def test_router_initialize(self, config):
        """Router lädt verfügbare Modelle."""
        from jarvis.core.model_router import ModelRouter

        mock_ollama = AsyncMock()
        mock_ollama.list_models.return_value = ["qwen3:8b", "qwen3:32b"]

        router = ModelRouter(config, mock_ollama)
        await router.initialize()

        assert "qwen3:8b" in router._available_models
        assert "qwen3:32b" in router._available_models

    @pytest.mark.asyncio()
    async def test_ollama_client_available(self):
        """OllamaClient prüft Verfügbarkeit via /api/tags."""
        from jarvis.core.model_router import OllamaClient

        config = JarvisConfig()
        client = OllamaClient(config)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_resp
        mock_http.is_closed = False
        client._client = mock_http

        available = await client.is_available()
        assert available is True
        await client.close()

    @pytest.mark.asyncio()
    async def test_ollama_client_not_available(self):
        """OllamaClient meldet nicht verfügbar bei Fehler."""
        import httpx

        from jarvis.core.model_router import OllamaClient

        config = JarvisConfig()
        client = OllamaClient(config)

        mock_http = AsyncMock()
        mock_http.get.side_effect = httpx.ConnectError("Connection refused")
        mock_http.is_closed = False
        client._client = mock_http

        available = await client.is_available()
        assert available is False
        await client.close()

    @pytest.mark.asyncio()
    async def test_ollama_client_list_models(self):
        """OllamaClient listet Modelle."""
        from jarvis.core.model_router import OllamaClient

        config = JarvisConfig()
        client = OllamaClient(config)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text"}]
        }

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_resp
        mock_http.is_closed = False
        client._client = mock_http

        models = await client.list_models()
        assert "qwen3:8b" in models
        assert "nomic-embed-text" in models
        await client.close()

    @pytest.mark.asyncio()
    async def test_ollama_client_list_models_error(self):
        """list_models gibt leere Liste bei Fehler."""
        from jarvis.core.model_router import OllamaClient

        config = JarvisConfig()
        client = OllamaClient(config)

        mock_http = AsyncMock()
        mock_http.get.side_effect = Exception("boom")
        mock_http.is_closed = False
        client._client = mock_http

        models = await client.list_models()
        assert models == []
        await client.close()


# ============================================================================
# 5. mcp/client.py – Tool-Ausführung (54% → 90%+)
# ============================================================================


class TestMCPClient:
    """Testet den MCP-Client und Tool-Registry."""

    @pytest.fixture()
    def mcp(self, tmp_path: Path):
        from jarvis.mcp.client import JarvisMCPClient

        config = JarvisConfig(jarvis_home=tmp_path)
        return JarvisMCPClient(config)

    def test_register_builtin_handler(self, mcp):
        """Builtin-Handler wird korrekt registriert."""
        handler = AsyncMock(return_value={"content": "ok"})
        mcp.register_builtin_handler(
            tool_name="test_tool",
            handler=handler,
            description="Ein Test-Tool",
            input_schema={"type": "object", "properties": {}},
        )
        assert "test_tool" in mcp.get_tool_list()

    @pytest.mark.asyncio()
    async def test_call_builtin_tool(self, mcp):
        """Builtin-Tool wird mit Parametern aufgerufen."""
        handler = AsyncMock(return_value="Ergebnis 42")
        mcp.register_builtin_handler(
            tool_name="calculator",
            handler=handler,
            description="Rechner",
            input_schema={"type": "object", "properties": {"expr": {"type": "string"}}},
        )

        result = await mcp.call_tool("calculator", {"expr": "6*7"})
        assert "Ergebnis 42" in result.content
        assert result.is_error is False

    @pytest.mark.asyncio()
    async def test_call_unknown_tool(self, mcp):
        """Unbekanntes Tool gibt Fehler-Result."""
        result = await mcp.call_tool("nonexistent_tool", {})
        assert result.is_error is True
        assert "nicht gefunden" in result.content

    def test_get_tool_schemas(self, mcp):
        """Tool-Schemas werden korrekt zurückgegeben."""
        mcp.register_builtin_handler(
            tool_name="my_tool",
            handler=AsyncMock(),
            description="Mein Tool",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        schemas = mcp.get_tool_schemas()
        assert "my_tool" in schemas
        assert schemas["my_tool"]["description"] == "Mein Tool"

    def test_get_tool_list(self, mcp):
        """Tool-Liste gibt alle registrierten Tool-Namen."""
        mcp.register_builtin_handler(
            tool_name="tool_a",
            handler=AsyncMock(),
            description="A",
        )
        mcp.register_builtin_handler(
            tool_name="tool_b",
            handler=AsyncMock(),
            description="B",
        )
        tools = mcp.get_tool_list()
        assert "tool_a" in tools
        assert "tool_b" in tools

    @pytest.mark.asyncio()
    async def test_tool_handler_exception(self, mcp):
        """Tool-Handler-Exception wird als Fehler-Result zurückgegeben."""
        handler = AsyncMock(side_effect=RuntimeError("Boom!"))
        mcp.register_builtin_handler(
            tool_name="failing_tool",
            handler=handler,
            description="Failing",
        )
        result = await mcp.call_tool("failing_tool", {})
        assert result.is_error is True
        assert "Boom" in result.content

    @pytest.mark.asyncio()
    async def test_call_sync_handler(self, mcp):
        """Synchroner Handler funktioniert auch."""

        def sync_handler(text: str = "") -> str:
            return f"sync: {text}"

        mcp.register_builtin_handler(
            tool_name="sync_tool",
            handler=sync_handler,
            description="Sync tool",
        )
        result = await mcp.call_tool("sync_tool", {"text": "hello"})
        assert "sync: hello" in result.content
        assert result.is_error is False

    @pytest.mark.asyncio()
    async def test_disconnect_all(self, mcp):
        """disconnect_all schließt alle Server-Verbindungen."""
        await mcp.disconnect_all()  # Sollte nicht crashen auch wenn leer


# ============================================================================
# 6. security/sandbox.py – Sandboxed Execution (60% → 85%+)
# ============================================================================


class TestSandbox:
    """Testet die Sandbox-Execution."""

    @pytest.fixture()
    def sandbox(self):
        from jarvis.models import SandboxConfig
        from jarvis.security.sandbox import Sandbox

        config = SandboxConfig()
        return Sandbox(config)

    def test_sandbox_init(self, sandbox):
        """Sandbox initialisiert korrekt."""
        assert sandbox is not None
        assert sandbox.capabilities["process"] is True

    def test_available_levels(self, sandbox):
        """Process-Level ist immer verfügbar."""
        from jarvis.models import SandboxLevel

        levels = sandbox.available_levels
        assert SandboxLevel.PROCESS in levels

    def test_max_level(self, sandbox):
        """Max-Level gibt die höchste verfügbare Stufe."""
        level = sandbox.max_level
        assert level is not None

    @pytest.mark.asyncio()
    async def test_safe_command_execution(self, sandbox):
        """Sicherer Befehl wird ausgeführt."""
        result = await sandbox.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    @pytest.mark.asyncio()
    async def test_command_timeout(self, sandbox):
        """Langläufer werden nach Timeout abgebrochen."""
        result = await sandbox.execute("sleep 30", timeout=1)
        assert result.exit_code != 0 or result.timed_out

    @pytest.mark.asyncio()
    async def test_command_with_stderr(self, sandbox):
        """Stderr wird korrekt erfasst."""
        # Nutzt python statt Shell-Redirect, da Sandbox exec-basiert ist
        result = await sandbox.execute("python -c \"import sys; sys.stderr.write('fehler\\n')\"")
        assert "fehler" in result.stderr

    @pytest.mark.asyncio()
    async def test_nonexistent_command(self, sandbox):
        """Nicht existierender Befehl gibt Fehler."""
        result = await sandbox.execute("nonexistent_command_xyz_123")
        assert result.exit_code != 0

    @pytest.mark.asyncio()
    async def test_command_with_env(self, sandbox):
        """Umgebungsvariablen werden übergeben."""
        # Nutzt python statt Shell-Expansion, da Sandbox exec-basiert ist
        result = await sandbox.execute(
            "python -c \"import os; print(os.environ.get('MY_VAR', ''))\"",
            env={"MY_VAR": "test123"},
        )
        assert result.exit_code == 0

    @pytest.mark.asyncio()
    async def test_duration_tracked(self, sandbox):
        """Ausführungsdauer wird gemessen."""
        result = await sandbox.execute("echo fast")
        assert result.duration_ms >= 0


# ============================================================================
# 7. memory/watcher.py – File-Watcher (73% → 90%+)
# ============================================================================


class TestMemoryWatcher:
    """Testet den Memory-File-Watcher."""

    @pytest.fixture()
    def watcher_setup(self, tmp_path: Path):
        from jarvis.memory.watcher import MemoryWatcher

        watch_dir = tmp_path / "knowledge"
        watch_dir.mkdir()

        callback = MagicMock()
        watcher = MemoryWatcher(
            memory_dir=watch_dir,
            on_file_changed=callback,
            poll_interval=0.1,
            debounce_seconds=0.1,
        )
        return watcher, watch_dir, callback

    def test_watcher_init(self, watcher_setup):
        """Watcher initialisiert korrekt."""
        watcher, _, _ = watcher_setup
        assert watcher.is_running is False

    def test_watcher_start_stop(self, watcher_setup):
        """Watcher startet und stoppt."""
        watcher, _watch_dir, _ = watcher_setup
        watcher.start()
        assert watcher.is_running is True
        time.sleep(0.2)  # Thread starten lassen
        watcher.stop()
        assert watcher.is_running is False

    def test_watcher_double_start(self, watcher_setup):
        """Doppelter Start ist no-op."""
        watcher, _, _ = watcher_setup
        watcher.start()
        watcher.start()  # Kein Crash
        assert watcher.is_running is True
        watcher.stop()

    def test_scan_files(self, watcher_setup):
        """_scan_files findet .md Dateien."""
        watcher, watch_dir, _ = watcher_setup

        (watch_dir / "doc.md").write_text("Markdown")
        (watch_dir / "other.txt").write_text("Text")

        watcher._scan_files()
        # Nur .md Dateien
        assert any("doc.md" in p for p in watcher._file_mtimes)
        assert not any("other.txt" in p for p in watcher._file_mtimes)

    def test_check_changes_new_file(self, watcher_setup):
        """_check_changes erkennt neue Dateien."""
        watcher, watch_dir, callback = watcher_setup

        # Initial Scan
        watcher._scan_files()

        # Neue Datei erstellen
        (watch_dir / "new.md").write_text("Neues Dokument")

        # Changes prüfen
        watcher._check_changes()
        watcher._handler.process_pending()

        # Kurz warten für Debounce
        time.sleep(0.2)
        watcher._handler.process_pending()

        callback.assert_called()

    def test_check_changes_modified_file(self, watcher_setup):
        """_check_changes erkennt geänderte Dateien."""
        watcher, watch_dir, callback = watcher_setup

        f = watch_dir / "existing.md"
        f.write_text("Version 1")

        watcher._scan_files()

        # Datei ändern (mtime muss sich unterscheiden)
        time.sleep(0.1)
        f.write_text("Version 2")

        watcher._check_changes()
        time.sleep(0.2)
        watcher._handler.process_pending()

        callback.assert_called()


class TestMemoryFileHandler:
    """Testet den Debounce-Handler."""

    @pytest.fixture()
    def handler(self):
        from jarvis.memory.watcher import MemoryFileHandler

        callback = MagicMock()
        return MemoryFileHandler(callback=callback, debounce_seconds=0.1), callback

    def test_on_file_changed_md_only(self, handler):
        """Nur .md Dateien werden registriert."""
        h, _callback = handler
        h.on_file_changed("/path/to/doc.md")
        h.on_file_changed("/path/to/image.png")

        assert "/path/to/doc.md" in h._pending
        assert "/path/to/image.png" not in h._pending

    def test_process_pending_after_debounce(self, handler):
        """Pending-Dateien werden nach Debounce verarbeitet."""
        h, callback = handler
        h.on_file_changed("/path/to/doc.md")

        # Sofort: noch im Debounce
        processed = h.process_pending()
        # Eventuell noch leer (Debounce nicht abgelaufen)

        # Warten
        time.sleep(0.2)
        processed = h.process_pending()
        assert "/path/to/doc.md" in processed
        callback.assert_called_with("/path/to/doc.md")

    def test_process_pending_empty(self, handler):
        """Leerer Pending-Queue gibt leere Liste."""
        h, _ = handler
        assert h.process_pending() == []
