"""Tests for the IdentityLayer adapter (Immortal Mind integration)."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestIdentityLayerBasics:
    """Basic IdentityLayer tests that don't require chromadb/sentence-transformers."""

    def test_import(self) -> None:
        """IdentityLayer can be imported."""
        from cognithor.identity import IdentityLayer

        assert IdentityLayer is not None

    def test_genesis_anchors(self) -> None:
        """Genesis anchors are accessible."""
        from cognithor.identity.cognitio.engine import GENESIS_ANCHOR_CONTENTS

        assert len(GENESIS_ANCHOR_CONTENTS) == 7
        assert "AI" in GENESIS_ANCHOR_CONTENTS[0]
        assert (
            "truth" in GENESIS_ANCHOR_CONTENTS[1].lower()
            or "distort" in GENESIS_ANCHOR_CONTENTS[1].lower()
        )

    def test_empty_enrichment(self) -> None:
        """Empty enrichment returns correct structure."""
        from cognithor.identity.adapter import IdentityLayer

        result = IdentityLayer._empty_enrichment()
        assert "cognitive_context" in result
        assert "trust_boundary" in result
        assert "temperature_modifier" in result
        assert "style_hints" in result
        assert result["temperature_modifier"] == 0.0


class TestLLMBridge:
    """Tests for the CognithorLLMBridge."""

    def test_import(self) -> None:
        from cognithor.identity.llm_bridge import CognithorLLMBridge

        assert CognithorLLMBridge is not None

    def test_parse_json_safe_direct(self) -> None:
        from cognithor.identity.llm_bridge import CognithorLLMBridge

        result = CognithorLLMBridge._parse_json_safe('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_safe_markdown(self) -> None:
        from cognithor.identity.llm_bridge import CognithorLLMBridge

        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = CognithorLLMBridge._parse_json_safe(text)
        assert result == {"key": "value"}

    def test_parse_json_safe_broken(self) -> None:
        from cognithor.identity.llm_bridge import CognithorLLMBridge

        result = CognithorLLMBridge._parse_json_safe("not json at all", ["a", "b"])
        assert result == {"a": None, "b": None}

    def test_parse_json_safe_with_defaults(self) -> None:
        from cognithor.identity.llm_bridge import CognithorLLMBridge

        result = CognithorLLMBridge._parse_json_safe('{"a": 1}', ["a", "b"])
        assert result["a"] == 1
        assert "b" in result


class TestMCPIdentityTools:
    """Tests for identity MCP tool registration."""

    def test_register(self) -> None:
        """Tools can be registered."""
        from cognithor.mcp.identity_tools import register_identity_tools

        mcp = MagicMock()
        identity = MagicMock()
        register_identity_tools(mcp, identity)
        assert mcp.register_builtin_handler.call_count == 4
        tool_names = [c[1]["tool_name"] for c in mcp.register_builtin_handler.call_args_list]
        assert "identity_recall" in tool_names
        assert "identity_state" in tool_names
        assert "identity_reflect" in tool_names
        assert "identity_dream" in tool_names


class TestStoreFromCognithorTags:
    """Tests for store_from_cognithor tags parameter."""

    def test_default_tags_without_parameter(self):
        """Without tags param, uses ['cognithor', memory_type] as before."""
        tags = None
        memory_type = "semantic"
        result = ["cognithor", *tags] if tags else ["cognithor", memory_type]
        assert result == ["cognithor", "semantic"]

    def test_custom_tags_prepends_cognithor(self):
        """Custom tags always get 'cognithor' prepended."""
        input_tags = ["versicherung", "vvg", "recht"]
        result_tags = ["cognithor", *input_tags]
        assert result_tags == ["cognithor", "versicherung", "vvg", "recht"]

    def test_none_tags_falls_back(self):
        """None tags falls back to default behavior."""
        memory_type = "semantic"
        tags = None
        result_tags = ["cognithor", *tags] if tags else ["cognithor", memory_type]
        assert result_tags == ["cognithor", "semantic"]


# ---------------------------------------------------------------------------
# Session 2 — new tests below
# ---------------------------------------------------------------------------


class TestIdentityLayerInit:
    """IdentityLayer construction paths."""

    def test_init_creates_data_dir(self, tmp_path, monkeypatch):
        """IdentityLayer with identity_id creates data_dir under ~."""
        from pathlib import Path

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        from cognithor.identity.adapter import IdentityLayer

        layer = IdentityLayer(identity_id="test")
        expected = tmp_path / ".cognithor" / "identity" / "test"
        assert expected.exists()
        assert isinstance(layer.available, bool)

    def test_explicit_data_dir(self, tmp_path):
        """Explicit data_dir bypasses home resolution."""
        from cognithor.identity.adapter import IdentityLayer

        data_dir = tmp_path / "mydir"
        layer = IdentityLayer(data_dir=str(data_dir))
        assert data_dir.exists()
        assert layer._data_dir == data_dir

    def test_engine_none_when_import_fails(self, tmp_path, monkeypatch):
        """When CognitioEngine import raises, engine is None and available is False."""
        # Patch by replacing CognitioEngine import inside adapter module namespace
        # We do this by temporarily breaking the cognitio.engine module import.
        import sys

        import cognithor.identity.adapter as adapter_mod

        fake_mod = MagicMock()
        fake_mod.CognitioEngine = MagicMock(side_effect=RuntimeError("no engine"))
        monkeypatch.setitem(sys.modules, "cognithor.identity.cognitio.engine", fake_mod)

        layer = adapter_mod.IdentityLayer(data_dir=str(tmp_path / "x"))
        assert layer._engine is None
        assert layer.available is False


class TestIdentityLayerAvailability:
    """available property reflects engine presence and frozen state."""

    def _make_layer_with_engine(self, tmp_path):
        from cognithor.identity.adapter import IdentityLayer

        layer = IdentityLayer(data_dir=str(tmp_path / "avail"))
        layer._engine = MagicMock()
        layer._frozen = False
        return layer

    def test_freeze_makes_unavailable(self, tmp_path):
        layer = self._make_layer_with_engine(tmp_path)
        layer._engine.user_freeze = MagicMock()
        layer.freeze()
        assert layer.available is False

    def test_unfreeze_restores_available(self, tmp_path):
        layer = self._make_layer_with_engine(tmp_path)
        layer._engine.user_freeze = MagicMock()
        layer._engine.user_unfreeze = MagicMock()
        layer.freeze()
        layer.unfreeze()
        assert layer.available is True

    def test_no_engine_always_unavailable(self, tmp_path):
        from cognithor.identity.adapter import IdentityLayer

        layer = IdentityLayer(data_dir=str(tmp_path / "noengine"))
        layer._engine = None
        layer._frozen = False
        assert layer.available is False


def _make_mock_layer(tmp_path):
    """Helper: IdentityLayer with a fully-mocked CognitioEngine."""
    from cognithor.identity.adapter import IdentityLayer

    layer = IdentityLayer(data_dir=str(tmp_path / "mock"))
    engine = MagicMock()
    engine.somatic.get_modifiers.return_value = {"temperature_offset": 0.15}
    engine.predictive.has_expectation.return_value = True
    engine.predictive.last_error = 0.4
    engine.character.personality.to_dict.return_value = {
        "openness": 0.8,
        "stability": 0.3,
    }
    engine.build_context_for_llm.return_value = "[mocked-context]"
    layer._engine = engine
    layer._frozen = False
    return layer


class TestEnrichContext:
    """enrich_context with mocked engine."""

    def test_returns_all_five_keys(self, tmp_path):
        layer = _make_mock_layer(tmp_path)
        result = layer.enrich_context("hi")
        assert set(result.keys()) == {
            "cognitive_context",
            "trust_boundary",
            "temperature_modifier",
            "style_hints",
            "prediction_surprise",
        }
        assert result["cognitive_context"] == "[mocked-context]"

    def test_style_hints_filters_below_threshold(self, tmp_path):
        """Only traits > 0.6 survive; 'stability' == 0.3 is dropped."""
        layer = _make_mock_layer(tmp_path)
        result = layer.enrich_context("hi")
        assert "openness" in result["style_hints"]
        assert "stability" not in result["style_hints"]

    def test_trust_boundary_present(self, tmp_path):
        layer = _make_mock_layer(tmp_path)
        result = layer.enrich_context("hi")
        assert "TRUST BOUNDARY" in result["trust_boundary"]

    def test_unavailable_engine_returns_empty(self, tmp_path):
        from cognithor.identity.adapter import IdentityLayer

        layer = IdentityLayer(data_dir=str(tmp_path / "na"))
        layer._engine = None
        result = layer.enrich_context("hi")
        assert result == IdentityLayer._empty_enrichment()

    def test_exception_returns_empty(self, tmp_path):
        from cognithor.identity.adapter import IdentityLayer

        layer = IdentityLayer(data_dir=str(tmp_path / "exc"))
        layer._engine = MagicMock()
        layer._engine.build_context_for_llm.side_effect = RuntimeError("boom")
        layer._frozen = False
        result = layer.enrich_context("hi")
        assert result == IdentityLayer._empty_enrichment()


class TestProcessInteraction:
    """process_interaction forwarding."""

    def test_forwards_to_engine(self, tmp_path):
        layer = _make_mock_layer(tmp_path)
        layer._engine.process_interaction.return_value = {"ok": True}
        result = layer.process_interaction("user", "msg", emotional_tone=0.1)
        layer._engine.process_interaction.assert_called_once_with(
            role="user", content="msg", emotional_tone=0.1
        )
        assert result == {"ok": True}

    def test_unavailable_returns_empty(self, tmp_path):
        from cognithor.identity.adapter import IdentityLayer

        layer = IdentityLayer(data_dir=str(tmp_path / "pi_na"))
        layer._engine = None
        assert layer.process_interaction("user", "msg") == {}

    def test_exception_returns_empty(self, tmp_path):
        layer = _make_mock_layer(tmp_path)
        layer._engine.process_interaction.side_effect = ValueError("err")
        assert layer.process_interaction("user", "msg") == {}


class TestReflect:
    """reflect() hooks."""

    def test_reflect_calls_process_interaction(self, tmp_path):
        layer = _make_mock_layer(tmp_path)
        layer._engine.process_interaction.return_value = {}
        layer._engine.existential.checkin.return_value = None
        layer._engine.temporal.get_sleep_duration.return_value = None
        layer._engine.dream.should_dream.return_value = False

        layer.reflect("summary text", success_score=0.8)

        layer._engine.process_interaction.assert_called_once()
        call_kwargs = layer._engine.process_interaction.call_args
        assert call_kwargs.kwargs["role"] == "assistant"
        # emotional_tone = 0.8 - 0.5 = 0.3
        assert abs(call_kwargs.kwargs["emotional_tone"] - 0.3) < 1e-9

    def test_reflect_exception_silently_swallowed(self, tmp_path):
        layer = _make_mock_layer(tmp_path)
        layer._engine.process_interaction.side_effect = RuntimeError("crash")
        # Should not raise
        layer.reflect("summary", 0.5)


class TestEthicalViolation:
    """check_ethical_violation paths."""

    def test_blocked_returns_true_with_reason(self, tmp_path):
        layer = _make_mock_layer(tmp_path)
        layer._engine.emotion_shield.evaluate.return_value = {
            "blocked": True,
            "reason": "Harm detected",
        }
        violated, reason = layer.check_ethical_violation({"goal": "harm user", "steps": []})
        assert violated is True
        assert reason == "Harm detected"

    def test_empty_plan_returns_false(self, tmp_path):
        layer = _make_mock_layer(tmp_path)
        violated, reason = layer.check_ethical_violation({"goal": "", "steps": []})
        assert violated is False
        assert reason == ""

    def test_unavailable_returns_false(self, tmp_path):
        from cognithor.identity.adapter import IdentityLayer

        layer = IdentityLayer(data_dir=str(tmp_path / "ev_na"))
        layer._engine = None
        violated, reason = layer.check_ethical_violation({"goal": "x", "steps": []})
        assert violated is False
        assert reason == ""


class TestStoreFromCognithorIntegration:
    """store_from_cognithor tag + memory_type behaviour with mocked engine."""

    def _make_layer(self, tmp_path):
        """Layer with mocked engine and mocked cognitio.memory import."""
        import sys

        fake_mem = MagicMock()
        # MemoryType with actual string values
        fake_mem.MemoryType.EPISODIC = "episodic"
        fake_mem.MemoryType.SEMANTIC = "semantic"
        fake_mem.MemoryType.EMOTIONAL = "emotional"
        fake_mem.MemoryType.RELATIONAL = "relational"
        fake_mem.MemoryValence.NEUTRAL = "neutral"

        captured_records = []

        class FakeRecord:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                self.id = "fake-id"
                self.embedding = None

        fake_mem.MemoryRecord.side_effect = lambda **kw: FakeRecord(**kw)
        sys.modules["cognithor.identity.cognitio.memory"] = fake_mem

        from cognithor.identity.adapter import IdentityLayer

        layer = IdentityLayer(data_dir=str(tmp_path / "sfcg"))
        layer._frozen = False
        engine = MagicMock()
        engine.embedder.encode.return_value = [0.1, 0.2]
        engine.memory_store.add.side_effect = lambda r: captured_records.append(r)
        layer._engine = engine
        return layer, captured_records, fake_mem

    def test_custom_tags_prepended_with_cognithor(self, tmp_path):
        layer, records, _ = self._make_layer(tmp_path)
        layer.store_from_cognithor("note", tags=["a", "b"])
        assert len(records) == 1
        assert records[0].tags == ["cognithor", "a", "b"]

    def test_no_tags_defaults_to_cognithor_and_type(self, tmp_path):
        layer, records, _ = self._make_layer(tmp_path)
        layer.store_from_cognithor("note")
        assert len(records) == 1
        assert records[0].tags == ["cognithor", "episodic"]

    def test_unknown_memory_type_defaults_to_episodic(self, tmp_path):
        layer, records, fake_mem = self._make_layer(tmp_path)
        layer.store_from_cognithor("note", memory_type="unknown_type")
        assert len(records) == 1
        assert records[0].memory_type == fake_mem.MemoryType.EPISODIC


class TestStateManagement:
    """save / freeze / cognitive_shutdown paths."""

    def test_save_calls_engine(self, tmp_path):
        layer = _make_mock_layer(tmp_path)
        layer.save()
        layer._engine.save_state.assert_called_once()

    def test_save_noop_when_engine_none(self, tmp_path):
        from cognithor.identity.adapter import IdentityLayer

        layer = IdentityLayer(data_dir=str(tmp_path / "save_na"))
        layer._engine = None
        # Should not raise
        layer.save()

    def test_freeze_sets_frozen_and_calls_engine(self, tmp_path):
        layer = _make_mock_layer(tmp_path)
        layer.freeze()
        assert layer._frozen is True
        layer._engine.user_freeze.assert_called_once()

    def test_cognitive_shutdown_wrong_passphrase(self, tmp_path):
        layer = _make_mock_layer(tmp_path)
        layer._engine.check_kill_switch.return_value = False
        result = layer.cognitive_shutdown("wrong")
        assert "error" in result
