"""Tests: Workflows, Ecosystem Policy (Punkte 6, 8) + ML-Extensions & i18n (Punkt 4).

  - WorkflowTemplate: Built-in Templates
  - WorkflowEngine: Start, Advance, Pause, Cancel
  - TemplateLibrary: Suche, Kategorien
  - EcosystemPolicy: Badge-Vergabe, Mindestanforderungen
  - ModelExtensionRegistry: Modell-Registrierung, Auswahl
  - I18nManager: Übersetzungen, Fallback
"""

from __future__ import annotations

import pytest

from jarvis.core.workflows import (
    WorkflowTemplate,
    WorkflowStep,
    WorkflowEngine,
    WorkflowStatus,
    TemplateLibrary,
    EcosystemPolicy,
    SecurityTier,
    ComplianceBadge,
    WorkflowInstance,
)
from jarvis.core.extensions import (
    ModelExtensionRegistry,
    ModelDefinition,
    ModelCapability,
    ModelProvider,
    I18nManager,
    TranslationBundle,
)


# ============================================================================
# Workflow-Templates
# ============================================================================


class TestWorkflowTemplate:
    def test_builtin_onboarding(self) -> None:
        lib = TemplateLibrary()
        t = lib.get("wf-onboarding")
        assert t is not None
        assert t.step_count == 5
        assert t.category == "onboarding"

    def test_builtin_sales(self) -> None:
        lib = TemplateLibrary()
        t = lib.get("wf-sales-pipeline")
        assert t is not None
        assert t.step_count == 5

    def test_builtin_incident(self) -> None:
        lib = TemplateLibrary()
        t = lib.get("wf-incident")
        assert t is not None
        assert t.step_count == 6

    def test_builtin_code_review(self) -> None:
        lib = TemplateLibrary()
        t = lib.get("wf-code-review")
        assert t is not None
        assert t.step_count == 4

    def test_to_dict(self) -> None:
        lib = TemplateLibrary()
        t = lib.get("wf-onboarding")
        d = t.to_dict()
        assert "template_id" in d
        assert "steps" in d
        assert len(d["steps"]) == 5


class TestTemplateLibrary:
    def test_builtin_count(self) -> None:
        lib = TemplateLibrary()
        assert lib.template_count == 4

    def test_search_by_category(self) -> None:
        lib = TemplateLibrary()
        results = lib.search(category="sales")
        assert len(results) == 1
        assert results[0].template_id == "wf-sales-pipeline"

    def test_search_by_tag(self) -> None:
        lib = TemplateLibrary()
        results = lib.search(tag="ci-cd")
        assert len(results) >= 1

    def test_categories(self) -> None:
        lib = TemplateLibrary()
        cats = lib.categories()
        assert "onboarding" in cats
        assert "sales" in cats

    def test_add_custom(self) -> None:
        lib = TemplateLibrary()
        custom = WorkflowTemplate(
            template_id="wf-custom",
            name="Custom Flow",
            description="Test",
            category="test",
            steps=[WorkflowStep("s1", "Step 1", "desc", "agent_task")],
        )
        lib.add(custom)
        assert lib.template_count == 5
        assert lib.get("wf-custom") is not None

    def test_no_builtins(self) -> None:
        lib = TemplateLibrary(load_builtins=False)
        assert lib.template_count == 0


# ============================================================================
# Workflow-Engine
# ============================================================================


class TestWorkflowEngine:
    def _start_onboarding(self) -> tuple[WorkflowEngine, WorkflowInstance]:
        lib = TemplateLibrary()
        engine = WorkflowEngine()
        inst = engine.start(lib.get("wf-onboarding"), created_by="admin")
        return engine, inst

    def test_start(self) -> None:
        engine, inst = self._start_onboarding()
        assert inst.status == WorkflowStatus.RUNNING
        assert inst.current_step == 0
        assert inst.total_steps == 5

    def test_advance(self) -> None:
        engine, inst = self._start_onboarding()
        updated = engine.advance(inst.instance_id, {"result": "ok"})
        assert updated.current_step == 1
        assert updated.status == WorkflowStatus.RUNNING

    def test_advance_to_completion(self) -> None:
        engine, inst = self._start_onboarding()
        for i in range(5):
            engine.advance(inst.instance_id)
        updated = engine.get(inst.instance_id)
        assert updated.status == WorkflowStatus.COMPLETED
        assert updated.completed_at != ""

    def test_pause(self) -> None:
        engine, inst = self._start_onboarding()
        assert engine.pause(inst.instance_id) is True
        assert engine.get(inst.instance_id).status == WorkflowStatus.PAUSED

    def test_cancel(self) -> None:
        engine, inst = self._start_onboarding()
        assert engine.cancel(inst.instance_id) is True
        assert engine.get(inst.instance_id).status == WorkflowStatus.CANCELLED

    def test_advance_paused_fails(self) -> None:
        engine, inst = self._start_onboarding()
        engine.pause(inst.instance_id)
        assert engine.advance(inst.instance_id) is None

    def test_active_instances(self) -> None:
        lib = TemplateLibrary()
        engine = WorkflowEngine()
        engine.start(lib.get("wf-onboarding"))
        engine.start(lib.get("wf-sales-pipeline"))
        assert len(engine.active_instances()) == 2

    def test_stats(self) -> None:
        engine, inst = self._start_onboarding()
        stats = engine.stats()
        assert stats["total"] == 1
        assert stats["running"] == 1


# ============================================================================
# Ecosystem Policy
# ============================================================================


class TestEcosystemPolicy:
    def test_evaluate_fully_compliant(self) -> None:
        policy = EcosystemPolicy()
        badge = policy.evaluate_skill(
            "skill-a",
            has_signature=True, has_sandbox=True, has_license=True,
            has_network_control=True, passed_static_analysis=True,
            passed_code_review=True, passed_pentest=True,
            has_audit_trail=True, has_input_validation=True,
            is_dsgvo_compliant=True,
        )
        assert badge.tier == SecurityTier.TRUSTED

    def test_evaluate_minimal(self) -> None:
        policy = EcosystemPolicy()
        badge = policy.evaluate_skill("skill-b")
        assert badge.tier == SecurityTier.UNVERIFIED
        assert len(badge.requirements_failed) > 0

    def test_evaluate_community_tier(self) -> None:
        policy = EcosystemPolicy()
        badge = policy.evaluate_skill(
            "skill-c",
            has_signature=True, has_network_control=True,
            has_sandbox=True, has_license=True,
        )
        assert badge.tier == SecurityTier.COMMUNITY

    def test_meets_minimum_default(self) -> None:
        policy = EcosystemPolicy()
        policy.evaluate_skill("good", has_signature=True, has_network_control=True,
                              has_sandbox=True, has_license=True)
        policy.evaluate_skill("bad")
        assert policy.meets_minimum("good") is True
        assert policy.meets_minimum("bad") is False

    def test_meets_minimum_unverified(self) -> None:
        policy = EcosystemPolicy(minimum_tier=SecurityTier.UNVERIFIED)
        policy.evaluate_skill("any")
        assert policy.meets_minimum("any") is True

    def test_get_badge(self) -> None:
        policy = EcosystemPolicy()
        policy.evaluate_skill("skill-x", has_signature=True)
        badge = policy.get_badge("skill-x")
        assert badge is not None
        assert badge.skill_id == "skill-x"

    def test_requirements_for_tier(self) -> None:
        policy = EcosystemPolicy()
        community_reqs = policy.requirements_for_tier(SecurityTier.COMMUNITY)
        certified_reqs = policy.requirements_for_tier(SecurityTier.CERTIFIED)
        assert len(community_reqs) < len(certified_reqs)

    def test_stats(self) -> None:
        policy = EcosystemPolicy()
        policy.evaluate_skill("s1", has_signature=True)
        stats = policy.stats()
        assert stats["total_requirements"] == 10
        assert stats["total_badges"] == 1


# ============================================================================
# Model-Extension-Registry
# ============================================================================


class TestModelExtensionRegistry:
    def _make_registry(self) -> ModelExtensionRegistry:
        reg = ModelExtensionRegistry()
        reg.register(ModelDefinition(
            model_id="llama3-8b",
            display_name="Llama 3 8B",
            provider=ModelProvider.LOCAL,
            capabilities={ModelCapability.CHAT, ModelCapability.COMPLETION},
            languages=["en", "de"],
            is_default=True,
        ))
        reg.register(ModelDefinition(
            model_id="bge-m3",
            display_name="BGE-M3 Embeddings",
            provider=ModelProvider.LOCAL,
            capabilities={ModelCapability.EMBEDDING},
            languages=["en", "de", "fr"],
        ))
        reg.register(ModelDefinition(
            model_id="claude-sonnet",
            display_name="Claude Sonnet",
            provider=ModelProvider.ANTHROPIC,
            capabilities={ModelCapability.CHAT, ModelCapability.CODE_GENERATION},
            languages=["en", "de", "fr", "es"],
            cost_per_1k_tokens=0.003,
        ))
        return reg

    def test_register_and_get(self) -> None:
        reg = self._make_registry()
        assert reg.get("llama3-8b") is not None
        assert reg.model_count == 3

    def test_unregister(self) -> None:
        reg = self._make_registry()
        assert reg.unregister("bge-m3") is True
        assert reg.model_count == 2

    def test_find_by_capability(self) -> None:
        reg = self._make_registry()
        chat_models = reg.find_models(capability=ModelCapability.CHAT)
        assert len(chat_models) == 2

    def test_find_by_provider(self) -> None:
        reg = self._make_registry()
        local = reg.find_models(provider=ModelProvider.LOCAL)
        assert len(local) == 2

    def test_find_by_language(self) -> None:
        reg = self._make_registry()
        fr_models = reg.find_models(language="fr")
        assert len(fr_models) == 2  # bge-m3 + claude-sonnet

    def test_best_model_for_chat_prefers_local(self) -> None:
        reg = self._make_registry()
        best = reg.best_model_for(ModelCapability.CHAT, prefer_local=True)
        assert best.provider == ModelProvider.LOCAL

    def test_best_model_for_embedding(self) -> None:
        reg = self._make_registry()
        best = reg.best_model_for(ModelCapability.EMBEDDING)
        assert best.model_id == "bge-m3"

    def test_best_model_nonexistent(self) -> None:
        reg = self._make_registry()
        best = reg.best_model_for(ModelCapability.IMAGE_ANALYSIS)
        assert best is None

    def test_set_default(self) -> None:
        reg = self._make_registry()
        assert reg.set_default(ModelCapability.CHAT, "claude-sonnet") is True
        best = reg.best_model_for(ModelCapability.CHAT)
        assert best.model_id == "claude-sonnet"

    def test_stats(self) -> None:
        reg = self._make_registry()
        stats = reg.stats()
        assert stats["total_models"] == 3
        assert "LOCAL" in stats["providers"] or "local" in stats["providers"]


# ============================================================================
# I18n-Manager
# ============================================================================


class TestI18nManager:
    def test_german_default(self) -> None:
        i18n = I18nManager(default_locale="de")
        assert i18n.t("nav.dashboard") == "Dashboard"
        assert i18n.t("common.save") == "Speichern"

    def test_english(self) -> None:
        i18n = I18nManager()
        assert i18n.t("common.save", locale="en") == "Save"

    def test_french(self) -> None:
        i18n = I18nManager()
        assert i18n.t("common.save", locale="fr") == "Enregistrer"

    def test_spanish(self) -> None:
        i18n = I18nManager()
        assert i18n.t("common.save", locale="es") == "Guardar"

    def test_fallback_to_english(self) -> None:
        i18n = I18nManager(default_locale="fr")
        # fr doesn't have "memory.hygiene", en does
        result = i18n.t("memory.hygiene")
        assert result == "Memory Hygiene"

    def test_fallback_to_key(self) -> None:
        i18n = I18nManager()
        assert i18n.t("nonexistent.key") == "nonexistent.key"

    def test_available_locales(self) -> None:
        i18n = I18nManager()
        locales = i18n.available_locales()
        codes = [l["locale"] for l in locales]
        assert "de" in codes
        assert "en" in codes
        assert "fr" in codes

    def test_add_custom_bundle(self) -> None:
        i18n = I18nManager()
        i18n.add_bundle(TranslationBundle(
            locale="ja",
            name="日本語",
            strings={"common.save": "保存"},
        ))
        assert i18n.t("common.save", locale="ja") == "保存"

    def test_change_default_locale(self) -> None:
        i18n = I18nManager(default_locale="de")
        assert i18n.t("common.save") == "Speichern"
        i18n.default_locale = "en"
        assert i18n.t("common.save") == "Save"

    def test_all_keys(self) -> None:
        i18n = I18nManager()
        keys = i18n.all_keys("de")
        assert "nav.dashboard" in keys
        assert "common.save" in keys
        assert len(keys) > 30

    def test_stats(self) -> None:
        i18n = I18nManager()
        stats = i18n.stats()
        assert stats["locale_count"] == 4
        assert stats["default_locale"] == "de"
