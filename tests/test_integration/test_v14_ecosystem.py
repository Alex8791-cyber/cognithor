"""Tests: Skill-CLI, Installer, Performance.

v14-Module für Ecosystem, Onboarding und Skalierbarkeit.
"""

from __future__ import annotations

import pytest
from typing import Any

# ============================================================================
# 1. Skill CLI
# ============================================================================

from jarvis.tools.skill_cli import (
    LintSeverity,
    PublishStatus,
    RewardSystem,
    ScaffoldResult,
    SkillCLI,
    SkillLinter,
    SkillPublisher,
    SkillScaffolder,
    SkillTester,
    TemplateType,
)


class TestSkillScaffolder:
    def test_scaffold_basic(self) -> None:
        s = SkillScaffolder()
        result = s.scaffold("Wetter Abfrage", TemplateType.BASIC, author="alex")
        assert result.slug == "wetter_abfrage"
        assert len(result.files_created) == 4

    def test_scaffold_api(self) -> None:
        s = SkillScaffolder()
        result = s.scaffold("Gmail Sync", TemplateType.API_INTEGRATION)
        assert any("manifest.json" in f for f in result.files_created)

    def test_scaffold_automation(self) -> None:
        s = SkillScaffolder()
        result = s.scaffold("Backup", TemplateType.AUTOMATION)
        assert result.template_used == "TPL-AUTO"

    def test_templates_available(self) -> None:
        s = SkillScaffolder()
        assert s.template_count >= 3

    def test_stats(self) -> None:
        s = SkillScaffolder()
        s.scaffold("Test", TemplateType.BASIC)
        assert s.stats()["skills_created"] == 1


class TestSkillLinter:
    def test_valid_skill(self) -> None:
        linter = SkillLinter()
        files = {
            "SKILL.md": "# My Skill\n\n## Beschreibung\nEin toller Skill der viel kann und gut ist.",
            "skill.py": "from jarvis.skills.base import BaseSkill\nclass MySkill(BaseSkill): pass",
            "manifest.json": '{"name": "my", "version": "0.1.0", "permissions": []}',
            "test_skill.py": "def test_it(): pass",
        }
        issues = linter.lint(files)
        assert linter.is_valid(files)

    def test_missing_file(self) -> None:
        linter = SkillLinter()
        issues = linter.lint({"skill.py": "x"})
        errors = [i for i in issues if i.severity == LintSeverity.ERROR]
        assert len(errors) >= 2  # SKILL.md + manifest.json fehlen

    def test_missing_base_class(self) -> None:
        linter = SkillLinter()
        files = {
            "SKILL.md": "# X\n## Beschreibung\nBlah blah blah blah blah blah",
            "skill.py": "class MySkill: pass",
            "manifest.json": '{"name": "x", "version": "0.1", "permissions": []}',
        }
        issues = linter.lint(files)
        assert any(i.rule == "no-base-class" for i in issues)

    def test_no_tests_warning(self) -> None:
        linter = SkillLinter()
        files = {
            "SKILL.md": "# X\n## Beschreibung\nAusführliche Beschreibung des Skills",
            "skill.py": "from jarvis.skills.base import BaseSkill\nclass X(BaseSkill): pass",
            "manifest.json": '{"name": "x", "version": "0.1", "permissions": []}',
        }
        issues = linter.lint(files)
        assert any(i.rule == "no-tests" for i in issues)


class TestSkillTester:
    def test_with_tests(self) -> None:
        tester = SkillTester()
        result = tester.test_skill("myskill", "def test_one(): pass\ndef test_two(): pass")
        assert result.total_tests == 2
        assert result.success

    def test_no_tests(self) -> None:
        tester = SkillTester()
        result = tester.test_skill("empty", "# no tests here")
        assert not result.success

    def test_stats(self) -> None:
        tester = SkillTester()
        tester.test_skill("a", "def test_x(): pass")
        assert tester.stats()["total_runs"] == 1


class TestSkillPublisher:
    def test_create_and_publish(self) -> None:
        pub = SkillPublisher()
        req = pub.create_request("Weather", "1.0.0", "alex")
        pub.run_checks(req.request_id, lint=True, tests=True, security=True)
        assert pub.publish(req.request_id)
        assert req.status == PublishStatus.PUBLISHED

    def test_cant_publish_without_checks(self) -> None:
        pub = SkillPublisher()
        req = pub.create_request("Bad", "1.0", "bob")
        assert not pub.publish(req.request_id)

    def test_reject(self) -> None:
        pub = SkillPublisher()
        req = pub.create_request("Spam", "1.0", "eve")
        pub.reject(req.request_id, "Malware erkannt")
        assert req.status == PublishStatus.REJECTED


class TestRewardSystem:
    def test_award_points(self) -> None:
        rs = RewardSystem()
        points = rs.award_points("alice", "skill_published")
        assert points == 100

    def test_first_skill_bonus(self) -> None:
        rs = RewardSystem()
        rs.award_points("alice", "skill_published")
        cr = rs.get_or_create("alice")
        assert cr.points >= 200  # 100 + 200 Bonus
        assert "🌱 Erster Skill" in cr.badges

    def test_leaderboard(self) -> None:
        rs = RewardSystem()
        rs.award_points("alice", "skill_published")
        rs.award_points("bob", "review_given")
        board = rs.leaderboard()
        assert board[0].contributor == "alice"

    def test_stats(self) -> None:
        rs = RewardSystem()
        rs.award_points("alice", "skill_published")
        stats = rs.stats()
        assert stats["contributors"] == 1


class TestSkillCLI:
    def test_cmd_new(self) -> None:
        cli = SkillCLI()
        result = cli.cmd_new("Test Skill", "basic", "dev")
        assert result.slug == "test_skill"

    def test_full_pipeline(self) -> None:
        cli = SkillCLI()
        files = {
            "SKILL.md": "# Test\n## Beschreibung\nEin vollwertiger Test-Skill für alles",
            "skill.py": "from jarvis.skills.base import BaseSkill\nclass TestSkill(BaseSkill): pass",
            "manifest.json": '{"name": "test", "version": "0.1.0", "permissions": []}',
            "test_skill.py": "def test_one(): pass",
        }
        result = cli.full_pipeline("test", "0.1.0", "dev", files)
        assert result["lint"]["ok"]
        assert result["pipeline_success"]


# ============================================================================
# 2. Installer / Setup Wizard
# ============================================================================

from jarvis.core.installer import (
    ChannelConfigurator,
    ChannelType,
    HardwareDetector,
    HardwareProfile,
    ModelRecommender,
    PresetConfig,
    PresetLevel,
    PRESETS,
    SetupStep,
    SetupWizard,
)


class TestHardwareDetector:
    def test_from_specs(self) -> None:
        hw = HardwareDetector.from_specs(
            cpu="Ryzen 9950X3D", cores=16, ram_gb=64,
            gpu_name="RTX 5090", vram_gb=32, disk_gb=1000,
        )
        assert hw.tier == "power"
        assert hw.gpu.cuda_available

    def test_minimal_specs(self) -> None:
        hw = HardwareDetector.from_specs(cpu="i5", cores=4, ram_gb=8)
        assert hw.tier == "minimal"

    def test_detect(self) -> None:
        d = HardwareDetector()
        hw = d.detect()
        assert hw.cpu_cores > 0


class TestModelRecommender:
    def test_recommend_gpu(self) -> None:
        r = ModelRecommender()
        hw = HardwareDetector.from_specs(gpu_name="RTX 3080", vram_gb=10, ram_gb=32)
        recs = r.recommend(hw, top_n=3)
        assert len(recs) >= 1
        assert all(rec.vram_required_gb <= 10 for rec in recs)

    def test_recommend_cpu_only(self) -> None:
        r = ModelRecommender()
        hw = HardwareDetector.from_specs(ram_gb=16)
        recs = r.recommend(hw)
        assert all(rec.vram_required_gb == 0 for rec in recs)

    def test_recommend_power(self) -> None:
        r = ModelRecommender()
        hw = HardwareDetector.from_specs(gpu_name="RTX 5090", vram_gb=32, ram_gb=64)
        recs = r.recommend(hw, top_n=5)
        assert any(rec.quality_score >= 9 for rec in recs)


class TestPresets:
    def test_all_presets_exist(self) -> None:
        for level in PresetLevel:
            assert level in PRESETS

    def test_minimal_preset(self) -> None:
        p = PRESETS[PresetLevel.MINIMAL]
        assert p.max_agents == 1
        assert not p.enable_federation

    def test_enterprise_preset(self) -> None:
        p = PRESETS[PresetLevel.ENTERPRISE]
        assert p.max_agents == 50
        assert p.enable_federation


class TestChannelConfigurator:
    def test_configure(self) -> None:
        cc = ChannelConfigurator()
        ch = cc.configure(ChannelType.TELEGRAM, {"bot_token": "123:ABC"})
        assert ch.status == "configured"

    def test_missing_config(self) -> None:
        cc = ChannelConfigurator()
        ch = cc.configure(ChannelType.TELEGRAM, {})
        assert ch.status == "unconfigured"

    def test_stats(self) -> None:
        cc = ChannelConfigurator()
        cc.configure(ChannelType.WEB, {"port": 8080, "host": "0.0.0.0"})
        assert cc.stats()["configured"] == 1


class TestSetupWizard:
    def test_auto_setup(self) -> None:
        wiz = SetupWizard()
        state = wiz.auto_setup()
        assert state.is_complete
        assert state.progress_percent == 100

    def test_step_by_step(self) -> None:
        wiz = SetupWizard()
        hw = wiz.step_hardware(HardwareDetector.from_specs(cores=8, ram_gb=32, gpu_name="RTX 4070", vram_gb=12))
        assert hw.tier == "standard"
        recs = wiz.step_model()
        assert len(recs) > 0
        preset = wiz.step_preset()
        assert preset.name == "Standard"

    def test_generate_config(self) -> None:
        wiz = SetupWizard()
        wiz.auto_setup()
        config = wiz.generate_config()
        assert "jarvis" in config
        assert "channels" in config

    def test_power_setup(self) -> None:
        wiz = SetupWizard()
        wiz.step_hardware(HardwareDetector.from_specs(
            cpu="Ryzen 9950X3D", cores=16, ram_gb=64,
            gpu_name="RTX 5090", vram_gb=32,
        ))
        wiz.step_model()
        preset = wiz.step_preset()
        assert preset.level == PresetLevel.POWER


# ============================================================================
# 3. Performance
# ============================================================================

from jarvis.core.performance import (
    Backend,
    BalancingStrategy,
    CloudFallback,
    CloudProvider,
    FallbackConfig,
    LatencyTracker,
    LoadBalancer,
    PerformanceManager,
    QueryDecomposer,
    ResourceOptimizer,
    VectorBackend,
    VectorStore,
)


class TestVectorStore:
    def test_add_and_search(self) -> None:
        vs = VectorStore(dimension=3)
        vs.add("Hallo", [1.0, 0.0, 0.0])
        vs.add("Welt", [0.0, 1.0, 0.0])
        results = vs.search([1.0, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].entry.text == "Hallo"  # Höchste Ähnlichkeit

    def test_cosine_similarity(self) -> None:
        assert VectorStore._cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert VectorStore._cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_delete(self) -> None:
        vs = VectorStore(dimension=2)
        e = vs.add("x", [1.0, 0.0])
        assert vs.delete(e.entry_id)
        assert vs.entry_count == 0

    def test_stats(self) -> None:
        vs = VectorStore()
        vs.add("test", [0.1] * 384)
        assert vs.stats()["entries"] == 1


class TestQueryDecomposer:
    def test_simple_query(self) -> None:
        d = QueryDecomposer()
        subs = d.decompose("Was ist der Preis?")
        assert len(subs) == 1

    def test_compound_query(self) -> None:
        d = QueryDecomposer()
        subs = d.decompose("Was kostet die BU und welche Leistungen sind enthalten?")
        assert len(subs) == 2

    def test_classify_comparison(self) -> None:
        d = QueryDecomposer()
        subs = d.decompose("Vergleich zwischen WWK und R&V Tarifen")
        assert subs[0].query_type == "comparison"


class TestLoadBalancer:
    def test_add_backend(self) -> None:
        lb = LoadBalancer()
        lb.add_backend(Backend("b1", "Local Ollama", "http://localhost:11434", max_concurrent=4))
        assert lb.backend_count == 1

    def test_select_backend(self) -> None:
        lb = LoadBalancer(BalancingStrategy.LEAST_CONNECTIONS)
        lb.add_backend(Backend("b1", "Local", "http://localhost", current_load=3, max_concurrent=4))
        lb.add_backend(Backend("b2", "Backup", "http://backup", current_load=0, max_concurrent=4))
        selected = lb.select_backend()
        assert selected.backend_id == "b2"

    def test_latency_based(self) -> None:
        lb = LoadBalancer(BalancingStrategy.LATENCY_BASED)
        lb.add_backend(Backend("b1", "Slow", "http://slow", avg_latency_ms=2000, max_concurrent=4))
        lb.add_backend(Backend("b2", "Fast", "http://fast", avg_latency_ms=100, max_concurrent=4))
        assert lb.select_backend().backend_id == "b2"

    def test_health_check(self) -> None:
        lb = LoadBalancer()
        lb.add_backend(Backend("b1", "Dead", "http://x", healthy=False))
        assert len(lb.health_check()) == 1

    def test_stats(self) -> None:
        lb = LoadBalancer()
        lb.add_backend(Backend("b1", "X", "http://x", max_concurrent=4))
        assert lb.stats()["backends"] == 1


class TestCloudFallback:
    def test_disabled(self) -> None:
        cf = CloudFallback()
        assert not cf.should_fallback(10000, 95)

    def test_enabled_triggers(self) -> None:
        cfg = FallbackConfig(enabled=True, provider=CloudProvider.ANTHROPIC, model="haiku",
                             trigger_latency_ms=3000)
        cf = CloudFallback(cfg)
        assert cf.should_fallback(5000, 50)

    def test_daily_limit(self) -> None:
        cfg = FallbackConfig(enabled=True, provider=CloudProvider.OPENAI, max_daily_requests=2)
        cf = CloudFallback(cfg)
        cf.record_fallback(0.01)
        cf.record_fallback(0.01)
        assert not cf.should_fallback(10000, 95)

    def test_stats(self) -> None:
        cf = CloudFallback()
        assert cf.stats()["enabled"] is False


class TestResourceOptimizer:
    def test_snapshot(self) -> None:
        ro = ResourceOptimizer()
        snap = ro.snapshot(cpu=45.0, ram_used=8, ram_total=16)
        assert snap.ram_percent == 50.0

    def test_alerts(self) -> None:
        ro = ResourceOptimizer()
        ro.snapshot(cpu=95.0, ram_used=15, ram_total=16, gpu=92)
        alerts = ro.alerts()
        assert len(alerts) == 3  # CPU + RAM + GPU

    def test_recommendations(self) -> None:
        ro = ResourceOptimizer()
        ro.snapshot(cpu=95, ram_used=15, ram_total=16, gpu=10)
        recs = ro.recommendations()
        assert any("GPU-Offloading" in r for r in recs)


class TestLatencyTracker:
    def test_record_and_percentiles(self) -> None:
        lt = LatencyTracker()
        for ms in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            lt.record(ms)
        assert lt.p50 >= 500
        assert lt.p95 >= 900
        assert 540 <= lt.avg <= 560

    def test_by_operation(self) -> None:
        lt = LatencyTracker()
        lt.record(100, "inference")
        lt.record(50, "rag")
        data = lt.by_operation("rag")
        assert data["avg"] == 50

    def test_stats(self) -> None:
        lt = LatencyTracker()
        lt.record(100)
        assert lt.stats()["total_samples"] == 1


class TestPerformanceManager:
    def test_process_query(self) -> None:
        pm = PerformanceManager()
        pm.balancer.add_backend(Backend("b1", "Local", "http://local", max_concurrent=4))
        result = pm.process_query("Wie viel kostet eine BU?")
        assert result["backend"] == "Local"
        assert result["latency_ms"] >= 0

    def test_vector_rag(self) -> None:
        pm = PerformanceManager()
        pm.vector_store.add("BU kostet 50€/Monat", [1.0, 0.5, 0.0])
        pm.balancer.add_backend(Backend("b1", "X", "http://x", max_concurrent=4))
        result = pm.process_query("BU Preis?", [1.0, 0.4, 0.1])
        assert result["rag_results"] == 1

    def test_health(self) -> None:
        pm = PerformanceManager()
        health = pm.health()
        assert "vector_store" in health
        assert "balancer" in health
        assert "latency" in health
