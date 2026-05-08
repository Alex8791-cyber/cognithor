"""Layer 4 — Pareto Multi-Objective Solver.

Inputs:  Manifest (L3) + Capabilities (L2) + UserObjective (user-pref).
Outputs: Pareto-optimal tuple of `Solution` objects sorted by composite score.

Determinism: same input → same output. Tie-breakers are lexical over `tier_id`.

The solver is NEVER the place to add new tiers or models — that happens in
the YAML manifest. The solver is also NEVER the place to add new
capabilities — that happens in L2. Solver is pure constraint-checking +
scoring + Pareto-filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cognithor.system.capabilities import Capabilities
from cognithor.system.manifest_models import Manifest, PricingManifest, Tier
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "OBJECTIVE_PRESETS",
    "Solution",
    "UserObjective",
    "solve",
]


# ─────────────────────────────────────────────────────────────────────────
# UserObjective
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UserObjective:
    """User's multi-objective preference. Weights should sum to ~1.0."""

    weight_quality: float = 0.4
    weight_speed: float = 0.3
    weight_cost: float = 0.2
    weight_privacy: float = 0.1
    max_disk_gb: float | None = None
    max_setup_minutes: int | None = None
    max_cloud_eur_per_month: float | None = None
    require_offline_capable: bool = False


OBJECTIVE_PRESETS: dict[str, UserObjective] = {
    "balanced": UserObjective(
        weight_quality=0.4, weight_speed=0.3, weight_cost=0.2, weight_privacy=0.1
    ),
    "quality": UserObjective(
        weight_quality=0.7, weight_speed=0.15, weight_cost=0.05, weight_privacy=0.10
    ),
    "speed": UserObjective(
        weight_quality=0.25, weight_speed=0.6, weight_cost=0.1, weight_privacy=0.05
    ),
    "privacy": UserObjective(
        weight_quality=0.25,
        weight_speed=0.15,
        weight_cost=0.1,
        weight_privacy=0.5,
        require_offline_capable=True,
    ),
    "cost": UserObjective(
        weight_quality=0.15, weight_speed=0.15, weight_cost=0.6, weight_privacy=0.1
    ),
}


# ─────────────────────────────────────────────────────────────────────────
# Solution
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Solution:
    tier_id: str
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    estimated_first_response_s: float = 0.0
    estimated_disk_gb: float = 0.0
    estimated_setup_minutes: int = 0
    estimated_cost_eur_per_month: float = 0.0
    rule_id: str = "solver.match.exact"  # for TRUST-2

    @property
    def is_immediately_runnable(self) -> bool:
        return not self.blockers


# ─────────────────────────────────────────────────────────────────────────
# Constraint-checking (capabilities ⊇ tier-requires)
# ─────────────────────────────────────────────────────────────────────────


def _check_blockers(tier: Tier, caps: Capabilities) -> tuple[str, ...]:
    blockers: list[str] = []
    for req in tier.requires_capabilities:
        if not caps.satisfies(req):
            blockers.append(req)
    return tuple(blockers)


# ─────────────────────────────────────────────────────────────────────────
# Score components — each in 0..1 range
# ─────────────────────────────────────────────────────────────────────────


def _quality_score(tier: Tier, manifest: Manifest) -> float:
    """Average quality_tier of tier's planner+coder+executor models."""
    quality_map = {"S": 0.4, "M": 0.7, "L": 0.9, "XL": 1.0}
    roles = ("planner", "coder", "executor")
    scores: list[float] = []
    for role in roles:
        model_id = getattr(tier.model_set, role)
        model = manifest.models.get(model_id)
        if model is None:
            continue
        scores.append(quality_map.get(model.quality_tier, 0.5))
    return sum(scores) / len(scores) if scores else 0.5


def _speed_score(tier: Tier) -> float:
    """Higher is better. Use planner_tok_s_p50 with reasonable normalization."""
    pe = tier.performance_estimates
    # Normalize: 200 tok/s = 1.0, 5 tok/s = 0.025
    return min(1.0, max(0.0, pe.planner_tok_s_p50 / 200.0))


def _privacy_score(tier: Tier) -> float:
    """1.0 = local-only, 0.0 = cloud-only, between for hybrid."""
    if tier.backend in ("ollama", "vllm", "lmstudio", "llama_cpp"):
        return 1.0
    if tier.backend in ("anthropic", "openai", "gemini", "groq", "deepseek", "xai"):
        return 0.0
    return 0.5


def _cost_score(tier: Tier, pricing: PricingManifest | None) -> float:
    """1.0 = no cloud cost (local), lower for higher cloud spend."""
    monthly = _estimate_monthly_cost_eur(tier, pricing)
    if monthly == 0.0:
        return 1.0
    # Diminishing penalty: €5/mo = 0.9, €20/mo = 0.6, €100/mo = 0.1
    return max(0.0, 1.0 - monthly / 110.0)


def _estimate_monthly_cost_eur(tier: Tier, pricing: PricingManifest | None) -> float:
    if pricing is None:
        return 0.0
    if tier.backend in ("ollama", "vllm", "lmstudio", "llama_cpp"):
        return 0.0  # electricity not in this scope; covered separately
    profile = pricing.default_usage_profile or {}
    rpd = int(profile.get("requests_per_day", 80))
    avg_in = int(profile.get("avg_input_tokens", 1500))
    avg_out = int(profile.get("avg_output_tokens", 800))

    provider_pricing = pricing.providers.get(tier.backend, {})
    if not provider_pricing:
        return 0.0
    # Use first model in provider — solver-level approximation
    model_pricing = next(iter(provider_pricing.values()))
    monthly_in_mtok = (rpd * 30 * avg_in) / 1_000_000
    monthly_out_mtok = (rpd * 30 * avg_out) / 1_000_000
    return (
        monthly_in_mtok * model_pricing.input_eur_per_mtok
        + monthly_out_mtok * model_pricing.output_eur_per_mtok
    )


# ─────────────────────────────────────────────────────────────────────────
# Composite score + Pareto filter
# ─────────────────────────────────────────────────────────────────────────


def _score_solution(
    tier: Tier,
    manifest: Manifest,
    caps: Capabilities,
    objective: UserObjective,
    pricing: PricingManifest | None,
) -> Solution:
    quality = _quality_score(tier, manifest)
    speed = _speed_score(tier)
    cost = _cost_score(tier, pricing)
    privacy = _privacy_score(tier)

    score = (
        objective.weight_quality * quality
        + objective.weight_speed * speed
        + objective.weight_cost * cost
        + objective.weight_privacy * privacy
    )

    blockers = _check_blockers(tier, caps)
    warnings: list[str] = []

    # Apply hard constraints
    if objective.max_disk_gb is not None and tier.estimated_disk_gb > objective.max_disk_gb:
        blockers = (*blockers, f"disk_required>{objective.max_disk_gb}gb")
    if (
        objective.max_setup_minutes is not None
        and tier.estimated_setup_minutes > objective.max_setup_minutes
    ):
        warnings.append(f"setup_time>{objective.max_setup_minutes}min")
    if objective.require_offline_capable and tier.backend not in (
        "ollama",
        "vllm",
        "lmstudio",
        "llama_cpp",
    ):
        blockers = (*blockers, "offline_capable_required")

    monthly = _estimate_monthly_cost_eur(tier, pricing)
    if (
        objective.max_cloud_eur_per_month is not None
        and monthly > objective.max_cloud_eur_per_month
    ):
        blockers = (*blockers, f"cost>{objective.max_cloud_eur_per_month}eur")

    return Solution(
        tier_id=tier.id,
        score=score,
        score_breakdown={
            "quality": quality,
            "speed": speed,
            "cost": cost,
            "privacy": privacy,
        },
        blockers=blockers,
        warnings=tuple(warnings),
        estimated_first_response_s=tier.performance_estimates.first_token_ms_p50 / 1000.0,
        estimated_disk_gb=tier.estimated_disk_gb,
        estimated_setup_minutes=tier.estimated_setup_minutes,
        estimated_cost_eur_per_month=monthly,
        rule_id="solver.match.exact" if not blockers else "solver.match.with_blockers",
    )


def _pareto_filter(solutions: list[Solution]) -> list[Solution]:
    """Remove dominated solutions. A solution is dominated if another beats
    it on every breakdown component."""
    # Only score Pareto-frontier among NON-blocked solutions
    runnable = [s for s in solutions if s.is_immediately_runnable]
    blocked = [s for s in solutions if not s.is_immediately_runnable]

    pareto: list[Solution] = []
    for s in runnable:
        dominated = False
        for other in runnable:
            if other.tier_id == s.tier_id:
                continue
            if _dominates(other, s):
                dominated = True
                break
        if not dominated:
            pareto.append(s)
    # Always include the best blocked solution (so user sees what's possible
    # if blockers were resolved)
    if blocked:
        blocked_sorted = sorted(blocked, key=lambda s: (-s.score, s.tier_id))
        pareto.append(blocked_sorted[0])
    return pareto


def _dominates(a: Solution, b: Solution) -> bool:
    """True iff `a` is at-least-as-good in every breakdown AND strictly
    better in at least one."""
    keys = ("quality", "speed", "cost", "privacy")
    at_least = all(a.score_breakdown.get(k, 0) >= b.score_breakdown.get(k, 0) for k in keys)
    strictly = any(a.score_breakdown.get(k, 0) > b.score_breakdown.get(k, 0) for k in keys)
    return at_least and strictly


# ─────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────


def solve(
    manifest: Manifest,
    capabilities: Capabilities,
    objective: UserObjective,
    *,
    pricing: PricingManifest | None = None,
    max_solutions: int = 5,
) -> tuple[Solution, ...]:
    """Compute Pareto-optimal solutions sorted by composite score.

    Determinism guarantee: identical inputs produce identical outputs.
    Tie-breaker: lexical over tier_id.
    """
    all_solutions = [
        _score_solution(t, manifest, capabilities, objective, pricing) for t in manifest.tiers
    ]

    # Pareto-filter then sort by score (desc) with tier_id as tie-breaker (asc)
    pareto = _pareto_filter(all_solutions)
    pareto.sort(key=lambda s: (-s.score, s.tier_id))

    # If no solution at all matches, fall back to the cloud-only / lowest-bar tier
    if not pareto:
        log.warning("solver_no_pareto_emit_fallback")
        all_solutions.sort(key=lambda s: (-s.score, s.tier_id))
        return tuple(all_solutions[:max_solutions])

    return tuple(pareto[:max_solutions])
