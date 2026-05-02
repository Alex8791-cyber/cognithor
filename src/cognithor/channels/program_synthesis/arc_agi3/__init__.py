# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Cognithor PSE — ARC-AGI-3 Game-Agent integration (Sprint-11).

ARC-AGI-3 is the interactive-game successor to the static
input/output ARC-AGI-1 benchmark we covered in Sprint-9/10. This
sub-package is the Cognithor side of the integration: protocol
classes that mirror the official ``arcengine`` API surface, an
abstract :class:`CognithorPSEAgent` that subclasses the official
``Agent`` ABC contract, and concrete agents that build on the
Sprint-10 DSL + the Sprint-1 LLM-Prior.

Wave-1 (foundation) ships protocol + abstract base + a smoke-baseline
``RandomActionAgent``. Subsequent waves (PR-2..6) add the
:class:`FrameBridge`, :class:`ActionDecoder`, the DSL-search agent,
and the LLM-reasoning agent.

Cognithor's code is typed against the local :mod:`.protocol` types,
not directly against ``arcengine`` — so the package imports cleanly
without ``arc-agi``/``arcengine`` installed. A thin adapter (Wave-5)
plugs in the live arcengine types when running against the official
harness.
"""

from cognithor.channels.program_synthesis.arc_agi3.action_decoder import (
    ActionDecoder,
    UniformActionDecoder,
)
from cognithor.channels.program_synthesis.arc_agi3.agent import (
    CognithorPSEAgent,
    RandomActionAgent,
)
from cognithor.channels.program_synthesis.arc_agi3.audit import (
    ArcAuditEvent,
    ArcAuditTrail,
)
from cognithor.channels.program_synthesis.arc_agi3.click_target_sampler import (
    ClickTargetSampler,
)
from cognithor.channels.program_synthesis.arc_agi3.dsl_action_decoder import (
    DSLActionDecoder,
)
from cognithor.channels.program_synthesis.arc_agi3.dsl_agent import Sprint10DSLAgent
from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    ChangeDetector,
    EpisodeMemory,
    EpisodeStep,
    FrameChange,
    StuckDetector,
    count_actions,
)
from cognithor.channels.program_synthesis.arc_agi3.fast_grid_planner import (
    Cluster,
    detect_toggle_pair,
    find_clusters,
    is_level_complete,
    plan_click_solution,
    simulate_combo,
    simulate_toggle,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_analyzer import (
    FrameAnalyzer,
    MovementInfo,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import (
    ClampPolicy,
    FrameBridge,
)
from cognithor.channels.program_synthesis.arc_agi3.game_profile import (
    GameProfile,
    StrategyMetrics,
)
from cognithor.channels.program_synthesis.arc_agi3.harness_shim import (
    cognithor_agent_factory,
)
from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
    ChoiceFn,
    FrameContext,
    LLMActionDecoder,
    render_grid,
    summarise_history,
)
from cognithor.channels.program_synthesis.arc_agi3.llm_agent import (
    LLMReasoningAgent,
    build_inprocess_vllm_choice_fn,
    build_vllm_choice_fn,
)
from cognithor.channels.program_synthesis.arc_agi3.protocol import (
    FrameDataProtocol,
    GameActionProtocol,
    GameStateProtocol,
)
from cognithor.channels.program_synthesis.arc_agi3.scorecard import (
    GameResult,
    ScorecardSummary,
    parse_scorecard,
    summarise,
)
from cognithor.channels.program_synthesis.arc_agi3.state_graph import (
    StateEdge,
    StateGraphNavigator,
    StateNode,
)

__all__ = [
    "ActionDecoder",
    "ArcAuditEvent",
    "ArcAuditTrail",
    "ChangeDetector",
    "ChoiceFn",
    "ClampPolicy",
    "ClickTargetSampler",
    "Cluster",
    "CognithorPSEAgent",
    "DSLActionDecoder",
    "EpisodeMemory",
    "EpisodeStep",
    "FrameAnalyzer",
    "FrameBridge",
    "FrameChange",
    "FrameContext",
    "FrameDataProtocol",
    "GameActionProtocol",
    "GameProfile",
    "GameResult",
    "GameStateProtocol",
    "LLMActionDecoder",
    "LLMReasoningAgent",
    "MovementInfo",
    "RandomActionAgent",
    "ScorecardSummary",
    "Sprint10DSLAgent",
    "StateEdge",
    "StateGraphNavigator",
    "StateNode",
    "StrategyMetrics",
    "StuckDetector",
    "UniformActionDecoder",
    "build_inprocess_vllm_choice_fn",
    "build_vllm_choice_fn",
    "cognithor_agent_factory",
    "count_actions",
    "detect_toggle_pair",
    "find_clusters",
    "is_level_complete",
    "parse_scorecard",
    "plan_click_solution",
    "render_grid",
    "simulate_combo",
    "simulate_toggle",
    "summarise",
    "summarise_history",
]
