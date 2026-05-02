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
from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    ChangeDetector,
    EpisodeMemory,
    EpisodeStep,
    FrameChange,
    StuckDetector,
    count_actions,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import (
    ClampPolicy,
    FrameBridge,
)
from cognithor.channels.program_synthesis.arc_agi3.protocol import (
    FrameDataProtocol,
    GameActionProtocol,
    GameStateProtocol,
)

__all__ = [
    "ActionDecoder",
    "ChangeDetector",
    "ClampPolicy",
    "CognithorPSEAgent",
    "EpisodeMemory",
    "EpisodeStep",
    "FrameBridge",
    "FrameChange",
    "FrameDataProtocol",
    "GameActionProtocol",
    "GameStateProtocol",
    "RandomActionAgent",
    "StuckDetector",
    "UniformActionDecoder",
    "count_actions",
]
