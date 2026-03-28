"""Autonomous Evolution Engine — self-improving idle-time learning."""

from jarvis.evolution.deep_learner import DeepLearner
from jarvis.evolution.idle_detector import IdleDetector
from jarvis.evolution.loop import EvolutionLoop
from jarvis.evolution.models import LearningPlan, SubGoal
from jarvis.evolution.resume import EvolutionResumer, ResumeState
from jarvis.evolution.strategy_planner import StrategyPlanner

__all__ = [
    "DeepLearner",
    "EvolutionLoop",
    "EvolutionResumer",
    "IdleDetector",
    "LearningPlan",
    "ResumeState",
    "StrategyPlanner",
    "SubGoal",
]
