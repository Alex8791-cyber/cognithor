"""Task 20 — Public API stability test.

Guards against accidental breakage / pollution of the ``cognithor.crew``
public surface. New symbols must be added to ``__all__`` AND to the
required set below. Removed symbols must be deprecated via a semver bump.
"""

from __future__ import annotations


def test_top_level_reexports_match_subpackage():
    """`cognithor.Crew` must be identical to `cognithor.crew.Crew`."""
    from cognithor import Crew as TopCrew
    from cognithor.crew import Crew as PkgCrew

    assert TopCrew is PkgCrew


def test_frozen_public_surface():
    """Guard against accidental public-API additions without a version bump."""
    from cognithor import crew as m

    public = {n for n in dir(m) if not n.startswith("_")}
    required = {
        "Crew",
        "CrewAgent",
        "CrewTask",
        "CrewProcess",
        "CrewOutput",
        "TaskOutput",
        "TokenUsageDict",
        "LLMConfig",
        "GuardrailFailure",
        "ToolNotFoundError",
        "CrewError",
        "CrewCompilationError",
        "GuardrailCallable",
    }
    missing = required - public
    assert not missing, f"Missing exports: {missing}"
