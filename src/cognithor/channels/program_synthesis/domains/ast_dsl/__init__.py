"""AST/Code synthesis domain (Sprint-26.3).

Synthesises Python function bodies from (input_args, output) examples.
The verifier executes synthesised functions in a subprocess sandbox
with hard timeout + memory + no-network constraints (HumanEval-Plus
target benchmark requires real execution).
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.domains.ast_dsl.domain import (
    AstDomain,
    register_ast_domain,
)
from cognithor.channels.program_synthesis.domains.ast_dsl.sandbox import (
    SandboxConfig,
    SandboxResult,
    run_in_sandbox,
)
from cognithor.channels.program_synthesis.domains.ast_dsl.types import (
    AST_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.ast_dsl.verifier import (
    AstVerifier,
    AstVerifierError,
)

__all__ = [
    "AST_TYPE_TAGS",
    "AstDomain",
    "AstVerifier",
    "AstVerifierError",
    "SandboxConfig",
    "SandboxResult",
    "register_ast_domain",
    "run_in_sandbox",
]
