"""``AstDomain`` — wires AST verifier into the Sprint-26.1 ``DomainRegistry``.

The AST domain has no primitive catalog — it relies on the LLM-prior
to emit valid Python. The catalog method returns a stub object so
the Domain Protocol stays satisfied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.domains.ast_dsl.types import (
    AST_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.ast_dsl.verifier import (
    AstVerifier,
)
from cognithor.channels.program_synthesis.domains.base import (
    DomainCapability,
    DomainMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from cognithor.channels.program_synthesis.domains.registry import (
        DomainRegistry,
    )


AST_DOMAIN_METADATA = DomainMetadata(
    name="ast",
    display_name="AST / Code",
    description=(
        "Synthesises Python function bodies from (input_args, output) "
        "examples. Verifier executes the function in a subprocess "
        "sandbox (timeout=2s, memory=128MB, no-net advisory)."
    ),
    capabilities=frozenset(
        {
            DomainCapability.SYNTHESISE,
            DomainCapability.EXECUTE,
            DomainCapability.SANDBOX,
            DomainCapability.PROPERTY,
        }
    ),
    type_tags=AST_TYPE_TAGS,
    benchmark_name="humaneval-plus",
    benchmark_target=0.45,
    few_shot_bank_path="prompts/pse/ast/examples.jsonl",
)


class _NullCatalog:
    """Stub catalog — AST has no enumerated primitives at synthesis time."""

    def get(self, name: str) -> object:
        msg = f"AST domain has no enumerable primitives ({name!r})"
        raise KeyError(msg)

    @staticmethod
    def names() -> list[str]:
        return []

    def __len__(self) -> int:
        return 0

    def __contains__(self, name: object) -> bool:
        return False


class AstDomain:
    """:class:`Domain` implementation for AST/Code synthesis."""

    def __init__(self) -> None:
        self._catalog = _NullCatalog()
        self._verifier = AstVerifier()

    @property
    def metadata(self) -> DomainMetadata:
        return AST_DOMAIN_METADATA

    def primitives(self) -> _NullCatalog:
        return self._catalog

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        return self._verifier.verify(program, examples)


def register_ast_domain(registry: DomainRegistry) -> None:
    """Register :class:`AstDomain` with ``registry``."""
    registry.register(AST_DOMAIN_METADATA, lambda: AstDomain())
