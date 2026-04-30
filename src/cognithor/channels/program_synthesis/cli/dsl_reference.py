# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""DSL reference Markdown generator.

Reads :data:`REGISTRY` and produces a single Markdown document with
one row per primitive, grouped by output type. The output is
deterministic so it can be checked in and verified by drift checks
(``verify_readme_claims.py`` style).

Wired into the CLI as ``pse dsl reference --output <path>``; if no
output is given, prints to stdout.
"""

from __future__ import annotations

from collections import defaultdict
from typing import IO

from cognithor.channels.program_synthesis.core.version import (
    DSL_VERSION,
    PSE_VERSION,
)
from cognithor.channels.program_synthesis.dsl.predicates import (
    PREDICATE_CONSTRUCTORS,
)
from cognithor.channels.program_synthesis.dsl.registry import (
    REGISTRY,
    PrimitiveSpec,
)

# Order primitives appear in the generated reference. Anything not in
# this list lands in the "Other" group at the bottom.
GROUP_ORDER: tuple[str, ...] = (
    "Grid",
    "Color",
    "Mask",
    "Object",
    "ObjectSet",
    "Lambda",
    "Bool",
    "Int",
    "AlignMode",
    "SortKey",
)


def _group_for(spec: PrimitiveSpec) -> str:
    """Map output_type → display group."""
    return spec.signature.output


def _format_signature(spec: PrimitiveSpec) -> str:
    if not spec.signature.inputs:
        return f"() → {spec.signature.output}"
    return f"({', '.join(spec.signature.inputs)}) → {spec.signature.output}"


def _format_primitive_row(spec: PrimitiveSpec) -> str:
    """One Markdown table row."""
    desc = (spec.description or "").replace("|", "\\|").replace("\n", " ")
    return f"| `{spec.name}` | `{_format_signature(spec)}` | {spec.cost:.2f} | {desc} |"


def _format_examples(spec: PrimitiveSpec) -> str:
    if not spec.examples:
        return ""
    lines = []
    for inp, out in spec.examples:
        lines.append(f"  - `{inp}` → `{out}`")
    return "\n".join(lines)


def render_dsl_reference() -> str:
    """Build the full Markdown document.

    The output is deterministic across runs — primitives are sorted
    by (group_index, name); examples render verbatim.
    """
    grouped: dict[str, list[PrimitiveSpec]] = defaultdict(list)
    for spec in REGISTRY.all_primitives():
        grouped[_group_for(spec)].append(spec)
    for specs in grouped.values():
        specs.sort(key=lambda s: s.name)

    out: list[str] = []
    out.append("# Cognithor PSE — ARC-DSL Reference")
    out.append("")
    out.append(f"_Auto-generated. PSE version `{PSE_VERSION}`, DSL version `{DSL_VERSION}`._")
    out.append("")
    total = len(REGISTRY)
    out.append(
        f"**{total} primitives** registered, plus {len(PREDICATE_CONSTRUCTORS)} "
        "predicate constructors and the closed Lambda / AlignMode / "
        "SortKey enums."
    )
    out.append("")
    out.append(
        "Run `cognithor pse dsl describe <name>` for any primitive to see "
        "its full record (signature + cost + description + examples)."
    )
    out.append("")
    out.append("## Catalog")
    out.append("")

    rendered_groups: list[str] = []
    for group in GROUP_ORDER:
        specs = grouped.pop(group, [])
        if not specs:
            continue
        rendered_groups.append(group)
        out.append(f"### Output type: `{group}`")
        out.append("")
        out.append("| Name | Signature | Cost | Description |")
        out.append("|---|---|---|---|")
        for spec in specs:
            out.append(_format_primitive_row(spec))
        out.append("")

    # Anything left over goes under "Other".
    other = sorted(grouped.items())
    if other:
        out.append("### Output type: other")
        out.append("")
        out.append("| Name | Signature | Cost | Description |")
        out.append("|---|---|---|---|")
        for _, specs in other:
            for spec in sorted(specs, key=lambda s: s.name):
                out.append(_format_primitive_row(spec))
        out.append("")

    # Predicate constructor list — closed set, useful reference.
    out.append("## Predicate constructors (closed set)")
    out.append("")
    out.append(
        "Higher-order primitives like `filter_objects` accept a "
        "`Predicate` argument. The constructor names below are the only "
        "predicates the search engine may construct (free Python "
        "lambdas are forbidden — sandbox guarantee, see spec §6.4)."
    )
    out.append("")
    out.append("| Constructor | Arity | Notes |")
    out.append("|---|---|---|")
    for name, arity in sorted(PREDICATE_CONSTRUCTORS.items()):
        notes = ""
        if name in {"not", "and", "or"}:
            notes = "combinator"
        elif name in {"is_largest_in", "is_smallest_in"}:
            notes = "needs ObjectSet context"
        elif name == "touches_border":
            notes = "needs grid_shape context"
        out.append(f"| `{name}` | {arity} | {notes} |")
    out.append("")

    return "\n".join(out) + "\n"


def write_dsl_reference(path: str) -> int:
    """Write the rendered document to *path*. Returns byte count."""
    body = render_dsl_reference()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    return len(body.encode("utf-8"))


def cmd_dsl_reference(stream: IO[str], output: str | None) -> int:
    """``pse dsl reference [--output PATH]`` — emit the reference."""
    body = render_dsl_reference()
    if output:
        with open(output, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        print(f"wrote {len(body)} chars to {output}", file=stream)
    else:
        print(body, file=stream)
    return 0


__all__ = [
    "cmd_dsl_reference",
    "render_dsl_reference",
    "write_dsl_reference",
]
