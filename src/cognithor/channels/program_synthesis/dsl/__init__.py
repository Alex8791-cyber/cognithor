# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ARC-DSL: typed grid-transformation primitives.

Phase 1 ships ~50 base primitives + 5 higher-order primitives + 12
predicate constructors. See spec §7 for the full catalog.
"""

from __future__ import annotations

# Importing the primitives module has the side effect of registering all
# primitives into the module-level REGISTRY. The re-export keeps the import
# from being flagged as unused.
from cognithor.channels.program_synthesis.dsl import primitives as primitives

# Sprint-22 — Regex/Pattern-DSL family. Layered on top of the String family
# with character-class filters, run extraction, pattern extraction (email /
# URL), and slug rewrites.
from cognithor.channels.program_synthesis.dsl import regex_primitives as regex_primitives

# Sprint-22 — String-DSL family. Same import-side-effect pattern: pulling
# the module triggers ``@primitive`` decoration on every string primitive,
# which adds them to the singleton REGISTRY. Type filtering at search time
# keeps grid and string searches disjoint (a Grid InputRef never matches a
# String-input primitive's signature).
from cognithor.channels.program_synthesis.dsl import string_primitives as string_primitives
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.dsl.signatures import Signature
from cognithor.channels.program_synthesis.dsl.types_grid import Mask, Object, ObjectSet

__all__ = [
    "REGISTRY",
    "Mask",
    "Object",
    "ObjectSet",
    "Signature",
    "primitives",
    "regex_primitives",
    "string_primitives",
]
