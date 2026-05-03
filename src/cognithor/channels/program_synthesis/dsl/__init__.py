# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ARC-DSL: typed grid-transformation primitives.

Phase 1 ships ~50 base primitives + 5 higher-order primitives + 12
predicate constructors. See spec §7 for the full catalog.
"""

from __future__ import annotations

# Sprint-22 — List/Sequence-DSL family. Reductions, reorderings, and
# bridges over both ``StringList`` (PR#1) and the new ``IntList`` type.
from cognithor.channels.program_synthesis.dsl import list_primitives as list_primitives

# Sprint-22 — Number/Int-DSL family. Brings ``Int`` to first-class
# status with arithmetic primitives plus the two bridge conversions
# (``int_to_string`` / ``string_to_int``) that connect the int-shaped
# and str-shaped families.
from cognithor.channels.program_synthesis.dsl import number_primitives as number_primitives

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
    "list_primitives",
    "number_primitives",
    "primitives",
    "regex_primitives",
    "string_primitives",
]
