"""Quality-gate tests — run on dedicated CI jobs (mutation, soak, chaos).

These are intentionally NOT part of the default `pytest tests/` invocation
because they're heavyweight (mutation testing alone is hours-long) and
non-deterministic in throughput. The standard PR pipeline runs unit + integration
tests; these run nightly / weekly on dedicated workflows.
"""
