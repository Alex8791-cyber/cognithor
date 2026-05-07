"""Schema migration round-trip tests — Sprint 2.2.

For each module that owns a SQLite database, the round-trip test:

1. Creates an empty DB on the *current* schema.
2. Inserts a fixture row.
3. Roll backward to the previous schema (if any).
4. Roll forward to the latest.
5. Asserts the fixture row survives lossless.

Forward-only is the easy part. The interesting test is **forward then
backward then forward again** — the round-trip catches "backward
forgets to drop a column" or "forward forgets to migrate data into
the new column". Both are common production-killers.
"""
