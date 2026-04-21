---
name: test-skill
description: "Extended diagnostic skill that validates the full Cognithor skill lifecycle — SkillRegistry scanning, keyword matching (exact + fuzzy at 70% threshold), and async execution. Use when testing skill discovery, debugging keyword matching accuracy, validating the SkillRegistry scan cycle, or verifying async execute() behavior."
---

# Test Skill

Extended diagnostic that validates the full skill lifecycle: directory scan → SkillRegistry registration → keyword matching → async `execute()`.

## Steps

1. **Trigger the skill** — Cognithor must match via keyword similarity:
   ```
   cognithor test_skill <eingabe>
   ```
2. **Verify registration** — confirm the skill appears in the registry:
   ```
   User > Zeige alle registrierten Skills
   # Expected: test_skill listed with NAME="test_skill", VERSION="0.1.0"
   ```
3. **Check keyword matching** — the SkillRegistry uses exact match (case-insensitive) then fuzzy match (70% threshold). Verify the input routes to this skill, not the sibling `test` skill
4. **Inspect the response** — `execute()` returns:
   ```json
   {"status": "ok", "result": "TODO"}
   ```
5. **Confirm async execution** — the skill runs via `async def execute(self, params)`, so verify no blocking calls appear in logs

## Example

```
User > cognithor test_skill Prüfe den Lifecycle
Cognithor > {"status": "ok", "result": "TODO"}

# Success: skill was discovered, matched, and executed asynchronously
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Routes to `test` instead | Fuzzy match overlap | Use the full slug `test_skill` to force exact match |
| `ModuleNotFoundError` | `__init__.py` missing | Verify `skills/test_skill/__init__.py` exists |
| Timeout on execute | Event loop blocked | Check for synchronous calls in `skill.py` |

## Notes

Diagnostic-only skill — `"result": "TODO"` is intentional. Differs from `test` by validating keyword disambiguation and the full scan-to-execute lifecycle.
