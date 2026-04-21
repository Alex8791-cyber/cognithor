---
name: test
description: "Diagnostic smoke-test skill that validates Cognithor's skill loading, SkillRegistry registration, and Planner-Gatekeeper-Executor pipeline. Use when running a framework health check, verifying skill setup works end-to-end, debugging skill registration, or smoke-testing after configuration changes."
---

# Test

Diagnostic smoke-test that sends a probe through the full Planner → Gatekeeper → Executor pipeline and returns a status response.

## Steps

1. **Invoke the skill** with any input:
   ```
   cognithor test <eingabe>
   ```
2. **Planner** receives the input and identifies this skill via SkillRegistry matching
3. **Gatekeeper** validates the request against the skill's empty permission set (always passes)
4. **Executor** runs `TestSkill.execute(params)` and returns:
   ```json
   {"status": "ok", "result": "TODO"}
   ```
5. **Verify success** — confirm the response contains `"status": "ok"`

## Example

```
User > cognithor test Hallo Welt
Cognithor > {"status": "ok", "result": "TODO"}

# Healthy pipeline — skill loaded, registered, and executed successfully
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Skill not found | Not registered in SkillRegistry | Restart Cognithor to re-scan `skills/` |
| Import error | `cognithor.skills.base` missing | Verify installation: `pip install -e ".[all]"` |
| No response | Executor timeout | Check logs at `$COGNITHOR_HOME/logs/` for stack traces |

## Notes

This is a diagnostic-only skill — the `"result": "TODO"` placeholder confirms the pipeline works without performing real work.
