# Cognithor Skills Guide

> How to create, install, and manage skills.
> For the skill file format reference, see [DEVELOPER.md](DEVELOPER.md#creating-a-skill).

## What Are Skills?

Skills are Markdown files with YAML frontmatter that teach Cognithor *how* to
handle specific types of requests. When a user message matches a skill's
trigger keywords, the skill body is injected into the Planner's context as
instructions.

Skills are **not code** — they are natural language procedures that guide the
Planner's reasoning. The Planner then uses MCP tools to execute the steps.

---

## Built-in Skills

| Skill | Category | Priority | Triggers |
|-------|----------|----------|----------|
| `morgen-briefing` | productivity | 4 | Briefing, Morgen, Tagesplan, Ueberblick |
| `email-triage` | productivity | 5 | Email, Postfach, sortieren, Inbox |
| `meeting-vorbereitung` | productivity | 5 | Meeting, Besprechung, Agenda, vorbereiten |
| `meeting-protokoll` | productivity | 6 | Protokoll, Mitschrift, Action Items, Nachbereitung |
| `tages-report` | productivity | 6 | Tagesreport, Tagesbericht, was habe ich gemacht, EOD |
| `todo-management` | productivity | 5 | Aufgabe, Todo, Liste, Planung |
| `marketplace-monitor` | productivity | 5 | marketplace, Preis ueberwachen, Angebot, Deal |
| `web-recherche` | research | 5 | Recherche, Internet, Web-Suche |
| `kontakt-recherche` | research | 5 | Kontakt, Person, Firma, Hintergrund |
| `dokument-analyse` | analysis | 5 | Dokument, analysieren, PDF, Vertrag, zusammenfassen |
| `wissens-synthese` | analysis | 6 | Synthese, zusammenfuehren, Gesamtbild, Faktencheck |
| `projekt-setup` | development | 5 | Projekt, Setup, neues Projekt, initialisieren |
| `code-review` | development | 6 | Code Review, Pull Request, PR, Bugs finden |
| `vertrag-pruefer` | legal | 6 | Vertrag, pruefen, AGB, Klausel, Risiko, NDA |
| `workflow-recorder` | automation | 7 | Automatisierung, Ablauf aufnehmen, Skill erstellen |

---

## Creating a Skill

### 1. Choose a location

| Location | Purpose |
|----------|---------|
| `~/.jarvis/skills/` | Personal skills |
| `data/procedures/` | Built-in (ships with Cognithor) |

### 2. Create the skill file

```bash
mkdir -p ~/.jarvis/skills/my-skill
```

Create `~/.jarvis/skills/my-skill/skill.md`:

```markdown
---
name: my-skill
trigger_keywords: [Keyword1, Keyword2, "multi word trigger"]
tools_required: [web_search, write_file]
category: research
priority: 5
description: "What this skill does in one sentence"
enabled: true
---
# Skill Title

## When to Apply
Describe when the Planner should activate this skill.

## Steps
1. First step...
2. Second step...
3. Final step...

## Known Pitfalls
- Edge cases to watch for

## Quality Criteria
- How to evaluate success
```

### 3. Restart Cognithor

The SkillRegistry scans skill directories at startup. No registration code needed.

---

## Skill Matching

When a user sends a message, the SkillRegistry:

1. Extracts keywords from the message
2. Matches against each skill's `trigger_keywords`
   - Exact match (case-insensitive)
   - Fuzzy match (70% similarity threshold)
3. Scores by overlap count + success rate bonus
4. Injects the best match into Working Memory

The Planner then sees the skill body as part of its system prompt and follows
the instructions.

### Debugging Matches

Use the `list_skills` tool or check logs:
```
User > Zeige alle registrierten Skills
```

---

## YAML Frontmatter Reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | yes | — | Unique skill identifier |
| `trigger_keywords` | list | yes | — | Keywords that activate this skill |
| `tools_required` | list | no | `[]` | MCP tools the skill needs |
| `category` | string | no | `general` | Category for filtering |
| `priority` | int | no | `0` | Higher = preferred at tie |
| `description` | string | no | — | Short description |
| `enabled` | bool | no | `true` | Enable/disable |
| `model_preference` | string | no | — | Preferred LLM model |
| `agent` | string | no | — | Route to specific agent |

---

## Community Marketplace

### Browsing Skills

```
User > Suche Community-Skills zum Thema Datenanalyse
```

Or via the MCP tool:
```
search_community_skills(query="data analysis")
```

### Installing

```
User > Installiere den Community-Skill "data-analysis"
```

Or via tool:
```
install_community_skill(name="data-analysis")
```

Installed skills go to `~/.jarvis/skills/community/<name>/`.

### Security Chain

Community skills go through a 5-step validation:

1. **Syntax check** — Valid Markdown + YAML frontmatter
2. **Injection scan** — No prompt injection patterns
3. **Tool declaration** — All `tools_required` are valid MCP tools
4. **Safety check** — No dangerous patterns (file deletion, etc.)
5. **Hash verification** — SHA-256 content hash matches registry

At runtime, the **ToolEnforcer** restricts community skills to only their
declared `tools_required` — they cannot escalate to tools they didn't declare.

### Publishing Your Skills

Share your skills with the community:

```
User > Veroeffentliche meinen Skill "rechnungs-generator"
```

Or via tool:
```
publish_skill(name="rechnungs-generator")
```

This will:
1. Validate the skill locally (5 checks)
2. Compute SHA-256 content hash
3. Upload `skill.md` + `manifest.json` to the [skill-registry](https://github.com/Alex8791-cyber/skill-registry)
4. Update `registry.json` so other users can find and install it

**Requirements:**
- Skill must have `trigger_keywords` and `tools_required` in frontmatter
- GitHub credentials (via git credential manager or `GITHUB_TOKEN` env var)

### Reporting Issues

```
User > Melde den Skill "suspicious-skill" als problematisch
```

Reports are tracked by the governance system and may trigger a recall.

### Registry Structure

The [skill-registry](https://github.com/Alex8791-cyber/skill-registry) on GitHub:
```
skill-registry/
  registry.json          # Index of all available skills
  publishers.json        # Publisher profiles and trust levels
  skills/
    meeting-protokoll/
      skill.md           # Skill content (YAML frontmatter + Markdown)
      manifest.json      # Metadata + SHA-256 content hash
    code-review/
      skill.md
      manifest.json
    ...
```

---

## Tips for Good Skills

1. **Be specific** — "When the user asks about X" is better than "General helper"
2. **Use concrete tool names** — `web_search`, `write_file`, not "search the internet"
3. **Include failure modes** — What to do when a tool returns no results
4. **Set quality criteria** — How the Planner knows the task is complete
5. **Keep it focused** — One skill, one purpose. Don't try to handle everything
6. **Test your triggers** — Make sure keywords are distinctive enough to not
   overlap with other skills
