# Cognithor MCP Tools Reference

> Complete reference for all 123 MCP tools grouped by module.
> For how to add new tools, see [DEVELOPER.md](DEVELOPER.md#adding-an-mcp-tool).

## Table of Contents

- [Filesystem](#filesystem)
- [Shell](#shell)
- [Web](#web)
- [Search (Files)](#search-files)
- [Code](#code)
- [Git](#git)
- [Database](#database)
- [Memory](#memory)
- [Vault](#vault)
- [Media](#media)
- [Synthesis](#synthesis)
- [Skills](#skills)
- [Notifications](#notifications)
- [Desktop](#desktop)
- [Docker](#docker)
- [Email](#email)
- [Calendar](#calendar)
- [Charts](#charts)
- [API Hub](#api-hub)
- [Browser](#browser)
- [Background Tasks](#background-tasks)
- [Computer Use](#computer-use)
- [Deep Research](#deep-research)
- [Verified Lookup](#verified-lookup)
- [Identity](#identity)
- [Remote Shell](#remote-shell)
- [A2A Delegation](#a2a-delegation)
- [Feedback](#feedback)
- [Risk Classification](#risk-classification)

---

## Filesystem

Module: `mcp/filesystem.py`

| Tool | Description | Risk |
|------|-------------|------|
| `read_file` | Read file content with optional line range | GREEN |
| `write_file` | Create or overwrite file atomically | YELLOW |
| `edit_file` | Replace text in file (str_replace pattern) | YELLOW |
| `list_directory` | List directory contents as tree | GREEN |

### read_file

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `path` | string | yes | — | File path |
| `line_start` | int | no | 0 | First line (0-based) |
| `line_end` | int | no | -1 | Last line (-1 = end) |

### write_file

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `path` | string | yes | — | File path |
| `content` | string | yes | — | File content |

### edit_file

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `path` | string | yes | — | File path |
| `old_text` | string | yes | — | Text to find |
| `new_text` | string | yes | — | Replacement text |

### list_directory

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `path` | string | yes | — | Directory path |
| `depth` | int | no | 2 | Max recursion depth |

---

## Shell

Module: `mcp/shell.py`

| Tool | Description | Risk |
|------|-------------|------|
| `exec_command` | Execute shell command in sandbox | YELLOW |

### exec_command

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `command` | string | yes | — | Shell command |
| `working_dir` | string | no | workspace | Working directory |
| `timeout` | int | no | 30 | Timeout in seconds |

---

## Web

Module: `mcp/web.py`

| Tool | Description | Risk |
|------|-------------|------|
| `web_search` | Search the web (multi-backend fallback) | GREEN |
| `web_fetch` | Fetch and extract text from URL | GREEN |
| `search_and_read` | Combined search + fetch top results | GREEN |
| `web_news_search` | Search news with time filters | GREEN |
| `http_request` | Generic HTTP request | YELLOW |

### web_search

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | yes | — | Search query |
| `num_results` | int | no | 5 | Max results |
| `language` | string | no | de | Language code |
| `timelimit` | string | no | — | Time filter (d/w/m/y) |

### web_fetch

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | string | yes | — | URL to fetch |
| `extract_text` | bool | no | true | Extract readable text |
| `max_chars` | int | no | 20000 | Max output chars |
| `reader_mode` | bool | no | — | Force reader mode |

### search_and_read

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | yes | — | Search query |
| `num_results` | int | no | 3 | Results to fetch |
| `language` | string | no | de | Language code |
| `cross_check` | bool | no | — | Cross-check sources |

### http_request

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | string | yes | — | Request URL |
| `method` | string | no | GET | HTTP method |
| `headers` | object | no | — | Request headers |
| `body` | string | no | — | Request body |
| `timeout_seconds` | int | no | — | Timeout |

---

## Search (Files)

Module: `mcp/search_tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `search_files` | Find files by glob pattern | GREEN |
| `find_in_files` | Search file contents (regex support) | GREEN |
| `find_and_replace` | Search and replace in files | YELLOW |

### search_files

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `pattern` | string | yes | — | Glob pattern |
| `path` | string | no | workspace | Search root |
| `max_results` | int | no | 100 | Max results |

### find_in_files

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | yes | — | Search text or regex |
| `path` | string | no | workspace | Search root |
| `glob` | string | no | — | File filter |
| `max_results` | int | no | 50 | Max results |
| `context_lines` | int | no | 2 | Context lines |
| `regex` | bool | no | — | Enable regex |

### find_and_replace

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | yes | — | Search text |
| `replacement` | string | yes | — | Replacement text |
| `path` | string | no | workspace | Search root |
| `glob` | string | no | — | File filter |
| `dry_run` | bool | no | true | Preview only |
| `regex` | bool | no | — | Enable regex |

---

## Code

Module: `mcp/code_tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `run_python` | Execute Python in sandbox | YELLOW |
| `analyze_code` | Analyze code for smells and security | GREEN |

### run_python

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `code` | string | yes | — | Python code |
| `timeout` | int | no | — | Timeout (max 120s) |
| `working_dir` | string | no | — | Working directory |

### analyze_code

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `code` | string | no | — | Code string |
| `file_path` | string | no | — | File to analyze |

---

## Git

Module: `mcp/git_tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `git_status` | Show working tree status | GREEN |
| `git_diff` | Show diffs (staged/unstaged/commit) | GREEN |
| `git_log` | Show commit history | GREEN |
| `git_commit` | Stage files and create commit | YELLOW |
| `git_branch` | Branch operations | YELLOW |

### git_commit

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | yes | — | Commit message |
| `files` | array | yes | — | Files to stage |
| `amend` | bool | no | false | Amend last commit |
| `path` | string | no | workspace | Repo path |

### git_branch

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `action` | string | no | list | list/create/switch/delete |
| `name` | string | no | — | Branch name |
| `path` | string | no | workspace | Repo path |

---

## Database

Module: `mcp/database_tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `db_query` | Execute SELECT query (read-only) | GREEN |
| `db_schema` | Show database schema | GREEN |
| `db_execute` | Execute write query | ORANGE |
| `db_connect` | Test database connection | GREEN |

### db_query

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `database` | string | yes | — | Database path or URL |
| `sql` | string | yes | — | SELECT query |
| `params` | array | no | — | Query parameters |
| `limit` | int | no | — | Row limit |

---

## Memory

Module: `mcp/memory_server.py`

| Tool | Description | Risk |
|------|-------------|------|
| `search_memory` | Hybrid search across all tiers | GREEN |
| `save_to_memory` | Save to memory tier | YELLOW |
| `get_entity` | Load entity from knowledge graph | GREEN |
| `add_entity` | Create entity in knowledge graph | YELLOW |
| `add_relation` | Create relationship between entities | YELLOW |
| `get_core_memory` | Return CORE.md (identity) | GREEN |
| `get_recent_episodes` | Get recent daily log entries | GREEN |
| `search_procedures` | Search learned procedures | GREEN |
| `record_procedure_usage` | Report procedure success/failure | YELLOW |
| `memory_stats` | Memory system statistics | GREEN |

### search_memory

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | yes | — | Search query |
| `top_k` | int | no | 6 | Max results |
| `tier` | string | no | — | Filter by tier |

---

## Vault

Module: `mcp/vault.py`

| Tool | Description | Risk |
|------|-------------|------|
| `vault_save` | Create note in Knowledge Vault | YELLOW |
| `vault_search` | Full-text search notes | GREEN |
| `vault_list` | List notes with filters | GREEN |
| `vault_read` | Read individual note | GREEN |
| `vault_update` | Append to note | YELLOW |
| `vault_link` | Create link between notes | YELLOW |

---

## Media

Module: `mcp/media.py`

| Tool | Description | Risk |
|------|-------------|------|
| `media_transcribe_audio` | Audio → text (Whisper) | GREEN |
| `media_analyze_image` | Image → description (multimodal LLM) | GREEN |
| `media_extract_text` | Extract text from PDF/DOCX/TXT | GREEN |
| `media_convert_audio` | Audio format conversion (ffmpeg) | YELLOW |
| `media_image_resize` | Resize image (Pillow) | YELLOW |
| `media_tts` | Text → speech (Piper/eSpeak) | YELLOW |
| `analyze_document` | LLM-powered document analysis | GREEN |

### media_transcribe_audio

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audio_path` | string | yes | — | Audio file path |
| `language` | string | no | de | Language code |
| `model` | string | no | base | Whisper model |

Timeout: 120s

### media_analyze_image

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image_path` | string | yes | — | Image file path |
| `prompt` | string | no | — | Analysis prompt |

Timeout: 180s

---

## Synthesis

Module: `mcp/synthesis.py`

| Tool | Description | Risk |
|------|-------------|------|
| `knowledge_synthesize` | Complete knowledge synthesis | GREEN |
| `knowledge_contradictions` | Find contradictions | GREEN |
| `knowledge_timeline` | Build chronological timeline | GREEN |
| `knowledge_gaps` | Identify missing knowledge | GREEN |

---

## Skills

Module: `mcp/skill_tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `create_skill` | Create new skill | YELLOW |
| `list_skills` | List registered skills | GREEN |
| `install_community_skill` | Install from community registry | YELLOW |
| `search_community_skills` | Search community registry | GREEN |
| `report_skill` | Report problematic skill | YELLOW |

---

## Notifications

Module: `mcp/notification_tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `set_reminder` | Set future reminder | YELLOW |
| `list_reminders` | List active reminders | GREEN |
| `send_notification` | Desktop notification | YELLOW |

---

## Desktop

Module: `mcp/desktop_tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `get_clipboard` | Read clipboard | GREEN |
| `set_clipboard` | Copy text to clipboard | YELLOW |
| `screenshot_desktop` | Full desktop screenshot | GREEN |
| `screenshot_region` | Screenshot of screen region | GREEN |

---

## Docker

Module: `mcp/docker_tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `docker_ps` | List containers | GREEN |
| `docker_logs` | Get container logs | GREEN |
| `docker_inspect` | Inspect container/image | GREEN |
| `docker_run` | Start container | ORANGE |
| `docker_stop` | Stop container | ORANGE |
| `docker_remove` | Remove container | ORANGE |
| `docker_exec` | Execute command in container | ORANGE |

---

## Email

Module: `mcp/email_tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `email_read_inbox` | Read inbox emails | GREEN |
| `email_search` | Search emails | GREEN |
| `email_send` | Send email via SMTP | ORANGE |
| `email_summarize` | Summarize inbox | GREEN |

---

## Calendar

Module: `mcp/calendar_tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `calendar_today` | Today's events | GREEN |
| `calendar_upcoming` | Upcoming events | GREEN |
| `calendar_create_event` | Create event | YELLOW |
| `calendar_check_availability` | Find free slots | GREEN |

---

## Charts

Module: `mcp/chart_tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `create_chart` | Create chart (bar/line/pie/scatter) | YELLOW |
| `create_table_image` | Render table as PNG | YELLOW |
| `chart_from_csv` | Create chart from CSV file | YELLOW |

---

## API Hub

Module: `mcp/api_hub.py`

| Tool | Description | Risk |
|------|-------------|------|
| `api_list` | List configured APIs | GREEN |
| `api_connect` | Configure API integration | YELLOW |
| `api_call` | Make authenticated API request | YELLOW |
| `api_disconnect` | Remove API integration | YELLOW |

---

## Browser

Module: `mcp/browser.py`

| Tool | Description | Risk |
|------|-------------|------|
| `browse_url` | Navigate to URL | YELLOW |
| `browse_screenshot` | Screenshot current page | GREEN |
| `browse_click` | Click element by selector | YELLOW |
| `browse_fill` | Fill form field | YELLOW |
| `browse_execute_js` | Execute JavaScript | ORANGE |
| `browse_page_info` | Get page metadata | GREEN |

---

## A2A Delegation

Module: `gateway/phases/tools.py`

| Tool | Description | Risk |
|------|-------------|------|
| `list_remote_agents` | List registered remote agents | GREEN |
| `delegate_to_remote_agent` | Send task to remote agent | YELLOW |

---

## Risk Classification

The Gatekeeper classifies every tool call into four risk levels:

| Level | Behavior | Examples |
|-------|----------|---------|
| **GREEN** | Execute immediately | `read_file`, `web_search`, `get_entity` |
| **YELLOW** | Execute + notify user | `write_file`, `exec_command`, `save_to_memory` |
| **ORANGE** | User must approve first | `email_send`, `docker_run`, `db_execute` |
| **RED** | Blocked, logged | Destructive patterns, path violations |

Unknown tools default to **ORANGE**. When adding new tools, register them in
`core/gatekeeper.py` → `_classify_risk()`.

### Timeouts

| Tool | Timeout |
|------|---------|
| `media_analyze_image` | 180s |
| `media_transcribe_audio`, `media_extract_text`, `media_tts` | 120s |
| `run_python` | 120s |
| All others | 30s |

### Output Limits

All tool output is capped at **50 KB** per call.

---

## Background Tasks

Long-running shell commands with monitoring. Introduced in v0.55.0.

| Tool | Risk | Description |
|------|------|-------------|
| `start_background` | YELLOW | Start a shell command in the background. Returns job_id for monitoring. |
| `list_background_jobs` | GREEN | List all background jobs (active and recent completed). |
| `check_background_job` | GREEN | Check status + last 20 lines output of a background job. |
| `read_background_log` | GREEN | Read log with tail/head/offset/limit/grep support. |
| `stop_background_job` | YELLOW | Stop a running background job (SIGTERM, then SIGKILL). |
| `wait_background_job` | GREEN | Wait for a background job to complete (with timeout). |

**ProcessMonitor** polls active jobs every 30 seconds with 5 verification methods:
1. Process-alive check
2. Exit-code detection
3. Output-stall detection (no new output for 2 intervals)
4. Timeout enforcement
5. Resource check (optional, via psutil)

## Computer Use

Desktop automation via screenshots and coordinate clicking. Requires `pip install cognithor[desktop]`. Disabled by default — enable via `config.tools.computer_use_enabled: true`.

| Tool | Risk | Description |
|------|------|-------------|
| `computer_screenshot` | GREEN | Take desktop screenshot, describe visible UI elements with coordinates. |
| `computer_click` | GREEN | Click at specific pixel coordinates (left/right/middle, single/double). |
| `computer_type` | GREEN | Type text using keyboard. Unicode via clipboard fallback. |
| `computer_hotkey` | GREEN | Press keyboard shortcut (e.g., ctrl+c, alt+tab). |
| `computer_scroll` | GREEN | Scroll at specific coordinates (up/down). |
| `computer_drag` | GREEN | Drag from start to end coordinates. |

## Deep Research

Perplexity-style iterative search with multi-source verification.

| Tool | Risk | Description |
|------|------|-------------|
| `deep_research` | GREEN | Multi-source search with fact consensus (5+ sources). |
| `deep_research_v2` | GREEN | Up to 25 search rounds with query decomposition, cross-verification, and confidence scoring. |

## Verified Lookup

Multi-agent fact verification.

| Tool | Risk | Description |
|------|------|-------------|
| `verified_web_lookup` | GREEN | Cross-check a claim against 3+ independent sources. Returns verdict + confidence. |

## Identity

Cognitive identity layer (Immortal Mind Protocol).

| Tool | Risk | Description |
|------|------|-------------|
| `identity_recall` | GREEN | Recall a memory from the identity layer. |
| `identity_state` | GREEN | Get current cognitive state. |
| `identity_reflect` | GREEN | Trigger self-reflection. |
| `identity_dream` | GREEN | Trigger dream-mode consolidation. |

## Remote Shell

SSH to remote hosts. Requires `pip install cognithor[remote]`.

| Tool | Risk | Description |
|------|------|-------------|
| `remote_shell` | ORANGE | Execute command on remote host via SSH. |
| `remote_list_hosts` | GREEN | List configured SSH hosts. |
| `remote_test_connection` | GREEN | Test SSH connection to a host. |

## Feedback

User satisfaction tracking. Introduced in v0.56.0.

| Tool | Risk | Description |
|------|------|-------------|
| POST `/api/v1/feedback` | — | Submit thumbs up/down rating for a message. |
| PATCH `/api/v1/feedback/{id}` | — | Add comment to existing feedback (follow-up). |
| GET `/api/v1/feedback/stats` | — | Feedback statistics (satisfaction rate, counts). |
| GET `/api/v1/feedback/recent` | — | Recent feedback entries. |

Note: Feedback is submitted via WebSocket (`type: "feedback"`) from the Flutter chat UI, not as MCP tools.

## Skills (updated v0.56.0)

| Tool | Risk | Description |
|------|------|-------------|
| `create_skill` | YELLOW | Create a new skill locally (Markdown + YAML frontmatter). |
| `list_skills` | GREEN | List all registered skills (builtin + personal + community). |
| `install_community_skill` | YELLOW | Install a skill from the community registry with 5-stage validation. |
| `search_community_skills` | GREEN | Search the community registry by query and category. |
| `report_skill` | YELLOW | Report a community skill for abuse. |
| `publish_skill` | YELLOW | Publish a local skill to the community registry on GitHub. |
