# Cognithor API Reference

> Complete REST API and WebSocket protocol reference.
> For architecture overview, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Table of Contents

- [Authentication](#authentication)
- [WebSocket Protocol](#websocket-protocol)
- [Core API](#core-api)
- [Control Center](#control-center)
- [Skills Marketplace](#skills-marketplace)
- [A2A Protocol](#a2a-protocol)

---

## Authentication

### Token Acquisition

```
GET /api/v1/bootstrap
```

No authentication required. Returns a per-session Bearer token:

```json
{"token": "abc123..."}
```

### HTTP Authentication

Include the token in the `Authorization` header:

```
Authorization: Bearer <token>
```

### WebSocket Authentication

The **first message** after connecting must be an auth message:

```json
{"type": "auth", "token": "<token>"}
```

If authentication fails, the server closes the connection with code **4001**.
The client should invalidate its cached token and fetch a new one from
`/api/v1/bootstrap` before reconnecting.

### Unprotected Endpoints

- `GET /api/v1/bootstrap` — Token acquisition
- `GET /api/v1/health` — Health check
- `GET /.well-known/agent.json` — A2A agent discovery
- Skills marketplace search endpoints

---

## WebSocket Protocol

### Connection

```
ws://<host>:<port>/ws/<session_id>
```

The `session_id` is client-generated and used to resume conversations across
page navigations.

### Client → Server Messages

| Type | Fields | Description |
|------|--------|-------------|
| `auth` | `token` | First message — authenticate |
| `user_message` | `text`, `session_id`, `metadata?` | Send user message |
| `approval_response` | `id`, `approved`, `session_id` | Respond to ORANGE action |
| `ping` | — | Heartbeat (send every 30s) |

#### File Upload (via metadata)

```json
{
  "type": "user_message",
  "text": "[File: report.pdf]",
  "session_id": "...",
  "metadata": {
    "file_name": "report.pdf",
    "file_type": "application/pdf",
    "file_base64": "<base64>"
  }
}
```

#### Voice Upload (via metadata)

```json
{
  "type": "user_message",
  "text": "[Voice message]",
  "session_id": "...",
  "metadata": {
    "audio_base64": "<base64>",
    "audio_type": "audio/webm"
  }
}
```

Max upload size: 50 MB.

### Server → Client Messages

| Type | Fields | Description |
|------|--------|-------------|
| `stream_token` | `token` | Single token during streaming |
| `stream_end` | — | Streaming complete |
| `assistant_message` | `text` | Complete response (non-streaming) |
| `tool_start` | `tool`, `args` | Tool execution started |
| `tool_result` | `tool`, `data` | Tool execution finished |
| `approval_request` | `id`, `tool`, `reason`, `params` | ORANGE action needs approval |
| `status_update` | `status`, `text` | Progress update |
| `transcription` | `text` | Voice message transcribed |
| `canvas_push` | `html`, `title` | Push HTML to canvas panel |
| `canvas_reset` | — | Clear canvas |
| `canvas_eval` | `script` | Execute JS in canvas iframe |
| `error` | `message` | Error occurred |
| `pong` | — | Heartbeat response |

### Status Types

Status updates indicate which PGE phase is active:

| Status | Meaning |
|--------|---------|
| `THINKING` | Planner is reasoning |
| `SEARCHING` | Web search in progress |
| `EXECUTING` | Running a tool |
| `RETRYING` | Retrying after transient error |
| `PROCESSING` | General processing |
| `FINISHING` | Formulating response |

---

## Core API

Base URL: `http://localhost:8741`

### Health

```
GET /api/v1/health
```

No auth. Returns:

```json
{
  "status": "ok",
  "version": "0.99.0",
  "uptime_seconds": 3600,
  "active_sessions": 2
}
```

> The `version` field tracks `pyproject.toml` `[project] version`; clients use this for the major.minor compatibility check (frontend `kFrontendVersion` in `flutter_app/lib/providers/connection_provider.dart` enforces it).

### Send Message (REST)

```
POST /api/v1/message
Authorization: Bearer <token>
Content-Type: application/json

{
  "text": "Search for Python best practices",
  "session_id": "optional-session-id",
  "metadata": {}
}
```

Response:

```json
{
  "text": "Here are the top Python best practices...",
  "session_id": "web_1710000000_abc123",
  "timestamp": "2026-03-15T10:30:00Z",
  "duration_ms": 4500,
  "tools_used": ["web_search", "web_fetch"]
}
```

### Sessions

```
GET /api/v1/sessions
Authorization: Bearer <token>
```

### Approvals

```
GET  /api/v1/approvals/pending
POST /api/v1/approvals/respond

{"request_id": "...", "approved": true}
```

---

## Control Center

The Control Center API provides system management endpoints. All require
Bearer token authentication.

### System Status

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/status` | System status overview |
| GET | `/api/v1/overview` | System overview summary |
| GET | `/api/v1/agents` | List configured agents |
| GET | `/api/v1/config` | Current configuration |
| PATCH | `/api/v1/config` | Update configuration (deep merge) |
| POST | `/api/v1/config/reload` | Reload config from disk |

### Credentials

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/credentials` | List stored credentials |
| POST | `/api/v1/credentials` | Store credential |
| DELETE | `/api/v1/credentials/{service}/{key}` | Delete credential |

### Memory & Knowledge Graph

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/memory/hygiene/scan` | Run memory hygiene scan |
| GET | `/api/v1/memory/hygiene/stats` | Memory hygiene statistics |
| GET | `/api/v1/memory/integrity` | Memory integrity check |
| GET | `/api/v1/memory/graph/stats` | Knowledge graph statistics |
| GET | `/api/v1/memory/graph/entities` | List entities |
| GET | `/api/v1/memory/graph/entities/{id}/relations` | Entity relationships |

### Security & Compliance

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/security/redteam/scan` | Run red-team scan |
| GET | `/api/v1/security/redteam/status` | Red-team scan status |
| GET | `/api/v1/compliance/report` | Compliance report |
| GET | `/api/v1/compliance/decisions` | Decision audit log |
| GET | `/api/v1/compliance/stats` | Compliance statistics |
| GET | `/api/v1/explainability/trails` | Decision audit trails |

### Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/monitoring/dashboard` | Monitoring dashboard |
| GET | `/api/v1/monitoring/metrics` | All metrics |
| GET | `/api/v1/monitoring/events` | Recent events |
| GET | `/api/v1/monitoring/audit` | Audit log |
| GET | `/api/v1/monitoring/stream` | SSE live event stream |
| GET | `/metrics` | Prometheus metrics (no auth) |

### Workflows

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/workflows/templates` | List workflow templates |
| GET | `/api/v1/workflows/instances` | List running workflows |
| POST | `/api/v1/workflows/instances` | Create workflow |
| POST | `/api/v1/workflows/start` | Start workflow execution |
| GET | `/api/v1/workflows/stats` | Workflow statistics |

### System Control

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/system/status` | System status |
| POST | `/api/v1/system/start` | Start system |
| POST | `/api/v1/system/stop` | Stop system |

### Additional Endpoints

The Control Center exposes 170+ endpoints covering sandbox, isolation, RBAC,
internationalization, governance, economics, prompt evolution, performance,
cron jobs, MCP servers, and more. Use the OpenAPI schema at
`/docs` (Swagger UI) or `/redoc` for the complete interactive reference.

---

## Skills Marketplace

### Built-in Marketplace

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/skills/search?query=...` | Search skills |
| GET | `/api/v1/skills/featured` | Featured skills |
| GET | `/api/v1/skills/trending` | Trending skills |
| GET | `/api/v1/skills/categories` | List categories |
| GET | `/api/v1/skills/installed` | Installed skills |
| GET | `/api/v1/skills/{id}` | Skill details |
| POST | `/api/v1/skills/{id}/install` | Install skill |
| DELETE | `/api/v1/skills/{id}` | Uninstall skill |
| GET | `/api/v1/skills/{id}/reviews` | Skill reviews |
| POST | `/api/v1/skills/{id}/reviews` | Submit review |

### Community Marketplace

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/skills/community/search` | Search community skills |
| GET | `/api/v1/skills/community/recalls` | Active recalls |
| GET | `/api/v1/skills/community/{name}` | Skill details |
| POST | `/api/v1/skills/community/{name}/install` | Install |
| DELETE | `/api/v1/skills/community/{name}` | Uninstall |
| POST | `/api/v1/skills/community/{name}/report` | Report abuse |
| POST | `/api/v1/skills/community/sync` | Force registry sync |

---

## A2A Protocol

Agent-to-Agent protocol (RC v1.0) for inter-agent communication.

### Discovery

```
GET /.well-known/agent.json
```

Returns the agent card with capabilities, supported protocols, and metadata.

### Task Dispatch

```
POST /a2a
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "tasks/send",
  "params": {
    "task": "Summarize this document",
    "context": {}
  }
}
```

JSON-RPC 2.0 protocol. Error codes:

| Code | Meaning |
|------|---------|
| `-32700` | Parse error |
| `-32000` | Server error / rate limited |
| `-32004` | Unauthorized |
| `-32005` | Incompatible version |

### Streaming

```
POST /a2a/stream
```

Returns Server-Sent Events for long-running tasks.

### Health

```
GET /a2a/health
```

```json
{
  "status": "ok",
  "protocol_version": "1.0.0",
  "enabled": true,
  "server_running": true
}
```
