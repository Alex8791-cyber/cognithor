# Crew-Traces REST API

Endpoints for reading the cognithor.crew Hashline-Guard audit chain.

All endpoints require a valid owner token via `X-Cognithor-User` header.
Non-owner requests return HTTP 403 with body `{"detail": {"error": "owner_only", ...}}`.

## `GET /api/crew/traces`

List all traces with derived metadata, newest first.

**Query parameters:**
- `status` (string, optional) — filter by `running` / `completed` / `failed`.
- `since` (ISO-8601 timestamp, optional) — only traces started at-or-after this time.
- `limit` (int, default 50, max 1000) — max number of traces returned.

**Response:**

```json
{
  "traces": [
    {
      "trace_id": "abc...",
      "status": "running",
      "started_at": "2026-04-26T10:00:00Z",
      "ended_at": null,
      "duration_ms": null,
      "n_tasks": 4,
      "total_tokens": 1213,
      "agent_count": 1,
      "n_failed_guardrails": 0
    }
  ],
  "meta": {"skipped_lines": 0}
}
```

`meta.skipped_lines` is the count of corrupt JSONL lines encountered during the read.

## `GET /api/crew/trace/{trace_id}`

Return the full event list for one trace.

**Response:**

```json
{
  "trace_id": "abc...",
  "events": [
    {"hash": "...", "timestamp": "...", "session_id": "abc...", "event_type": "crew_kickoff_started", "details": {...}}
  ],
  "meta": {"skipped_lines": 0}
}
```

**404 response:** `{"detail": {"error": "trace_not_found", "trace_id": "..."}}`

## `GET /api/crew/trace/{trace_id}/stats`

Return derived aggregates for the Stats Sidebar.

**Response:**

```json
{
  "total_tokens": 4612,
  "total_duration_ms": 23010.0,
  "agent_breakdown": {"researcher": 1213, "analyzer": 3399},
  "guardrail_summary": {"pass": 2, "fail": 1, "retries": 1}
}
```

## Source

All endpoints read from `~/.cognithor/logs/audit.jsonl` (Hashline-Guard chain).
Corrupt lines are silently skipped; the count is surfaced via `meta.skipped_lines`.
