---
name: gmail-sync
description: "API integration skill that fetches Gmail messages, syncs inbox labels, and processes attachments via the Gmail REST API. Use when fetching new emails, syncing inbox state, triaging unread messages, exporting mail data, or integrating Gmail into automated workflows."
---

# Gmail Sync

Fetches Gmail messages, syncs labels, and processes attachments via the Gmail REST API using `httpx`. Requires network access.

## Steps

1. **Authenticate** — obtain or refresh the OAuth2 token:
   ```python
   headers = {"Authorization": f"Bearer {access_token}"}
   ```
2. **Fetch new messages** since the last sync timestamp:
   ```python
   async with httpx.AsyncClient() as client:
       resp = await client.get(
           f"{API_BASE}/messages",
           headers=headers,
           params={"q": "after:{last_sync_epoch}"}
       )
       messages = resp.json().get("messages", [])
   ```
3. **Parse each message** — extract sender, subject, body, labels, and attachment metadata
4. **Store locally** — persist parsed data under `$COGNITHOR_HOME/data/gmail/` as JSON
5. **Return summary** — the skill returns `{"data": <parsed_response>}` with message counts

## Example

```
User > Synchronisiere meine Gmail-Inbox
Cognithor > 14 neue Nachrichten synchronisiert (3 ungelesen, 2 mit Anhängen)
         Letzte Synchronisation: 2026-04-20 14:00 UTC
```

## Error Handling

- **401 Unauthorized**: Token expired — trigger OAuth refresh flow before retrying
- **429 Rate Limited**: Back off exponentially (1s, 2s, 4s) up to 3 retries
- **Network timeout**: Log the failure, preserve last-known state, report to user
- **Empty response**: Confirm query parameters are correct; check `last_sync_epoch` is valid
