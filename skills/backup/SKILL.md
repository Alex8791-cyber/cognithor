---
name: backup
description: "Automated hourly backup skill that creates timestamped archives of Cognithor config files, skill definitions, and data directories on a cron schedule. Use when setting up automated backups, restoring from snapshots, scheduling data protection, or configuring backup retention policies."
---

# Backup

Automated backup skill that creates timestamped archives of Cognithor configuration, skills, and data on a cron schedule (`0 * * * *` — hourly by default).

## Steps

1. **Check prerequisites** — verify backup target exists and has write access:
   ```bash
   BACKUP_DIR="${COGNITHOR_HOME:-$HOME/.cognithor}/backups"
   mkdir -p "$BACKUP_DIR" && test -w "$BACKUP_DIR"
   ```
2. **Create timestamped archive** of config, skills, and data:
   ```bash
   STAMP=$(date +%Y%m%d_%H%M%S)
   tar czf "$BACKUP_DIR/cognithor_$STAMP.tar.gz" \
     -C "${COGNITHOR_HOME:-$HOME/.cognithor}" \
     config/ skills/ data/
   ```
3. **Validate integrity** — verify the archive is readable:
   ```bash
   tar tzf "$BACKUP_DIR/cognithor_$STAMP.tar.gz" > /dev/null
   ```
4. **Prune old backups** — keep the last 24 hourly snapshots:
   ```bash
   ls -1t "$BACKUP_DIR"/cognithor_*.tar.gz | tail -n +25 | xargs rm -f
   ```
5. **Return status** — the skill returns `{"status": "ok", "automated": true}` on success

## Example

```
User > Starte ein Backup meiner Cognithor-Daten
Cognithor > Backup erstellt: cognithor_20260420_140000.tar.gz (12 MB)
         Nächstes automatisches Backup: 15:00 Uhr
```

## Error Handling

- **Disk full**: Log warning, skip archive creation, alert user to free space
- **Permission denied**: Verify `$BACKUP_DIR` ownership and retry with correct permissions
- **Corrupt archive**: Re-run backup immediately; if persistent, check filesystem health
