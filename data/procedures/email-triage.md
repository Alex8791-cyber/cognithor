---
name: email-triage
trigger_keywords: [Email, E-Mail, Postfach, sortieren, triage, Eingang, Inbox]
tools_required: [read_file, write_file]
success_count: 0
failure_count: 0
total_uses: 0
avg_score: 0.0
last_used: null
learned_from: [initial-setup]
---
# E‑Mail‑Triage

## Wann anwenden
Wenn der Benutzer sein lokales E‑Mail‑Postfach schnell sortieren oder priorisieren möchte.
Typische Trigger: „Sortiere meine E‑Mails“, „E‑Mail‑Triage durchführen“, „Posteingang aufräumen“.

## Voraussetzungen
- Lokale Datei oder Verzeichnis mit den E‑Mail‑Dateien (z. B. mbox, EML)
- Kategorien (z. B. Wichtig, Normal, Spam). Falls nicht angegeben, Standardkategorien verwenden.

## Ablauf
1. **Pfad abfragen** — Den Benutzer nach dem Speicherort seiner E‑Mails fragen, falls nicht bekannt.
2. **E‑Mails einlesen** — Das Tool `read_file` nutzen, um die E‑Mails einzulesen. Bei Verzeichnissen rekursiv alle Dateien einlesen.
3. **Kategorisieren** — Jede E‑Mail analysieren (Absender, Betreff, Keywords). Kategorien zuweisen (z. B. „Wichtig“ bei bekannten Absendern oder dringenden Begriffen; „Spam“ bei unerwünschten Absendern).
4. **Zusammenfassung erstellen** — Anzahl der E‑Mails pro Kategorie und eine Liste der wichtigsten Betreffzeilen.
5. **Ergebnis speichern** — Bericht als Datei in `~/.jarvis/workspace/email-triage-{datum}.md` speichern. Tool: `write_file`.

## Bekannte Fallstricke
- Datenschutz: Keine vertraulichen Inhalte in Logs speichern.
- E‑Mail‑Formate können variieren; Parsing‑Fehler abfangen.
- Kategorien vorab mit dem Benutzer abstimmen.

## Qualitätskriterien
- Alle E‑Mails erfasst und richtig kategorisiert
- Bericht mit Kategorien, wichtigsten E‑Mails und nächsten Schritten
- Datei im Workspace gespeichert