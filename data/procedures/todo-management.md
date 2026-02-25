---
name: todo-management
trigger_keywords: [To-Do, Aufgaben, ToDo, Liste, Aufgabenliste, organisieren, Planung]
tools_required: [search_memory, write_file]
success_count: 0
failure_count: 0
total_uses: 0
avg_score: 0.0
last_used: null
learned_from: [initial-setup]
---
# Aufgaben‑ und To‑Do‑Management

## Wann anwenden
Wenn der Benutzer seine Aufgaben organisieren, priorisieren oder eine neue To‑Do‑Liste erstellen möchte.
Typische Trigger: „Organisiere meine Aufgabenliste“, „Neue Aufgabe hinzufügen“, „Was steht noch an?“. 

## Voraussetzungen
- Falls vorhanden, eine bestehende Aufgabenliste (z. B. als Markdown-Datei im Workspace)
- Priorisierungskriterien (z. B. Wichtigkeit, Deadline)

## Ablauf
1. **Bestehende Liste laden** — Mit `search_memory` oder `read_file` nach vorhandenen Aufgabenlisten suchen (z. B. Dateien im Workspace oder im episodischen Gedächtnis).
2. **Neue Aufgaben abfragen** — Den Benutzer nach neuen Aufgaben, Deadlines und Prioritäten fragen.
3. **Liste aktualisieren** — Neue Aufgaben einfügen, bestehende aktualisieren und nach Priorität sortieren.
4. **Zusammenfassung erstellen** — Eine strukturierte To‑Do‑Liste mit Kategorien wie „Heute“, „Diese Woche“, „Später“.
5. **Speichern** — Liste als Markdown‑Datei im Workspace ablegen (`todo-{datum}.md`). Tool: `write_file`.

## Bekannte Fallstricke
- Überfällige Aufgaben kennzeichnen und priorisieren.
- Unterschiedliche Prioritätsbegriffe klären (z. B. „hoch“, „dringend“).

## Qualitätskriterien
- Vollständige Aufgabenliste mit Prioritäten und Deadlines
- Klarer Überblick über kurzfristige und langfristige Aufgaben
- Datei im Workspace gespeichert