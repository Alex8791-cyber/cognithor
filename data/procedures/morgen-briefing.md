---
name: morgen-briefing
trigger_keywords: [Briefing, Morgen, Tagesplan, Überblick, Zusammenfassung, was steht an]
tools_required: [get_recent_episodes, search_memory, list_directory, search_procedures]
success_count: 0
failure_count: 0
total_uses: 0
avg_score: 0.0
last_used: null
learned_from: [initial-setup]
---
# Morgen-Briefing

## Wann anwenden
Wenn der Benutzer den Tag startet und einen Überblick haben möchte.
Typische Trigger: "Was steht heute an?", "Morgen-Briefing", "Zusammenfassung",
"Was war gestern?", "Überblick".

## Voraussetzungen
- Keine besonderen Voraussetzungen (alles aus Memory abrufbar)

## Ablauf

1. **Gestrige Episoden laden** — Was wurde gestern gemacht?
   Tool: `get_recent_episodes` mit count=2 (gestern + vorgestern für Kontext).

2. **Offene Punkte suchen** — Aus den Episoden offene Aufgaben extrahieren.
   Tool: `search_memory` mit "offen TODO ausstehend nächster Schritt".

3. **Workspace prüfen** — Gibt es aktuelle Arbeitsdateien?
   Tool: `list_directory` mit `~/.jarvis/workspace/`.

4. **Briefing formulieren** — Strukturierte Zusammenfassung:
   - Was gestern erledigt wurde
   - Was offen ist
   - Vorschläge für heute
   - Erinnerungen (Termine, Fristen, Deadlines)

## Format der Ausgabe

```
Guten Morgen.

**Gestern erledigt:**
- [Zusammenfassung der gestrigen Aktivitäten]

**Offen:**
- [Aufgaben die noch ausstehen]

**Für heute:**
- [Priorisierte Vorschläge]
```

## Bekannte Fallstricke
- Am Montag: Freitag statt "gestern" als Referenz
- Wenn keine Episoden vorhanden: Ehrlich sagen, nicht erfinden
- Nicht zu lang — Briefing soll schnell erfassbar sein

## Qualitätskriterien
- Maximal 200 Wörter
- Offene Punkte priorisiert
- Konkrete Handlungsvorschläge
