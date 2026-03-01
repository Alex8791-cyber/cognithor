---
name: meeting-vorbereitung
trigger_keywords: [Meeting, Besprechung, Termin, vorbereiten, Agenda, Teilnehmer, Gesprächsvorbereitung]
tools_required: [search_memory, web_search, write_file]
category: productivity
priority: 5
success_count: 0
failure_count: 0
total_uses: 0
avg_score: 0.0
last_used: null
learned_from: [initial-setup]
---
# Meeting-Vorbereitung

## Wann anwenden
Wenn der Benutzer sich auf ein Meeting oder Gespräch vorbereiten möchte.
Typische Trigger: „Bereite das Meeting mit X vor", „Agenda für morgen",
„Gesprächsvorbereitung für X", „Was muss ich für den Termin wissen?".

## Voraussetzungen
- Gesprächspartner oder Thema
- Optional: Datum, Ort, Ziel des Meetings

## Ablauf

1. **Kontext aus Memory laden** — Gibt es bereits Informationen zum
   Gesprächspartner oder Thema?
   Tool: `search_memory` mit Name/Thema.

2. **Hintergrund recherchieren** — Falls nötig, öffentliche Informationen
   zum Gesprächspartner oder Thema sammeln.
   Tool: `web_search` (nur bei Bedarf und öffentlichen Informationen).

3. **Agenda erstellen** — Strukturierte Gesprächsagenda:
   - Begrüßung und Kontext
   - Hauptthemen (priorisiert)
   - Offene Fragen
   - Nächste Schritte

4. **Gesprächsnotiz erstellen** — Kompaktes Vorbereitungsdokument.
   Tool: `write_file` → `~/.jarvis/workspace/meeting-{name}-{datum}.md`
   Inhalt:
   - Zusammenfassung der Recherche
   - Agenda mit Zeitplan
   - 3 Kernfragen für das Gespräch
   - Gewünschtes Ergebnis

## Bekannte Fallstricke
- Datenschutz: Nur öffentlich verfügbare Informationen recherchieren
- Agenda nicht überladen — maximal 3–5 Hauptpunkte
- Immer ein klares Ziel für das Meeting definieren

## Qualitätskriterien
- Kompaktes Vorbereitungsdokument erstellt
- Agenda mit priorisierten Themen
- Mindestens 3 konkrete Gesprächsfragen
- Klares Ziel und nächste Schritte definiert
