---
name: kontakt-recherche
trigger_keywords: [Kontakt, Recherche, Person, Firma, Hintergrund, Gesprächspartner, nachschlagen]
tools_required: [search_memory, web_search, add_entity, save_to_memory, write_file]
category: research
priority: 5
success_count: 0
failure_count: 0
total_uses: 0
avg_score: 0.0
last_used: null
learned_from: [initial-setup]
---
# Kontakt-Recherche

## Wann anwenden
Wenn der Benutzer Informationen über eine Person oder Firma sammeln möchte.
Typische Trigger: „Recherchiere [Person/Firma]", „Was wissen wir über [Name]?",
„Hintergrund zu [Kontakt]".

## Voraussetzungen
- Name der Person oder Firma
- Optional: Kontext (woher bekannt, Anlass)

## Ablauf

1. **Bereits bekannt?** — Prüfen ob der Kontakt schon in Memory existiert.
   Tool: `search_memory` mit Name.
   Tool: `search_memory` mit Firmenname (wenn bekannt).

2. **Öffentliche Recherche** — Öffentlich verfügbare Informationen sammeln.
   Tool: `web_search` mit Name + Firma/Branche.
   Relevante Daten: Branche, Standort, öffentlich sichtbare Rollen.

3. **Kontakt-Profil anlegen** — Strukturiert in Memory speichern.
   Tool: `add_entity` mit:
   - Name, Firma, Position
   - Kontaktweg und Datum
   - Quelle (Empfehlung, Event, Website etc.)

4. **Zusammenfassung erstellen** — Kompakte Notiz.
   Tool: `write_file` → `~/.jarvis/workspace/kontakt-{name}-{datum}.md`
   Inhalt:
   - Zusammenfassung der Recherche
   - Relevante Hintergrundinformationen
   - 3 Gesprächseinstiegsfragen
   - Nächster Schritt

## Bekannte Fallstricke
- Datenschutz: Nur öffentlich verfügbare Informationen recherchieren
- Nicht zu viel vermuten — Recherche-Ergebnisse klar als solche markieren
- Quelle immer dokumentieren

## Qualitätskriterien
- Kontakt in Memory gespeichert
- Kompakte Zusammenfassung erstellt
- Mindestens 3 konkrete Gesprächseinstiegsfragen
- Klarer nächster Schritt definiert
