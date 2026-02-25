---
name: projekt-setup
trigger_keywords: [Projekt, Setup, neues Projekt, anlegen, Projektstruktur, Ordner, initialisieren]
tools_required: [write_file, list_directory, save_to_memory]
success_count: 0
failure_count: 0
total_uses: 0
avg_score: 0.0
last_used: null
learned_from: [initial-setup]
---
# Neues Projekt anlegen

## Wann anwenden
Wenn der Benutzer ein neues Projekt starten und eine strukturierte Ablage dafür anlegen möchte.
Typische Trigger: „Neues Projekt anlegen", „Projektstruktur für X erstellen",
„Projekt X initialisieren".

## Voraussetzungen
- Projektname
- Optional: Kurzbeschreibung, Deadline, beteiligte Personen

## Ablauf

1. **Projektname und Ziel klären** — Falls nicht angegeben, nachfragen.

2. **Verzeichnisstruktur erstellen** — Im Workspace:
   Tool: `write_file` für jede Datei.
   ```
   ~/.jarvis/workspace/projekte/{projektname}/
   ├── README.md          (Projektziel, Beteiligte, Status)
   ├── notizen/           (Besprechungsnotizen, Ideen)
   ├── recherche/         (Hintergrundmaterial)
   └── aufgaben.md        (To-Do-Liste mit Prioritäten)
   ```

3. **README.md befüllen** — Projektname, Ziel, Beteiligte, Startdatum.

4. **Im Memory speichern** — Projekt als Entität anlegen.
   Tool: `save_to_memory` mit Projektname, Status „aktiv", Startdatum.

5. **Erste Aufgaben anlegen** — Falls der Benutzer bereits Aufgaben nennt,
   in `aufgaben.md` eintragen.

## Bekannte Fallstricke
- Keine Duplikate: Prüfen ob ein Projekt mit gleichem Namen schon existiert
- README kurz halten — wird laufend aktualisiert
- Deadline immer dokumentieren, wenn genannt

## Qualitätskriterien
- Verzeichnisstruktur angelegt
- README.md mit Basisinformationen befüllt
- Projekt im Memory gespeichert
- Nächster Schritt klar definiert
