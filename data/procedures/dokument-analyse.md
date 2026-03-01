---
name: dokument-analyse
trigger_keywords: [Dokument, analysieren, PDF, Vertrag, Angebot, zusammenfassen, DOCX, Analyse, auswerten, Bericht, Protokoll]
tools_required: [media_extract_text, analyze_document, vault_save]
category: analysis
priority: 5
success_count: 0
failure_count: 0
total_uses: 0
avg_score: 0.0
last_used: null
learned_from: [initial-setup]
---
# Dokument-Analyse & Zusammenfassung

## Wann anwenden
Wenn der Benutzer ein Dokument (PDF, DOCX, HTML, TXT) analysieren, zusammenfassen oder auswerten möchte.
Typische Trigger: "Analysiere dieses Dokument", "Fasse den Vertrag zusammen", "Was steht in der PDF?", "Werte das Angebot aus", "Prüfe den Bericht".

## Voraussetzungen
- Dateipfad zum Dokument
- Optional: Art der Analyse (vollständig, Zusammenfassung, Risiken, To-Dos)

## Ablauf

### 1. Text extrahieren
Text aus dem Dokument extrahieren mit `media_extract_text`.
Falls das Ergebnis sehr kurz ist oder einen OCR-Hinweis enthält: Den Benutzer darauf hinweisen, dass es sich möglicherweise um ein gescanntes PDF handelt und OCR-Software benötigt wird.

### 2. Dokumenttyp erkennen
Anhand des Inhalts den Dokumenttyp bestimmen:
- **Vertrag** — rechtliche Vereinbarungen, Klauseln, Parteien
- **Angebot** — Preise, Leistungen, Konditionen
- **Bericht** — Analyse, Ergebnisse, Empfehlungen
- **Protokoll** — Teilnehmer, Beschlüsse, Aufgaben
- **Rechnung** — Positionen, Beträge, Fälligkeiten
- **Brief/Schreiben** — Absender, Empfänger, Anliegen
- **Sonstiges** — allgemeine Zusammenfassung

### 3. Strukturierte Analyse erstellen
Verwende `analyze_document` für die vollständige Analyse oder erstelle manuell eine Analyse mit folgenden 6 Abschnitten:

#### Zusammenfassung
2-3 Sätze, die den Kern des Dokuments erfassen.

#### Kernaussagen
Maximal 7 priorisierte Punkte — die wichtigsten Informationen aus dem Dokument.

#### Risiken & Bedenken
Bewertung mit Stufen:
- **HOCH** — sofortiger Handlungsbedarf
- **MITTEL** — sollte beachtet werden
- **NIEDRIG** — zur Kenntnis nehmen

#### Handlungsbedarf / To-Dos
Konkrete Aktionspunkte mit Priorität (Hoch/Mittel/Niedrig).

#### Entscheidungsprotokoll
- Bereits getroffene Entscheidungen
- Offene Entscheidungen die noch ausstehen

#### Metadaten
- Dokumenttyp
- Datum (falls erkennbar)
- Beteiligte Parteien
- Seitenzahl / Umfang

### 4. Optional: Im Vault speichern
Wenn der Benutzer es wünscht oder `save_to_vault=True`, die Analyse mit `vault_save` speichern:
- Ordner: `research`
- Tags: dokumenttyp, relevante Schlagwörter
- Quellen: Dateipfad des Originals

### 5. Schlüssel-Entitäten merken
Wichtige Personen, Firmen, Beträge oder Termine in das Semantic Memory speichern (falls verfügbar).

## Bekannte Fallstricke
- **Gescannte PDFs**: Wenn `media_extract_text` wenig/keinen Text liefert, ist OCR nötig. Hinweis an den Benutzer geben.
- **Rechtsdokumente**: Bei Verträgen, AGBs und juristischen Texten immer den Disclaimer anfügen: "Dies ist eine automatische Analyse und ersetzt keine rechtliche Beratung."
- **Große Dokumente**: Bei sehr langen Texten (>15.000 Zeichen) wird der Text gekürzt. Den Benutzer darauf hinweisen.
- **Vertraulichkeit**: Dokumente werden nur lokal verarbeitet, keine Cloud-Uploads.

## Qualitätskriterien
- Klare Struktur mit allen 6 Abschnitten
- Konkrete, actionable To-Dos (nicht vage)
- Risikobewertung mit Begründung
- Disclaimer bei Rechts-/Finanzdokumenten
