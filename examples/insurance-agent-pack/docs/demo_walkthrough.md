# Demo Walkthrough

A typical interactive session looks like:

```text
$ insurance-agent-pack run --interview

=== Versicherungs-Pre-Beratung ===
Alle Eingaben sind synthetisch. Diese Software ist keine §34d-Beratung.

Vorname: Alex
Alter: 45
Berufsstatus (GGF/selbständig/angestellt/freiberufler): GGF
Bestehende Policen (kurz, kommasepariert oder 'keine'): Hausrat, Haftpflicht

[Crew kickoff: 4 Agenten in sequenzieller Verarbeitung]

[needs-assessor] Erstellt JSON-Bedarfsprofil...
[policy-analyst] Keine PDF-Anhänge gefunden — überspringe Extraktion.
[compliance-gatekeeper] Anliegen klassifiziert: pre_advisory_question (allowed=true)
[report-generator] Erstelle Pre-Beratungs-Report...

## Pre-Beratungs-Report

### Was ich beobachte

- 45 Jahre, GGF — Sozialversicherung oft befreit.
- Bestand: Hausrat + Haftpflicht.

### Mögliche Lücken (Themen für ein §34d-Gespräch)

- Berufsunfähigkeitsversicherung (BU) — gesetzlicher Schutz greift bei GGF oft nicht.
- Altersvorsorge: bAV-Direktversicherung, Pensionszusage, oder private Rürup-Rente.
- PKV-vs-GKV-Entscheidung — bei GGF oft sinnvoll, individuelle Prüfung erforderlich.

### Worüber Sie mit einem §34d-Vermittler sprechen sollten

- Konkrete BU-Tarife (Eintrittsalter 45 ist relevant für Beitragshöhe).
- Altersvorsorge-Architektur — welche Bausteine kombinieren?
- PKV-Wechsel — Altersrückstellungen, Wartezeiten, Selbstbehalt.

— Ende des Reports —
```

The full asciinema recording will be linked once captured.
