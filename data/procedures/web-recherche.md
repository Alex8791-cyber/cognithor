---
name: web-recherche
trigger_keywords: [Recherche, recherchieren, Zusammenfassung, Internet, Web-Suche, Suche]
tools_required: [web_search, write_file]
category: research
priority: 5
success_count: 0
failure_count: 0
total_uses: 0
avg_score: 0.0
last_used: null
learned_from: [initial-setup]
---
# Web‑Recherche & Zusammenfassung

## Wann anwenden
Wenn der Benutzer eine kurze Recherche zu einem Thema benötigt und eine strukturierte Zusammenfassung wünscht.
Typische Trigger: „Recherchiere das Thema X“, „Erstelle eine Zusammenfassung zu [Thema]“, „Finde Informationen zu …“, „Internet‑Suche zu …“.

## Voraussetzungen
- Klare Fragestellung oder Thema
- Optional: gewünschte Anzahl von Quellen

## Ablauf
1. **Fragestellung präzisieren** — Wenn das Thema unklar ist, den Benutzer nach Details fragen. Verwende die Triggerwörter im Kontext.
2. **Websuche durchführen** — Mit dem Tool `web_search` nach relevanten Ergebnissen suchen. Verwende die vom Benutzer gegebene Fragestellung, top 3–5 Treffer.
3. **Ergebnisse lesen und extrahieren** — Die wichtigsten Erkenntnisse aus den Seiten extrahieren. Vermeide Werbung und irrelevante Inhalte.
4. **Zusammenfassung schreiben** — Eine strukturierte Zusammenfassung mit den wichtigsten Punkten, Zitaten und ggf. Links erstellen.
5. **Datei speichern** — Die Zusammenfassung als Markdown‑Datei im Workspace ablegen, z. B. `recherche-{thema}-{datum}.md`. Tool: `write_file`.

## Bekannte Fallstricke
- Veraltete Informationen: Prüfe Veröffentlichungsdaten der Quellen.
- Bestätigungsfehler: Ergebnisse kritisch hinterfragen, keine Fakten erfinden.
- Rechte: Nur öffentlich zugängliche Quellen verwenden.

## Qualitätskriterien
- Klare Struktur (Einleitung, Hauptpunkte, Fazit)
- Mindestens zwei unterschiedliche Quellen
- Datei im Workspace gespeichert