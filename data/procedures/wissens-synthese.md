---
name: wissens-synthese
trigger_keywords: [Synthese, zusammenführen, Überblick, Gesamtbild, was wissen wir, Wissensstand, Widersprüche, Zeitlinie, Wissenslücken, Faktencheck]
tools_required: [knowledge_synthesize, knowledge_contradictions, knowledge_timeline, knowledge_gaps, vault_save]
category: analysis
priority: 6
success_count: 0
failure_count: 0
total_uses: 0
avg_score: 0.0
last_used: null
learned_from: [initial-setup]
---
# Wissens-Synthese & Analyse

## Wann anwenden
Wenn der Benutzer ein umfassendes Verständnis zu einem Thema benötigt — nicht nur einzelne Fakten, sondern das Gesamtbild: Was wissen wir? Was hat sich verändert? Was fehlt noch? Was widerspricht sich?

Typische Trigger:
- "Was wissen wir über X?"
- "Gib mir einen Überblick zu Y"
- "Stimmt das noch, was wir zu Z gespeichert haben?"
- "Erstelle eine Zeitlinie zu ..."
- "Was fehlt uns noch zu ...?"
- "Führe alles zu X zusammen"

## Voraussetzungen
- Klares Thema oder Fragestellung
- Optional: Gewünschte Tiefe (schnell, standard, tief)
- Optional: Sprache (Deutsch/Englisch)

## Ablauf

### 1. Thema einordnen
Welche Art von Synthese ist gefragt?

| Anfrage | Tool | Modus |
|---------|------|-------|
| Gesamtüberblick, "was wissen wir" | `knowledge_synthesize` | standard oder deep |
| Faktencheck, "stimmt das noch" | `knowledge_contradictions` | — |
| Chronologie, "wie hat sich X entwickelt" | `knowledge_timeline` | — |
| Wissenslücken, "was fehlt noch" | `knowledge_gaps` | — |

### 2. Synthese durchführen

**Für Gesamtüberblick:**
```
knowledge_synthesize(topic="...", depth="standard", save_to_vault=true)
```
- `depth="quick"` — Nur Memory + Vault, keine Web-Recherche (schnell)
- `depth="standard"` — Memory + Vault + 3 Web-Ergebnisse (Standard)
- `depth="deep"` — Memory + Vault + 5 Web-Ergebnisse, detailliert

**Für Widerspruchserkennung:**
```
knowledge_contradictions(topic="...")
```
Vergleicht gespeichertes Wissen mit aktuellen Web-Informationen.

**Für Zeitlinien:**
```
knowledge_timeline(topic="...")
```
Extrahiert Datumsangaben und baut Kausalketten auf.

**Für Wissenslücken:**
```
knowledge_gaps(topic="...")
```
Analysiert nur gespeichertes Wissen, identifiziert was fehlt.

### 3. Ergebnis auswerten
Die Synthese liefert:
- **Kernerkenntnisse** mit Konfidenz-Sternen (★★★ = sicher)
- **Quellenvergleich** — wo stimmen Quellen überein, wo nicht
- **Widersprüche** — was gespeichert war vs. was aktuell ist
- **Zeitliche Entwicklung** — Kausalketten und Trends
- **Wissenslücken** — mit priorisierten Recherche-Vorschlägen
- **Fazit** — konkreter nächster Schritt

### 4. Optional: Lücken schließen
Wenn die Synthese Wissenslücken identifiziert und der Benutzer zustimmt:
1. Die vorgeschlagenen Suchbegriffe aus der Lückenanalyse nutzen
2. `search_and_read` oder `web_search` für jede Lücke ausführen
3. Ergebnisse mit `vault_save` speichern
4. Neue Synthese erstellen um das aktualisierte Gesamtbild zu sehen

### 5. Ergebnis speichern
Wenn nicht bereits `save_to_vault=true` gesetzt:
Die Synthese manuell mit `vault_save` im Ordner "knowledge" speichern.

## Bekannte Fallstricke
- **Veraltetes Memory**: Wenn Informationen im Semantic Memory veraltet sind, wird die Synthese darauf hinweisen. Ggf. Memory bereinigen.
- **Zu breites Thema**: Bei sehr generischen Themen ("Politik", "Wirtschaft") die Anfrage einschränken.
- **Keine gespeicherten Infos**: Bei neuen Themen liefert `knowledge_gaps` die besten Einstiegspunkte.
- **Rate-Limits**: Bei `depth="deep"` werden bis zu 5 Webseiten abgerufen — kann bei DuckDuckGo zu Rate-Limits führen.

## Qualitätskriterien
- Alle verfügbaren Quellen einbezogen (Memory, Vault, Web)
- Konfidenz transparent markiert
- Widersprüche explizit benannt, nicht stillschweigend übergangen
- Konkrete, actionable Empfehlungen
- Ergebnis im Vault gespeichert für spätere Referenz
