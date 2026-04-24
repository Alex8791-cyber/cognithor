# 07 · Troubleshooting

Die häufigsten Stolperstellen aus spec §8 + den bisherigen Nutzer-Reviews — mit Fix in einer Zeile wo immer möglich.

**Voraussetzungen:** keine — diese Seite ist Nachschlagewerk.
**Zeitbedarf:** 3 Minuten bis gezielt.

---

## Ollama läuft nicht

```
ConnectionError: Could not reach Ollama at http://127.0.0.1:11434
```

**Fix:**

```bash
# Linux/macOS:
ollama serve &

# Windows:
# 1. Start-Menü → "Ollama" starten
# 2. oder als Dienst: net start ollama
```

Verifikation: `curl http://127.0.0.1:11434/api/version` liefert JSON.

## Model-Pull fehlgeschlagen

```
Error: model 'qwen3:8b' not found locally
```

**Fix:**

```bash
ollama pull qwen3:8b
# Download ca. 4,7 GB — braucht ca. 5 min auf 50 Mbit/s

# Disk-Space prüfen:
# Linux/macOS:  df -h /
# Windows:      wmic logicaldisk get size,freespace,caption
```

## Port-Konflikt

```
OSError: [Errno 98] Address already in use — port 8741
```

**Fix:**

```bash
# Wer blockiert den Port?
# Linux/macOS:  lsof -i :8741
# Windows:      netstat -ano | findstr :8741

# Alternative: anderen Port nutzen
export COGNITHOR_API_PORT=9000
cognithor
# oder
cognithor --api-port 9000
```

## GuardrailFailure

```
cognithor.crew.errors.GuardrailFailure: Guardrail failed after 3 attempt(s): zu lang
```

**Ursachen:**
1. Der Agent produziert konstant Output, der nicht den Guardrail erfüllt (zu lang / PII / nicht Schema-konform).
2. Das Guardrail-Feedback ist zu vage — der Agent versteht nicht, was zu ändern ist.

**Fix:**
- **Feedback präziser machen:** Statt "zu lang" → `f"Output hat {count} Wörter, maximal {max} erlaubt"`.
- **`max_retries` erhöhen:** Default 2, mit komplexen Guardrails ggf. 5.
- **Guardrail logisch prüfen:** Schreibe einen Unit-Test der nur die Guardrail-Funktion aufruft — sonst debuggst du im Blindflug.

Details: [`docs/guardrails.md`](../guardrails.md).

## Unknown Tool

```
cognithor.crew.errors.ToolNotFoundError: Tool 'web_seach' nicht gefunden. Meintest du 'web_search'?
```

**Fix:**

```bash
# Alle verfügbaren Tools listen
cognithor tools list

# Oder via Python:
python -c "from cognithor.crew.tool_resolver import available_tool_names; \
           from cognithor.mcp.tool_registry_db import ToolRegistryDB; \
           print(available_tool_names(ToolRegistryDB('~/.cognithor/db/tool_registry.db')))"
```

Der `did_you_mean` Vorschlag in der Fehlermeldung zeigt oft direkt den Tippfehler.

## Template Name Collision

```
Error: Directory 'my-project' already exists and is not empty
```

**Fix:**

```bash
# Anderen Namen wählen
cognithor init my-project-v2 --template research

# Oder überschreiben
cognithor init my-project --template research --force
```

Vorsicht: `--force` löscht **nichts** sondern merged. Eigene Änderungen können überschrieben werden — in einer leeren Directory starten ist sicherer.

## Planner-Model zu langsam

```
TimeoutError: Planner took longer than 60s
```

**Fix:**
- Kleineres Modell nutzen: `COGNITHOR_MODEL_PLANNER=qwen3:8b` statt `qwen3:32b`.
- GPU-Unterstützung prüfen: `ollama ps` zeigt Load-Typ (CPU vs GPU).
- `cognithor --lite` startet mit weniger Hintergrund-Services.

## "cognithor_home nicht initialisiert"

```
RuntimeWarning: cognithor config load failed (...); using temp-dir tool registry
```

**Fix:**

```bash
# Einmalige Bootstrap-Initialisierung
cognithor --init-only

# Oder: manuell aufräumen falls korrupt
rm -rf ~/.cognithor/.cognithor_initialized
cognithor --init-only
```

## Ollama + Windows-Dienst installiert, `ollama` CLI nicht im PATH

```
'ollama' wird nicht als interner oder externer Befehl erkannt
```

**Fix:** Starte eine neue Shell (PATH-Update braucht Neustart). Sonst: `where.exe ollama` — falls leer, neu installieren.

## Mehr Hilfe

- [`docs/quickstart/EXTERNAL_REVIEW_RESULTS.md`](EXTERNAL_REVIEW_RESULTS.md) — aktuelle bekannte Stolperstellen aus externen Reviews.
- [GitHub Issues](https://github.com/Alex8791-cyber/cognithor/issues) — Suche nach der Fehlermeldung, häufig schon gelöst.

---

**Ende des Quickstart.** Zurück zur [Übersicht](README.md).
