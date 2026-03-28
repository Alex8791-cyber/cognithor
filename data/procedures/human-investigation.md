---
name: human-investigation
description: OSINT-Recherche und Verifikation von Personen, Projekten und Organisationen
trigger_keywords:
  - Investigation
  - OSINT
  - Background
  - Due Diligence
  - Recherche
  - wer ist
  - Hintergrund
  - Verifikation
  - trust score
  - Person recherchieren
  - Unternehmen prüfen
tools_required:
  - investigate_person
  - investigate_project
  - investigate_org
category: research
priority: 5
enabled: true
---

# Human Investigation Module (HIM)

## Wann anwenden
Wenn der Nutzer eine Person, ein Projekt oder eine Organisation recherchieren, verifizieren oder einen Background-Check durchfuehren moechte.

## Vorgehensweise

1. **Ziel identifizieren**: Name, GitHub-Username, Claims die geprueft werden sollen
2. **Tool waehlen**:
   - `investigate_person` fuer Personen
   - `investigate_project` fuer Projekte/Repos
   - `investigate_org` fuer Organisationen
3. **Claims als komma-separierte Liste angeben**
4. **Justification angeben** (DSGVO-Pflicht): Warum wird recherchiert?
5. **Report auswerten**: Trust Score (0-100), Claim-Status, Red Flags

## Beispiel

```
investigate_person(
  target_name="Terry Zhang",
  target_github="dinnar1407-code",
  claims="works at Anthropic, built Agent Nexus",
  justification="Collaboration request received, verifying credentials"
)
```

## Trust Score Interpretation
- 75-100 (HIGH): Glaubwuerdig, normale Zusammenarbeit moeglich
- 40-74 (MIXED): Teilweise verifiziert, zusaetzliche Belege anfordern
- 0-39 (LOW): Erhebliche Bedenken, Claims unabhaengig pruefen
