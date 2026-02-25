# ============================================================================
# Jarvis · Agent OS – Makefile
# ============================================================================
.DEFAULT_GOAL := help
SHELL := /bin/bash
VENV := $(HOME)/.jarvis/venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
JARVIS := $(VENV)/bin/jarvis

# ============================================================================
# Hilfe
# ============================================================================

.PHONY: help
help: ## Zeigt diese Hilfe
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Installation
# ============================================================================

.PHONY: install
install: ## Vollständige Installation (interaktiv)
	@./install.sh

.PHONY: install-minimal
install-minimal: ## Minimal-Installation (nur Core)
	@./install.sh --minimal

.PHONY: install-full
install-full: ## Komplette Installation inkl. Voice
	@./install.sh --full

.PHONY: venv
venv: ## Erstellt Virtual Environment
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

.PHONY: dev
dev: venv ## Installiert im Entwicklungsmodus
	$(PIP) install -e ".[all,dev]"

# ============================================================================
# Starten
# ============================================================================

.PHONY: run
run: ## Startet Jarvis CLI
	$(JARVIS)

.PHONY: run-debug
run-debug: ## Startet mit Debug-Logging
	$(JARVIS) --log-level DEBUG

.PHONY: init
init: ## Erstellt nur Verzeichnisstruktur
	$(JARVIS) --init-only

# ============================================================================
# Systemd
# ============================================================================

.PHONY: systemd-install
systemd-install: ## Installiert Systemd-Services
	@./install.sh --systemd

.PHONY: start
start: ## Startet Jarvis als Systemd-Service
	systemctl --user start jarvis

.PHONY: stop
stop: ## Stoppt Jarvis-Service
	systemctl --user stop jarvis

.PHONY: restart
restart: ## Neustart Jarvis-Service
	systemctl --user restart jarvis

.PHONY: status
status: ## Zeigt Service-Status
	systemctl --user status jarvis

.PHONY: logs
logs: ## Zeigt Live-Logs
	journalctl --user -u jarvis -f

.PHONY: enable
enable: ## Aktiviert Autostart
	systemctl --user enable jarvis

# ============================================================================
# Tests
# ============================================================================

.PHONY: test
test: ## Alle Tests ausführen
	$(PYTHON) -m pytest tests/ -q

.PHONY: test-v
test-v: ## Tests mit ausführlicher Ausgabe
	$(PYTHON) -m pytest tests/ -v --tb=short

.PHONY: test-cov
test-cov: ## Tests mit Coverage-Report
	$(PYTHON) -m pytest tests/ --cov=jarvis --cov-report=term-missing --cov-report=html

.PHONY: test-fast
test-fast: ## Nur schnelle Unit-Tests (ohne Integration)
	$(PYTHON) -m pytest tests/ -q -m "not slow and not integration"

.PHONY: test-integration
test-integration: ## Nur Integration-Tests
	$(PYTHON) -m pytest tests/test_integration/ -v --tb=short

.PHONY: smoke
smoke: ## Smoke-Test (Installation prüfen)
	$(PYTHON) scripts/smoke_test.py

.PHONY: first-boot
first-boot: ## First Boot — Erster Start mit echtem Ollama (komplett)
	$(PYTHON) scripts/first_boot.py

.PHONY: first-boot-quick
first-boot-quick: ## First Boot — Nur System + Modelle (kein Agent-Loop)
	$(PYTHON) scripts/first_boot.py --quick

.PHONY: live
live: ## Live Smoke-Test (mit echtem Ollama)
	$(PYTHON) scripts/live_smoke_test.py --verbose

.PHONY: live-quick
live-quick: ## Live Smoke-Test ohne LLM
	$(PYTHON) scripts/live_smoke_test.py --skip-llm

.PHONY: health
health: ## Health-Check (Laufzeit)
	$(PYTHON) scripts/health_check.py

.PHONY: health-json
health-json: ## Health-Check als JSON
	$(PYTHON) scripts/health_check.py --json

# ============================================================================
# Code-Qualität
# ============================================================================

.PHONY: lint
lint: ## Ruff Linter
	$(PYTHON) -m ruff check src/ tests/

.PHONY: lint-fix
lint-fix: ## Ruff Auto-Fix
	$(PYTHON) -m ruff check --fix src/ tests/

.PHONY: format
format: ## Ruff Formatter
	$(PYTHON) -m ruff format src/ tests/

.PHONY: typecheck
typecheck: ## MyPy Type-Checking
	$(PYTHON) -m mypy src/jarvis/ --ignore-missing-imports

.PHONY: check
check: lint typecheck test ## Alles prüfen (lint + types + tests)

# ============================================================================
# Metriken
# ============================================================================

.PHONY: stats
stats: ## Projekt-Statistiken
	@echo ""
	@echo "═══ Jarvis · Projekt-Statistiken ═══"
	@echo ""
	@echo "Source Code:"
	@find src/jarvis -name "*.py" | xargs wc -l | tail -1
	@echo ""
	@echo "Tests:"
	@find tests -name "*.py" | xargs wc -l | tail -1
	@echo ""
	@echo "Module:"
	@find src/jarvis -name "*.py" | wc -l
	@echo ""
	@echo "Test-Dateien:"
	@find tests -name "test_*.py" | wc -l
	@echo ""

# ============================================================================
# Aufräumen
# ============================================================================

.PHONY: clean
clean: ## Build-Artefakte entfernen
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov

.PHONY: uninstall
uninstall: ## Deinstallation
	@./install.sh --uninstall
