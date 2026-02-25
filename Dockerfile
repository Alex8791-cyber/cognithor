# ============================================================================
# Jarvis · Agent OS – Dockerfile
# ============================================================================
# Multi-Stage Build: Sauberes, kleines Image.
#
# Nutzung:
#   docker build -t jarvis .
#   docker run -it --name jarvis \
#     -e JARVIS_OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#     -v jarvis-data:/home/jarvis/.jarvis \
#     jarvis
#
# Wichtig: Ollama muss AUSSERHALB des Containers laufen (GPU-Zugriff).
# ============================================================================

# ── Stage 1: Build ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System-Deps für Compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Abhängigkeiten zuerst (besseres Layer-Caching)
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install ".[all]"

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Alexander Söllner"
LABEL description="Jarvis Agent OS – Lokales autonomes Agent-Betriebssystem"
LABEL version="0.1.0"

# Nicht als root laufen
RUN groupadd -r jarvis && useradd -r -g jarvis -m -s /bin/bash jarvis

# Runtime-Deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python-Pakete aus Builder kopieren
COPY --from=builder /install /usr/local

# Jarvis-Source
COPY --chown=jarvis:jarvis src/ /app/src/
COPY --chown=jarvis:jarvis config.yaml.example /app/
COPY --chown=jarvis:jarvis scripts/ /app/scripts/

WORKDIR /app

# Als jarvis-User laufen
USER jarvis

# Jarvis-Home als Volume
VOLUME ["/home/jarvis/.jarvis"]

# Environment
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    JARVIS_HOME=/home/jarvis/.jarvis \
    JARVIS_LOGGING_LEVEL=INFO \
    JARVIS_OLLAMA_BASE_URL=http://host.docker.internal:11434

# Health-Check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python scripts/health_check.py --quick --json || exit 1

# Init + Start
ENTRYPOINT ["python", "-m", "jarvis"]
CMD []
