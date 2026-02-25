"""Jarvis 5-Tier Cognitive Memory System. [B§4]

Public API:
    MemoryManager      — Zentrale Schnittstelle (nutze diese!)
    CoreMemory         — Tier 1: Identität (CORE.md)
    EpisodicMemory     — Tier 2: Tageslog
    SemanticMemory     — Tier 3: Wissens-Graph
    ProceduralMemory   — Tier 4: Gelernte Skills
    WorkingMemoryManager — Tier 5: Session-Kontext
    HybridSearch       — 3-Kanal Suche
    MemoryIndex        — SQLite Index
    EmbeddingClient    — Embedding-Generierung
"""

from jarvis.memory.chunker import chunk_file, chunk_text
from jarvis.memory.core_memory import CoreMemory
from jarvis.memory.embeddings import EmbeddingClient, cosine_similarity
from jarvis.memory.episodic import EpisodicMemory
from jarvis.memory.indexer import MemoryIndex
from jarvis.memory.manager import MemoryManager
from jarvis.memory.procedural import ProceduralMemory
from jarvis.memory.search import HybridSearch, recency_decay
from jarvis.memory.semantic import SemanticMemory
from jarvis.memory.watcher import MemoryWatcher
from jarvis.memory.hygiene import MemoryHygieneEngine
from jarvis.memory.integrity import (
    ContradictionDetector,
    DecisionExplainer,
    DuplicateDetector,
    IntegrityChecker,
    MemoryVersionControl,
    PlausibilityChecker,
)
from jarvis.memory.working import WorkingMemoryManager

__all__ = [
    "CoreMemory",
    "EmbeddingClient",
    "EpisodicMemory",
    "HybridSearch",
    "MemoryHygieneEngine",
    "MemoryIndex",
    "MemoryManager",
    "MemoryWatcher",
    "ProceduralMemory",
    "SemanticMemory",
    "WorkingMemoryManager",
    "chunk_file",
    "chunk_text",
    "cosine_similarity",
    "recency_decay",
]
