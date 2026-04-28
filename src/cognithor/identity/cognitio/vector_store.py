"""
cognitio/vector_store.py

ChromaDB wrapper — vector-based memory indexing and ANN search.

Two-stage retrieval:
    Stage 1 (ChromaDB): Coarse filtering, O(log N), pure semantic similarity
    Stage 2 (Cognitio): Fine-grained, O(k), bias-weighted attention
                        (recency, emotion, entrenchment)

This approach produces cognitively weighted results in milliseconds
even across millions of records.
"""

import logging
import os

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB wrapper — vector-based memory indexing.

    Operates in persistent mode: index is preserved across restarts.

    Parameters:
        persist_dir: ChromaDB persistent storage directory
        collection_name: ChromaDB collection name
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "memories",
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb

            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Cosine similarity
            )
            logger.info(
                "VectorStore initialized: collection=%s, record_count=%d",
                self.collection_name,
                self._collection.count(),
            )
        except ImportError:
            logger.error("chromadb not installed: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"VectorStore initialization failed: {e}")
            raise

    def add(
        self,
        memory_id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """
        Add a memory record to ChromaDB.

        Parameters:
            memory_id: Unique memory ID
            embedding: Vector representation
            metadata: Additional metadata (memory_type, emotional_intensity,
                      entrenchment, etc.)
        """
        try:
            # ChromaDB metadata values must be string, int, float, or bool
            clean_metadata = self._clean_metadata(metadata)

            # ChromaDB 1.5+ rejects empty metadata dicts with ValueError.
            # Inject a sentinel so the call still succeeds; callers that
            # pass {} get a single benign field instead of a 500.
            if not clean_metadata:
                clean_metadata = {"_meta_empty": True}

            # Check for existing record
            existing = self._collection.get(ids=[memory_id])
            if existing["ids"]:
                # Update
                self._collection.update(
                    ids=[memory_id],
                    embeddings=[embedding],
                    metadatas=[clean_metadata],
                )
                logger.debug(f"Record updated: {memory_id[:8]}")
            else:
                # Add
                self._collection.add(
                    ids=[memory_id],
                    embeddings=[embedding],
                    metadatas=[clean_metadata],
                )
                logger.debug(f"Record added: {memory_id[:8]}")

        except Exception as e:
            logger.error(f"VectorStore.add error (id={memory_id[:8]}): {e}")
            raise

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 50,
        where: dict | None = None,
    ) -> list[str]:
        """
        Return the nearest n_results memory_ids via ANN search.

        This is the candidate pool that Cognitio will work with.
        n_results should be larger than final top_k (e.g., n_results=50 for top_k=10).

        Parameters:
            query_embedding: Query embedding vector
            n_results: Maximum number of results to return
            where: ChromaDB metadata filter (e.g., {"memory_type": "relational"})
                   No filter applied if None.

        Returns:
            list[str]: List of memory_ids (in order of semantic proximity)
        """
        try:
            total = self._collection.count()
            if total == 0:
                return []

            actual_n = min(n_results, total)
            query_kwargs: dict = {
                "query_embeddings": [query_embedding],
                "n_results": actual_n,
                "include": ["metadatas"],
            }
            if where:
                query_kwargs["where"] = where

            results = self._collection.query(**query_kwargs)
            return results["ids"][0] if results["ids"] else []

        except Exception as e:
            logger.error(f"VectorStore.query error: {e}")
            return []

    def update_metadata(self, memory_id: str, metadata: dict) -> None:
        """
        Update metadata for an existing record.

        Updates changing fields such as entrenchment and last_accessed.
        Embedding is preserved.

        Parameters:
            memory_id: ID of the memory to update
            metadata: Metadata fields to update
        """
        try:
            existing = self._collection.get(ids=[memory_id], include=["metadatas"])
            if not existing["ids"]:
                logger.warning(f"update_metadata: record not found: {memory_id[:8]}")
                return

            current = existing["metadatas"][0] if existing["metadatas"] else {}
            current.update(self._clean_metadata(metadata))

            self._collection.update(
                ids=[memory_id],
                metadatas=[current],
            )
        except Exception as e:
            logger.error(f"VectorStore.update_metadata error: {e}")

    def delete(self, memory_id: str) -> None:
        """
        Delete a record from ChromaDB.

        Called by the garbage collector.
        The permanent copy on Arweave is preserved.

        Parameters:
            memory_id: ID of the memory to delete
        """
        try:
            self._collection.delete(ids=[memory_id])
            logger.debug(f"Record deleted: {memory_id[:8]}")
        except Exception as e:
            logger.error(f"VectorStore.delete error: {e}")

    def count(self) -> int:
        """Total record count."""
        try:
            return self._collection.count()
        except Exception as e:
            logger.error(f"VectorStore.count error: {e}")
            return 0

    def exists(self, memory_id: str) -> bool:
        """Does a specific record exist?"""
        try:
            result = self._collection.get(ids=[memory_id])
            return bool(result["ids"])
        except Exception:
            return False

    def get_all_ids(self) -> list[str]:
        """Retrieve all memory IDs."""
        try:
            result = self._collection.get(include=[])
            return result["ids"]
        except Exception as e:
            logger.error(f"VectorStore.get_all_ids error: {e}")
            return []

    def clear(self) -> None:
        """
        Delete all records. Drop and recreate the collection.

        Called by soft_reset() — Genesis anchors are re-added
        separately by the engine.
        """
        try:
            name = self._collection.name
            self._client.delete_collection(name)
            self._collection = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("VectorStore cleared.")
        except Exception as e:
            logger.error("VectorStore clear error: %s", e)

    _META_MAX_STR = 1024  # max chars for a single string metadata value
    _META_MAX_LIST = 100  # max items in a list before truncation

    @staticmethod
    def _clean_metadata(metadata: dict) -> dict:
        """
        Clean metadata for ChromaDB.
        Only string, int, float, and bool values are allowed.
        String values are capped at 1024 chars; lists at 100 items.
        """
        max_str = VectorStore._META_MAX_STR
        max_list = VectorStore._META_MAX_LIST
        clean = {}
        for key, value in metadata.items():
            if isinstance(value, int | float | bool):
                clean[key] = value
            elif isinstance(value, str):
                clean[key] = value[:max_str]
            elif isinstance(value, list):
                clean[key] = ",".join(str(v) for v in value[:max_list])
            elif value is None:
                clean[key] = ""
            else:
                clean[key] = str(value)[:max_str]
        return clean
