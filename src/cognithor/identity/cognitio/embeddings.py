"""
cognitio/embeddings.py

Embedding computation engine.

Converts text → vector using sentence-transformers.
All memory records and query contexts are embedded through this module.

Model: all-MiniLM-L6-v2 (lightweight, fast, 384-dimensional)
"""

import logging
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Engine for computing text embeddings.

    Parameters:
        model_name: sentence-transformers model name
        device: Compute device ('cpu' or 'cuda')
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    _model: Any

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        logger.info(f"EmbeddingEngine initialized: model={model_name}, device={device}")

    def _load_model(self) -> None:
        """Lazy-load the model (loaded on first use)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Model loaded: {self.model_name}")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def encode(self, text: str) -> list[float]:
        """
        Encode a single text into an embedding.

        Parameters:
            text: Text to encode

        Returns:
            list[float]: Embedding vector (384-dimensional)
        """
        self._load_model()
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return cast("list[float]", embedding.tolist())

        except Exception as e:
            logger.error(f"Encoding error: {e}")
            # Return zero vector on error
            return [0.0] * self.EMBEDDING_DIM

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Encode multiple texts in a batch.

        Parameters:
            texts: List of texts to encode

        Returns:
            list[list[float]]: List of embedding vectors
        """
        self._load_model()
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True, batch_size=32)
            return cast("list[list[float]]", embeddings.tolist())

        except Exception as e:
            logger.error(f"Batch encoding error: {e}")
            return [[0.0] * self.EMBEDDING_DIM for _ in texts]

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Parameters:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity (-1.0 to 1.0, typically 0.0–1.0)
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Parameters:
            text1: First text
            text2: Second text

        Returns:
            float: Similarity score (0.0–1.0)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        similarity = self.cosine_similarity(emb1, emb2)
        # Normalize cosine similarity to 0–1 range
        return max(0.0, similarity)
