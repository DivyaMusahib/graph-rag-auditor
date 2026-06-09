"""FAISS-backed vector index for Python AST semantic chunks.

Manages the full lifecycle: encoding, building, persisting, loading,
and nearest-neighbour search.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Tuple

from core.ast_parser import ASTNodePayload
from core.exceptions import VectorStorageError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
_STORAGE_DIR: str = "storage"
_FAISS_INDEX_PATH: str = os.path.join(_STORAGE_DIR, "code_index.faiss")
_PAYLOAD_PATH: str = os.path.join(_STORAGE_DIR, "payloads.json")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
_VECTOR_DIMENSION: int = 384


class CodeIndexer:
    """Manages the lifecycle of the FAISS index and chunk embeddings.

    Typical usage::

        indexer = CodeIndexer()
        indexer.build_and_save_index(chunks)
        results = indexer.search("SQL injection vulnerability", top_k=10)
    """

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        self.index = None  # faiss.Index | None
        self.semantic_chunk_matrix: List[ASTNodePayload] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_storage_dir(self) -> None:
        os.makedirs(_STORAGE_DIR, exist_ok=True)

    def _encode(self, texts: List[str]):
        """Returns a float32 numpy array of shape (n, 384)."""
        return self.embedding_model.encode(texts, convert_to_numpy=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_and_save_index(self, semantic_chunk_matrix: List[ASTNodePayload]) -> None:
        """Embeds every chunk, builds a flat L2 FAISS index, and persists it.

        Args:
            semantic_chunk_matrix: Parsed AST payloads from the uploaded codebase.

        Raises:
            VectorStorageError: On any FAISS or I/O failure.
        """
        self.semantic_chunk_matrix = semantic_chunk_matrix
        if not semantic_chunk_matrix:
            logger.warning("build_and_save_index called with an empty chunk list.")
            return

        try:
            import faiss

            texts = [chunk["source_code"] for chunk in self.semantic_chunk_matrix]
            embeddings = self._encode(texts)

            self.index = faiss.IndexFlatL2(_VECTOR_DIMENSION)
            self.index.add(embeddings)

            self._ensure_storage_dir()
            faiss.write_index(self.index, _FAISS_INDEX_PATH)

            with open(_PAYLOAD_PATH, "w", encoding="utf-8") as fh:
                json.dump(self.semantic_chunk_matrix, fh, ensure_ascii=False)

            logger.info(
                "Built and saved FAISS index: %d vectors → %s",
                len(embeddings),
                _FAISS_INDEX_PATH,
            )
        except Exception as exc:
            logger.error("FAISS build failed: %s", exc)
            raise VectorStorageError(f"FAISS indexing failed: {exc}") from exc

    def load_existing_index(self) -> bool:
        """Loads an existing FAISS index and payload file from disk.

        Returns:
            ``True`` if both files were found and loaded successfully.
            ``False`` if either file is missing (no error raised).

        Raises:
            VectorStorageError: If the files exist but cannot be read.
        """
        if not os.path.exists(_FAISS_INDEX_PATH) or not os.path.exists(_PAYLOAD_PATH):
            return False

        try:
            import faiss

            self.index = faiss.read_index(_FAISS_INDEX_PATH)
            with open(_PAYLOAD_PATH, "r", encoding="utf-8") as fh:
                self.semantic_chunk_matrix = json.load(fh)

            logger.info(
                "Loaded existing FAISS index (%d vectors) from disk.",
                self.index.ntotal,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load existing FAISS index: %s", exc)
            raise VectorStorageError(f"FAISS read failed: {exc}") from exc

    def search(
        self, query: str, top_k: int = 15
    ) -> List[Tuple[float, ASTNodePayload]]:
        """Performs a dense vector search against the FAISS index.

        Args:
            query:  Natural-language or code query string.
            top_k:  Number of nearest neighbours to return.

        Returns:
            A list of ``(l2_distance, ASTNodePayload)`` tuples, ordered by
            ascending distance (most similar first).

        Raises:
            VectorStorageError: If the index is not initialised or search fails.
        """
        if self.index is None or not self.semantic_chunk_matrix:
            raise VectorStorageError(
                "Cannot search: index is not initialised. Upload a codebase first."
            )

        try:
            import faiss

            query_vec = self._encode([query])
            distances, indices = self.index.search(query_vec, top_k)

            results: List[Tuple[float, ASTNodePayload]] = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.semantic_chunk_matrix):
                    results.append((float(dist), self.semantic_chunk_matrix[idx]))

            return results
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            raise VectorStorageError(f"Search error: {exc}") from exc
