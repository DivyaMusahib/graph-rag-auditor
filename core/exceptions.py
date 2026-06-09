"""Custom exception hierarchy for Graph-RAG Auditor."""

from __future__ import annotations


class GraphRagError(Exception):
    """Base exception for all Graph-RAG Auditor errors."""


class ZipExtractionError(GraphRagError):
    """Raised when the uploaded ZIP cannot be opened or a member cannot be read."""


class ASTParsingFailure(GraphRagError):
    """Raised when AST parsing of a Python file fails (syntax error, etc.)."""


class VectorStorageError(GraphRagError):
    """Raised when FAISS indexing, serialisation, or search fails."""


class LLMInferenceError(GraphRagError):
    """Raised when the LLM API returns an error during inference."""
