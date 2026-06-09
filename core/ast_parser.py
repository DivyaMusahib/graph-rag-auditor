"""AST parser for Python codebases.

Extracts structured semantic chunks (Module, ClassDef, FunctionDef,
AsyncFunctionDef) from every .py file inside an uploaded ZIP archive.
"""

from __future__ import annotations

import ast
import io
import logging
import zipfile
from typing import Dict, List, Tuple, TypedDict

from core.exceptions import ASTParsingFailure, ZipExtractionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / directories to skip inside the ZIP
# ---------------------------------------------------------------------------
_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", "venv", ".venv", "env",
    ".env", "dist", "build", "node_modules", ".tox",
    ".mypy_cache", ".pytest_cache", "site-packages",
})


# ---------------------------------------------------------------------------
# Public TypedDict
# ---------------------------------------------------------------------------
class ASTNodePayload(TypedDict):
    """A semantic chunk extracted from a Python source file."""
    file_path: str
    node_type: str       # "Module" | "ClassDef" | "FunctionDef" | "AsyncFunctionDef"
    node_name: str       # qualified name within the file
    source_code: str
    start_line: int
    end_line: int
    outgoing_calls: List[str]
    child_nodes: List[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _ext(filename: str) -> str:
    """Returns the lowercase file extension including the dot."""
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot != -1 else ""


def _should_skip(filename: str) -> bool:
    """Returns True if the file resides in a directory we want to ignore."""
    parts = filename.replace("\\", "/").split("/")
    return any(p in _SKIP_DIRS for p in parts)


def _call_name(node: ast.AST) -> str:
    """Recursively extracts the dotted call name from an ast.Call node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


# ---------------------------------------------------------------------------
# Python parser
# ---------------------------------------------------------------------------
def _parse_python(raw_source: str, file_path: str) -> List[ASTNodePayload]:
    """Parses a single Python source file into a list of ASTNodePayload chunks.

    One chunk is emitted per top-level / nested function or class, plus a
    Module-level chunk that captures imports and the file overview.

    Args:
        raw_source: The UTF-8 source text of the file.
        file_path:  The logical path inside the ZIP (used as an identifier).

    Returns:
        A list of ASTNodePayload dicts.

    Raises:
        ASTParsingFailure: On syntax errors or unexpected parse failures.
    """
    try:
        tree = ast.parse(raw_source, filename=file_path)
    except SyntaxError as exc:
        raise ASTParsingFailure(
            f"Syntax error in {file_path} at line {exc.lineno}: {exc.msg}"
        ) from exc
    except Exception as exc:
        raise ASTParsingFailure(
            f"Unexpected parse error in {file_path}: {exc}"
        ) from exc

    source_lines = raw_source.splitlines()
    chunks: List[ASTNodePayload] = []

    # ---- Module chunk -------------------------------------------------------
    module_calls: List[str] = []
    module_children: List[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_children.append(node.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_calls.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                module_calls.append(f"{node.module}.{alias.name}")

    _MAX_MODULE_SRC = 1_500
    module_src = (
        raw_source[:_MAX_MODULE_SRC] + "\n... (truncated)"
        if len(raw_source) > _MAX_MODULE_SRC
        else raw_source
    )
    chunks.append(
        ASTNodePayload(
            file_path=file_path,
            node_type="Module",
            node_name="<module>",
            source_code=module_src,
            start_line=1,
            end_line=len(source_lines),
            outgoing_calls=module_calls,
            child_nodes=module_children,
        )
    )

    # ---- Function / Class chunks --------------------------------------------
    _FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        outgoing_calls: List[str] = []
        child_nodes: List[str] = []

        if isinstance(node, ast.ClassDef):
            # Bases become outgoing calls (inheritance)
            for base in node.bases:
                name = _call_name(base)
                if name:
                    outgoing_calls.append(name)
            # Direct methods become child nodes
            for child in ast.iter_child_nodes(node):
                if isinstance(child, _FUNC_TYPES):
                    child_nodes.append(child.name)

        # All function calls within this scope
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                name = _call_name(sub.func)
                if name:
                    outgoing_calls.append(name)

        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", len(source_lines))
        source = "\n".join(source_lines[start - 1 : end])

        chunks.append(
            ASTNodePayload(
                file_path=file_path,
                node_type=type(node).__name__,
                node_name=node.name,
                source_code=source,
                start_line=start,
                end_line=end,
                outgoing_calls=list(dict.fromkeys(outgoing_calls)),  # dedupe, preserve order
                child_nodes=child_nodes,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_zip_to_ast_chunks(
    zip_bytes: bytes,
) -> Tuple[List[ASTNodePayload], Dict[str, str]]:
    """Extracts a ZIP archive and parses every .py file into AST chunks.

    Non-Python files and common noise directories are silently skipped.

    Args:
        zip_bytes: Raw bytes of the uploaded ZIP file.

    Returns:
        A tuple of:
        - ``semantic_chunk_matrix``: all semantic chunks across all Python files.
        - ``raw_files``: mapping of ``file_path → raw source text``.

    Raises:
        ZipExtractionError: If the ZIP cannot be opened or a member cannot be read.
        ASTParsingFailure:  If a .py file contains a syntax error.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as exc:
        raise ZipExtractionError(f"Failed to open ZIP: {exc}") from exc

    semantic_chunk_matrix: List[ASTNodePayload] = []
    raw_files: Dict[str, str] = {}
    parsed = skipped = 0

    for info in archive.infolist():
        # Skip directories
        if info.filename.endswith("/"):
            continue

        if _ext(info.filename) != ".py" or _should_skip(info.filename):
            skipped += 1
            continue

        try:
            with archive.open(info) as fh:
                raw_source = fh.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise ZipExtractionError(
                f"Cannot read {info.filename} from ZIP: {exc}"
            ) from exc

        raw_files[info.filename] = raw_source

        try:
            chunks = _parse_python(raw_source, info.filename)
            semantic_chunk_matrix.extend(chunks)
            parsed += 1
            logger.info("Parsed %s → %d chunks", info.filename, len(chunks))
        except ASTParsingFailure as exc:
            # One bad file should not abort the entire session
            logger.warning("Skipping %s (parse error): %s", info.filename, exc)
            skipped += 1

    logger.info(
        "ZIP processed: %d .py files parsed, %d skipped, %d total chunks",
        parsed,
        skipped,
        len(semantic_chunk_matrix),
    )
    return semantic_chunk_matrix, raw_files
