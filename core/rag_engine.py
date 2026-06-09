"""LangGraph-powered RAG orchestration engine.

Pipeline:
  1. retrieve  - FAISS vector search + graph neighbour expansion
  2. map_evaluate - LLM evaluates each retrieved chunk for relevance
  3. reduce_generate - LLM synthesises a final Markdown report

A separate ``audit_full_codebase`` method runs a Map-Reduce sweep over all
indexed chunks for a comprehensive security and quality audit.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional, Set, TypedDict

import networkx as nx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential

from core.ast_parser import ASTNodePayload
from core.exceptions import LLMInferenceError
from core.indexer import CodeIndexer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit guard: free-tier APIs choke above ~2 RPM for large models
# ---------------------------------------------------------------------------
_INTER_BATCH_SLEEP: float = float(os.environ.get("API_BATCH_DELAY", 2.0))
_MAP_BATCH_SIZE: int = 5         # chunks per map-phase LLM call
_AUDIT_BATCH_SIZE: int = 5       # chunks per audit LLM call
_MAX_BATCH_CHARS: int = 40000    # max chars per batch


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class RagState(TypedDict):
    """Immutable-style state dict threaded through the LangGraph pipeline."""

    user_query: str
    context_payloads: Dict[str, str]
    combined_context: str
    mapped_findings: List[str]
    llm_response: str
    top_k: int
    expansion_depth: int


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class RagEngine:
    """Orchestrates FAISS retrieval, graph expansion, and LLM inference.

    Args:
        indexer: An initialised (or empty) :class:`~core.indexer.CodeIndexer`.
    """

    def __init__(self, indexer: CodeIndexer) -> None:
        self.indexer = indexer
        self.dependency_graph: nx.MultiDiGraph = nx.MultiDiGraph()

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set - inference will fail until a key is provided.")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.2,
            google_api_key=api_key,
        )
        self.workflow = self._build_workflow()

    # ------------------------------------------------------------------
    # LLM management
    # ------------------------------------------------------------------

    def update_llm(
        self,
        provider: str,
        api_key: str,
        model_name: Optional[str] = None,
    ) -> None:
        """Hot-swaps the underlying LLM (Gemini ↔ Groq).

        Args:
            provider:   ``"Gemini"`` or ``"Groq"`` (case-insensitive).
            api_key:    Provider API key.
            model_name: Optional model override; falls back to a sensible default.
        """
        if not api_key:
            logger.warning("update_llm called without an API key for %s.", provider)

        if provider.lower() == "groq":
            self.llm = ChatGroq(
                model=model_name or "llama-3.3-70b-versatile",
                temperature=0.2,
                groq_api_key=api_key,
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name or "gemini-2.5-pro",
                temperature=0.2,
                google_api_key=api_key,
            )

        logger.info("LLM updated → provider=%s model=%s", provider, model_name)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self, semantic_chunk_matrix: List[ASTNodePayload]) -> None:
        """Builds a directed dependency graph from parsed AST payloads.

        Nodes represent individual code entities (modules, classes, functions).
        Edges encode ``call``, ``contains``, and ``method`` relationships.

        Args:
            semantic_chunk_matrix: Output of :func:`~core.ast_parser.parse_zip_to_ast_chunks`.
        """
        self.dependency_graph.clear()

        # First pass: add all nodes and build name → node_id lookup
        name_to_ids: Dict[str, List[str]] = {}
        for payload in semantic_chunk_matrix:
            node_id = f"{payload['file_path']}::{payload['node_name']}"
            if not self.dependency_graph.has_node(node_id):
                self.dependency_graph.add_node(
                    node_id,
                    file_path=payload["file_path"],
                    node_type=payload["node_type"],
                    source_code=payload["source_code"],
                )
            name_to_ids.setdefault(payload["node_name"], []).append(node_id)

        # Second pass: add edges
        for payload in semantic_chunk_matrix:
            node_id = f"{payload['file_path']}::{payload['node_name']}"

            for call in payload.get("outgoing_calls", []):
                if call in name_to_ids:
                    for target in name_to_ids[call]:
                        self.dependency_graph.add_edge(node_id, target, edge_type="call")
                else:
                    # Try suffix match (e.g. "self.helper" → "helper")
                    resolved = False
                    for name, ids in name_to_ids.items():
                        if call.endswith(f".{name}"):
                            for target in ids:
                                self.dependency_graph.add_edge(node_id, target, edge_type="call")
                            resolved = True
                            break
                    if not resolved:
                        # External / stdlib call - add as a lightweight stub node
                        self.dependency_graph.add_edge(node_id, call, edge_type="call")

            for child in payload.get("child_nodes", []):
                child_id = f"{payload['file_path']}::{child}"
                edge_type = "contains" if payload["node_type"] == "Module" else "method"
                self.dependency_graph.add_edge(node_id, child_id, edge_type=edge_type)

        logger.info(
            "Dependency graph: %d nodes, %d edges",
            self.dependency_graph.number_of_nodes(),
            self.dependency_graph.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

    def _node_retrieve(self, state: RagState) -> RagState:
        """Retrieves the top-k most relevant chunks and expands via graph neighbours."""
        top_k = state.get("top_k", 15)
        expansion_depth = state.get("expansion_depth", 3)

        top_matches = self.indexer.search(state["user_query"], top_k=top_k)

        context_nodes: Set[str] = set()
        context_payloads: Dict[str, str] = {}

        for _dist, payload in top_matches:
            node_id = f"{payload['file_path']}::{payload['node_name']}"
            context_nodes.add(node_id)
            context_payloads[node_id] = payload["source_code"]

            if self.dependency_graph.has_node(node_id):
                neighbours = (
                    list(self.dependency_graph.successors(node_id))[:expansion_depth]
                    + list(self.dependency_graph.predecessors(node_id))[:expansion_depth]
                )
                for nb in neighbours:
                    if nb not in context_nodes:
                        context_nodes.add(nb)
                        src = self.dependency_graph.nodes[nb].get("source_code", "")
                        if src:
                            context_payloads[nb] = src

        combined_context = "\n\n".join(
            f"--- {nid} ---\n{src}" for nid, src in context_payloads.items()
        )

        return {
            **state,
            "context_payloads": context_payloads,
            "combined_context": combined_context,
            "mapped_findings": [],
            "llm_response": "",
        }

    def _node_map_evaluate(self, state: RagState) -> RagState:
        """Batches context payloads through the LLM to find relevant findings."""
        payloads = list(state["context_payloads"].items())
        user_query = state["user_query"]

        system_prompt = (
            "You are a Python code auditor.\n"
            "Analyse the provided Python code chunks and determine whether each chunk "
            "contains or is directly related to the issue described in the USER QUERY.\n"
            "For any relevant chunk: describe the problem clearly and provide a corrected "
            "version of the code.\n"
            "If none of the chunks in this batch are relevant, reply EXACTLY with 'NO_ISSUE'.\n"
            "Do NOT fabricate issues that are not present."
        )

        mapped_findings: List[str] = []

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        def _invoke_with_retry(messages):
            return self.llm.invoke(messages)

        batches = []
        current_batch = []
        current_chars = 0
        for nid, src in payloads:
            chunk_chars = len(nid) + len(src)
            if current_batch and (len(current_batch) >= _MAP_BATCH_SIZE or current_chars + chunk_chars > _MAX_BATCH_CHARS):
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append((nid, src))
            current_chars += chunk_chars
        if current_batch:
            batches.append(current_batch)

        for i, batch in enumerate(batches):
            batch_ctx = "\n\n".join(
                f"--- {nid} ---\n{src}" for nid, src in batch
            )
            user_prompt = (
                f"USER QUERY: {user_query}\n\n"
                f"PYTHON CODE CHUNKS:\n{batch_ctx}"
            )
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            try:
                response = _invoke_with_retry(messages)
                content = str(response.content).strip()
                if "NO_ISSUE" not in content.upper() and len(content) > 10:
                    mapped_findings.append(content)
            except Exception as exc:
                logger.error("Map-phase batch %d failed after retries: %s", i + 1, exc)

            if _INTER_BATCH_SLEEP > 0 and i < len(batches) - 1:
                time.sleep(_INTER_BATCH_SLEEP)

        return {**state, "mapped_findings": mapped_findings}

    def _node_reduce_generate(self, state: RagState) -> RagState:
        """Synthesises all map-phase findings into a single Markdown report."""
        findings = state["mapped_findings"]

        if not findings:
            return {
                **state,
                "llm_response": (
                    "After analysing the most relevant parts of the codebase, "
                    "no instances of the queried issue were found."
                ),
            }

        system_prompt = (
            "You are a senior Python engineer compiling a code-audit report.\n"
            "Synthesise the provided findings into a clear, well-structured Markdown report.\n"
            "Group related issues, list every distinct occurrence, and provide exact, "
            "ready-to-apply code fixes for each. Use severity labels: "
            "[Critical], [High], [Medium], or [Low]."
        )
        combined = "\n\n---\n\n".join(findings)
        user_prompt = (
            f"USER QUERY: {state['user_query']}\n\n"
            f"FINDINGS FROM MAP PHASE:\n{combined}"
        )

        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = self.llm.invoke(messages)
            return {**state, "llm_response": str(response.content)}
        except Exception as exc:
            logger.error("Reduce-phase failed: %s", exc)
            raise LLMInferenceError(f"Reduce phase error: {exc}") from exc

    # ------------------------------------------------------------------
    # Full codebase audit
    # ------------------------------------------------------------------

    def audit_full_codebase(
        self,
        progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
    ) -> str:
        """Runs a comprehensive Map-Reduce audit over all indexed Python chunks.

        Processes chunks in batches of :data:`_AUDIT_BATCH_SIZE` to stay within
        API rate limits. A master reduction pass is skipped when there are no
        findings (clean codebase).

        Args:
            progress_callback: Optional ``(current, total, status_msg)`` callable
                               called after each batch.

        Returns:
            A Markdown string with all findings, or a "no issues" message.
        """
        chunks = self.indexer.semantic_chunk_matrix
        if not chunks:
            return "No codebase loaded. Please upload a Python ZIP first."

        system_prompt = (
            "You are a senior Python security and quality auditor.\n"
            "Review the following batch of Python code chunks.\n"
            "Identify ALL bugs, security vulnerabilities, anti-patterns, and "
            "code-quality issues present in this batch.\n"
            "If there are absolutely no issues, reply EXACTLY with 'NO_ISSUES'.\n"
            "Format output in strict Markdown. Tag every finding with a severity "
            "label: [Critical], [High], [Medium], or [Low].\n"
            "Include exact, corrected code for every issue you identify."
        )

        all_reports: List[str] = []

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        def _invoke_with_retry(messages):
            return self.llm.invoke(messages)

        batches = []
        current_batch = []
        current_chars = 0
        for c in chunks:
            chunk_chars = len(c['node_name']) + len(c['file_path']) + len(c['source_code'])
            if current_batch and (len(current_batch) >= _AUDIT_BATCH_SIZE or current_chars + chunk_chars > _MAX_BATCH_CHARS):
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(c)
            current_chars += chunk_chars
        if current_batch:
            batches.append(current_batch)

        total_batches = len(batches)
        chunks_processed = 0

        for i, batch in enumerate(batches):
            batch_num = i + 1

            combined = "\n\n".join(
                f"--- {c['node_name']} ({c['file_path']}) ---\n{c['source_code']}"
                for c in batch
            )
            user_prompt = f"CODE BATCH {batch_num} of {total_batches}:\n{combined}"

            if progress_callback:
                progress_callback(chunks_processed, len(chunks), f"Auditing batch {batch_num} of {total_batches}…")

            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
                response = _invoke_with_retry(messages)
                content = str(response.content).strip()
                if content and "NO_ISSUES" not in content.upper():
                    all_reports.append(content)
            except Exception as exc:
                logger.error("Audit batch %d failed after retries: %s", batch_num, exc)
                all_reports.append(f"**⚠️ Error in batch {batch_num}:** {exc}")

            chunks_processed += len(batch)
            if progress_callback:
                progress_callback(chunks_processed, len(chunks), None)

            if _INTER_BATCH_SLEEP > 0 and batch_num < total_batches:
                time.sleep(_INTER_BATCH_SLEEP)

        if not all_reports:
            return "✅ No issues found - the codebase passed all audit checks."

        return "### 🔍 Comprehensive Audit Findings\n\n" + "\n\n---\n\n".join(all_reports)

    # ------------------------------------------------------------------
    # Workflow construction
    # ------------------------------------------------------------------

    def _build_workflow(self) -> StateGraph:
        """Assembles and compiles the three-node LangGraph pipeline."""
        wf = StateGraph(RagState)

        wf.add_node("retrieve", self._node_retrieve)
        wf.add_node("map_evaluate", self._node_map_evaluate)
        wf.add_node("reduce_generate", self._node_reduce_generate)

        wf.add_edge(START, "retrieve")
        wf.add_edge("retrieve", "map_evaluate")
        wf.add_edge("map_evaluate", "reduce_generate")
        wf.add_edge("reduce_generate", END)

        return wf.compile()

    def query(
        self,
        user_query: str,
        top_k: int = 15,
        expansion_depth: int = 3,
    ) -> str:
        """Public convenience wrapper for :meth:`workflow.invoke`.

        Args:
            user_query:      Natural-language query or bug description.
            top_k:           Number of FAISS results to retrieve.
            expansion_depth: Graph hops to expand from each retrieved node.

        Returns:
            Markdown-formatted LLM response.

        Raises:
            LLMInferenceError: If the workflow execution fails.
        """
        initial: RagState = {
            "user_query": user_query,
            "context_payloads": {},
            "combined_context": "",
            "mapped_findings": [],
            "llm_response": "",
            "top_k": top_k,
            "expansion_depth": expansion_depth,
        }

        try:
            return self.workflow.invoke(initial)["llm_response"]
        except Exception as exc:
            logger.error("LangGraph workflow failed: %s", exc)
            raise LLMInferenceError(f"Workflow error: {exc}") from exc
