"""Graph-RAG Auditor - main Streamlit entry point.

Structure-aware static analysis and AI-powered bug remediation for Python
codebases. Parses uploaded .py files into AST chunks, indexes them with
FAISS, builds a NetworkX dependency graph, then lets you query the system
with an LLM-backed Map-Reduce RAG pipeline.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
import zipfile

import git
import streamlit as st
from dotenv import load_dotenv

# ── Page config must be the very first Streamlit call ─────────────────────
st.set_page_config(
    page_title="Graph-RAG Auditor",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ── Heavy imports behind a spinner so the page renders immediately ─────────
with st.spinner("Booting Graph-RAG Engine ..."):
    from core.ast_parser import parse_zip_to_ast_chunks
    from core.indexer import CodeIndexer
    from core.rag_engine import RagEngine
    from ui.components import (
        apply_custom_css,
        clear_sidebar_cache,
        render_ast_chunks_explorer,
        render_codebase_explorer,
        render_dependency_graph,
        render_index_metrics,
        render_query_results,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-25s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

clear_sidebar_cache()


# ---------------------------------------------------------------------------
# Cached engine initialisation
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_engine() -> RagEngine:
    return RagEngine(CodeIndexer())


# ---------------------------------------------------------------------------
# Left Panel
# ---------------------------------------------------------------------------

def _render_left_panel(engine: RagEngine) -> None:
    with st.container(border=True):

        # ── Codebase Upload ────────────────────────────────────────────
        st.markdown(
            "<span class='panel-section-label' id='step-1'>Step 1</span>"
            "<span class='panel-section-title'>Load Codebase</span>",
            unsafe_allow_html=True,
        )

        upload_tab, gh_tab, samples_tab = st.tabs(["ZIP", "GitHub", "Samples"])

        with upload_tab:
            uploaded = st.file_uploader(
                "Python codebase (.zip)",
                type=["zip"],
                label_visibility="collapsed",
            )
            process_zip = st.button("⚡ Process ZIP", use_container_width=True)

        with samples_tab:
            _APP_DIR = os.path.dirname(os.path.abspath(__file__))
            _SAMPLES = {
                "📦 Repo Alpha · small": os.path.join(_APP_DIR, "samples", "small_error_optimized.zip"),
                "📦 Repo Beta · medium": os.path.join(_APP_DIR, "samples", "medium_error_optimized.zip"),
                "📦 Repo Gamma · large": os.path.join(_APP_DIR, "samples", "large_error_optimized.zip"),
            }
            for sample_name, path in _SAMPLES.items():
                if st.button(sample_name, use_container_width=True):
                    if os.path.exists(path):
                        with open(path, "rb") as fh:
                            zip_bytes_sample = fh.read()
                        st.session_state["_pending_zip"] = zip_bytes_sample
                        st.rerun()
                    else:
                        st.error(f"Sample not found: {path}")

        with gh_tab:
            gh_url = st.text_input(
                "Repository URL",
                value="https://github.com/DivyaMusahib/python-sample-testing-code-for-rag-project",
                label_visibility="collapsed",
            )
            process_gh = st.button("Clone & Process", use_container_width=True)

        zip_bytes: bytes | None = None

        if process_zip and uploaded:
            zip_bytes = uploaded.read()

        if process_gh and gh_url:
            with st.spinner("Cloning repository…"):
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        git.Repo.clone_from(gh_url, tmp)
                        buf = io.BytesIO()
                        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                            for root, _, files in os.walk(tmp):
                                if ".git" in root.replace("\\", "/").split("/"):
                                    continue
                                for fname in files:
                                    full = os.path.join(root, fname)
                                    arcname = os.path.relpath(full, tmp).replace("\\", "/")
                                    zf.write(full, arcname)
                        zip_bytes = buf.getvalue()
                except Exception as exc:
                    logger.error("Clone failed: %s", exc)
                    st.error(f"Clone failed: {exc}")

        if zip_bytes is None and "_pending_zip" in st.session_state:
            zip_bytes = st.session_state.pop("_pending_zip")

        if zip_bytes:
            with st.spinner("Parsing AST chunks and building index…"):
                try:
                    if "raw_files" in st.session_state:
                        del st.session_state["raw_files"]

                    chunks, raw_files = parse_zip_to_ast_chunks(zip_bytes)

                    st.session_state["raw_files"] = raw_files
                    st.session_state["semantic_chunk_matrix"] = chunks

                    engine.indexer.build_and_save_index(chunks)
                    engine.build_graph(chunks)

                    py_files = len({c["file_path"] for c in chunks})
                    st.success(f"Indexed {len(chunks)} chunks from {py_files} files.")
                except Exception as exc:
                    logger.exception("Processing failed")
                    st.error(f"Processing failed: {exc}")

        # ── Reset button lives here in the load section ────────────────
        if st.button("↺ Reset & Clear", use_container_width=True):
            engine.indexer.index = None
            engine.indexer.semantic_chunk_matrix = []
            engine.dependency_graph.clear()
            st.cache_resource.clear()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            if os.path.exists("storage"):
                shutil.rmtree("storage")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):

        # ── LLM Settings ──────────────────────────────────────────────
        st.markdown(
            "<span class='panel-section-label'>Step 2</span>"
            "<span class='panel-section-title'>Model Settings</span>",
            unsafe_allow_html=True,
        )

        provider = st.radio(
            "Provider",
            ["Gemini", "Groq"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if provider == "Groq":
            raw_key = st.text_input(
                "Groq API Key",
                type="password",
                value="",
                placeholder="sk-… (blank = site default key)",
                help="Leave blank to use the site developer's default Groq API key.",
            )
            key = raw_key or os.environ.get("GROQ_API_KEY", "")
            model = st.selectbox(
                "Model",
                ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"],
            )
        else:
            raw_key = st.text_input(
                "Gemini API Key",
                type="password",
                value="",
                placeholder="AI… (blank = site default key)",
                help="Leave blank to use the site developer's default Gemini API key.",
            )
            key = raw_key or os.environ.get("GEMINI_API_KEY", "")
            model = st.selectbox(
                "Model",
                ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
            )

        if key:
            engine.update_llm(provider, key, model)
        else:
            st.caption(f"No key provided - using default {provider} key.")

        if engine.indexer.index is not None:
            n_chunks = len(engine.indexer.semantic_chunk_matrix)
            n_files = len({c["file_path"] for c in engine.indexer.semantic_chunk_matrix})
            st.success(f"Index ready - {n_chunks} chunks · {n_files} files")


# ---------------------------------------------------------------------------
# Tab: Graph View
# ---------------------------------------------------------------------------

def _tab_graph(engine: RagEngine) -> None:
    render_dependency_graph(engine.dependency_graph)


# ---------------------------------------------------------------------------
# Tab: RAG Query
# ---------------------------------------------------------------------------

def _tab_rag_query(engine: RagEngine) -> None:
    st.markdown(
        "<div class='section-header'>◎ Targeted Auditor Query</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Retrieval Settings", expanded=False):
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            top_k = st.slider("Top K chunks", min_value=3, max_value=50, value=15, step=1)
        with rcol2:
            expansion_depth = st.slider("Graph expansion depth", min_value=0, max_value=10, value=3, step=1)

    user_query = st.text_input(
        "Query",
        placeholder="e.g. Find SQL injection vulnerabilities in the database layer",
        label_visibility="visible",
    )

    run_query = st.button("Run Query", type="primary", use_container_width=False)

    if run_query and user_query:
        with st.spinner("Retrieving context and running analysis…"):
            try:
                final_state = engine.workflow.invoke(
                    {
                        "user_query": user_query,
                        "context_payloads": {},
                        "combined_context": "",
                        "mapped_findings": [],
                        "llm_response": "",
                        "top_k": top_k,
                        "expansion_depth": expansion_depth,
                    }
                )
                render_query_results(
                    final_state.get("combined_context", ""),
                    final_state.get("llm_response", ""),
                )
            except Exception as exc:
                logger.exception("Query execution failed")
                st.error(f"Query failed: {exc}")
    elif run_query and not user_query:
        st.warning("Enter a query first.")


# ---------------------------------------------------------------------------
# Tab: Full Audit
# ---------------------------------------------------------------------------

def _tab_full_audit(engine: RagEngine) -> None:
    st.markdown(
        "<div class='section-header'>⚑ Comprehensive Codebase Audit</div>",
        unsafe_allow_html=True,
    )

    n_chunks = len(engine.indexer.semantic_chunk_matrix)
    n_batches = (n_chunks + 4) // 5
    st.info(f"**{n_chunks}** chunks → **{n_batches}** batches. ~2 s cooldown per batch for API rate limits.")

    if st.button("Run Full Audit", type="primary"):
        progress = st.progress(0.0)
        status = st.empty()

        def _cb(current: int, total: int, msg: str | None) -> None:
            if msg:
                status.text(msg)
            if total > 0:
                progress.progress(min(1.0, current / total))

        try:
            report = engine.audit_full_codebase(progress_callback=_cb)
            status.empty()
            progress.empty()
            st.markdown("---")
            st.markdown(report)
        except Exception as exc:
            logger.exception("Full audit failed")
            st.error(f"Audit failed: {exc}")


# ---------------------------------------------------------------------------
# Tab: Codebase Explorer
# ---------------------------------------------------------------------------

def _tab_explorer() -> None:
    raw_files = st.session_state.get("raw_files", {})
    render_codebase_explorer(raw_files)


# ---------------------------------------------------------------------------
# Tab: AST Chunks
# ---------------------------------------------------------------------------

def _tab_ast_chunks() -> None:
    st.markdown(
        "<div style='margin-bottom: 1.25rem;'>"
        "<span style='font-size: 0.65rem; font-weight: 700; color: #FF8C00; letter-spacing: 0.12em; text-transform: uppercase;'>AST CHUNKS</span>"
        "<h2 style='margin: 0.15rem 0 0; font-size: 1.35rem; font-weight: 700; letter-spacing:-0.03em;'>Structure-Aware Chunk Inspector</h2>"
        "</div>",
        unsafe_allow_html=True,
    )
    chunks = st.session_state.get("semantic_chunk_matrix", [])
    render_ast_chunks_explorer(chunks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_custom_css()

    # Top Navbar
    st.markdown(
        """
        <div class="top-navbar">
            <div class="navbar-logo">
                <span class="logo-icon">⬡</span>
                <span>Graph-RAG Auditor</span>
            </div>
            <div class="navbar-links">
                <a href="#how-it-works" style="color:inherit;text-decoration:none;" title="How Graph-RAG works">How It Works</a>
                <a href="#tech-stack" style="color:inherit;text-decoration:none;" title="Tech Stack">Tech Stack</a>
                <a href="https://github.com/DivyaMusahib/graph-rag-auditor" style="color:inherit;text-decoration:none;" title="View on GitHub">GitHub</a>
            </div>
            <a href="#step-1" class="navbar-btn" style="text-decoration:none;display:inline-block;color:#ffffff !important;">Upload Codebase &rarr;</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero header
    st.markdown(
        "<div style='text-align: center;margin-top:1rem;'>"
        "<span style='font-size:0.7rem; color:#FF8C00; font-weight:700; letter-spacing:0.1em; background:rgba(255,140,0,0.1); padding:0.22rem 0.8rem; border-radius:999px; border:1px solid rgba(255,140,0,0.2);'>Graph-RAG Engine</span>"
        "</div>"
        "<h1 class='rag-hero-title'>AI-Powered Python<br><span class='gradient-text'>Code Auditor</span></h1>"
        "<p class='rag-hero-sub'>Upload any Python codebase - ZIP or GitHub repo - and get structure-aware<br>bug detection, security audits, and dependency graph exploration powered by LLMs.</p>",
        unsafe_allow_html=True,
    )

    # How it works flowchart - single line, no wrap
    st.markdown(
        """
        <div class="hero-flow">
            <div class="hero-flow-step"><span class="step-icon">📦</span><span class="step-label">Upload ZIP / GitHub</span></div>
            <span class="hero-flow-arrow">→</span>
            <div class="hero-flow-step"><span class="step-icon">🌳</span><span class="step-label">AST Parsing</span></div>
            <span class="hero-flow-arrow">→</span>
            <div class="hero-flow-step"><span class="step-icon">⬡</span><span class="step-label">Dependency Graph</span></div>
            <span class="hero-flow-arrow">→</span>
            <div class="hero-flow-step"><span class="step-icon">🔍</span><span class="step-label">FAISS Index</span></div>
            <span class="hero-flow-arrow">→</span>
            <div class="hero-flow-step"><span class="step-icon">🤖</span><span class="step-label">LLM Map-Reduce</span></div>
            <span class="hero-flow-arrow">→</span>
            <div class="hero-flow-step"><span class="step-icon">📋</span><span class="step-label">Audit Report</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    engine = _get_engine()

    if engine.indexer.index is None:
        if engine.indexer.load_existing_index():
            engine.build_graph(engine.indexer.semantic_chunk_matrix)
            logger.info("Restored index from disk.")

    if (
        "semantic_chunk_matrix" not in st.session_state
        and engine.indexer.semantic_chunk_matrix
    ):
        st.session_state["semantic_chunk_matrix"] = engine.indexer.semantic_chunk_matrix
        st.session_state["raw_files"] = {
            "ℹ️ Notice": (
                "Raw files not available from cached index.\n"
                "Re-upload your ZIP to use the Codebase Explorer."
            )
        }

    col_left, col_main = st.columns([1, 2.8], gap="large")

    with col_left:
        _render_left_panel(engine)

    with col_main:
        index_ready = (
            engine.indexer.index is not None
            and engine.dependency_graph.number_of_nodes() > 0
        )

        if index_ready:
            st.markdown(
                "<span style='font-size: 0.65rem; font-weight: 700; color: #FF8C00; letter-spacing: 0.12em; text-transform: uppercase;'>CODEBASE METRICS</span>"
                "<h2 style='margin: 0.15rem 0 1rem; font-size: 1.35rem; font-weight: 700; letter-spacing:-0.03em;'>Live Snapshot</h2>",
                unsafe_allow_html=True,
            )
            chunks = engine.indexer.semantic_chunk_matrix
            py_files = len({c["file_path"] for c in chunks})
            render_index_metrics(
                num_files=py_files,
                num_chunks=len(chunks),
                num_nodes=engine.dependency_graph.number_of_nodes(),
                num_edges=engine.dependency_graph.number_of_edges(),
            )

            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                [
                    "⬡ Graph",
                    "◎ Query",
                    "⚑ Full Audit",
                    "🗂 Explorer",
                    "⬡ AST Chunks",
                ]
            )

            with tab1:
                _tab_graph(engine)
            with tab2:
                _tab_rag_query(engine)
            with tab3:
                _tab_full_audit(engine)
            with tab4:
                _tab_explorer()
            with tab5:
                _tab_ast_chunks()

        else:
            st.markdown(
                """
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                              d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>
                    </svg>
                    <p style="font-size:1rem;font-weight:700;color:#fff;margin-bottom:0.4rem;">No codebase loaded</p>
                    <p style="font-size:0.83rem;opacity:0.5;">Upload a Python ZIP or clone a GitHub repo using the panel on the left.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _render_info_section()


def _render_info_section() -> None:
    """Renders the bottom 'Why this project' information section."""
    st.html(
        """
        <div style="margin-top:5rem; padding-top:2.5rem; border-top: 1px solid rgba(0,149,255,0.1);">

        <div style="text-align:center; margin-bottom:3rem;" id="how-it-works">
            <span style="font-size:0.65rem;font-weight:700;color:#FF8C00;letter-spacing:0.14em;text-transform:uppercase;">Under the Hood</span>
            <h2 style="margin:0.4rem 0 0.6rem;font-size:1.9rem;font-weight:700;letter-spacing:-0.04em;color:#fff;">Why Graph-RAG - not just an LLM API call?</h2>
            <p style="font-size:0.9rem;color:rgba(255,255,255,0.5);max-width:580px;margin:0 auto;line-height:1.65;">
                A raw LLM call sees a flat blob of text. Graph-RAG understands the <em>structure</em> of your code
                - the call graph, the class hierarchy, the import chain - and uses that topology to pull the right context.
            </p>
        </div>

        <!-- Comparison cards -->
        <div style="display:flex;gap:1.2rem;margin-bottom:3rem;flex-wrap:wrap;">

            <div style="flex:1;min-width:260px;padding:1.5rem;border-radius:14px;background:rgba(255,60,60,0.04);border:1px solid rgba(255,60,60,0.15);">
                <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#FF4B4B;text-transform:uppercase;margin-bottom:0.75rem;">❌ Naive LLM API Call</div>
                <ul style="font-size:0.84rem;color:rgba(255,255,255,0.6);line-height:2;padding-left:1.2rem;margin:0;">
                    <li>Paste code → get generic answer</li>
                    <li>No awareness of which functions call which</li>
                    <li>Token limit forces arbitrary truncation</li>
                    <li>No cross-file context or import tracing</li>
                    <li>Same analysis quality on 5 lines or 500</li>
                    <li>Hallucinations with no code grounding</li>
                </ul>
            </div>

            <div style="flex:1;min-width:260px;padding:1.5rem;border-radius:14px;background:rgba(0,149,255,0.05);border:1px solid rgba(0,149,255,0.2);">
                <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#38BEFF;text-transform:uppercase;margin-bottom:0.75rem;">✓ Graph-RAG Auditor</div>
                <ul style="font-size:0.84rem;color:rgba(255,255,255,0.6);line-height:2;padding-left:1.2rem;margin:0;">
                    <li>AST-parsed structure-aware chunks</li>
                    <li>Dependency graph traces call paths</li>
                    <li>FAISS retrieves only relevant chunks</li>
                    <li>Graph expansion pulls caller/callee context</li>
                    <li>Map-Reduce scales to thousands of chunks</li>
                    <li>Grounded answers with file + line citations</li>
                </ul>
            </div>

            <div style="flex:1;min-width:260px;padding:1.5rem;border-radius:14px;background:rgba(255,140,0,0.04);border:1px solid rgba(255,140,0,0.15);">
                <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#FF8C00;text-transform:uppercase;margin-bottom:0.75rem;">⚡ What Makes It Fast</div>
                <ul style="font-size:0.84rem;color:rgba(255,255,255,0.6);line-height:2;padding-left:1.2rem;margin:0;">
                    <li>FAISS ANN search - sub-ms retrieval</li>
                    <li>Cached vector index survives restarts</li>
                    <li>Parallel Map step across chunk batches</li>
                    <li>Graph traversal in pure NetworkX (no DB)</li>
                    <li>Streamlit cache_resource for zero re-init</li>
                    <li>Gemini Flash Lite for cost-efficient queries</li>
                </ul>
            </div>
        </div>

        <!-- Tech stack row -->
        <div style="margin-bottom:3rem;" id="tech-stack">
            <div style="text-align:center;margin-bottom:1.5rem;">
                <span style="font-size:0.65rem;font-weight:700;color:#FF8C00;letter-spacing:0.14em;text-transform:uppercase;">Tech Stack</span>
            </div>
            <div style="display:flex;gap:0.75rem;flex-wrap:wrap;justify-content:center;">
                {tech_pills}
            </div>
        </div>

        <!-- Architecture note -->
        <div style="padding:1.5rem;border-radius:14px;background:rgba(0,149,255,0.04);border:1px solid rgba(0,149,255,0.1);margin-bottom:3rem;">
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#38BEFF;text-transform:uppercase;margin-bottom:0.75rem;">Architecture Note - Map-Reduce RAG</div>
            <p style="font-size:0.84rem;color:rgba(255,255,255,0.6);line-height:1.8;margin:0;">
                After retrieval, chunks are grouped into batches of 5 and each batch is sent to the LLM in parallel (the <strong style="color:#fff;">Map</strong> step).
                Each partial analysis is then merged in a second <strong style="color:#fff;">Reduce</strong> call that consolidates findings,
                deduplicates issues, and formats a final structured report. This lets the auditor handle codebases with 500+ chunks
                that would never fit in a single LLM context window - and keeps individual API calls small and fast.
            </p>
        </div>

        <!-- Footer line -->
        <div style="text-align:center;padding-bottom:2rem;font-size:0.75rem;color:rgba(255,255,255,0.2);">
            Graph-RAG Auditor · Built with Streamlit, LangGraph, FAISS, NetworkX · MIT License
        </div>

        </div>
        """.replace(
            "{tech_pills}",
            "".join(
                f"<span style='padding:0.3rem 0.85rem;border-radius:999px;font-size:0.72rem;font-weight:600;"
                f"background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.55);'>{t}</span>"
                for t in [
                    "Python 3.11+", "Streamlit", "LangGraph", "LangChain",
                    "FAISS", "NetworkX", "Google Gemini", "Groq / LLaMA",
                    "streamlit-agraph", "GitPython", "Python AST",
                ]
            ),
        )
    )


if __name__ == "__main__":
    main()
