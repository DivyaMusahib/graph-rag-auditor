"""Streamlit UI component library for Graph-RAG Auditor.

All CSS uses Streamlit's native theming variables so components render
correctly in both light and dark mode without any manual theme detection.
"""

from __future__ import annotations

import hashlib
import json
from typing import List

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import Config, Edge, Node, agraph

from core.ast_parser import ASTNodePayload


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700;800&display=swap');

        /* ── Base typography & Background ── */
        html, body, [class*="css"], .stApp {
            font-family: 'Space Grotesk', 'Inter', sans-serif !important;
            background-color: #080C14 !important;
            color: #E2E8F0 !important;
        }

        /* ── Grid background ── */
        .stApp::before {
            content: '';
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(0, 149, 255, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 149, 255, 0.04) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none;
            z-index: 0;
        }

        /* Hide default header */
        header[data-testid="stHeader"] {
            display: none !important;
        }

        /* ── Top Navbar ── */
        .top-navbar {
            position: fixed;
            top: 0.75rem;
            left: 0;
            right: 0;
            margin: 0 auto;
            width: max-content;
            max-width: 90vw;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(10, 15, 28, 0.75);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(0, 149, 255, 0.15);
            border-radius: 999px;
            padding: 0.45rem 1.1rem;
            z-index: 99999;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(0,149,255,0.08);
            gap: 2rem;
        }

        .navbar-logo {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 700;
            font-size: 0.9rem;
            color: #ffffff;
            letter-spacing: -0.01em;
        }

        .navbar-logo .logo-icon {
            background: linear-gradient(135deg, #0095FF 0%, #FF8C00 100%);
            padding: 0.22rem 0.3rem;
            border-radius: 8px;
            font-size: 1rem;
            line-height: 1;
        }

        .navbar-links {
            display: flex;
            gap: 1.5rem;
            font-size: 0.82rem;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.55);
        }

        .navbar-links span:hover {
            color: #ffffff;
            cursor: pointer;
        }

        .navbar-btn {
            background: linear-gradient(135deg, #33A8FF, #0077FF);
            color: #ffffff !important;
            border: none;
            padding: 0.38rem 1.1rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: 'Space Grotesk', sans-serif;
        }

        .navbar-btn:hover {
            box-shadow: 0 0 18px rgba(0, 149, 255, 0.5);
            transform: translateY(-1px);
        }

        /* ── Hero section ── */
        .rag-hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            text-align: center;
            font-size: clamp(2.2rem, 4.5vw, 3.6rem);
            margin-top: 5.5rem;
            margin-bottom: 0.5rem;
            letter-spacing: -0.04em;
            line-height: 1.08;
            color: #ffffff;
        }

        .rag-hero-title span.gradient-text {
            background: linear-gradient(135deg, #0095FF 0%, #FF8C00 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .rag-hero-sub {
            text-align: center;
            opacity: 0.55;
            margin-bottom: 2rem;
            font-size: 0.92rem;
            font-weight: 400;
            color: #cbd5e1;
            line-height: 1.6;
        }

        /* ── Hero flowchart - single line ── */
        .hero-flow {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0;
            margin: 0 auto 3rem;
            flex-wrap: nowrap;
            overflow-x: auto;
            max-width: 100%;
            padding: 0.5rem 1rem;
        }
        .hero-flow-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.3rem;
            padding: 0.65rem 0.85rem;
            background: rgba(0, 149, 255, 0.07);
            border: 1px solid rgba(0, 149, 255, 0.18);
            border-radius: 10px;
            min-width: 90px;
            max-width: 110px;
            flex-shrink: 0;
            transition: border-color 0.2s;
            white-space: nowrap;
        }
        .hero-flow-step:hover {
            border-color: rgba(0, 149, 255, 0.45);
        }
        .hero-flow-step .step-icon { font-size: 1.2rem; line-height: 1; }
        .hero-flow-step .step-label {
            font-size: 0.63rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.65);
            text-align: center;
            white-space: normal;
        }
        .hero-flow-arrow {
            color: rgba(0, 149, 255, 0.45);
            font-size: 1rem;
            padding: 0 0.25rem;
            flex-shrink: 0;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            display: none !important;
        }

        /* Glassmorphic border containers */
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background-color: rgba(10, 15, 28, 0.55) !important;
            border: 1px solid rgba(0, 149, 255, 0.1) !important;
            border-radius: 14px !important;
            padding: 1.25rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        }

        /* ── Metric cards ── */
        .metric-row {
            display: flex;
            gap: 0.9rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }
        .metric-card {
            flex: 1;
            min-width: 130px;
            padding: 1.1rem 1.2rem;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            background: rgba(10, 15, 28, 0.7);
            text-align: left;
            position: relative;
            overflow: hidden;
            transition: all 0.25s ease;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            border-radius: 14px 14px 0 0;
        }
        .metric-card.accent-blue::before   { background: linear-gradient(90deg, #0095FF, #00C6FF); }
        .metric-card.accent-orange::before { background: linear-gradient(90deg, #FF8C00, #FFB347); }
        .metric-card.accent-green::before  { background: linear-gradient(90deg, #00C896, #2ECC71); }
        .metric-card.accent-purple::before { background: linear-gradient(90deg, #7C3AED, #A855F7); }

        .metric-card:hover {
            border-color: rgba(0, 149, 255, 0.25);
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        }
        .metric-card .metric-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.7rem;
        }
        .metric-card .metric-icon {
            font-size: 1rem;
            padding: 0.3rem;
            border-radius: 7px;
        }
        .metric-card .metric-label {
            font-size: 0.68rem;
            opacity: 0.55;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
        }
        .metric-card.accent-blue   .metric-value { color: #38BEFF; }
        .metric-card.accent-orange .metric-value { color: #FFB347; }
        .metric-card.accent-green  .metric-value { color: #2ECC71; }
        .metric-card.accent-purple .metric-value { color: #A855F7; }
        .metric-card .metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.4rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 0;
            letter-spacing: -0.03em;
        }

        .metric-card.accent-blue   .metric-icon { background: rgba(0,149,255,0.12); }
        .metric-card.accent-orange .metric-icon { background: rgba(255,140,0,0.12); }
        .metric-card.accent-green  .metric-icon { background: rgba(46,204,113,0.12); }
        .metric-card.accent-purple .metric-icon { background: rgba(168,85,247,0.12); }

        /* ── VS Code style file tree ── */
        .vsc-explorer {
            background: rgba(10, 15, 28, 0.8);
            border: 1px solid rgba(0, 149, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        .vsc-explorer-header {
            padding: 0.5rem 0.75rem;
            background: rgba(0, 149, 255, 0.06);
            border-bottom: 1px solid rgba(0, 149, 255, 0.08);
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.45);
        }
        .vsc-file-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.3rem 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: rgba(255,255,255,0.65);
            cursor: pointer;
            border-left: 2px solid transparent;
            transition: all 0.15s;
        }
        .vsc-file-item:hover {
            background: rgba(0, 149, 255, 0.08);
            color: #fff;
        }
        .vsc-file-item.active {
            background: rgba(0, 149, 255, 0.12);
            border-left-color: #0095FF;
            color: #fff;
        }
        .vsc-file-icon { font-size: 0.9rem; flex-shrink: 0; }

        /* ── File tree (Radio Override for VSCode look) ── */
        div[data-testid="stRadio"] label > div:first-child {
            display: none !important;
        }
        div[data-testid="stRadio"] label {
            display: flex !important;
            align-items: center;
            padding: 0.28rem 0.75rem !important;
            border-radius: 0 !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.79rem !important;
            cursor: pointer;
            transition: background 0.15s ease;
            border-left: 2px solid transparent !important;
            color: rgba(255,255,255,0.6) !important;
            gap: 0.4rem;
        }
        div[data-testid="stRadio"] label:hover {
            background: rgba(0, 149, 255, 0.08) !important;
            color: rgba(255,255,255,0.9) !important;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            background: rgba(0, 149, 255, 0.12) !important;
            border-left-color: #0095FF !important;
            color: #fff !important;
        }

        /* ── Section headers ── */
        .section-header {
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            color: #FF8C00;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        /* ── Chunk badges ── */
        .chunk-badge {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }
        .badge-module   { background: rgba(0,149,255,0.12); color: #38BEFF; border: 1px solid rgba(0,149,255,0.25); }
        .badge-class    { background: rgba(168,85,247,0.12); color: #C084FC; border: 1px solid rgba(168,85,247,0.25); }
        .badge-function { background: rgba(46,204,113,0.12); color: #4ADE80; border: 1px solid rgba(46,204,113,0.25); }
        .badge-async    { background: rgba(255,140,0,0.12); color: #FFA726; border: 1px solid rgba(255,140,0,0.25); }

        /* ── AST Chunk card pop-beside layout ── */
        .chunk-row {
            display: flex;
            gap: 0;
            margin-bottom: 0.5rem;
        }
        .chunk-item {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.5rem 0.75rem;
            background: rgba(10, 15, 28, 0.6);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.82rem;
        }
        .chunk-item:hover {
            border-color: rgba(0, 149, 255, 0.3);
            background: rgba(0, 149, 255, 0.06);
        }

        /* ── Meta card for AST chunk detail ── */
        .ast-meta-card {
            background: rgba(10, 15, 28, 0.8);
            border: 1px solid rgba(0, 149, 255, 0.15);
            border-radius: 10px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
        }
        .ast-meta-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.3rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .ast-meta-row:last-child { border-bottom: none; }
        .ast-meta-key { color: rgba(255,255,255,0.4); font-size: 0.7rem; }
        .ast-meta-val { color: #38BEFF; font-size: 0.75rem; font-weight: 600; }

        /* ── Query result panels ── */
        .result-panel {
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 1.1rem;
            margin-top: 0.5rem;
            background: rgba(255, 255, 255, 0.02);
        }
        .result-panel-header {
            font-size: 0.72rem;
            font-weight: 700;
            opacity: 0.55;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.75rem;
        }

        /* ── Buttons - bright blue primary, orange secondary ── */
        .stButton > button {
            border-radius: 8px !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.83rem !important;
            background-color: rgba(0, 149, 255, 0.06) !important;
            border: 1px solid rgba(0, 149, 255, 0.25) !important;
            color: #fff !important;
            transition: all 0.2s ease !important;
            letter-spacing: 0.01em;
        }
        .stButton > button:hover {
            border-color: #0095FF !important;
            background-color: rgba(0, 149, 255, 0.12) !important;
            box-shadow: 0 0 16px rgba(0, 149, 255, 0.22) !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0095FF 0%, #0060CC 100%) !important;
            border: none !important;
            color: #ffffff !important;
            box-shadow: 0 2px 12px rgba(0, 149, 255, 0.3) !important;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 4px 22px rgba(0, 149, 255, 0.5) !important;
            transform: translateY(-1px);
        }

        /* Orange accent buttons (samples) */
        .stButton > button[data-variant="sample"] {
            border-color: rgba(255, 140, 0, 0.3) !important;
            color: #FFB347 !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
            background-color: rgba(10, 15, 28, 0.6);
            padding: 0.35rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 1.5rem;
            display: inline-flex;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600;
            font-size: 0.78rem;
            border: none !important;
            background: transparent;
            padding: 0.3rem 0.75rem;
            border-radius: 8px;
            color: rgba(255,255,255,0.45) !important;
            letter-spacing: 0.01em;
            transition: all 0.2s;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0,149,255,0.18), rgba(0,96,204,0.1)) !important;
            color: #ffffff !important;
            box-shadow: 0 0 12px rgba(0,149,255,0.2);
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        /* ── Inputs ── */
        .stTextInput input, .stSelectbox [data-baseweb="select"] {
            background-color: rgba(10, 15, 28, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 8px !important;
            color: #E2E8F0 !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }
        .stTextInput input:focus {
            border-color: rgba(0, 149, 255, 0.5) !important;
            box-shadow: 0 0 0 2px rgba(0, 149, 255, 0.15) !important;
        }

        /* ── Expanders ── */
        .streamlit-expanderHeader {
            font-weight: 600;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-family: 'Space Grotesk', sans-serif !important;
        }

        /* ── Empty state ── */
        .empty-state {
            text-align: center;
            padding: 5rem 2rem;
            background: rgba(0, 149, 255, 0.02);
            border: 1px dashed rgba(0, 149, 255, 0.15);
            border-radius: 16px;
        }
        .empty-state svg {
            width: 3.5rem;
            height: 3.5rem;
            margin-bottom: 1rem;
            color: #0095FF;
            opacity: 0.6;
        }

        /* Load codebase section label */
        .panel-section-label {
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.3);
            display: block;
            margin-bottom: 0.15rem;
        }
        .panel-section-title {
            font-size: 1rem;
            font-weight: 700;
            color: #fff;
            line-height: 1.2;
            display: block;
            margin-bottom: 0.75rem;
        }

        /* Tab buttons size fix */
        .stTabs [data-baseweb="tab-list"] button {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 130px;
        }

        /* Provider radio */
        div[data-testid="stRadio"] > div {
            gap: 0.4rem;
        }

        /* Warnings/Info styling */
        .stAlert {
            border-radius: 10px !important;
            font-size: 0.83rem !important;
        }

        /* Code blocks */
        .stCodeBlock {
            border-radius: 8px !important;
        }
        pre {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.8rem !important;
        }

        /* Caption text */
        .stCaption {
            font-size: 0.75rem !important;
            opacity: 0.55 !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metric row
# ---------------------------------------------------------------------------

def render_index_metrics(
    num_files: int,
    num_chunks: int,
    num_nodes: int,
    num_edges: int,
) -> None:
    st.markdown(
        '<div class="metric-row">'
        '<div class="metric-card accent-blue">'
        '<div class="metric-header"><span class="metric-icon">🗂</span><span class="metric-label">Python Files</span></div>'
        '<div class="metric-value">' + f"{num_files:,}" + '</div>'
        '</div>'

        '<div class="metric-card accent-orange">'
        '<div class="metric-header"><span class="metric-icon">⬡</span><span class="metric-label">AST Chunks</span></div>'
        '<div class="metric-value">' + f"{num_chunks:,}" + '</div>'
        '</div>'

        '<div class="metric-card accent-green">'
        '<div class="metric-header"><span class="metric-icon">◉</span><span class="metric-label">Graph Nodes</span></div>'
        '<div class="metric-value">' + f"{num_nodes:,}" + '</div>'
        '</div>'

        '<div class="metric-card accent-purple">'
        '<div class="metric-header"><span class="metric-icon">↗</span><span class="metric-label">Graph Edges</span></div>'
        '<div class="metric-value">' + f"{num_edges:,}" + '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------------

_FILE_COLOR_CACHE: dict[str, str] = {}


def _file_color(file_path: str) -> str:
    """Deterministically maps a file path to a bright, saturated hex colour."""
    if file_path not in _FILE_COLOR_CACHE:
        digest = hashlib.md5(file_path.encode()).hexdigest()
        h = int(digest[0:3], 16) % 360  # hue 0–359
        # Bright saturated palette via HSL → RGB approximation
        import colorsys
        r, g, b = colorsys.hls_to_rgb(h / 360.0, 0.62, 0.92)
        _FILE_COLOR_CACHE[file_path] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return _FILE_COLOR_CACHE[file_path]


def render_dependency_graph(graph: nx.DiGraph) -> None:
    if not graph or graph.number_of_nodes() == 0:
        st.info("No dependency graph available. Upload a codebase to generate one.")
        return

    st.markdown(
        "<div class='section-header'>⬡ Interactive Code Topology</div>",
        unsafe_allow_html=True,
    )

    total_nodes = graph.number_of_nodes()
    safe_max_val = max(2, min(500, total_nodes))

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        max_nodes = st.slider(
            "Max nodes to display",
            min_value=1,
            max_value=safe_max_val,
            value=min(150, total_nodes),
            step=10 if safe_max_val > 20 else 1,
        )
    with col_b:
        layout_physics = st.checkbox("Physics simulation", value=True)
    with col_c:
        show_external = st.checkbox("External nodes", value=False)

    nodes: list[Node] = []
    edges: list[Edge] = []

    internal_nodes = {
        nid
        for nid, data in graph.nodes(data=True)
        if data.get("file_path", "Unknown") != "Unknown"
    }
    candidate_nodes = internal_nodes if not show_external else set(graph.nodes())
    by_degree = sorted(
        candidate_nodes, key=lambda n: graph.degree(n), reverse=True
    )[:max_nodes]
    display_set = set(by_degree)

    for node_id in display_set:
        data = graph.nodes[node_id]
        node_type = data.get("node_type", "External")
        file_path = data.get("file_path", "Unknown")

        color = "#555577" if file_path == "Unknown" else _file_color(file_path)
        size = 32 if node_type in ("ClassDef", "Module") else 22

        display_name = node_id.split("::")[-1] if "::" in node_id else node_id
        label = f"{display_name}"

        nodes.append(
            Node(
                id=node_id,
                label=label,
                size=size,
                color=color,
                shape="dot",
                font={"size": 11, "face": "Space Grotesk", "color": "#FFFFFF"},
                title=f"📄 {file_path}\n🏷 {node_type}\n⬡ {display_name}",
            )
        )

    for src, tgt, data in graph.edges(data=True):
        if src not in display_set or tgt not in display_set:
            continue
        edge_type = data.get("edge_type", "call")
        color_map = {"call": "#0095FF", "contains": "#FF8C00", "method": "#A855F7"}
        edges.append(
            Edge(
                source=src,
                target=tgt,
                color=color_map.get(edge_type, "#666688"),
                width=1.5 if edge_type == "call" else 2.5,
                title=edge_type,
            )
        )

    config = Config(
        width="100%",
        height=540,
        directed=True,
        physics=layout_physics,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#FFFFFF",
        collapsible=False,
    )

    agraph(nodes=nodes, edges=edges, config=config)

    st.caption(
        "🔵 Call edge  ·  🟠 Contains edge  ·  🟣 Method edge  ·  Node colour = source file  ·  Node size = importance"
    )


# ---------------------------------------------------------------------------
# RAG query results
# ---------------------------------------------------------------------------

def render_query_results(combined_context: str, llm_response: str) -> None:
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            "<div class='result-panel-header'>📎 Retrieved Context</div>",
            unsafe_allow_html=True,
        )
        st.code(combined_context, language="python", line_numbers=True)

    with col2:
        st.markdown(
            "<div class='result-panel-header'>🤖 AI Audit Report</div>",
            unsafe_allow_html=True,
        )
        st.markdown(llm_response)


# ---------------------------------------------------------------------------
# Chunk badge helper
# ---------------------------------------------------------------------------

_BADGE_MAP = {
    "Module": "badge-module",
    "ClassDef": "badge-class",
    "FunctionDef": "badge-function",
    "AsyncFunctionDef": "badge-async",
}


def _chunk_badge(node_type: str) -> str:
    cls = _BADGE_MAP.get(node_type, "badge-module")
    label = {
        "Module": "module",
        "ClassDef": "class",
        "FunctionDef": "def",
        "AsyncFunctionDef": "async def",
    }.get(node_type, node_type.lower())
    return f"<span class='chunk-badge {cls}'>{label}</span>"


# ---------------------------------------------------------------------------
# AST chunk explorer  – collapsed by default, meta card beside
# ---------------------------------------------------------------------------

def render_ast_chunks_explorer(chunks: List[ASTNodePayload]) -> None:
    if not chunks:
        st.info("No AST chunks available. Upload a Python codebase to populate this view.")
        return

    # ── Filters ───────────────────────────────────────────────────────────
    fcol1, fcol2, fcol3 = st.columns([3, 2, 2])
    with fcol1:
        search_q = st.text_input(
            "Filter",
            placeholder="Search by name, file, or keyword…",
            label_visibility="collapsed",
        )
    with fcol2:
        type_filter = st.multiselect(
            "Type",
            options=["Module", "ClassDef", "FunctionDef", "AsyncFunctionDef"],
            default=[],
            label_visibility="collapsed",
            placeholder="All types",
        )
    with fcol3:
        all_files = sorted({c["file_path"] for c in chunks})
        file_filter = st.selectbox(
            "File",
            options=["All files"] + all_files,
            label_visibility="collapsed",
        )

    # ── Apply filters ──────────────────────────────────────────────────────
    filtered = chunks
    if search_q:
        q = search_q.lower()
        filtered = [
            c for c in filtered
            if q in c["node_name"].lower()
            or q in c["file_path"].lower()
            or q in c["source_code"].lower()
        ]
    if type_filter:
        filtered = [c for c in filtered if c["node_type"] in type_filter]
    if file_filter != "All files":
        filtered = [c for c in filtered if c["file_path"] == file_filter]

    st.caption(f"Showing **{len(filtered)}** of **{len(chunks)}** chunks")

    # ── Chunk list: collapsed, meta info INSIDE expander ─────────────────
    for i, chunk in enumerate(filtered):
        badge_html = _chunk_badge(chunk["node_type"])
        lines = chunk["end_line"] - chunk["start_line"] + 1
        calls = chunk.get("outgoing_calls", [])
        children = chunk.get("child_nodes", [])
        calls_preview = ", ".join(calls[:6]) + ("…" if len(calls) > 6 else "")
        children_preview = ", ".join(children[:4]) + ("…" if len(children) > 4 else "")

        with st.expander(
            f"#{i+1}  {chunk['node_name']}  ·  {chunk['file_path'].split('/')[-1]}  L{chunk['start_line']}–{chunk['end_line']}",
            expanded=False,
        ):
            # Header line with badge + path
            st.markdown(
                f"{badge_html} &nbsp;"
                f"<code style='font-family:JetBrains Mono,monospace;font-size:0.82rem;'>{chunk['node_name']}</code>"
                f"<span style='opacity:0.4;font-size:0.75rem;margin-left:0.6rem;'>{chunk['file_path']} L{chunk['start_line']}–{chunk['end_line']}</span>",
                unsafe_allow_html=True,
            )

            # Metadata row - inline pills
            st.markdown(
                f'<div class="ast-meta-card" style="margin:0.75rem 0;">'
                f'<div class="ast-meta-row"><span class="ast-meta-key">TYPE</span><span class="ast-meta-val">{chunk["node_type"]}</span></div>'
                f'<div class="ast-meta-row"><span class="ast-meta-key">FILE</span><span class="ast-meta-val" style="color:#FF8C00;">{chunk["file_path"]}</span></div>'
                f'<div class="ast-meta-row"><span class="ast-meta-key">LINES</span><span class="ast-meta-val">{chunk["start_line"]} – {chunk["end_line"]} &nbsp;({lines} total)</span></div>'
                f'<div class="ast-meta-row"><span class="ast-meta-key">CALLS</span><span class="ast-meta-val" style="color:#A855F7;">{calls_preview or "-"}</span></div>'
                f'<div class="ast-meta-row"><span class="ast-meta-key">CHILDREN</span><span class="ast-meta-val">{children_preview or "-"}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Source code
            st.code(chunk["source_code"], language="python", line_numbers=True)


# ---------------------------------------------------------------------------
# Codebase explorer - VS Code-style directory tree
# ---------------------------------------------------------------------------

def _build_tree(file_paths: list[str]) -> dict:
    """Build a nested dict tree from flat file paths."""
    tree: dict = {}
    for path in file_paths:
        parts = path.replace("\\", "/").split("/")
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
    return tree


def _render_tree_html(tree: dict, prefix: str = "", depth: int = 0) -> str:
    """Recursively render the tree as HTML list items."""
    html = ""
    items = sorted(tree.items(), key=lambda x: (len(x[1]) == 0, x[0]))  # dirs first
    for name, subtree in items:
        is_file = len(subtree) == 0
        indent = depth * 16
        if is_file:
            icon = "🐍" if name.endswith(".py") else "📄"
            full_path = (prefix + "/" + name).lstrip("/")
            html += (
                f"<div class='vsc-file-item' data-path='{full_path}' "
                f"style='padding-left:{12 + indent}px;' "
                f"onclick=\"selectFile('{full_path}')\">"
                f"<span class='vsc-file-icon'>{icon}</span>{name}</div>"
            )
        else:
            full_prefix = (prefix + "/" + name).lstrip("/")
            folder_id = full_prefix.replace("/", "_").replace(".", "_")
            html += (
                f"<div class='vsc-dir-item' style='padding-left:{12 + indent}px;' "
                f"onclick=\"toggleDir('{folder_id}')\">"
                f"<span class='vsc-dir-arrow' id='arr_{folder_id}'>▶</span>"
                f"<span style='margin-right:0.4rem;'>📁</span>{name}</div>"
                f"<div id='dir_{folder_id}' style='display:none;'>"
                + _render_tree_html(subtree, full_prefix, depth + 1)
                + "</div>"
            )
    return html


def render_codebase_explorer(raw_files: dict[str, str]) -> None:
    if not raw_files:
        st.info("No files available. Upload a Python codebase to populate this view.")
        return

    st.markdown(
        "<div style='margin-bottom: 1.25rem;'>"
        "<span style='font-size: 0.65rem; font-weight: 700; color: #FF8C00; letter-spacing: 0.12em; text-transform: uppercase;'>CODEBASE EXPLORER</span>"
        "<h2 style='margin: 0.15rem 0 0; font-size: 1.35rem; font-weight: 700; letter-spacing:-0.03em;'>Browse Indexed Source</h2>"
        "</div>",
        unsafe_allow_html=True,
    )

    sorted_files = sorted(raw_files.keys())
    tree = _build_tree(sorted_files)
    tree_html = _render_tree_html(tree)

    col_tree, col_view = st.columns([1, 2.8], gap="large")

    with col_tree:
        st.markdown("<div class='vsc-explorer-header'>⬡ Explorer</div>", unsafe_allow_html=True)
        
        # Load the custom component
        import os
        from streamlit.components import v1 as components
        tree_component = components.declare_component(
            "codebase_tree", 
            path=os.path.join(os.path.dirname(__file__), "tree_component")
        )
        
        selected_file = tree_component(tree_html=tree_html, key="explorer_selected_file")

    with col_view:
        if selected_file and selected_file in raw_files:
            content = raw_files[selected_file]
            file_size = len(content.encode("utf-8")) / 1024
            line_count = content.count("\n") + 1

            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"background:rgba(10,15,28,0.85);padding:0.45rem 1rem;"
                f"border-radius:8px 8px 0 0;border:1px solid rgba(0,149,255,0.12);border-bottom:none;'>"
                f"<span style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#E2E8F0;'>🐍 {selected_file}</span>"
                f"<span style='font-size:0.7rem;color:rgba(255,255,255,0.3);'>{line_count} lines · {file_size:.1f} KB · Python</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if len(content) > 15_000:
                st.warning(f"Large file ({len(content):,} chars) - syntax highlighting off.")
                st.text_area("Source", content, height=600, label_visibility="collapsed")
            else:
                st.code(content, language="python", line_numbers=True)


# ---------------------------------------------------------------------------
# Sidebar cache clear
# ---------------------------------------------------------------------------

def clear_sidebar_cache() -> None:
    st.html(
        """
        <script>
        (function clearSidebarCache() {
            try {
                const ls = window.parent.localStorage;
                const toRemove = [];
                for (let i = 0; i < ls.length; i++) {
                    const k = ls.key(i);
                    if (k && (k.includes("sidebar") || k.includes("stActiveSidebarState"))) {
                        toRemove.push(k);
                    }
                }
                toRemove.forEach(k => ls.removeItem(k));
            } catch (e) { /* sandboxed iframe - ignore */ }
        })();
        </script>
        """
    )
