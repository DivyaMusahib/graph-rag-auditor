---
title: My RAG App
emoji: 🕵️‍♂️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Graph-RAG Auditor 🔍

Structure-aware static analysis and AI-powered bug remediation for **Python** codebases.

## What it does

1. **Parses** uploaded Python ZIP archives into typed AST chunks (Module, ClassDef, FunctionDef, AsyncFunctionDef) using Python's built-in `ast` module - no external tree-sitter dependency.
2. **Indexes** chunks into a FAISS flat-L2 vector store using `all-MiniLM-L6-v2` embeddings.
3. **Builds** a directed NetworkX dependency graph capturing call, contains, and method relationships.
4. **Queries** via a LangGraph Map-Reduce pipeline:
   - **Retrieve** - FAISS nearest-neighbour search + graph-neighbour expansion
   - **Map** - LLM evaluates each retrieved chunk for relevance to the query
   - **Reduce** - LLM synthesises a final Markdown audit report
5. **Full audit** - sweeps every indexed chunk in batches for a comprehensive security and quality report.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
streamlit run app.py
```

## Environment variables

| Variable         | Description                        |
|------------------|------------------------------------|
| `GEMINI_API_KEY` | Google Gemini API key              |
| `GROQ_API_KEY`   | Groq API key (optional)            |

## Project structure

```
graph-rag-auditor/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env                    # API keys (not committed)
├── core/
│   ├── ast_parser.py       # Python AST → ASTNodePayload chunks
│   ├── indexer.py          # FAISS vector index lifecycle
│   ├── rag_engine.py       # LangGraph pipeline + full audit
│   └── exceptions.py       # Custom exception hierarchy
└── ui/
    └── components.py       # Streamlit UI components (theme-aware)
```

## Tabs

| Tab | Description |
|-----|-------------|
| 🕸️ Graph View | Interactive agraph visualisation of the dependency graph |
| 💬 RAG Query | Targeted query with inline retrieval settings (Top K, expansion depth) |
| 🔍 Full Audit | Comprehensive Map-Reduce audit over the entire codebase |
| 📁 Codebase Explorer | Syntax-highlighted Python file viewer |
| 🧩 AST Chunks | Searchable, filterable list of all parsed chunks |
