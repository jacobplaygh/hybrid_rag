"""Run the A/B harness against the local HybridRAG query pipeline."""

import asyncio
from pathlib import Path
from typing import Any, Dict

from api.config import get_settings
from data.vector_store import ChromaVectorStore
from rag.hybrid_rag import HybridRAG


_rag: HybridRAG | None = None


def _chroma_path() -> str:
    configured = Path(get_settings().CHROMA_PERSIST_DIR)
    if configured.is_absolute():
        return str(configured)
    project_root = Path(__file__).resolve().parents[1]
    if configured.parts and configured.parts[0].lower() == project_root.name.lower():
        return str(project_root.parent / configured)
    return str(project_root / configured)


def _get_rag() -> HybridRAG:
    global _rag
    if _rag is None:
        vector_store = ChromaVectorStore(persist_dir=_chroma_path())
        _rag = HybridRAG(vector_store=vector_store)
        # Keep each variant independent when evaluating the same query set.
        _rag.semantic_cache = None
        _rag.query_cache.clear()
    return _rag


def _run(query: str, *, agentic: bool) -> Dict[str, Any]:
    rag = _get_rag()
    loop = rag.agentic_loop
    rag.query_cache.clear()
    rag.agentic_loop = loop if agentic else None
    try:
        result = asyncio.run(rag.query(query=query, mode="simple"))
    finally:
        rag.agentic_loop = loop

    if "error" in result:
        return {
            "status": "error",
            "confidence": 0.0,
            "latency_ms": result.get("query_time_ms", 0.0),
            "response": result.get("error", ""),
            "retrieved_docs": result.get("retrieved_docs", []),
        }
    return {
        "status": "success",
        "confidence": (
            result.get("agentic_loop", {}).get("confidence", 0.0)
            if agentic and result.get("agentic_loop")
            else result.get("confidence_score", 0.0)
        ) or 0.0,
        "latency_ms": result.get("query_time_ms", 0.0) or 0.0,
        "response": result.get("response", ""),
        "retrieved_docs": result.get("retrieved_docs", []),
    }


def baseline_run(query: str) -> Dict[str, Any]:
    """Evaluate a query with the agentic loop disabled."""
    return _run(query, agentic=False)


def agentic_run(query: str) -> Dict[str, Any]:
    """Evaluate a query with the configured agentic loop enabled."""
    return _run(query, agentic=True)
