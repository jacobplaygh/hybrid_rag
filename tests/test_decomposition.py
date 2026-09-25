import asyncio
import logging
from unittest.mock import AsyncMock, Mock

import pytest

from rag.hybrid_rag import HybridRAG
from api.config import get_settings
from rag.retrieval import HybridRetriever
from rag.reranker import CrossEncoderReranker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def validate_decomposition_recall():
    # 1. Setup Environment
    settings = get_settings()
    
    # Use the actual VectorStore implementation
    try:
        from data.vector_store import ChromaVectorStore
        vector_store = ChromaVectorStore()
    except ImportError:
        logger.error("ChromaVectorStore not found. Please ensure dependencies are installed.")
        return

    # Initialize real LLMs from settings
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    
    
    chat_llm = ChatNVIDIA(model=settings.DEFAULT_CHAT_MODEL)
    reasoning_llm = ChatNVIDIA(model=settings.DEFAULT_CHAT_MODEL)
    structured_llm = ChatNVIDIA(model=settings.DEFAULT_CHAT_MODEL)
    
    # Initialize HybridRAG with real LLMs
    rag = HybridRAG(
        vector_store=vector_store,
        chat_llm=chat_llm,
        reasoning_llm=reasoning_llm,
        structured_llm=structured_llm
    )
    await rag.ensure_keyword_index()

    # 2. Define a complex query that requires information from multiple sections
    # This query is designed to test the architecture described in overview.md
    complex_query = "Compare the performance of the new engine model with the previous version and explain how the cooling system was improved."

    # 3. Run in 'simple' mode
    logger.info("--- Running in SIMPLE mode ---")
    simple_result = await rag.query(query=complex_query, mode="simple")
    simple_docs = simple_result.get("retrieved_docs", [])
    simple_doc_ids = {doc.get("id") if isinstance(doc, dict) else doc.id for doc in simple_docs}
    
    logger.info(f"Simple mode retrieved {len(simple_docs)} documents.")

    # 4. Run in 'decompose' mode
    logger.info("--- Running in DECOMPOSE mode ---")
    decompose_result = await rag.query(query=complex_query, mode="decompose")
    decompose_docs = decompose_result.get("retrieved_docs", [])
    decompose_doc_ids = {doc.get("id") if isinstance(doc, dict) else doc.id for doc in decompose_docs}
    
    logger.info(f"Decompose mode retrieved {len(decompose_docs)} documents.")

    # 5. Analysis
    union_docs = simple_doc_ids.union(decompose_doc_ids)
    new_docs = decompose_doc_ids - simple_doc_ids
    
    logger.info("--- Analysis Results ---")
    logger.info(f"Unique documents found in simple mode: {len(simple_doc_ids)}")
    logger.info(f"Unique documents found in decompose mode: {len(decompose_doc_ids)}")
    logger.info(f"Additional documents found by decomposition: {len(new_docs)}")
    
    if len(new_docs) > 0:
        logger.info("✅ SUCCESS: Query decomposition improved recall by finding additional relevant documents.")
    else:
        logger.info("ℹ️ INFO: Query decomposition did not find additional unique documents, but may have improved ranking.")

    # 6. Compare Synthesis
    logger.info("\n--- Response Comparison ---")
    simple_answer = simple_result.get('answer') or "No answer"
    decompose_answer = decompose_result.get('answer') or "No answer"
    logger.info(f"Simple Response: {simple_answer[:200]}...")
    logger.info(f"Decompose Response: {decompose_answer[:200]}...")

@pytest.mark.asyncio
async def test_decompose_query_uses_default_top_k():
    """The decompose helper should not reference an undefined top_k variable."""
    rag = HybridRAG.__new__(HybridRAG)
    rag.chains = {
        "decompose": Mock(
            decompose=AsyncMock(return_value=["What is Python?"]),
            rewrite=AsyncMock(return_value="What is Python?"),
            invoke=AsyncMock(return_value="synthetic answer"),
        ),
        "simple": Mock(),
    }
    rag.retriever = Mock()
    rag.retriever.retrieve = AsyncMock(return_value=([{"content": "python doc", "source": "doc"}], "retrieval_id"))
    rag.reranker = Mock()
    rag.reranker.rerank = Mock(return_value=[{"content": "python doc", "source": "doc"}])
    rag._build_context_text = Mock(return_value="python context")
    rag._simple_rag = AsyncMock(return_value="answer")

    result = await rag._decompose_query("What is Python?", "context", llm=None)

    assert result == "synthetic answer"
    rag.retriever.retrieve.assert_awaited_once_with("What is Python?", top_k=5, alpha=0.7)


if __name__ == "__main__":
    asyncio.run(validate_decomposition_recall())
