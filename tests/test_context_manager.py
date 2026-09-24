import pytest
from unittest.mock import MagicMock
from rag.context_manager import ContextManager

class MockDoc:
    def __init__(self, content, source="doc", doc_id=None, score=None):
        self.content = content
        self.source = source
        self.doc_id = doc_id
        self.score = score

def test_context_manager_truncation_respects_limit():
    # Set a very small token limit to force truncation
    # 'Hello world' is roughly 2-3 tokens. 
    # We'll set limit to 5 tokens.
    max_tokens = 5
    cm = ContextManager(max_tokens=max_tokens, system_prompt_reserve=0)
    
    # Create documents that definitely exceed 5 tokens
    docs = [
        MockDoc(content="This is a very long document that should be truncated.", source="doc1"),
        MockDoc(content="This is another long document that should also be truncated.", source="doc2"),
    ]
    
    # We don't provide query or history to keep it simple
    truncated_docs = cm.truncate_context(docs, query=None, history=None)
    
    # Calculate total tokens of the result
    total_tokens = cm.estimate_tokens(" ".join([d.content for d in truncated_docs]))
    
    assert total_tokens <= max_tokens, f"Total tokens {total_tokens} exceeded limit {max_tokens}"

def test_context_manager_no_truncation_needed():
    max_tokens = 1000
    cm = ContextManager(max_tokens=max_tokens, system_prompt_reserve=0)
    
    docs = [
        MockDoc(content="Short doc.", source="doc1"),
        MockDoc(content="Another short doc.", source="doc2"),
    ]
    
    truncated_docs = cm.truncate_context(docs, query=None, history=None)
    
    # Should keep all documents if they fit
    assert len(truncated_docs) == 2
    assert truncated_docs[0].content == "Short doc."
    assert truncated_docs[1].content == "Another short doc."

def test_context_manager_handles_empty_docs():
    cm = ContextManager(max_tokens=100)
    assert cm.truncate_context([], None, None) == []

def test_context_manager_handles_none_content():
    cm = ContextManager(max_tokens=100, system_prompt_reserve=0)
    docs = [MockDoc(content=None, source="doc1")]
    # Should not crash and handle None as empty string
    truncated_docs = cm.truncate_context(docs, None, None)
    assert len(truncated_docs) == 1


def test_context_manager_keeps_small_explicit_budget_usable():
    cm = ContextManager(max_tokens=20)
    docs = [MockDoc(content="A useful context passage.", source="doc1")]

    selected_docs = cm.truncate_context(docs, query="question", history=None)

    assert selected_docs
    assert selected_docs[0].content


def test_context_manager_deduplicates_retrieved_chunks_by_id():
    cm = ContextManager(max_tokens=1000, system_prompt_reserve=0)
    docs = [
        MockDoc("first copy", source="doc1", doc_id="chunk-1"),
        MockDoc("duplicate copy", source="doc1", doc_id="chunk-1"),
        MockDoc("different chunk", source="doc1", doc_id="chunk-2"),
    ]

    selected_docs = cm.truncate_context(docs, query=None, history=None)

    assert [doc.content for doc in selected_docs] == ["first copy", "different chunk"]


def test_context_manager_deduplicates_generic_documents_by_source_and_content():
    cm = ContextManager(max_tokens=1000, system_prompt_reserve=0)
    docs = [
        {"content": "same text", "source": "doc1"},
        {"content": "same text", "source": "doc1"},
        {"content": "same text", "source": "doc2"},
    ]

    selected_docs = cm.truncate_context(docs, query=None, history=None)

    assert selected_docs == [
        {"content": "same text", "source": "doc1"},
        {"content": "same text", "source": "doc2"},
    ]


def test_context_manager_limits_documents_per_source():
    cm = ContextManager(max_tokens=1000, system_prompt_reserve=0, max_docs_per_source=2)
    docs = [
        MockDoc("first", source="doc1", doc_id="chunk-1"),
        MockDoc("second", source="doc1", doc_id="chunk-2"),
        MockDoc("third", source="doc1", doc_id="chunk-3"),
        MockDoc("other source", source="doc2", doc_id="chunk-4"),
    ]

    selected_docs = cm.truncate_context(docs, query=None, history=None)

    assert [doc.content for doc in selected_docs] == ["first", "second", "other source"]


def test_context_manager_can_disable_source_limit():
    cm = ContextManager(max_tokens=1000, system_prompt_reserve=0, max_docs_per_source=None)
    docs = [
        MockDoc("first", source="doc1", doc_id="chunk-1"),
        MockDoc("second", source="doc1", doc_id="chunk-2"),
        MockDoc("third", source="doc1", doc_id="chunk-3"),
    ]

    selected_docs = cm.truncate_context(docs, query=None, history=None)

    assert len(selected_docs) == 3


def test_context_manager_reports_selected_and_dropped_documents():
    cm = ContextManager(max_tokens=1000, system_prompt_reserve=0, max_docs_per_source=1)
    docs = [
        MockDoc("first", source="doc1", doc_id="chunk-1"),
        MockDoc("duplicate", source="doc1", doc_id="chunk-1"),
        MockDoc("source limit", source="doc1", doc_id="chunk-2"),
        MockDoc("selected", source="doc2", doc_id="chunk-3"),
    ]

    cm.truncate_context(docs, query=None, history=None)

    assert cm.last_selection_report["selected"] == [
        {"id": "chunk-1", "source": "doc1", "truncated": False},
        {"id": "chunk-3", "source": "doc2", "truncated": False},
    ]
    assert cm.last_selection_report["dropped"] == [
        {"id": "chunk-1", "reason": "duplicate"},
        {"id": "chunk-2", "reason": "source_limit"},
    ]


def test_context_manager_reports_context_overhead_exhaustion():
    cm = ContextManager(max_tokens=5, system_prompt_reserve=5)
    docs = [MockDoc("content", source="doc1", doc_id="chunk-1")]

    assert cm.truncate_context(docs, query="a very long question", history=None) == []
    assert cm.last_selection_report["reason"] == "context_overhead"
    assert cm.last_selection_report["dropped"] == [
        {"id": "chunk-1", "reason": "context_overhead"},
    ]


def test_context_manager_orders_scored_documents_descending():
    cm = ContextManager(max_tokens=1000, system_prompt_reserve=0)
    docs = [
        MockDoc("low", source="doc1", doc_id="low", score=0.2),
        MockDoc("high", source="doc2", doc_id="high", score=0.9),
        MockDoc("middle", source="doc3", doc_id="middle", score=0.5),
    ]

    selected_docs = cm.truncate_context(docs, query=None, history=None)

    assert [doc.content for doc in selected_docs] == ["high", "middle", "low"]
    assert cm.last_selection_report["ordering"] == "score_descending"


def test_context_manager_preserves_order_when_scores_are_unavailable():
    cm = ContextManager(max_tokens=1000, system_prompt_reserve=0)
    docs = [
        {"content": "first", "source": "doc1"},
        {"content": "second", "source": "doc2", "score": "unknown"},
    ]

    selected_docs = cm.truncate_context(docs, query=None, history=None)

    assert [doc["content"] for doc in selected_docs] == ["first", "second"]
    assert cm.last_selection_report["ordering"] == "retrieval_order"
