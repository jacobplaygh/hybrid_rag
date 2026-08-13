import pytest
from unittest.mock import MagicMock
from rag.context_manager import ContextManager

class MockDoc:
    def __init__(self, content, source="doc"):
        self.content = content
        self.source = source

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
