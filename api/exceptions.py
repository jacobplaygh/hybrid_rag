"""Custom exceptions for RAG API."""


class RAGException(Exception):
    """Base exception for RAG system."""
    
    def __init__(self, message: str, error_code: str = None, status_code: int = 500):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.status_code = status_code
        super().__init__(self.message)


class QueryParseError(RAGException):
    """Query cannot be parsed or understood."""
    
    def __init__(self, message: str):
        super().__init__(message, "QUERY_PARSE_ERROR", 400)


class RetrievalError(RAGException):
    """Retrieval operation failed."""
    
    def __init__(self, message: str):
        super().__init__(message, "RETRIEVAL_ERROR", 500)


class GenerationError(RAGException):
    """LLM generation failed."""
    
    def __init__(self, message: str):
        super().__init__(message, "GENERATION_ERROR", 500)


class RateLimitError(RAGException):
    """Rate limit exceeded."""
    
    def __init__(self, message: str):
        super().__init__(message, "RATE_LIMIT_EXCEEDED", 429)


class ContextWindowExceededError(RAGException):
    """Context too large for model."""
    
    def __init__(self, message: str):
        super().__init__(message, "CONTEXT_WINDOW_EXCEEDED", 413)


class VectorStoreError(RAGException):
    """Vector store operation failed."""
    
    def __init__(self, message: str):
        super().__init__(message, "VECTOR_STORE_ERROR", 500)
