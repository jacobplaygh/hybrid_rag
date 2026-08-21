"""Prometheus metrics for observability."""

import logging

try:
    from prometheus_client import Counter, Histogram, Gauge
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

logger = logging.getLogger(__name__)


# Query Metrics
if HAS_PROMETHEUS:
    query_counter = Counter(
        "rag_queries_total",
        "Total number of RAG queries",
        ["mode", "status"],
    )

    query_latency = Histogram(
        "rag_query_duration_seconds",
        "Query execution time",
        ["mode"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    retrieval_latency = Histogram(
        "rag_retrieval_duration_seconds",
        "Document retrieval time",
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
    )

    stream_first_chunk_latency = Histogram(
        "rag_stream_first_chunk_duration_seconds",
        "Time from stream start until the first response chunk",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    stream_duration = Histogram(
        "rag_stream_duration_seconds",
        "Total streaming response duration",
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )

    tokens_used = Counter(
        "rag_tokens_used_total",
        "Total tokens used in LLM calls",
        ["model"],
    )

    # Cache Metrics
    cache_hits = Counter(
        "rag_cache_hits_total",
        "Cache hits",
    )

    cache_misses = Counter(
        "rag_cache_misses_total",
        "Cache misses",
    )

    # Document Metrics
    documents_indexed = Gauge(
        "rag_documents_indexed",
        "Number of indexed documents",
    )

    document_chunks = Gauge(
        "rag_document_chunks_total",
        "Total number of document chunks",
    )

    # Index Metrics
    index_rebuild_counter = Counter(
        "rag_index_rebuilds_total",
        "Total index rebuilds",
    )

    # Quality Metrics
    ndcg_gauge = Gauge(
        "rag_ndcg_score",
        "Normalized Discounted Cumulative Gain score",
        ["mode"],
    )

    confidence_gauge = Gauge(
        "rag_confidence_score",
        "Overall response confidence score",
        ["mode"],
    )

    # Error Metrics
    errors_total = Counter(
        "rag_errors_total",
        "Total errors",
        ["error_type"],
    )
else:
    query_counter = None
    query_latency = None
    retrieval_latency = None
    stream_first_chunk_latency = None
    stream_duration = None
    tokens_used = None
    cache_hits = None
    cache_misses = None
    documents_indexed = None
    document_chunks = None
    index_rebuild_counter = None
    errors_total = None
    ndcg_gauge = None
    confidence_gauge = None


def record_query(mode: str, duration: float, status: str = "success", tokens: int = 0, model: str = "default"):
    """Record query metrics."""
    if not HAS_PROMETHEUS:
        return
    query_counter.labels(mode=mode, status=status).inc()
    query_latency.labels(mode=mode).observe(duration)
    if tokens > 0:
        tokens_used.labels(model=model).inc(tokens)


def record_retrieval(duration: float, doc_count: int = 0):
    """Record retrieval metrics."""
    if not HAS_PROMETHEUS:
        return
    retrieval_latency.observe(duration)


def record_error(error_type: str):
    """Record error metric."""
    if not HAS_PROMETHEUS:
        return
    errors_total.labels(error_type=error_type).inc()


def update_index_stats(doc_count: int, chunk_count: int):
    """Update index statistics."""
    if not HAS_PROMETHEUS:
        return
    documents_indexed.set(doc_count)
    document_chunks.set(chunk_count)
