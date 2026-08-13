# RAG Quality Observability Dashboard

This document describes the Grafana dashboard configuration for monitoring the quality and confidence of the Hybrid RAG system.

## 1. Dashboard Overview
The Quality Dashboard provides real-time visibility into the retrieval and generation performance of the RAG pipeline, allowing developers to detect regressions in ranking quality or LLM confidence.

## 2. Key Panels & Metrics

### A. Retrieval Quality (NDCG)
- **Metric**: `rag_ndcg_score`
- **Visualization**: Time Series / Gauge
- **Description**: Tracks the Normalized Discounted Cumulative Gain (NDCG) across different query modes.
- **PromQL**: `avg(rag_ndcg_score) by (mode)`
- **Insight**: A drop in NDCG indicates that the hybrid retriever or reranker is failing to place the most relevant documents at the top of the results.

### B. Response Confidence
- **Metric**: `rag_confidence_score`
- **Visualization**: Time Series / Heatmap
- **Description**: Tracks the overall confidence score assigned by the `ConfidenceScorer`.
- **PromQL**: `avg(rag_confidence_score) by (mode)`
- **Insight**: Low confidence scores suggest that the LLM is struggling to find a definitive answer in the provided context or that the context is insufficient.

### C. Quality Correlation
- **Visualization**: Scatter Plot
- **X-Axis**: `rag_ndcg_score`
- **Y-Axis**: `rag_confidence_score`
- **Insight**: Helps identify if high retrieval quality (NDCG) consistently leads to high response confidence.

### D. System Health Context
- **Latency**: `rag_query_duration_seconds`
- **Throughput**: `rate(rag_queries_total[5m])`
- **Error Rate**: `rate(rag_errors_total[5m])`

## 3. Alerting Rules

| Alert Name | Condition | Severity | Action |
|-----------|-----------|----------|--------|
| Low NDCG | `avg(rag_ndcg_score) < 0.6` | Warning | Review recent index updates or reranker model. |
| Confidence Drop | `avg(rag_confidence_score) < 0.4` | Critical | Investigate LLM prompt or document coverage. |
| High Error Rate | `rate(rag_errors_total[1m]) > 5%` | Critical | Check LLM API availability and system logs. |

## 4. Implementation Notes
- Metrics are exposed via the `/metrics` endpoint using the `prometheus_client` library.
- The `HybridRAG` orchestrator updates these gauges on every successful query.
- Dashboard is configured to filter by `mode` (simple, multi-turn, decompose, structured).

## 5. Query Analytics Integration
In addition to real-time metrics, the system implements a persistent analytics store:
- **Storage**: Queries are logged to `data/analytics/query_logs.jsonl`.
- **Analysis**: The `/analytics/summary` endpoint provides distributions of intents, modes, and cache hit rates.
- **Failure Analysis**: The `/analytics/failures` endpoint allows developers to isolate low-confidence or errored queries for targeted dataset improvement.
