# RAG quality metrics and dashboard suggestions

This document lists available Prometheus metrics and example panels for a future Grafana dashboard. It does not represent a provisioned dashboard or active alert rules.

`api/main.py` mounts `/metrics` when `prometheus_client` is importable. The `PROMETHEUS_ENABLED` setting currently does not gate that mount. Confirm that a metric is being updated before relying on it for an alert; the NDCG and confidence gauges are defined but are not currently updated by the query path.

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

## 3. Example alert thresholds (not configured)

| Alert Name | Condition | Severity | Action |
|-----------|-----------|----------|--------|
| Low NDCG | `avg(rag_ndcg_score) < 0.6` | Warning | Review recent index updates or reranker model. |
| Confidence Drop | `avg(rag_confidence_score) < 0.4` | Critical | Investigate LLM prompt or document coverage. |
| High Error Rate | `sum(rate(rag_errors_total[5m])) / clamp_min(sum(rate(rag_queries_total[5m])), 1) > 0.05` | Critical | Check model availability and system logs. |

## 4. Implementation Notes
- The metrics endpoint is mounted at `/metrics` in `api/main.py`.
- Metric definitions and labels are in `observability/metrics.py`.
- The NDCG and confidence gauges have a `mode` label. Query latency uses a `mode` label; query totals use `mode` and `status`.
- This repository does not provision Grafana panels or alert rules.

## 5. Query Analytics Integration
In addition to real-time metrics, the system implements a persistent analytics store:
- **Analysis**: `GET /analytics/summary` provides aggregate query-pattern analysis.
- **Latency**: `GET /analytics/latency` returns stage-level latency and cache correlation.
- **Failure Analysis**: `GET /analytics/failures` returns recorded failed or low-confidence queries.
