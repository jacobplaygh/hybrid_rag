"""Query analytics for analyzing user patterns and system performance."""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

class QueryAnalytics:
    """
    Handles logging and analysis of RAG queries to identify patterns,
    common failures, and intent distributions.
    """
    
    def __init__(self, storage_path: str = "data/analytics/query_logs.jsonl"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _create_log_entry(self) -> Dict[str, Any]:
        return {
            "timestamp": None,
            "query_id": None,
            "query": None,
            "mode": None,
            "intent": None,
            "response_time_ms": None,
            "tokens_used": None,
            "cache_status": None,
            "confidence_score": None,
            "status": None,
            "error": None
        }

    def log_query(self, query_data: Dict[str, Any]):
        """Log a query execution record to the analytics store."""
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **query_data
            }
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log query analytics: {e}")

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze logged queries to find common patterns and issues."""
        if not self.storage_path.exists():
            return {"error": "No analytics data available"}

        queries = []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                queries.append(json.loads(line))

        if not queries:
            return {"error": "No queries found in logs"}

        total = len(queries)
        intents = Counter([q.get("intent", "UNKNOWN") for q in queries])
        modes = Counter([q.get("mode", "UNKNOWN") for q in queries])
        statuses = Counter([q.get("status", "UNKNOWN") for q in queries])
        cache_stats = Counter([q.get("cache_status", "MISS") for q in queries])
        
        avg_latency = sum(q.get("response_time_ms", 0) for q in queries) / total
        
        return {
            "total_queries": total,
            "intent_distribution": {k: v/total for k, v in intents.items()},
            "mode_distribution": {k: v/total for k, v in modes.items()},
            "status_distribution": {k: v/total for k, v in statuses.items()},
            "cache_hit_rate": cache_stats.get("HIT", 0) / total,
            "average_latency_ms": avg_latency,
            "top_intents": intents.most_common(5)
        }

    def analyze_latency(self) -> Dict[str, Any]:
        """Summarize request, retrieval, and first-token latency by cache status."""
        if not self.storage_path.exists():
            return {"error": "No analytics data available"}

        queries = []
        with open(self.storage_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    queries.append(json.loads(line))

        if not queries:
            return {"error": "No queries found in logs"}

        def average(records: List[Dict[str, Any]], field: str) -> float:
            values = [float(record.get(field) or 0) for record in records]
            return sum(values) / len(values) if values else 0.0

        by_cache = {
            status: [
                query for query in queries
                if query.get("cache_status", "MISS") == status
            ]
            for status in ("HIT", "MISS")
        }

        return {
            "total_queries": len(queries),
            "average_query_time_ms": average(queries, "query_time_ms"),
            "average_response_time_ms": average(queries, "response_time_ms"),
            "average_retrieval_time_ms": average(queries, "retrieval_time_ms"),
            "average_first_token_time_ms": average(queries, "first_token_time_ms"),
            "cache_breakdown": {
                status: {
                    "queries": len(records),
                    "average_query_time_ms": average(records, "query_time_ms"),
                    "average_retrieval_time_ms": average(records, "retrieval_time_ms"),
                    "average_first_token_time_ms": average(records, "first_token_time_ms"),
                }
                for status, records in by_cache.items()
            },
        }

    def get_failed_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve a list of queries that resulted in errors or low confidence."""
        failures = []
        if not self.storage_path.exists():
            return []
            
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                if q.get("status") == "error" or q.get("confidence_score", 1.0) < 0.5:
                    failures.append(q)
        
        return failures[-limit:]
