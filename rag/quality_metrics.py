"""Quality metrics for evaluating retrieval results."""

from typing import List, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class QualityMetrics:
    """Holds quality metric values."""
    ndcg_at_3: float
    mrr: float
    precision_at_3: float
    recall_at_3: float
    
    def __repr__(self):
        return (
            f"QualityMetrics("
            f"ndcg@3={self.ndcg_at_3:.3f}, "
            f"mrr={self.mrr:.3f}, "
            f"precision@3={self.precision_at_3:.3f}, "
            f"recall@3={self.recall_at_3:.3f})"
        )


class QualityMetricsCalculator:
    """Calculate quality metrics for retrieval results."""
    
    @staticmethod
    def calculate_ndcg(results: List[Dict[str, Any]], k: int = 3) -> float:
        """
        Calculate NDCG@k (Normalized Discounted Cumulative Gain).
        
        NDCG measures ranking quality by comparing to ideal ranking.
        Higher values = better ranking quality.
        Range: 0.0 to 1.0
        
        Args:
            results: List of retrieved documents with relevance scores
            k: Number of results to consider
            
        Returns:
            NDCG@k score
        """
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, result in enumerate(results[:k]):
            # Use relevance_score if available, otherwise fallback to retrieval score
            relevance = float(result.get('relevance_score', result.get('score', 0)))
            # DCG formula: rel_i / log2(i+2)
            # Using i+2 so position 0 = log2(2) = 1
            dcg += relevance / math.log2(i + 2)
        
        # Calculate ideal DCG (perfect ranking)
        sorted_results = sorted(
            results,
            key=lambda x: float(x.get('relevance_score', x.get('score', 0))),
            reverse=True
        )
        idcg = 0.0
        for i, result in enumerate(sorted_results[:k]):
            relevance = float(result.get('relevance_score', result.get('score', 0)))
            idcg += relevance / math.log2(i + 2)
        
        # NDCG = DCG / IDCG (avoid division by zero)
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    @staticmethod
    def calculate_mrr(results: List[Dict[str, Any]], 
                     relevance_threshold: float = 0.5) -> float:
        """
        Calculate MRR (Mean Reciprocal Rank).
        
        MRR measures position of first relevant result.
        Range: 0.0 to 1.0
        
        Args:
            results: List of retrieved documents
            relevance_threshold: Minimum score for relevance
            
        Returns:
            MRR score
        """
        for i, result in enumerate(results):
            # Use relevance_score if available, otherwise fallback to retrieval score
            score = float(result.get('relevance_score', result.get('score', 0)))
            if score >= relevance_threshold:
                # Rank starts at 1
                return 1.0 / (i + 1)
        return 0.0  # No relevant result found
    
    @staticmethod
    def calculate_precision(results: List[Dict[str, Any]], 
                          k: int = 3,
                          relevance_threshold: float = 0.5) -> float:
        """
        Calculate Precision@k.
        
        Precision measures what fraction of retrieved docs are relevant.
        Range: 0.0 to 1.0
        
        Args:
            results: List of retrieved documents
            k: Number of results to consider
            relevance_threshold: Minimum score for relevance
            
        Returns:
            Precision@k score
        """
        if k == 0:
            return 0.0
        
        relevant_count = 0
        for result in results[:k]:
            # Use relevance_score if available, otherwise fallback to retrieval score
            score = float(result.get('relevance_score', result.get('score', 0)))
            if score >= relevance_threshold:
                relevant_count += 1
        
        return relevant_count / k
    
    @staticmethod
    def calculate_recall(results: List[Dict[str, Any]], 
                        total_relevant: int,
                        k: int = 3,
                        relevance_threshold: float = 0.5) -> float:
        """
        Calculate Recall@k.
        
        Recall measures what fraction of all relevant docs were found.
        Range: 0.0 to 1.0
        
        Args:
            results: List of retrieved documents
            total_relevant: Total number of relevant docs available
            k: Number of results to consider
            relevance_threshold: Minimum score for relevance
            
        Returns:
            Recall@k score
        """
        if total_relevant == 0:
            return 0.0
        
        relevant_count = 0
        for result in results[:k]:
            # Use relevance_score if available, otherwise fallback to retrieval score
            score = float(result.get('relevance_score', result.get('score', 0)))
            if score >= relevance_threshold:
                relevant_count += 1
        
        return relevant_count / total_relevant
    
    @classmethod
    def calculate_all(cls, 
                     results: List[Dict[str, Any]], 
                     total_relevant: int = None,
                     k: int = 3) -> QualityMetrics:
        """
        Calculate all quality metrics at once.
        
        Args:
            results: List of retrieved documents
            total_relevant: Total number of relevant docs (for recall)
            k: Number of results to consider
            
        Returns:
            QualityMetrics object with all metrics
        """
        ndcg = cls.calculate_ndcg(results, k=k)
        mrr = cls.calculate_mrr(results)
        precision = cls.calculate_precision(results, k=k)
        recall = cls.calculate_recall(
            results, 
            total_relevant or len(results),
            k=k
        )
        
        return QualityMetrics(
            ndcg_at_3=ndcg,
            mrr=mrr,
            precision_at_3=precision,
            recall_at_3=recall
        )
