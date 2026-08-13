import pytest
from rag.quality_metrics import QualityMetricsCalculator

def test_calculate_ndcg_perfect_ranking():
    """Test NDCG with a perfect ranking."""
    results = [
        {"score": 1.0, "relevance_score": 1.0},
        {"score": 0.8, "relevance_score": 0.8},
        {"score": 0.6, "relevance_score": 0.6},
    ]
    # Perfect ranking should result in NDCG = 1.0
    score = QualityMetricsCalculator.calculate_ndcg(results, k=3)
    assert score == pytest.approx(1.0)

def test_calculate_ndcg_worst_ranking():
    """Test NDCG with a worst-case ranking."""
    results = [
        {"score": 0.1, "relevance_score": 0.1},
        {"score": 0.2, "relevance_score": 0.2},
        {"score": 1.0, "relevance_score": 1.0},
    ]
    # Worst ranking should be significantly lower than 1.0
    score = QualityMetricsCalculator.calculate_ndcg(results, k=3)
    assert score < 1.0
    assert score > 0.0

def test_calculate_ndcg_zero_relevance():
    """Test NDCG when no results are relevant."""
    results = [
        {"score": 0.0, "relevance_score": 0.0},
        {"score": 0.0, "relevance_score": 0.0},
        {"score": 0.0, "relevance_score": 0.0},
    ]
    score = QualityMetricsCalculator.calculate_ndcg(results, k=3)
    assert score == 0.0

def test_calculate_mrr_first_relevant():
    """Test MRR when the first result is relevant."""
    results = [
        {"score": 0.9, "relevance_score": 0.9}, # Relevant
        {"score": 0.1, "relevance_score": 0.1},
    ]
    score = QualityMetricsCalculator.calculate_mrr(results, relevance_threshold=0.5)
    assert score == 1.0

def test_calculate_mrr_second_relevant():
    """Test MRR when the second result is relevant."""
    results = [
        {"score": 0.1, "relevance_score": 0.1},
        {"score": 0.9, "relevance_score": 0.9}, # Relevant
    ]
    score = QualityMetricsCalculator.calculate_mrr(results, relevance_threshold=0.5)
    assert score == 0.5

def test_calculate_mrr_none_relevant():
    """Test MRR when no results are relevant."""
    results = [
        {"score": 0.1, "relevance_score": 0.1},
        {"score": 0.2, "relevance_score": 0.2},
    ]
    score = QualityMetricsCalculator.calculate_mrr(results, relevance_threshold=0.5)
    assert score == 0.0
