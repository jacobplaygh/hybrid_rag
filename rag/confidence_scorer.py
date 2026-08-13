"""Confidence scoring for RAG responses."""

from typing import List, Dict, Any
from dataclasses import dataclass
import statistics


@dataclass
class ConfidenceScore:
    """Holds confidence scoring information."""
    overall_score: float  # 0.0 to 1.0
    relevance_score: float
    llm_confidence: float
    source_quality: float
    reasoning: List[str]
    
    def __repr__(self):
        return (
            f"ConfidenceScore(overall={self.overall_score:.2f}, "
            f"relevance={self.relevance_score:.2f}, "
            f"llm={self.llm_confidence:.2f}, "
            f"sources={self.source_quality:.2f})"
        )


class ConfidenceScorer:
    """Score confidence in RAG responses."""
    
    def __init__(self, 
                 relevance_weight: float = 0.4,
                 source_weight: float = 0.3,
                 llm_weight: float = 0.3):
        """
        Initialize scorer with weights.
        
        Args:
            relevance_weight: Weight for retrieval relevance (0.0-1.0)
            source_weight: Weight for source quality (0.0-1.0)
            llm_weight: Weight for LLM confidence (0.0-1.0)
        """
        total = relevance_weight + source_weight + llm_weight
        self.relevance_weight = relevance_weight / total
        self.source_weight = source_weight / total
        self.llm_weight = llm_weight / total
    
    def score_response(self,
                      retrieved_docs: List[Dict[str, Any]],
                      response: str,
                      context_length: int = None) -> ConfidenceScore:
        """
        Score overall confidence in response.
        
        Args:
            retrieved_docs: Documents used to generate response
            response: Generated response text
            context_length: Tokens used for generation
            
        Returns:
            ConfidenceScore object
        """
        reasoning = []
        
        # 1. Relevance Score: Based on retrieval quality
        relevance_score = self._score_relevance(retrieved_docs, reasoning)
        
        # 2. Source Quality Score: Based on document quality
        source_score = self._score_sources(retrieved_docs, reasoning)
        
        # 3. LLM Confidence: Based on response characteristics
        llm_score = self._score_llm_confidence(response, reasoning)
        
        # Overall: Weighted combination
        overall = (
            relevance_score * self.relevance_weight +
            source_score * self.source_weight +
            llm_score * self.llm_weight
        )
        
        return ConfidenceScore(
            overall_score=overall,
            relevance_score=relevance_score,
            llm_confidence=llm_score,
            source_quality=source_score,
            reasoning=reasoning
        )
    
    def _score_relevance(self, 
                        docs: List[Dict[str, Any]], 
                        reasoning: List[str]) -> float:
        """Score based on document relevance scores."""
        if not docs:
            reasoning.append("No documents retrieved")
            return 0.0
        
        scores = [float(d.get('relevance_score', 0)) for d in docs]
        avg_score = statistics.mean(scores)
        
        if avg_score < 0.3:
            reasoning.append("Low document relevance")
        elif avg_score < 0.6:
            reasoning.append("Moderate document relevance")
        else:
            reasoning.append("High document relevance")
        
        return avg_score
    
    def _score_sources(self, 
                      docs: List[Dict[str, Any]], 
                      reasoning: List[str]) -> float:
        """Score based on source document quality."""
        if not docs:
            return 0.0
        
        # Quality indicators: recent, from trusted sources, etc.
        quality_scores = []
        
        for doc in docs:
            quality = 0.8  # Default quality
            
            # Adjust for document properties
            if doc.get('is_verified'):
                quality += 0.15
            if doc.get('citation_count', 0) > 5:
                quality += 0.05
            
            quality_scores.append(min(quality, 1.0))
        
        avg_quality = statistics.mean(quality_scores)
        reasoning.append(f"Document quality: {avg_quality:.0%}")
        
        return avg_quality
    
    def _score_llm_confidence(self, 
                             response: str, 
                             reasoning: List[str]) -> float:
        """Score based on LLM response characteristics."""
        score = 0.7  # Default confidence
        
        # Penalty if response contains uncertainty markers
        uncertainty_markers = ['might', 'could', 'perhaps', 'uncertain', 'unclear']
        marker_count = sum(1 for marker in uncertainty_markers 
                          if marker.lower() in response.lower())
        
        if marker_count > 3:
            score -= 0.15
            reasoning.append("Response contains uncertainty language")
        
        # Penalty for very short responses
        if len(response) < 50:
            score -= 0.1
            reasoning.append("Short response may lack detail")
        
        return max(score, 0.0)
