"""Agentic retrieval loop for iterative context refinement.

Implements:
1. ContextSufficiencyEvaluator - Determines if retrieved docs are sufficient
2. QueryReformulator - Generates improved queries targeting missing aspects
3. AgenticRetrieverLoop - Orchestrates iterative retrieval with retry logic
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Strategies for query reformulation."""
    ADD_KEYWORDS = "add_keywords"
    BROADEN = "broaden"
    SYNONYMS = "synonyms"
    CLARIFY = "clarify"
    AUTO = "auto"


@dataclass
class AgenticLoopConfig:
    """Configuration for agentic retrieval loop."""
    max_retries: int = 2
    confidence_threshold: float = 0.7
    use_query_reformulation: bool = True
    retry_strategy: str = "auto"
    log_retry_attempts: bool = True
    timeout_seconds: int = 30
    enable_cache_bypass_on_retry: bool = True


@dataclass
class AgenticLoopState:
    """Track state across retrieval iterations."""
    iteration: int = 0
    confidence_history: List[float] = field(default_factory=list)
    query_history: List[str] = field(default_factory=list)
    retrieval_history: List[List[Dict[str, Any]]] = field(default_factory=list)
    reformulation_reasons: List[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    final_status: str = "pending"  # success, max_retries, low_confidence, timeout


@dataclass
class SufficiencyEvaluation:
    """Result of context sufficiency evaluation."""
    is_sufficient: bool
    confidence: float
    missing_aspects: List[str]
    reasoning: str
    metrics: Dict[str, float]  # term_coverage, sub_question_coverage, etc.


@dataclass
class QueryReformation:
    """Result of query reformulation."""
    reformulated_query: str
    strategy: str
    confidence: float
    reasoning: str


class ContextSufficiencyEvaluator:
    """
    Evaluates whether retrieved documents are sufficient to answer the query.
    
    Uses multiple heuristics:
    1. Term coverage: What % of query terms appear in docs?
    2. Sub-question coverage: How many decomposed sub-questions are covered?
    3. Diversity: Are multiple sources represented?
    4. Relevance: What are the average retrieval scores?
    """
    
    def __init__(self, 
                 confidence_scorer=None,
                 min_term_coverage: float = 0.5,
                 min_sub_q_coverage: float = 0.7):
        """
        Initialize evaluator.
        
        Args:
            confidence_scorer: Optional existing ConfidenceScorer instance
            min_term_coverage: Minimum % of query terms that must appear in docs
            min_sub_q_coverage: Minimum % of sub-questions covered by docs
        """
        self.confidence_scorer = confidence_scorer
        self.min_term_coverage = min_term_coverage
        self.min_sub_q_coverage = min_sub_q_coverage
    
    async def evaluate(self, 
                       query: str,
                       retrieved_docs: List[Dict[str, Any]],
                       query_decomposition: Optional[List[str]] = None) -> SufficiencyEvaluation:
        """
        Evaluate if retrieved documents are sufficient.
        
        Args:
            query: Original user query
            retrieved_docs: List of retrieved documents with scores
            query_decomposition: Optional list of decomposed sub-questions
            
        Returns:
            SufficiencyEvaluation with details on coverage and confidence
        """
        metrics = {}
        reasoning_parts = []
        
        # 1. Calculate term coverage
        term_coverage = self._calculate_term_coverage(query, retrieved_docs)
        metrics["term_coverage"] = term_coverage
        
        if term_coverage < self.min_term_coverage:
            reasoning_parts.append(
                f"Low term coverage ({term_coverage:.2%}): Key query concepts "
                f"may not be well represented in retrieved documents"
            )
        
        # 2. Calculate sub-question coverage (if decomposition available)
        sub_q_coverage = 1.0
        if query_decomposition:
            sub_q_coverage = self._calculate_sub_question_coverage(
                query_decomposition, retrieved_docs
            )
            metrics["sub_question_coverage"] = sub_q_coverage
            
            if sub_q_coverage < self.min_sub_q_coverage:
                reasoning_parts.append(
                    f"Incomplete sub-question coverage ({sub_q_coverage:.2%}): "
                    f"Not all aspects of the multi-part query are addressed"
                )
        
        # 3. Calculate diversity score
        diversity_score = self._calculate_diversity(retrieved_docs)
        metrics["diversity_score"] = diversity_score
        
        if diversity_score < 0.5:
            reasoning_parts.append(
                "Low source diversity: Multiple results from same source may indicate "
                "incomplete perspective coverage"
            )
        
        # 4. Calculate average relevance score
        relevance_avg = self._calculate_average_relevance(retrieved_docs)
        metrics["relevance_avg"] = relevance_avg
        
        if relevance_avg < 0.5:
            reasoning_parts.append(
                f"Low average retrieval scores ({relevance_avg:.2f}): "
                f"Documents may not be strongly relevant to the query"
            )
        
        # 5. Overall sufficiency (weighted combination)
        weights = {
            "term_coverage": 0.3,
            "sub_question_coverage": 0.3,
            "diversity_score": 0.2,
            "relevance_avg": 0.2
        }
        
        if query_decomposition:
            # Weight sub-questions more heavily for complex queries
            weights = {
                "term_coverage": 0.2,
                "sub_question_coverage": 0.5,
                "diversity_score": 0.15,
                "relevance_avg": 0.15
            }
        
        overall_confidence = (
            term_coverage * weights["term_coverage"] +
            sub_q_coverage * weights["sub_question_coverage"] +
            diversity_score * weights["diversity_score"] +
            relevance_avg * weights["relevance_avg"]
        )
        
        is_sufficient = overall_confidence >= 0.65
        
        if is_sufficient and not reasoning_parts:
            reasoning_parts.append(
                f"Document retrieval appears adequate: "
                f"Good coverage of query topics (score: {overall_confidence:.2f})"
            )
        
        missing_aspects = self._identify_missing_aspects(query, retrieved_docs)
        
        return SufficiencyEvaluation(
            is_sufficient=is_sufficient,
            confidence=overall_confidence,
            missing_aspects=missing_aspects,
            reasoning=" | ".join(reasoning_parts) if reasoning_parts else "Context appears sufficient",
            metrics=metrics
        )
    
    def _calculate_term_coverage(self, query: str, docs: List[Dict[str, Any]]) -> float:
        """Calculate what % of query terms appear in retrieved documents."""
        import re
        
        # Tokenize query into meaningful terms (remove stop words)
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        query_terms = query_terms - stop_words
        
        if not query_terms:
            return 1.0  # No terms to check
        
        # Check which terms appear in docs
        covered_terms = set()
        for doc in docs:
            doc_text = (doc.get("content") or doc.get("text") or "").lower()
            for term in query_terms:
                if term in doc_text:
                    covered_terms.add(term)
        
        return len(covered_terms) / len(query_terms) if query_terms else 1.0
    
    def _calculate_sub_question_coverage(self, 
                                        sub_questions: List[str],
                                        docs: List[Dict[str, Any]]) -> float:
        """Calculate coverage of decomposed sub-questions."""
        if not sub_questions:
            return 1.0
        
        import re
        
        coverage_count = 0
        doc_text_combined = " ".join(
            doc.get("content") or doc.get("text") or "" for doc in docs
        ).lower()
        
        for sub_q in sub_questions:
            # Simple heuristic: check if key terms from sub-question appear
            sub_q_terms = set(re.findall(r'\b\w+\b', sub_q.lower()))
            sub_q_terms = sub_q_terms - {'the', 'a', 'an', 'and', 'or', 'but'}
            
            # If at least 50% of sub-question terms appear in docs
            if sub_q_terms:
                term_match = sum(1 for t in sub_q_terms if t in doc_text_combined)
                if term_match / len(sub_q_terms) >= 0.5:
                    coverage_count += 1
        
        return coverage_count / len(sub_questions) if sub_questions else 1.0
    
    def _calculate_diversity(self, docs: List[Dict[str, Any]]) -> float:
        """Calculate source diversity in retrieved documents."""
        if not docs:
            return 0.0
        
        # Count unique sources
        sources = set()
        for doc in docs:
            source = doc.get("source") or doc.get("doc_id") or "unknown"
            sources.add(source)
        
        # Diversity: unique sources / total docs
        # Max 1.0 when all different, min approaches 0 when all same
        return len(sources) / len(docs)
    
    def _calculate_average_relevance(self, docs: List[Dict[str, Any]]) -> float:
        """Calculate average retrieval score from documents."""
        if not docs:
            return 0.0
        
        scores = []
        for doc in docs:
            score = doc.get("score") or doc.get("relevance_score") or 0.0
            # Normalize to 0-1 if needed
            if score > 1.0:
                score = score / 100.0
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _identify_missing_aspects(self, query: str, docs: List[Dict[str, Any]]) -> List[str]:
        """Identify topics/aspects not well covered in retrieved docs."""
        # Simple heuristic: common follow-up question patterns
        aspects = []
        query_lower = query.lower()
        
        # Pattern-based detection
        if "how" in query_lower and "why" not in " ".join(
            d.get("content", "").lower() for d in docs
        ):
            aspects.append("causation/reasoning")
        
        if "when" in query_lower and "time" not in " ".join(
            d.get("content", "").lower() for d in docs
        ):
            aspects.append("temporal context")
        
        if "who" in query_lower and "person" not in " ".join(
            d.get("content", "").lower() for d in docs
        ):
            aspects.append("entities/people")
        
        if "where" in query_lower and "location" not in " ".join(
            d.get("content", "").lower() for d in docs
        ):
            aspects.append("geographic context")
        
        return aspects


class QueryReformulator:
    """
    Reformulates queries to target missing aspects or improve retrieval.
    
    Strategies:
    - add_keywords: Append relevant keywords
    - broaden: Generalize the query
    - synonyms: Replace with alternative terms
    - auto: LLM-driven selection
    """
    
    def __init__(self, llm=None, use_llm: bool = True):
        """
        Initialize reformulator.
        
        Args:
            llm: Optional ChatNVIDIA instance for LLM-based reformulation
            use_llm: Whether to attempt LLM reformulation if available
        """
        self.llm = llm
        self.use_llm = use_llm and llm is not None
    
    async def reformulate(self,
                         original_query: str,
                         missing_aspects: List[str],
                         strategy: str = "auto") -> QueryReformation:
        """
        Reformulate query targeting missing aspects.
        
        Args:
            original_query: Original user query
            missing_aspects: List of identified missing aspects
            strategy: Reformulation strategy to use
            
        Returns:
            QueryReformation with reformulated query and metadata
        """
        start_time = time.time()
        
        # Select strategy
        if strategy == "auto":
            # Auto-select: use LLM if available, else default to add_keywords
            if self.use_llm:
                strategy = "llm"
            else:
                strategy = "add_keywords"
        
        # Execute reformulation
        if strategy == "add_keywords":
            reformulated = self._strategy_add_keywords(original_query, missing_aspects)
            confidence = 0.7
        elif strategy == "broaden":
            reformulated = self._strategy_broaden(original_query)
            confidence = 0.6
        elif strategy == "synonyms":
            reformulated = self._strategy_synonyms(original_query)
            confidence = 0.65
        elif strategy == "llm" and self.use_llm:
            reformulated = await self._strategy_llm(original_query, missing_aspects)
            confidence = 0.8
        else:
            # Fallback to original query
            reformulated = original_query
            confidence = 0.5
        
        latency_ms = (time.time() - start_time) * 1000
        
        reasoning = (
            f"Reformulated using {strategy} strategy to address missing aspects: "
            f"{', '.join(missing_aspects[:2])}{'...' if len(missing_aspects) > 2 else ''}"
        )
        
        return QueryReformation(
            reformulated_query=reformulated,
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _strategy_add_keywords(self, query: str, aspects: List[str]) -> str:
        """Rule-based: append keywords from missing aspects."""
        if not aspects:
            return query
        
        # Take up to 2 most relevant aspects and add as keywords
        keywords = " ".join(aspects[:2]).replace("_", " ")
        return f"{query} {keywords}"
    
    def _strategy_broaden(self, query: str) -> str:
        """Rule-based: generalize query by removing specific terms."""
        import re
        
        # Remove quoted phrases and very specific terms
        broadened = re.sub(r'"[^"]+"', '', query)
        broadened = re.sub(r'\b\d{4}\b', '', broadened)  # Remove years
        
        return broadened.strip() if broadened.strip() else query
    
    def _strategy_synonyms(self, query: str) -> str:
        """Rule-based: replace terms with common synonyms."""
        # Simple synonym map (in production, use a better thesaurus)
        synonyms = {
            "how": "ways to",
            "what": "describe",
            "effect": "impact",
            "cause": "reason",
            "problem": "issue",
        }
        
        reformulated = query
        for term, synonym in synonyms.items():
            if term in query.lower():
                reformulated = reformulated.replace(term, synonym)
                break
        
        return reformulated
    
    async def _strategy_llm(self, query: str, aspects: List[str]) -> str:
        """LLM-based: use ChatNVIDIA for intelligent reformulation."""
        if not self.llm:
            return query
        
        try:
            prompt = (
                f"Rephrase this query to better capture these missing aspects:\n"
                f"Query: {query}\n"
                f"Missing aspects: {', '.join(aspects)}\n"
                f"Reformulated query (concise, single sentence):"
            )
            
            # This would be an actual LLM call in production
            # For now, just add the aspects
            return f"{query} (also covering: {', '.join(aspects[:2])})"
        except Exception as e:
            logger.warning(f"LLM reformulation failed: {e}. Falling back to keywords.")
            return self._strategy_add_keywords(query, aspects)


class AgenticRetrieverLoop:
    """
    Orchestrates iterative retrieval with automatic retry on low confidence.
    
    Flow:
    1. Execute initial retrieval
    2. Evaluate context sufficiency
    3. If insufficient, reformulate query and retry
    4. Continue until: confidence > threshold OR max_retries reached
    """
    
    def __init__(self,
                 retriever,
                 confidence_scorer,
                 evaluator: ContextSufficiencyEvaluator,
                 reformulator: QueryReformulator,
                 config: Optional[AgenticLoopConfig] = None):
        """
        Initialize agentic loop.
        
        Args:
            retriever: HybridRetriever instance for document retrieval
            confidence_scorer: ConfidenceScorer instance
            evaluator: ContextSufficiencyEvaluator instance
            reformulator: QueryReformulator instance
            config: AgenticLoopConfig with retry parameters
        """
        self.retriever = retriever
        self.confidence_scorer = confidence_scorer
        self.evaluator = evaluator
        self.reformulator = reformulator
        self.config = config or AgenticLoopConfig()
    
    async def retrieve_with_retry(self,
                                  query: str,
                                  top_k: int = 5,
                                  query_decomposition: Optional[List[str]] = None,
                                  timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute retrieval with automatic retry on low confidence.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            query_decomposition: Optional pre-computed query decomposition
            timeout_seconds: Optional override for loop timeout
            
        Returns:
            {
                "documents": List[RetrievedDoc],
                "confidence": float,
                "iterations": int,
                "queries_tried": List[str],
                "final_status": "success" | "max_retries" | "low_confidence" | "timeout",
                "metrics": {...}
            }
        """
        loop_start = time.time()
        timeout = timeout_seconds or self.config.timeout_seconds
        state = AgenticLoopState(query_history=[query])
        
        try:
            while state.iteration < self.config.max_retries + 1:
                # Check timeout
                if time.time() - loop_start > timeout:
                    state.final_status = "timeout"
                    logger.warning(f"Agentic loop timeout after {state.iteration} iterations")
                    break
                
                # Execute one iteration
                iteration_result = await self._iteration(
                    query=query if state.iteration == 0 else state.query_history[-1],
                    iteration=state.iteration,
                    state=state
                )
                
                # Check if we're done
                if state.iteration < self.config.max_retries:
                    confidence = iteration_result["confidence"]
                    
                    if confidence >= self.config.confidence_threshold:
                        state.final_status = "success"
                        logger.info(
                            f"Agentic loop: Query resolved in {state.iteration + 1} "
                            f"iteration(s) with confidence {confidence:.2f}"
                        )
                        break
                    
                    # Reformulate and retry
                    if self.config.use_query_reformulation:
                        evaluation = iteration_result["evaluation"]
                        reformulation = await self.reformulator.reformulate(
                            original_query=state.query_history[-1],
                            missing_aspects=evaluation.missing_aspects,
                            strategy=self.config.retry_strategy
                        )
                        
                        state.query_history.append(reformulation.reformulated_query)
                        state.reformulation_reasons.append(reformulation.reasoning)
                        
                        logger.info(
                            f"Agentic loop iteration {state.iteration + 1}: "
                            f"Confidence {confidence:.2f} < threshold {self.config.confidence_threshold}. "
                            f"Reformulating query..."
                        )
                else:
                    state.final_status = "max_retries" if state.iteration > 0 else "success"
                    break
                
                state.iteration += 1
            
            # Build final result
            final_docs = iteration_result.get("documents", [])
            final_confidence = iteration_result.get("confidence", 0.0)
            
            state.total_latency_ms = (time.time() - loop_start) * 1000
            
            return {
                "documents": final_docs,
                "confidence": final_confidence,
                "iterations": state.iteration + 1,
                "queries_tried": state.query_history,
                "final_status": state.final_status,
                "metrics": {
                    "total_latency_ms": state.total_latency_ms,
                    "confidence_history": state.confidence_history,
                    "retrieval_history": state.retrieval_history,
                    "reformulation_reasons": state.reformulation_reasons,
                }
            }
        
        except Exception as e:
            logger.error(f"Error in agentic retrieval loop: {e}", exc_info=True)
            state.final_status = "error"
            return {
                "documents": [],
                "confidence": 0.0,
                "iterations": state.iteration + 1,
                "queries_tried": state.query_history,
                "final_status": "error",
                "error": str(e),
                "metrics": {
                    "total_latency_ms": (time.time() - loop_start) * 1000,
                }
            }
    
    async def _iteration(self,
                        query: str,
                        iteration: int,
                        state: AgenticLoopState) -> Dict[str, Any]:
        """
        Execute one iteration of retrieval + evaluation.
        
        Returns:
            {
                "documents": List[RetrievedDoc],
                "confidence": float,
                "evaluation": SufficiencyEvaluation,
            }
        """
        start_time = time.time()
        
        try:
            # Retrieve documents
            retrieval_start = time.time()
            retrieved_docs = await self.retriever.retrieve(query, top_k=5)
            retrieval_latency = time.time() - retrieval_start
            
            # Convert to dict format for evaluation
            docs_dict = [
                {
                    "doc_id": doc.doc_id if hasattr(doc, 'doc_id') else doc.get('doc_id'),
                    "content": doc.content if hasattr(doc, 'content') else doc.get('content'),
                    "source": doc.source if hasattr(doc, 'source') else doc.get('source'),
                    "score": doc.score if hasattr(doc, 'score') else doc.get('score'),
                }
                for doc in retrieved_docs
            ]
            
            state.retrieval_history.append(docs_dict)
            
            # Evaluate sufficiency
            if not docs_dict:
                evaluation = SufficiencyEvaluation(
                    is_sufficient=False,
                    confidence=0.0,
                    missing_aspects=["no_documents_retrieved"],
                    reasoning="No documents were retrieved for this query.",
                    metrics={"term_coverage": 0.0, "sub_question_coverage": 0.0, "diversity_score": 0.0, "relevance_avg": 0.0}
                )
            else:
                evaluation = await self.evaluator.evaluate(
                    query=query,
                    retrieved_docs=docs_dict,
                    query_decomposition=None
                )
            
            state.confidence_history.append(evaluation.confidence)
            
            logger.debug(
                f"Agentic loop iteration {iteration}: "
                f"query='{query}' confidence={evaluation.confidence:.2f} "
                f"sufficient={evaluation.is_sufficient} "
                f"latency={retrieval_latency*1000:.1f}ms"
            )
            
            return {
                "documents": retrieved_docs,
                "confidence": evaluation.confidence,
                "evaluation": evaluation,
            }
        
        except Exception as e:
            logger.error(f"Error in agentic loop iteration {iteration}: {e}")
            return {
                "documents": [],
                "confidence": 0.0,
                "evaluation": SufficiencyEvaluation(
                    is_sufficient=False,
                    confidence=0.0,
                    missing_aspects=["retrieval_error"],
                    reasoning=f"Retrieval failed: {str(e)}",
                    metrics={}
                ),
                "error": str(e)
            }
