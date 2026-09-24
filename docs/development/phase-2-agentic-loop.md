# Phase 2.1: Agentic Retrieval Loop Implementation

**Status:** In Development  
**Start Date:** August 31, 2026  
**Target Completion:** Week 2 of Phase 2 (2 weeks)  
**Priority:** 🔴 CRITICAL (Highest ROI)

---

## Objective

Implement an **iterative retrieval loop** where the system:
1. Executes initial retrieval
2. Evaluates context sufficiency via confidence scoring
3. If confidence is below threshold, reformulates query and retries
4. Continues until: confidence > threshold OR max_retries reached

**Expected Impact:**
- Query success rate: 85-90% → 95%+
- Automatic recovery from insufficient initial retrieval
- Observable retry attempts and decision-making

---

## Architecture Design

### High-Level Flow

```
User Query
    ↓
[Query Understanding] (existing)
    ↓
[Initial Retrieval] (existing hybrid retriever)
    ↓
[Confidence Scoring] (existing + enhanced)
    ↓
Confidence > Threshold?
    ├─ YES → Generate Response
    └─ NO → Reformulate Query
         ↓
         [Retry Retrieval] (max 2-3 retries)
         ↓
         [Score Again]
         ↓
         Confidence > Threshold?
              ├─ YES → Generate Response
              └─ NO → Fallback Response
```

### New Components

#### 1. **ContextSufficiencyEvaluator**
Evaluates if retrieved documents are sufficient for answering the query.

**Inputs:**
- Query (original)
- Retrieved documents
- Query decomposition (if available)

**Outputs:**
- `is_sufficient: bool` (True if documents likely contain answer)
- `confidence_score: float` (0.0-1.0)
- `missing_aspects: List[str]` (topics not covered in docs)
- `reasoning: str` (explanation of decision)

**Implementation approach:**
- Lexical overlap between query terms and document content
- Coverage of query decomposed sub-questions
- Topic diversity in retrieved documents
- Document quality/relevance scores (from reranker)

#### 2. **QueryReformulator**
Reformulates query to target missing aspects.

**Inputs:**
- Original query
- Retrieved documents
- Missing aspects identified by evaluator

**Outputs:**
- `reformulated_query: str` (new query targeting missing aspects)
- `strategy: str` (e.g., "add_keywords", "broaden_scope", "synonyms")
- `confidence_in_reform: float` (how likely this will help)

**Implementation approach:**
- Rule-based: add/replace keywords
- LLM-based: use ChatNVIDIA to generate alternative phrasing
- Hybrid: rule-based for speed, LLM as fallback

#### 3. **AgenticRetrieverLoop**
Orchestrates the iterative retrieval process.

**Configuration:**
```python
class AgenticLoopConfig:
    max_retries: int = 2
    confidence_threshold: float = 0.7
    use_query_reformulation: bool = True
    retry_strategy: str = "reformulate"  # or "broaden", "hybrid"
    log_retry_attempts: bool = True
```

**State tracking:**
```python
class AgenticLoopState:
    iteration: int = 0
    confidence_history: List[float] = []
    query_history: List[str] = []
    retrieval_history: List[List[RetrievedDoc]] = []
    reformulation_reasons: List[str] = []
    total_latency_ms: float = 0.0
```

---

## Implementation Plan

### Phase 2.1a: ContextSufficiencyEvaluator (Day 1-2)

**File:** `rag/agentic_loop.py`

**Key Methods:**
```python
class ContextSufficiencyEvaluator:
    def __init__(self, 
                 confidence_scorer: ConfidenceScorer,
                 use_llm: bool = False):
        """Initialize evaluator."""
        
    async def evaluate(self, 
                       query: str,
                       retrieved_docs: List[RetrievedDoc],
                       query_decomposition: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate if retrieved documents are sufficient.
        
        Returns:
            {
                "is_sufficient": bool,
                "confidence": float,
                "missing_aspects": List[str],
                "reasoning": str,
                "metrics": {
                    "term_coverage": float,
                    "sub_question_coverage": float,
                    "diversity_score": float,
                    "relevance_avg": float
                }
            }
        """
        
    def _calculate_term_coverage(self, query: str, docs: List[RetrievedDoc]) -> float:
        """Calculate what % of query terms appear in docs."""
        
    def _calculate_sub_question_coverage(self, 
                                         sub_questions: List[str],
                                         docs: List[RetrievedDoc]) -> float:
        """Calculate coverage of decomposed sub-questions."""
        
    def _identify_missing_aspects(self, query: str, docs: List[RetrievedDoc]) -> List[str]:
        """Identify topics/aspects not covered in retrieved docs."""
```

**Tests:** `tests/test_agentic_sufficiency.py`

---

### Phase 2.1b: QueryReformulator (Day 2-3)

**File:** `rag/agentic_loop.py` (same file)

**Key Methods:**
```python
class QueryReformulator:
    def __init__(self, llm: Optional[ChatNVIDIA] = None):
        """Initialize reformulator with optional LLM."""
        
    async def reformulate(self,
                         original_query: str,
                         missing_aspects: List[str],
                         strategy: str = "auto") -> Dict[str, Any]:
        """
        Reformulate query targeting missing aspects.
        
        Strategies:
        - "add_keywords": Append missing aspect keywords
        - "broaden": Generalize to broader scope
        - "synonyms": Try alternative terminology
        - "clarify": Ask for more specific requirements
        - "auto": LLM chooses best strategy
        
        Returns:
            {
                "reformulated_query": str,
                "strategy": str,
                "confidence": float,
                "reasoning": str
            }
        """
        
    def _strategy_add_keywords(self, query: str, aspects: List[str]) -> str:
        """Rule-based: append keywords."""
        
    def _strategy_broaden(self, query: str) -> str:
        """Rule-based: generalize query."""
        
    async def _strategy_llm(self, query: str, aspects: List[str]) -> str:
        """LLM-based: use ChatNVIDIA for reformulation."""
```

**Tests:** `tests/test_agentic_reformulation.py`

---

### Phase 2.1c: AgenticRetrieverLoop (Day 3-4)

**File:** `rag/agentic_loop.py` (same file)

**Key Methods:**
```python
class AgenticRetrieverLoop:
    def __init__(self,
                 retriever: HybridRetriever,
                 confidence_scorer: ConfidenceScorer,
                 evaluator: ContextSufficiencyEvaluator,
                 reformulator: QueryReformulator,
                 config: AgenticLoopConfig = None):
        """Initialize agentic loop."""
        
    async def retrieve_with_retry(self,
                                  query: str,
                                  top_k: int = 5,
                                  query_decomposition: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute retrieval with automatic retry on low confidence.
        
        Returns:
            {
                "documents": List[RetrievedDoc],
                "confidence": float,
                "iterations": int,
                "queries_tried": List[str],
                "final_status": "success" | "max_retries" | "low_confidence",
                "metrics": {
                    "total_latency_ms": float,
                    "confidence_history": List[float],
                    "retrieval_history": List[List[str]],
                    "reformulation_reasons": List[str]
                }
            }
        """
        
    async def _iteration(self,
                        query: str,
                        iteration: int,
                        state: AgenticLoopState) -> Dict[str, Any]:
        """Execute one iteration of retrieval + evaluation."""
```

**Tests:** `tests/test_agentic_loop.py`

---

### Phase 2.1d: Integration with HybridRAG (Day 4-5)

**File:** `rag/hybrid_rag.py` (modify existing)

**Changes:**
```python
class HybridRAG:
    def __init__(self, ...):
        # ... existing init ...
        
        # Add agentic loop components
        self.sufficiency_evaluator = ContextSufficiencyEvaluator(
            self.confidence_scorer
        )
        self.query_reformulator = QueryReformulator(
            llm=chat_llm if HAS_LANGCHAIN else None
        )
        self.agentic_loop = AgenticRetrieverLoop(
            retriever=self.retriever,
            confidence_scorer=self.confidence_scorer,
            evaluator=self.sufficiency_evaluator,
            reformulator=self.query_reformulator,
            config=AgenticLoopConfig(
                max_retries=settings.AGENTIC_MAX_RETRIES or 2,
                confidence_threshold=settings.AGENTIC_CONFIDENCE_THRESHOLD or 0.7,
            )
        )
        
    async def query(self, query: str, use_agentic_loop: bool = True, ...):
        """
        Main query method with optional agentic loop.
        """
        # ... existing code ...
        
        if use_agentic_loop and self.agentic_loop:
            # Use agentic loop for retrieval
            retrieval_result = await self.agentic_loop.retrieve_with_retry(
                query=query,
                top_k=top_k,
                query_decomposition=query_plan.get("decomposed_queries", None)
            )
            documents = retrieval_result["documents"]
            agentic_metrics = retrieval_result["metrics"]
        else:
            # Use existing single-pass retrieval
            # ... existing code ...
            agentic_metrics = {}
        
        # ... generate response using documents ...
        
        # Log agentic metrics
        if agentic_metrics:
            logger.info(f"Agentic loop: {agentic_metrics['iterations']} iterations, "
                       f"confidence: {retrieval_result['confidence']:.2f}")
```

---

## Configuration Parameters

Add to `api/config.py`:

```python
class Settings:
    # Agentic Loop Configuration
    AGENTIC_LOOP_ENABLED: bool = True
    AGENTIC_MAX_RETRIES: int = 2
    AGENTIC_CONFIDENCE_THRESHOLD: float = 0.7
    AGENTIC_REFORMULATION_STRATEGY: str = "auto"  # or "add_keywords", "broaden"
    AGENTIC_LOG_RETRIES: bool = True
    AGENTIC_TIMEOUT_SECONDS: int = 30
```

---

## Metrics & Observability

### New Prometheus Metrics

```python
agentic_loop_iterations = Histogram(
    "agentic_loop_iterations",
    "Number of retrieval iterations",
    buckets=[1, 2, 3, 4, 5]
)

agentic_loop_confidence_initial = Histogram(
    "agentic_loop_confidence_initial",
    "Initial retrieval confidence score",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)

agentic_loop_confidence_final = Histogram(
    "agentic_loop_confidence_final",
    "Final retrieval confidence score",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)

agentic_loop_reformulations = Counter(
    "agentic_loop_reformulations",
    "Total query reformulations attempted"
)

agentic_loop_success_rate = Gauge(
    "agentic_loop_success_rate",
    "Percentage of queries meeting confidence threshold"
)
```

### Logging

```python
logger.info(f"Agentic Loop: query='{query}' initial_confidence={conf:.2f}")
logger.info(f"Agentic Loop: Iteration {i} insufficient confidence, reformulating...")
logger.info(f"Agentic Loop: Reformulation strategy={strategy}, new_query='{new_q}'")
logger.info(f"Agentic Loop: Final confidence={final_conf:.2f} after {iterations} iterations")
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_agentic_loop.py`

1. **ContextSufficiencyEvaluator**
   - Test term coverage calculation
   - Test sub-question coverage
   - Test missing aspect identification
   - Edge case: empty docs, single word query

2. **QueryReformulator**
   - Test add_keywords strategy
   - Test broaden strategy
   - Test LLM reformulation (mock)
   - Test confidence in reformulation

3. **AgenticRetrieverLoop**
   - Test single-pass retrieval (no retry needed)
   - Test retry on low confidence
   - Test max_retries limit
   - Test query history tracking
   - Test latency measurement

### Integration Tests

**File:** `tests/test_agentic_integration.py`

1. Test with actual HybridRAG system
2. Test with query decomposition
3. Test end-to-end: query → retrieval → response generation
4. Test A/B: agentic loop vs. baseline

### Evaluation Cases

**File:** `hybrid_rag/tools/eval_agentic_loop.py`

Test queries:
```python
EVAL_CASES = [
    # Simple query (should not need retry)
    {
        "query": "What is the capital of France?",
        "expected_iterations": 1,
        "min_confidence": 0.8
    },
    # Complex multi-part query
    {
        "query": "How do photosynthesis and cellular respiration relate?",
        "expected_iterations": 1,  # or 2 if retrieval is poor
        "min_confidence": 0.7
    },
    # Obscure query (likely needs retry)
    {
        "query": "What is the third-order derivative of the Bessel function?",
        "expected_max_iterations": 3,
        "min_confidence": 0.6
    },
]
```

---

## Rollout Strategy

### Phase 2.1-1: Baseline (Day 1)
- Implement without agentic loop
- Measure baseline success rate, latency, confidence
- Establish metrics

### Phase 2.1-2: Launch with Flag (Day 5)
- Deploy agentic loop behind feature flag
- `AGENTIC_LOOP_ENABLED=False` by default
- Canary: 10% of traffic
- Monitor: iterations, confidence, latency impact

### Phase 2.1-3: A/B Test (Week 2)
- 50% traffic with loop, 50% without
- Compare: success rate, latency, cost (token usage)
- Measure user satisfaction (if applicable)

### Phase 2.1-4: Full Rollout (Week 3)
- Enable for 100% of traffic (if A/B positive)
- Or keep as opt-in if user preference exists
- Continue monitoring

---

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Avg Iterations per Query** | < 1.3 | Most queries should work on first try |
| **Success Rate Improvement** | +5-10% | From 85-90% → 95%+ |
| **Latency Impact (p95)** | < +200ms | Acceptable for quality gain |
| **Confidence Score Improvement** | +0.15 avg | Clear confidence boost |
| **Max Iterations Observed** | < 5 | No runaway loops |
| **Coverage of Missing Aspects** | > 80% | Reformulations target real gaps |

---

## Known Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Infinite retry loop | High | Max_retries hard limit; timeout protection |
| Query drift | Medium | Compare reformulated ↔ original; log divergence |
| Latency explosion | High | Monitor p95 latency; early exit thresholds |
| Token cost | Medium | Track token usage per iteration; budget alerts |
| Cache invalidation | Low | Bypass semantic cache for retries (or separate cache) |

---

## Files to Create/Modify

### New Files ✨
- `rag/agentic_loop.py` - Main implementation (ContextSufficiencyEvaluator, QueryReformulator, AgenticRetrieverLoop)
- `tests/test_agentic_loop.py` - Unit tests
- `tests/test_agentic_integration.py` - Integration tests
- `hybrid_rag/tools/eval_agentic_loop.py` - Evaluation harness

### Modified Files 🔧
- `rag/hybrid_rag.py` - Add agentic loop components and integration
- `api/config.py` - Add agentic loop configuration parameters
- `observability/metrics.py` - Add agentic loop metrics
- `observability/logging_config.py` - Add agentic loop logging

---

## Timeline

- **Days 1-2:** Implement ContextSufficiencyEvaluator + unit tests
- **Days 2-3:** Implement QueryReformulator + unit tests
- **Days 3-4:** Implement AgenticRetrieverLoop + integration tests
- **Days 4-5:** Integrate with HybridRAG + evaluation
- **Week 2:** A/B testing and optimization

---

## Next: Phase 2.2 (GraphRAG) 📅

After Phase 2.1 is complete and stable:
- Start Phase 2.2: Knowledge Graph Integration
- Parallel: Phase 2.3 (Dynamic Routing)
- Use agentic loop foundation for multi-hop queries

---

## References

- Phase 1-5 (Context Engineering): `context-engineering-plan.md`
- Existing Components: `rag/confidence_scorer.py`, `rag/retrieval.py`, `rag/query_understanding.py`
- Integration Plan: `IMPLEMENTATION_ROADMAP.md` (Phase 2)

