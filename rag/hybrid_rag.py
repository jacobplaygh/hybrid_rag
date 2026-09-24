"""Hybrid RAG orchestrator combining LangChain + LlamaIndex."""

import asyncio
import copy
import inspect
import logging
import re
import time
import uuid
from types import SimpleNamespace
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from rag.reranker import CrossEncoderReranker
from rag.confidence_scorer import ConfidenceScorer
from rag.quality_metrics import QualityMetricsCalculator
from rag.query_understanding import QueryUnderstanding
from rag.chains import create_chains, QueryDecomposerChain
from rag.retrieval import HybridRetriever
from rag.retrieval_router import RetrievalRouter
from rag.graph_rag import KnowledgeGraph
from rag.memory import ConversationMemory
from rag.semantic_cache import SemanticCache
from rag.context_manager import ContextManager
from rag.response_validator import ResponseValidator
from rag.analytics import QueryAnalytics
from rag.agentic_loop import (
    AgenticRetrieverLoop,
    AgenticLoopConfig,
    ContextSufficiencyEvaluator,
    QueryReformulator,
)

DEFAULT_MAX_CONTEXT_TOKENS = 120000

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

from rag.constrained_tools import ConstrainedToolExecutor, WorkflowPolicy
from api.config import get_settings
from observability.metrics import (
    query_counter, 
    query_latency, 
    tokens_used, 
    cache_hits, 
    cache_misses, 
    ndcg_gauge, 
    confidence_gauge, 
    errors_total,
    retrieval_latency
)
from observability.tracing import LangSmithTracer
from rag.chains import create_chains, QueryDecomposerChain
from rag.retrieval import HybridRetriever
from rag.memory import ConversationMemory
from rag.semantic_cache import SemanticCache
from rag.context_manager import ContextManager
from rag.response_validator import ResponseValidator
from rag.analytics import QueryAnalytics

logger = logging.getLogger(__name__)


class HybridRAG:
    """
    Orchestrator for hybrid RAG combining:
    - LangChain for chain orchestration and memory management
    - LlamaIndex for advanced document indexing and retrieval
    """
    
    def __init__(
        self,
        vector_store,
        chat_llm: ChatNVIDIA = None,
        reasoning_llm: ChatNVIDIA = None,
        structured_llm: ChatNVIDIA = None,
        tracer: LangSmithTracer = None,
    ):
        """Initialize hybrid RAG system."""
        self.vector_store = vector_store
        self.chat_llm = chat_llm
        self.reasoning_llm = reasoning_llm
        self.structured_llm = structured_llm
        self.tracer = tracer
        
        # Initialize components
        self.memory = ConversationMemory()
        self.chains = create_chains(chat_llm, self.memory) if chat_llm and HAS_LANGCHAIN else {}
        self.reranker = CrossEncoderReranker()
        self.confidence_scorer = ConfidenceScorer()
        self.metrics_calculator = QualityMetricsCalculator()
        self.query_understanding = QueryUnderstanding(chat_llm) if chat_llm else None

        settings = get_settings()
        self.retrieval_router = RetrievalRouter(enabled=getattr(settings, "DYNAMIC_ROUTING_ENABLED", True))
        self.graph_rag = KnowledgeGraph()

        # Initialize Context Manager
        self.context_manager = ContextManager(
            max_tokens=getattr(settings, "MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS)
        )

        # Initialize Response Validator
        self.validator = ResponseValidator(llm=reasoning_llm or chat_llm)

        # Initialize Analytics
        self.analytics = QueryAnalytics()

        # Initialize local caches and history before query execution.
        self.query_history: Dict[str, Dict[str, Any]] = {}
        self.query_cache: Dict[str, Dict[str, Any]] = {}

        # Initialize retrieval components early so they are available to the entire pipeline.
        self.retriever = None
        reranker_model = None
        if HAS_LANGCHAIN and settings.RETRIEVAL_RERANK:
            reranker_model = ChatNVIDIA(model=settings.RERANK_MODEL)
        self.retriever = HybridRetriever(
            vector_store,
            use_reranker=settings.RETRIEVAL_RERANK,
            reranker=reranker_model,
        )

        self.agentic_loop = None
        if settings.AGENTIC_LOOP_ENABLED:
            self.agentic_loop = AgenticRetrieverLoop(
                retriever=self.retriever,
                confidence_scorer=self.confidence_scorer,
                evaluator=ContextSufficiencyEvaluator(self.confidence_scorer),
                reformulator=QueryReformulator(llm=chat_llm if HAS_LANGCHAIN else None, use_llm=bool(chat_llm)),
                config=AgenticLoopConfig(
                    max_retries=settings.AGENTIC_MAX_RETRIES,
                    confidence_threshold=settings.AGENTIC_CONFIDENCE_THRESHOLD,
                    retry_strategy=settings.AGENTIC_REFORMULATION_STRATEGY,
                    timeout_seconds=settings.AGENTIC_TIMEOUT_SECONDS,
                ),
            )
            logger.info("🔁 Agentic retrieval loop enabled")
        
        if settings.SEMANTIC_CACHE_ENABLED:
            # Use the vector store's embedding function for the semantic cache
            embedding_model = getattr(self.vector_store, "embedding_fn", None)
            if embedding_model:
                self.semantic_cache = SemanticCache(embedding_model=embedding_model)
            else:
                logger.warning("Semantic cache enabled but no embedding model found in vector store")
                self.semantic_cache = None

    def register_graph_documents(self, documents: List[Any]) -> None:
        """Index a document set in the lightweight knowledge graph."""
        if self.graph_rag:
            self.graph_rag.index_documents(documents)

    async def execute_constrained_query(self, query: str, policy: WorkflowPolicy) -> Dict[str, Any]:
        """
        Executes a query using a constrained tool executor to enforce budget and policy.
        """
        # This is a simplified implementation for evaluation purposes.
        # In a full implementation, this would integrate with the ConstrainedWorkflowAgent.
        
        tools = {
            "document_search": self.vector_store.search,
        }
        
        executor = ConstrainedToolExecutor.from_policy(tools, policy)
        
        start_time = time.perf_counter()
        try:
            # Simulate the agent loop: 
            # 1. Understand query -> 2. Call tools (constrained) -> 3. Generate response
            
            # Step 1: Query Understanding
            query_plan = await self.query_understanding.analyze(query) if self.query_understanding else {"plan": ["document_search"]}
            
            # Step 2: Constrained Tool Execution
            context_fragments = []
            for tool_name in query_plan.get("plan", ["document_search"]):
                # We wrap the tool call in the executor
                result = await executor.execute(tool_name, query)
                context_fragments.append(result)
            
            # Step 3: Final Response Generation (Simulated for eval)
            # In reality, this would call the LLM with the gathered context
            response = f"Generated response based on {len(context_fragments)} tool calls."
            
            # Validation
            validation_result = await self.validator.validate(response, query)
            
            return {
                "response": response,
                "tokens_used": executor.current_tokens,
                "tool_calls": executor.call_count,
                "is_valid": validation_result.get("is_valid", True),
                "score": validation_result.get("score", 1.0)
            }
            
            logger.error(f"Constrained execution failed: {e}")
        finally:
            latency = time.perf_counter() - start_time
            self.analytics.log_query({
                "query": query,
                "response_time_ms": latency * 1000,
                "tool_calls": executor.call_count,
                "status": "success" if 'response' in locals() else "error"
            })

            # Use the vector store's embedding capability if available, 
            # or a dedicated embedding model. For now, we pass the vector_store
            # as it typically handles the embedding logic.
            self.semantic_cache = SemanticCache(
                embedding_model=self.vector_store, 
                threshold=settings.SEMANTIC_CACHE_THRESHOLD
            )
            logger.info(f"🧠 Semantic Cache initialized (threshold: {settings.SEMANTIC_CACHE_THRESHOLD})")

        reranker_model = None
        if HAS_LANGCHAIN and get_settings().RETRIEVAL_RERANK:
            settings = get_settings()
            reranker_model = ChatNVIDIA(model=settings.RERANK_MODEL)

        self.retriever = HybridRetriever(
            vector_store,
            use_reranker=get_settings().RETRIEVAL_RERANK,
            reranker=reranker_model,
        )
        self.query_history: Dict[str, Dict[str, Any]] = {}
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        
        if not HAS_LANGCHAIN or not chat_llm:
            logger.warning("⚠️ LLM not available; using document-based fallback responses")
        else:
            logger.info("🚀 Initialized Hybrid RAG System")
    
    async def query_stream(self, request: Any):
        """
        Wrapper for the query method to support streaming from the API.
        Unpacks QueryRequest and calls query(stream=True).
        """
        # Handle both Pydantic models and dicts
        if hasattr(request, "dict"):
            req_dict = request.dict()
        elif isinstance(request, dict):
            req_dict = request
        else:
            req_dict = vars(request)

        return await self.query(
            query=req_dict.get("query"),
            mode=req_dict.get("mode", "simple"),
            top_k=req_dict.get("top_k", 5),
            model=req_dict.get("model"),
            stream=True
        )

    async def ensure_keyword_index(self):
        """Ensure BM25 keyword index is initialized for hybrid retrieval."""
        if self.retriever.bm25 is not None:
            return

        try:
            from rag.indexing import DocumentIndexer

            settings = get_settings()
            indexer = DocumentIndexer(
                settings.UPLOAD_DIR,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            documents = indexer.load_documents(settings.UPLOAD_DIR)
            if documents:
                await self.retriever.index_documents(documents)
                logger.info("✅ Keyword index initialized for hybrid retrieval")
        except Exception as exc:
            logger.warning(f"Keyword index initialization failed: {exc}")

    def _doc_to_confidence_payload(self, doc: Any) -> Dict[str, Any]:
        """Normalize retrieved docs into a dict structure for confidence scoring."""
        if isinstance(doc, dict):
            payload = dict(doc)
        elif hasattr(doc, "to_dict"):
            payload = doc.to_dict()
        else:
            payload = {
                "content": getattr(doc, "content", ""),
                "source": getattr(doc, "source", "unknown"),
                "score": getattr(doc, "score", 0.0),
            }

        if "relevance_score" not in payload and "score" in payload:
            payload["relevance_score"] = float(payload.get("score", 0.0))
        return payload

    def _retrieval_confidence(self, documents: List[Any]) -> float:
        """Estimate retrieval confidence for a document set."""
        if not documents:
            return 0.0

        payload = [self._doc_to_confidence_payload(doc) for doc in documents]
        # We pass an empty string as response because we are scoring the retrieval phase, not the generation phase
        score = self.confidence_scorer.score_response(payload, "")
        
        # In retrieval phase, we primarily care about relevance and source quality
        # We weight these more heavily than the LLM confidence (which will be 0.7 default for empty response)
        retrieval_conf = (score.relevance_score * 0.6) + (score.source_quality * 0.4)
        
        return float(retrieval_conf)

    def _build_corrective_queries(self, query: str, analysis: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate alternative retrieval queries for corrective fallback."""
        candidate_queries = [query.strip()]
        analysis = analysis or {}

        entities = [entity for entity in (analysis.get("entities") or []) if isinstance(entity, str) and entity.strip()]
        if entities:
            candidate_queries.append(" ".join(entities))
            candidate_queries.extend(f"{entity} details" for entity in entities[:3])

        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]{2,}", query)
        if tokens:
            candidate_queries.append(" ".join(tokens[:5]))

        if " " in query:
            candidate_queries.append(f"{query} overview")
            candidate_queries.append(f"{query} explain")

        deduped: List[str] = []
        seen = set()
        for candidate in candidate_queries:
            normalized = candidate.strip()
            if normalized and normalized.lower() not in seen:
                deduped.append(normalized)
                seen.add(normalized.lower())
        return deduped[:5]

    async def _apply_corrective_retrieval(
        self,
        query: str,
        context_docs: List[Any],
        analysis: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        retrieval_alpha: float = 0.7,
    ) -> tuple[List[Any], Dict[str, Any]]:
        """Trigger corrective fallback retrieval when the initial context is under-confident."""
        settings = get_settings()
        if not getattr(settings, "CRAG_ENABLED", True):
            return context_docs, {"triggered": False, "confidence": self._retrieval_confidence(context_docs), "queries_tried": []}

        current_confidence = self._retrieval_confidence(context_docs)
        threshold = float(getattr(settings, "CRAG_CONFIDENCE_THRESHOLD", 0.55))
        if current_confidence >= threshold and len(context_docs) > 0:
            return context_docs, {"triggered": False, "confidence": current_confidence, "queries_tried": []}

        fallback_queries = self._build_corrective_queries(query, analysis)
        fallback_docs: List[Any] = []
        queries_tried: List[str] = []
        max_fallbacks = int(getattr(settings, "CRAG_MAX_FALLBACKS", 2))

        for fallback_query in fallback_queries[1:][:max_fallbacks + 1]:
            queries_tried.append(fallback_query)
            docs, _ = await self.retriever.retrieve(
                fallback_query,
                top_k=max(top_k, 3),
                alpha=max(0.2, min(1.0, retrieval_alpha * 0.8)),
            )
            if self.reranker:
                reranked = self.reranker.rerank(fallback_query, docs, top_k=top_k)
                if reranked is not None:
                    docs = reranked
            fallback_docs.extend(docs)

            if self._retrieval_confidence(fallback_docs) >= threshold:
                break

        if not fallback_docs:
            return context_docs, {"triggered": False, "confidence": current_confidence, "queries_tried": queries_tried}

        deduped_fallback: List[Any] = []
        seen_ids = set()
        for doc in fallback_docs:
            key = None
            if isinstance(doc, dict):
                key = doc.get("doc_id") or doc.get("source") or doc.get("content", "")
            else:
                key = getattr(doc, "doc_id", None) or getattr(doc, "source", "") or getattr(doc, "content", "")
            if key and key in seen_ids:
                continue
            deduped_fallback.append(doc)
            seen_ids.add(key)

        deduped_fallback.sort(
            key=lambda doc: float(self._doc_to_confidence_payload(doc).get("relevance_score", self._doc_to_confidence_payload(doc).get("score", 0.0))),
            reverse=True,
        )

        updated_confidence = self._retrieval_confidence(deduped_fallback)
        triggered = updated_confidence > current_confidence or len(deduped_fallback) > len(context_docs)
        return (deduped_fallback if triggered else context_docs), {
            "triggered": triggered,
            "confidence": updated_confidence,
            "queries_tried": queries_tried,
            "original_confidence": current_confidence,
            "fallback_threshold": threshold,
        }

    async def query(
        self,
        query: str,
        mode: str = "simple",
        top_k: int = 5,
        model: Optional[str] = None,
        stream: bool = False,
    ) -> Any:
        """
        Execute the Hybrid RAG pipeline.
        If stream=True, returns an async generator of chunks.
        """
        settings = get_settings()
        query_id = str(uuid.uuid4())
        query_start_time = datetime.now(timezone.utc)
        
        logger.info(f"🔍 Query Mode: {mode} | Query: {query}")
        if self.tracer:
            self.tracer.trace_query(query_id, query, metadata={"mode": mode})
        
        # 1. Try Semantic Cache first
        cache_status = "MISS"
        if self.semantic_cache:
            cached_result = await asyncio.to_thread(self.semantic_cache.get, query)
            if cached_result:
                logger.info("♻️ Semantic cache hit!")
                cache_status = "HIT"
                if cache_hits:
                    try:
                        cache_hits.inc()
                    except AttributeError:
                        pass
                
                response_data = copy.deepcopy(cached_result["response"])
                response_data.update({
                    "query_id": query_id,
                    "query": query,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cache": "semantic_hit"
                })
                
                self.query_history[query_id] = {
                    "query": query,
                    "mode": mode,
                    "response": response_data["response"],
                    "retrieval_id": response_data.get("retrieval_id"),
                    "documents_retrieved": len(response_data.get("retrieved_docs", [])),
                    "retrieval_time_ms": response_data.get("retrieval_time_ms", 0),
                    "retrieved_docs": response_data.get("retrieved_docs", []),
                    "query_time_ms": response_data.get("query_time_ms", 0),
                    "tokens_used": response_data.get("tokens_used", 0),
                    "timestamp": response_data["timestamp"],
                }
                
                if stream:
                    async def cache_stream():
                        yield response_data["response"]
                    return cache_stream()
                return response_data

        # 2. Fallback to exact match cache (legacy/backup)
        cache_key = self._build_cache_key(query, mode, top_k, model)
        if self._should_cache(mode) and cache_key in self.query_cache:
            logger.info("♻️ Exact match cache hit!")
            cache_status = "HIT"
            if cache_hits:
                cache_hits.inc()
                
            response_data = copy.deepcopy(self.query_cache[cache_key])
            response_data.update({
                "query_id": query_id,
                "query": query,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cache": "exact_hit"
            })
            
            self.query_history[query_id] = {
                "query": query,
                "mode": mode,
                "response": response_data["response"],
                "retrieval_id": response_data.get("retrieval_id"),
                "documents_retrieved": len(response_data.get("retrieved_docs", [])),
                "retrieval_time_ms": response_data.get("retrieval_time_ms", 0),
                "retrieved_docs": response_data.get("retrieved_docs", []),
                "query_time_ms": response_data.get("query_time_ms", 0),
                "tokens_used": response_data.get("tokens_used", 0),
                "timestamp": response_data["timestamp"],
            }
            
            if self.tracer:
                self.tracer.trace_query(query_id, query, metadata={"mode": mode, "cache": "exact_hit"})
            
            if stream:
                async def cache_stream():
                    yield response_data["response"]
                return cache_stream()
            return response_data

        # 3. Core RAG Pipeline
        try:
            if cache_misses:
                try:
                    cache_misses.inc()
                except AttributeError:
                    pass
                
            await self.ensure_keyword_index()
            
            # --- Query Understanding Phase ---
            # Skip query understanding for streaming simple queries to reduce latency
            analysis = {"intent": "FACTUAL", "entities": [], "sub_queries": []}
            if self.query_understanding and not (stream and mode == "simple"):
                analysis = await self.query_understanding.analyze(query)
                logger.info(f"🧠 Query Analysis: Intent={analysis['intent']}, Entities={analysis['entities']}")

            # Auto-switch to decompose mode if intent is COMPLEX
            if analysis.get("intent") == "COMPLEX" and mode == "simple":
                logger.info("🔄 Auto-switching to 'decompose' mode due to COMPLEX intent")
                mode = "decompose"

            routing = self.retrieval_router.route(query, analysis) if self.retrieval_router else None
            retrieval_alpha = getattr(routing, "alpha", 0.7)
            retrieval_strategy = getattr(routing, "strategy", "hybrid")
            retrieval_reason = getattr(routing, "reason", "default retrieval policy")
            if routing:
                logger.info(
                    "🧭 Retrieval route selected: strategy=%s, alpha=%.2f, reason=%s",
                    retrieval_strategy,
                    retrieval_alpha,
                    retrieval_reason,
                )
            # ---------------------------------

            retrieval_id = None
            retrieval_start = datetime.now(timezone.utc)
            
            if mode == "decompose":
                logger.info("🧩 Executing Decomposition Flow")
                sub_queries = analysis.get("sub_queries", [])
                if not sub_queries:
                    # Fallback: try to decompose using the chain if analysis didn't provide them
                    if "decomposer" in self.chains:
                        sub_queries = await self.chains["decomposer"].decompose(query)
                        logger.info(f"🛠️ Decomposed into {len(sub_queries)} sub-queries")
                    else:
                        logger.warning("⚠️ Decomposition requested but no sub-queries found and no decomposer chain available")
                        sub_queries = [query]
                
                all_context_docs = []
                sub_answers = []
                
                for i, sq in enumerate(sub_queries):
                    logger.info(f"  ↳ Processing sub-query {i+1}/{len(sub_queries)}: {sq}")
                    sq_docs, _ = await self.retriever.retrieve(
                        sq,
                        top_k=top_k * 2,
                        alpha=0.7,
                    )
                    # Rerank sub-query results
                    sq_reranked = self.reranker.rerank(sq, sq_docs, top_k=top_k)
                    sq_docs = sq_reranked if sq_reranked is not None else []
                    all_context_docs.extend(sq_docs)
                    
                    # Generate a brief answer for the sub-query to help final synthesis
                    sq_context_text = self._build_context_text(sq_docs, sq)
                    sq_llm = self._select_llm("simple", model)
                    if sq_llm and "simple" in self.chains:
                        sq_ans = await self.chains["simple"].invoke(sq, sq_context_text)
                        sub_answers.append(f"Sub-query: {sq}\\nAnswer: {sq_ans}")
                
                # Deduplicate documents
                unique_docs = []
                seen_ids = set()
                for doc in all_context_docs:
                    doc_id = getattr(doc, 'id', str(doc))
                    if doc_id not in seen_ids:
                        unique_docs.append(doc)
                        seen_ids.add(doc_id)
                context_docs = unique_docs
            else:
                # Standard retrieval flow
                context_docs = []
                retrieval_id = None
                agentic_loop_result = None
                
                if self.agentic_loop and settings.AGENTIC_LOOP_ENABLED and not (mode == "decompose"):
                    agentic_loop_result = await self.agentic_loop.retrieve_with_retry(
                        query=query,
                        top_k=top_k,
                        query_decomposition=analysis.get("sub_queries") or None,
                    )
                    # Agentic loop returns docs as dicts, we need to ensure they are compatible with the rest of the pipeline
                    # The loop already handles the retrieval and reranking internally via self.retriever
                    context_docs = agentic_loop_result.get("documents", [])
                    retrieval_id = f"agentic-{query_id}"
                    logger.info(
                        "🔁 Agentic loop retrieval complete: status=%s, confidence=%.2f, iterations=%s",
                        agentic_loop_result.get("final_status"),
                        agentic_loop_result.get("confidence", 0.0),
                        agentic_loop_result.get("iterations", 0),
                    )
                else:
                    # Fallback to standard hybrid retrieval
                    retrieval_start_time = time.time()
                    retrieval_result = await self.retriever.retrieve(
                        query, 
                        top_k=top_k, 
                        alpha=retrieval_alpha
                    )
                    
                    if isinstance(retrieval_result, tuple):
                        context_docs = retrieval_result[0]
                    else:
                        context_docs = retrieval_result
                        
                    if self.reranker:
                        reranked = self.reranker.rerank(query, context_docs, top_k=top_k)
                        context_docs = reranked if reranked is not None else context_docs
                    
                    retrieval_id = f"hybrid-{query_id}"
                    logger.info(f"🔍 Standard retrieval complete: {len(context_docs)} docs retrieved")
                    context_docs, retrieval_id = await self.retriever.retrieve(
                        query,
                        top_k=top_k * 5 if not (stream and mode == "simple") else top_k,
                        alpha=retrieval_alpha,
                    )

                    if not (stream and mode == "simple"):
                        reranked_docs = self.reranker.rerank(query, context_docs, top_k=top_k)
                        context_docs = reranked_docs if reranked_docs is not None else []
                    else:
                        context_docs = context_docs[:top_k]

                graph_trigger = analysis.get("intent") in {"COMPLEX", "COMPARATIVE"} or any(
                    token in query.lower() for token in ["related", "depends on", "connects to", "compares", "versus"]
                )
                if self.graph_rag and graph_trigger:
                    graph_results = self.graph_rag.retrieve(query, top_k=top_k)
                    if graph_results:
                        for result in graph_results:
                            graph_doc = SimpleNamespace(
                                content=result["content"],
                                source=result["source"],
                                score=result["score"],
                                metadata=result.get("metadata", {}),
                            )
                            graph_doc.to_dict = lambda r=result: {
                                "content": r["content"],
                                "source": r["source"],
                                "score": r["score"],
                                "metadata": r.get("metadata", {}),
                            }
                            context_docs.append(graph_doc)
            
            retrieval_end = datetime.now(timezone.utc)
            retrieval_time_ms = (retrieval_end - retrieval_start).total_seconds() * 1000
            if retrieval_latency:
                retrieval_latency.observe(retrieval_time_ms / 1000.0)
            
            # Use the new context manager for token budgeting
            history = None
            if mode == "multi-turn":
                history = self.memory.get_relevant_context("default_session", query=query)
            context_docs = self.context_manager.truncate_context(
                context_docs=context_docs,
                query=query,
                history=history
            )
            context_docs, crag_metadata = await self._apply_corrective_retrieval(
                query=query,
                context_docs=context_docs,
                analysis=analysis,
                top_k=top_k,
                retrieval_alpha=retrieval_alpha,
            )
            context_text = self._build_context_text(context_docs, query)
            
            # Select LLM and execute chain
            llm = self._select_llm(mode, model)
            
            try:
                if llm is None or not self.chains:
                    if stream:
                        async def fallback_stream():
                            yield self._fallback_response(query, context_docs, context_text)
                        response = fallback_stream()
                    else:
                        response = self._fallback_response(query, context_docs, context_text)
                elif mode == "simple":
                    if stream:
                        response = self.chains["simple"].astream(query, context_text)
                    else:
                        response = await self._simple_rag(query, context_text, llm)
                elif mode == "multi-turn":
                    if stream:
                        history = self.memory.get_relevant_context(
                            "default_session", query=query
                        )
                        response = self.chains["multi_turn"].astream(
                            query=query,
                            history=history,
                            context=context_text,
                        )
                    else:
                        response = await self._multi_turn_chat(query, context_text, llm)
                elif mode == "decompose":
                    # Use the specialized decompose chain for final synthesis
                    # It takes the original query, the aggregated sub-answers, and the full context
                    if stream:
                        response = self.chains["decomposer"].astream(
                            query, 
                            sub_answers if 'sub_answers' in locals() else [], 
                            context_text
                        )
                    else:
                        response = await self.chains["decomposer"].invoke(
                            query, 
                            sub_answers if 'sub_answers' in locals() else [], 
                            context_text
                        )
                elif mode == "structured":
                    if stream:
                        # Structured queries typically don't stream well, but we provide a wrapper
                        async def structured_stream():
                            res = await self._structured_query(query, context_text, llm)
                            yield res
                        response = structured_stream()
                    else:
                        response = await self._structured_query(query, context_text, llm)
                else:
                    response = f"Unknown mode: {mode}"
                    if stream:
                        async def unknown_stream():
                            yield response
                        response = unknown_stream()
                
                # --- Response Validation Phase ---
                # If streaming, we cannot validate the full response before sending.
                # We will skip validation for streams or handle it asynchronously.
                if not stream:
                    # Validate response against context to detect hallucinations
                    validation = await self.validator.validate(query, response, context_docs)
                    if not validation.is_valid:
                        logger.warning(f"⚠️ Hallucination detected (Score: {validation.score}): {validation.issues}")
                        if validation.sanitized_response:
                            response = validation.sanitized_response
                    
                    # Final sanitization
                    response = self.validator.sanitize(response)
                # ---------------------------------

            except Exception as exc:
                if errors_total:
                    errors_total.labels(error_type=type(exc).__name__).inc()
                logger.warning(
                    f"LLM execution failed for query {query_id}; falling back to document response: {exc}",
                    exc_info=True,
                )
                response = self._fallback_response(query, context_docs, context_text)
                if stream:
                    async def fallback_stream():
                        yield response
                    response = fallback_stream()
            
            retrieval_info = self.retriever.get_retrieval_diagnostics(retrieval_id)
            retrieval_time_ms = retrieval_info.get("retrieval_time_ms", 0) if isinstance(retrieval_info, dict) else 0
            
            if self.tracer:
                self.tracer.trace_retrieval(query, len(context_docs), retrieval_time_ms)
            
            # Calculate metrics
            query_end_time = datetime.now(timezone.utc)
            query_duration_ms = (query_end_time - query_start_time).total_seconds() * 1000
            
            if query_counter:
                try:
                    query_counter.labels(mode=mode, status="success").inc()
                except AttributeError:
                    pass
            if query_latency:
                try:
                    query_latency.labels(mode=mode).observe(query_duration_ms / 1000)
                except AttributeError:
                    pass

            prompt_text = self._build_prompt_text(mode, query, context_text)
            tokens_used_val = 0
            if not stream:
                tokens_used_val = self._estimate_tokens(prompt_text, response, llm)
                if tokens_used:
                    try:
                        tokens_used.labels(model=model or "default").inc(tokens_used_val)
                    except AttributeError:
                        pass
            
            # Calculate Confidence and Quality Metrics
            if not stream:
                confidence = self.confidence_scorer.score_response(context_docs, response)
            else:
                # For streaming, we can't score the response until it's fully generated.
                # We provide a default or skip scoring for the gauge.
                confidence = None
            
            quality = self.metrics_calculator.calculate_ndcg(context_docs, k=top_k)
            
            # Extract sources from context documents
            sources = [
                doc.get("source", "unknown") if isinstance(doc, dict) else getattr(doc, "source", "unknown")
                for doc in context_docs
            ]
            
            # Prepare result metadata
            result = {
                "query_id": query_id,
                "query": query,
                "response": response,
                "mode": mode,
                "retrieval_id": retrieval_id,
                "documents_retrieved": len(context_docs),
                "retrieved_docs": [doc if isinstance(doc, dict) else doc.to_dict() for doc in context_docs],
                "sources": sources,
                "query_time_ms": query_duration_ms,
                "tokens_used": tokens_used_val,
                "confidence_score": confidence,
                "quality_metrics": quality,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_used": llm.model if hasattr(llm, 'model') else "unknown",
                "retrieval_strategy": retrieval_strategy,
                "retrieval_alpha": retrieval_alpha,
                "retrieval_reason": retrieval_reason,
                "corrective_retrieval": crag_metadata,
                "agentic_loop": None,
            }
            if agentic_loop_result:
                result["agentic_loop"] = {
                    "enabled": True,
                    "status": agentic_loop_result.get("final_status"),
                    "confidence": agentic_loop_result.get("confidence"),
                    "iterations": agentic_loop_result.get("iterations", 0),
                    "queries_tried": agentic_loop_result.get("queries_tried", []),
                    "reformulation_reasons": agentic_loop_result.get("reformulation_reasons", []),
                    "missing_aspects": agentic_loop_result.get("missing_aspects", []),
                }

            if stream:
                return self._stream_response(response, result)

            # Store query history
            self.query_history[query_id] = result
            
            if self.tracer:
                self.tracer.trace_llm_call(
                    model=llm.model if hasattr(llm, 'model') else "unknown",
                    prompt=prompt_text,
                    response=response,
                    tokens=tokens_used_val,
                    confidence=confidence,
                )
            
            # Record metrics
            try:
                if query_latency:
                    query_latency.labels(mode=mode).observe(query_duration_ms / 1000.0)
                if query_counter:
                    query_counter.labels(mode=mode, status="success").inc()
                if tokens_used:
                    tokens_used.labels(model=llm.model if hasattr(llm, 'model') else "unknown").inc(tokens_used_val)
            except AttributeError as e:
                logger.warning(f"Prometheus metric update failed: {e}")
            
            logger.info(f"✅ Query completed in {query_duration_ms:.1f}ms")
            
            if self._should_cache(mode):
                self.query_cache[cache_key] = {
                    "query": query,
                    "response": response,
                    "mode": mode,
                    "retrieved_docs": [doc if isinstance(doc, dict) else doc.to_dict() for doc in context_docs],
                    "model_used": llm.model if hasattr(llm, 'model') else "unknown",
                    "query_time_ms": query_duration_ms,
                    "tokens_used": tokens_used_val,
                    "retrieval_time_ms": retrieval_time_ms,
                    "timestamp": result["timestamp"],
                    "retrieval_id": retrieval_id,
                }
                
                # Store in semantic cache if enabled
                if self.semantic_cache:
                    self.semantic_cache.set(query, result)

            return result
        
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            if stream:
                async def error_stream(error_msg=str(e)):
                    yield f"Error: {error_msg}"
                return error_stream()
            return {
                "query_id": query_id,
                "query": query,
                "error": str(e),
                "mode": mode,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _build_cache_key(self, query: str, mode: str, top_k: int, model: Optional[str]) -> str:
        """Build a stable cache key for query results."""
        model_key = model if model else "default"
        return f"{mode}::{top_k}::{model_key}::{query.strip()}"

    def _should_cache(self, mode: str) -> bool:
        """Only cache simple query results for now."""
        return mode == "simple"

    async def _stream_response(self, response_generator, result_metadata: Dict[str, Any]):
        """
        Helper to wrap an async generator for streaming, while ensuring metadata 
        is handled and the final response is cached/logged.
        """
        full_response = []
        stream_start = time.perf_counter()
        first_chunk_time = None
        async for chunk in self._iter_stream_chunks(response_generator):
            if first_chunk_time is None:
                first_chunk_time = (time.perf_counter() - stream_start) * 1000
            full_response.append(chunk)
            yield chunk
        
        # Update metadata with the actual full response
        result_metadata["response"] = "".join(full_response)
        result_metadata["first_token_time_ms"] = first_chunk_time or 0.0
        result_metadata["stream_duration_ms"] = (time.perf_counter() - stream_start) * 1000
        if hasattr(self, "analytics"):
            self.analytics.log_query({
                **result_metadata,
                "response_time_ms": result_metadata["stream_duration_ms"],
                "cache_status": result_metadata.get("cache_status", "MISS"),
                "status": "success",
            })

    async def _iter_stream_chunks(self, stream):
        """Flatten nested async streams and normalize message chunks to text."""
        async for chunk in stream:
            if inspect.isasyncgen(chunk) or hasattr(chunk, "__aiter__"):
                async for nested_chunk in self._iter_stream_chunks(chunk):
                    yield nested_chunk
                continue

            content = getattr(chunk, "content", chunk)
            if content is not None:
                yield str(content)

    def _wrap_chat_stream(self, response, result_metadata, session_id):
        """
        Returns an async generator that wraps _stream_response and updates memory upon completion.
        """
        async def generator():
            async for chunk in self._stream_response(response, result_metadata):
                yield chunk
            
            # After stream completes, add the full response to memory
            self.memory.add_message(session_id, "assistant", result_metadata["response"])
        
        return generator()

    async def chat(
        self,
        message: str,
        session_id: str,
        model: Optional[str] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Execute multi-turn chat with memory."""
        chat_start_time = datetime.now(timezone.utc)
        message_id = str(uuid.uuid4())
        
        logger.info(f"💬 Chat Session: {session_id} | Message: {message} | Stream: {stream}")
        
        try:
            # Add to memory
            self.memory.add_message(session_id, "user", message)
            
            await self.ensure_keyword_index()
            
            # Get conversation history
            history = self.memory.get_relevant_context(session_id, query=message)

            # Retrieve context
            context_docs, retrieval_id = await self.retriever.retrieve(
                message,
                top_k=3,
                alpha=0.7,
            )
            
            context_text = self._build_context_text(context_docs, message, history)
            
            # Execute multi-turn chain
            llm = self._select_llm("multi-turn", model)
            if llm is None or not self.chains:
                if stream:
                    async def fallback_stream():
                        yield self._fallback_response(message, context_docs, context_text)
                    response = fallback_stream()
                else:
                    response = self._fallback_response(message, context_docs, context_text)
            else:
                try:
                    if stream:
                        response = self.chains["multi_turn"].astream(
                            query=message,
                            history=history,
                            context=context_text,
                        )
                    else:
                        response = await self.chains["multi_turn"].invoke(
                            query=message,
                            history=history,
                            context=context_text,
                        )
                except Exception as exc:
                    logger.warning(
                        f"LLM execution failed for chat session {session_id}; falling back to document response: {exc}",
                        exc_info=True,
                    )
                    if stream:
                        async def fallback_stream():
                            yield self._fallback_response(message, context_docs, context_text)
                        response = fallback_stream()
                    else:
                        response = self._fallback_response(message, context_docs, context_text)
            
            # --- Response Validation Phase ---
            # If streaming, we cannot validate the full response before sending.
            if not stream:
                # Validate response against context to detect hallucinations
                validation = await self.validator.validate(message, response, context_docs)
                if not validation.is_valid:
                    logger.warning(f"⚠️ Hallucination detected in chat (Score: {validation.score}): {validation.issues}")
                    if validation.sanitized_response:
                        response = validation.sanitized_response
                
                # Final sanitization
                response = self.validator.sanitize(response)
            
            # Add response to memory (deferred for stream)
            if not stream:
                self.memory.add_message(session_id, "assistant", response)
            
            # Calculate metrics
            chat_time = (datetime.now(timezone.utc) - chat_start_time).total_seconds() * 1000
            prompt_text = self._build_prompt_text("multi-turn", message, context_text, history)
            tokens_used = self._estimate_tokens(prompt_text, response if not stream else "", llm)

            if stream:
                # Prepare metadata for _stream_response
                result_metadata = {
                    "query_id": message_id,
                    "query": message,
                    "response": "", # Will be filled by _stream_response
                    "mode": "multi-turn",
                    "retrieval_id": retrieval_id,
                    "documents_retrieved": len(context_docs),
                    "retrieved_docs": [doc if isinstance(doc, dict) else doc.to_dict() for doc in context_docs],
                    "sources": [doc.get("source", "unknown") if isinstance(doc, dict) else getattr(doc, "source", "unknown") for doc in context_docs],
                    "query_time_ms": chat_time,
                    "tokens_used": tokens_used,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_used": llm.model if hasattr(llm, 'model') else "unknown",
                }
                
                # Wrap the stream to handle memory update and metrics
                return self._wrap_chat_stream(response, result_metadata, session_id)

            logger.info(f"✅ Chat completed in {chat_time:.1f}ms")

            logger.info(f"✅ Chat completed in {chat_time:.1f}ms")
            
            if self.tracer:
                self.tracer.trace_llm_call(
                    model=llm.model if hasattr(llm, 'model') else "unknown",
                    prompt=prompt_text,
                    response=response,
                    tokens=tokens_used,
                )

            return {
                "session_id": session_id,
                "message_id": message_id,
                "response": response,
                "retrieved_docs": [doc.to_dict() for doc in context_docs],
                "model_used": llm.model if hasattr(llm, 'model') else "unknown",
                "chat_time_ms": chat_time,
                "tokens_used": tokens_used,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Chat failed: {e}", exc_info=True)
            return {
                "session_id": session_id,
                "message_id": message_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> tuple[List[Dict[str, Any]], str]:
        """Retrieve documents for a query."""
        await self.ensure_keyword_index()
        docs, query_id = await self.retriever.retrieve(query, top_k=top_k)
        if self.tracer:
            self.tracer.trace_retrieval(query, len(docs), 0.0)
        return [doc.to_dict() for doc in docs], query_id
    
    def get_query_diagnostics(self, query_id: str) -> Dict[str, Any]:
        """Get diagnostics for a query."""
        if query_id not in self.query_history:
            return {"error": "Query ID not found"}
        
        return self.query_history[query_id]
    
    async def rebuild_index(self, clear: bool = False) -> Dict[str, Any]:
        """Rebuild vector index from persisted documents."""
        try:
            from rag.indexing import DocumentIndexer

            settings = get_settings()
            indexer = DocumentIndexer(
                settings.UPLOAD_DIR,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )

            if clear:
                indexer.documents_metadata.clear()
                indexer._save_metadata()
                if hasattr(self.vector_store, "clear"):
                    await self.vector_store.clear()
                self.retriever.all_documents = []
                self.retriever.bm25 = None

            docs = indexer.load_documents(indexer.storage_path)
            if docs and hasattr(self.vector_store, "add_documents"):
                await self.vector_store.add_documents(docs)
                await self.retriever.index_documents(docs)

            return {
                "status": "success",
                "documents_loaded": len(docs),
                "total_tracked": indexer.get_document_count(),
            }
        except Exception as e:
            logger.error(f"HybridRAG rebuild_index failed: {e}")
            return {"error": str(e)}
    
    def _select_llm(self, mode: str, model_override: Optional[str] = None):
        """Select LLM based on mode."""
        if model_override and HAS_LANGCHAIN:
            try:
                return ChatNVIDIA(model=model_override)
            except Exception as exc:
                logger.warning(f"Model override failed: {exc}")
                return self.chat_llm
        
        if mode == "multi-turn":
            return self.chat_llm
        elif mode == "structured":
            return self.structured_llm
        return self.reasoning_llm or self.chat_llm
    
    def _fallback_response(self, query: str, context_docs: List[Any], context_text: str) -> str:
        """Generate a simple answer from retrieved documents when no LLM is available."""
        if not context_docs:
            return "No relevant documents were found. Please upload a document and try again."

        best_doc = context_docs[0]
        # Handle both object and dict representations of documents
        content = best_doc.content if hasattr(best_doc, "content") else best_doc.get("content", "")
        
        snippet = content.strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."

        return (
            f"I found relevant information in the uploaded document(s): {snippet}"
        )

    def _build_context_text(self, context_docs: List[Any], query: Optional[str] = None, history: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """Build a compact context string using the ContextManager for token budgeting."""
        # Use the context manager to truncate documents to fit the budget
        if max_tokens is not None:
            self.context_manager.max_tokens = max_tokens
        
        truncated_docs = self.context_manager.truncate_context(context_docs, query, history)
        
        parts: List[str] = []
        for doc in truncated_docs:
            content = doc.content if hasattr(doc, "content") else doc.get("content", "")
            source = getattr(doc, "source", "document") if hasattr(doc, "source") else doc.get("source", "document")
            parts.append(f"[{source}]\n{content}")

        return "\n\n".join(parts)

    # Removed _truncate_to_token_budget as it is now handled by ContextManager


    def _build_prompt_text(self, mode: str, query: str, context: str, history: Optional[str] = None) -> str:
        """Build a text prompt approximation for token estimation."""
        if mode == "multi-turn":
            return (
                f"Conversation History:\n{history}\n\n"
                f"Document Context:\n{context}\n\n"
                f"Current Question: {query}"
            )
        elif mode == "structured":
            return (
                f"Data Summary:\n{context}\n\n"
                f"Question: {query}"
            )
        return f"Context:\n{context}\n\nQuestion: {query}"

    def _estimate_tokens(self, prompt: str, response: str, llm=None) -> int:
        """Estimate token usage without invoking unknown model tokenizer implementations."""
        if llm is None:
            return len(prompt.split()) + len(response.split())

        model_name = getattr(llm, "model", "") or ""
        lower_name = model_name.lower()

        # Only use model-specific token counting for recognized GPT-family models.
        if "gpt" in lower_name and hasattr(llm, "get_num_tokens"):
            try:
                return llm.get_num_tokens(prompt) + llm.get_num_tokens(response)
            except Exception:
                pass

        # For unknown or non-GPT models, use a simple word-based approximation.
        return len(prompt.split()) + len(response.split())

    async def _simple_rag(self, query: str, context: str, llm) -> str:
        """Simple RAG response."""
        chain = self.chains["simple"]
        response = await chain.invoke(query, context)
        return response
    
    async def _multi_turn_chat(self, query: str, context: str, llm, session_id: str = "default_session") -> str:
        """Multi-turn chat response."""
        history = self.memory.get_relevant_context(session_id, query=query)
        response = await self.chains["multi_turn"].invoke(
            query=query,
            history=history,
            context=context,
        )
        self.memory.add_message(session_id, "user", query)
        self.memory.add_message(session_id, "assistant", response)
        return response
    
    async def _decompose_query(self, query: str, context: str, llm) -> str:
        """Decomposed query response using a multi-stage retrieve-rerank-synthesize pipeline."""
        chain = self.chains["query_decomposer"]
        
        # 1. Decompose the complex query into sub-queries
        sub_queries = await self.chains["decompose"].decompose(query)
        logger.info(f"Decomposed query into {len(sub_queries)} sub-queries: {sub_queries}")
        
        sub_answers = {}
        all_context_docs = []
        
        for i, sub_q in enumerate(sub_queries):
            # 2. Rewrite each sub-query for better retrieval
            rewritten_q = await self.chains["decompose"].rewrite(sub_q)
            if rewritten_q != sub_q:
                logger.info(f"Rewrote sub-query {i+1}: '{sub_q}' -> '{rewritten_q}'")
            
            # 3. Retrieve context for the rewritten sub-query
            sub_docs, sub_id = await self.retriever.retrieve(
                rewritten_q,
                top_k=top_k,
                alpha=0.7,
            )
            
            # Rerank sub-query results
            reranked_sub_docs = self.reranker.rerank(rewritten_q, sub_docs, top_k=top_k)
            final_sub_docs = reranked_sub_docs if reranked_sub_docs is not None else []
            all_context_docs.extend(final_sub_docs)
            
            # 4. Generate answer for the sub-query
            sub_context_text = self._build_context_text(final_sub_docs)
            sub_response = await self._simple_rag(rewritten_q, sub_context_text, llm)
            sub_answers[sub_q] = sub_response
        
        # 5. Synthesize all sub-answers into the final response
        response = await self.chains["decompose"].invoke(query, sub_answers)
        return response
    
    async def _structured_query(self, query: str, context: str, llm) -> str:
        """Structured data query response."""
        if "structured_data" in self.chains:
            return await self.chains["structured_data"].invoke(query, context, "N/A")
        return await self.chains["simple_rag"].invoke(query, context)
