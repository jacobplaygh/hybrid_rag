"""Hybrid RAG orchestrator combining LangChain + LlamaIndex."""

import copy
import logging
import re
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from rag.reranker import CrossEncoderReranker
from rag.confidence_scorer import ConfidenceScorer
from rag.quality_metrics import QualityMetricsCalculator
from rag.query_understanding import QueryUnderstanding
from rag.chains import create_chains, QueryDecomposerChain
from rag.retrieval import HybridRetriever
from rag.memory import ConversationMemory
from rag.semantic_cache import SemanticCache
from rag.context_manager import ContextManager
from rag.response_validator import ResponseValidator
from rag.analytics import QueryAnalytics

DEFAULT_MAX_CONTEXT_TOKENS = 120000

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

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

        # Initialize Context Manager
        settings = get_settings()
        self.context_manager = ContextManager(
            max_tokens=getattr(settings, "MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS)
        )

        # Initialize Response Validator
        self.validator = ResponseValidator(llm=reasoning_llm or chat_llm)

        # Initialize Analytics
        self.analytics = QueryAnalytics()

        # Initialize Semantic Cache
        self.semantic_cache = None
        if settings.SEMANTIC_CACHE_ENABLED:
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
        query_id = str(uuid.uuid4())
        query_start_time = datetime.now(timezone.utc)
        
        logger.info(f"🔍 Query Mode: {mode} | Query: {query}")
        if self.tracer:
            self.tracer.trace_query(query_id, query, metadata={"mode": mode})
        
        # 1. Try Semantic Cache first
        cache_status = "MISS"
        if self.semantic_cache:
            cached_result = self.semantic_cache.get(query)
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
            analysis = {"intent": "FACTUAL", "entities": [], "sub_queries": []}
            if self.query_understanding:
                analysis = await self.query_understanding.analyze(query)
                logger.info(f"🧠 Query Analysis: Intent={analysis['intent']}, Entities={analysis['entities']}")

            # Auto-switch to decompose mode if intent is COMPLEX
            if analysis.get("intent") == "COMPLEX" and mode == "simple":
                logger.info("🔄 Auto-switching to 'decompose' mode due to COMPLEX intent")
                mode = "decompose"
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
                context_docs, retrieval_id = await self.retriever.retrieve(
                    query,
                    top_k=top_k * 5,  # Retrieve more for reranking
                    alpha=0.7,  # 70% semantic, 30% keyword
                )
                
                # Rerank results to improve precision
                reranked_docs = self.reranker.rerank(query, context_docs, top_k=top_k)
                context_docs = reranked_docs if reranked_docs is not None else []
            
            retrieval_end = datetime.now(timezone.utc)
            retrieval_time_ms = (retrieval_end - retrieval_start).total_seconds() * 1000
            if retrieval_latency:
                retrieval_latency.observe(retrieval_time_ms / 1000.0)
            
            # Use the new context manager for token budgeting
            history = self.memory.get_history(query_id) if hasattr(self.memory, 'get_history') else None
            context_docs = self.context_manager.truncate_context(
                context_docs=context_docs,
                query=query,
                history=history
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
                        response = self.chains["multi_turn"].astream(query, context_text)
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
            }

            if stream:
                return self._stream_response(response, result)

            # Store query history
            self.query_history[query_id] = result
            
            if self.tracer:
                # ...existing code...
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
        async for chunk in response_generator:
            full_response.append(chunk)
            yield chunk
        
        final_text = "".join(full_response)
        
        # Update metadata with the actual full response
        result_metadata["response"] = final_text
        
        # --- Deferred Metrics and Analytics ---
        mode = result_metadata["mode"]
        query_id = result_metadata["query_id"]
        query_duration_ms = result_metadata["query_time_ms"]
        tokens_used_val = result_metadata["tokens_used"]
        confidence = result_metadata["confidence_score"]
        quality = result_metadata["quality_metrics"]
        
        # Log to Analytics
        try:
            self.analytics.log_query({
                "query_id": query_id,
                "query": result_metadata["query"],
                "mode": mode,
                "intent": "STREAMED", # Simplified for deferred log
                "response_time_ms": query_duration_ms,
                "tokens_used": tokens_used_val,
                "cache_status": "MISS",
                "confidence_score": confidence.overall_score if (confidence and hasattr(confidence, 'overall_score')) else (confidence if confidence else 0.0),
                "status": "success"
            })
        except Exception as e:
            logger.warning(f"Deferred analytics logging failed: {e}")

        # Record to Prometheus
        try:
            # We need to access the gauges from the class instance
            # Assuming these are initialized in __init__ as self.ndcg_gauge, etc.
            if hasattr(self, 'ndcg_gauge') and self.ndcg_gauge:
                self.ndcg_gauge.labels(mode=mode).set(quality)
            if hasattr(self, 'confidence_gauge') and self.confidence_gauge and confidence:
                confidence_val = confidence.overall_score if hasattr(confidence, 'overall_score') else confidence
                self.confidence_gauge.labels(mode=mode).set(confidence_val)
        except Exception as e:
            logger.warning(f"Deferred Prometheus update failed: {e}")
        # ---------------------------------------

        # Store in history and cache
        query_id = result_metadata["query_id"]
        self.query_history[query_id] = result_metadata
        
        # Cache if applicable
        if self._should_cache(mode):
            cache_key = self._build_cache_key(
                result_metadata["query"], 
                mode, 
                result_metadata.get("top_k", 5), 
                result_metadata["model_used"]
            )
            self.query_cache[cache_key] = {
                "query": result_metadata["query"],
                "response": final_text,
                "mode": mode,
                "retrieved_docs": result_metadata["retrieved_docs"],
                "model_used": result_metadata["model_used"],
                "query_time_ms": result_metadata["query_time_ms"],
                "tokens_used": result_metadata["tokens_used"],
                "timestamp": result_metadata["timestamp"],
                "retrieval_id": result_metadata["retrieval_id"],
            }
            if self.semantic_cache:
                self.semantic_cache.set(result_metadata["query"], result_metadata)

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
            history = self.memory.get_context(session_id)

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
                async def chat_stream_wrapper():
                    async for chunk in self._stream_response(response, result_metadata):
                        yield chunk
                    # After stream completes, add the full response to memory
                    self.memory.add_message(session_id, "assistant", result_metadata["response"])
                
                return chat_stream_wrapper()

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
        history = self.memory.get_context(session_id)
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
