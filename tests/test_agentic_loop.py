"""Unit tests for agentic retrieval loop components."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List, Dict, Any

from rag.agentic_loop import (
    ContextSufficiencyEvaluator,
    QueryReformulator,
    AgenticRetrieverLoop,
    AgenticLoopConfig,
    SufficiencyEvaluation,
    QueryReformation,
)


class TestContextSufficiencyEvaluator:
    """Test ContextSufficiencyEvaluator component."""
    
    @pytest.fixture
    def evaluator(self):
        return ContextSufficiencyEvaluator(min_term_coverage=0.5, min_sub_q_coverage=0.7)
    
    @pytest.fixture
    def sample_docs(self) -> List[Dict[str, Any]]:
        return [
            {
                "doc_id": "doc1",
                "content": "Python is a programming language. It is widely used for data science.",
                "source": "wikipedia",
                "score": 0.9,
            },
            {
                "doc_id": "doc2",
                "content": "Data science involves statistics, machine learning, and data analysis.",
                "source": "tutorial",
                "score": 0.85,
            },
        ]
    
    @pytest.mark.asyncio
    async def test_evaluate_sufficient_context(self, evaluator, sample_docs):
        """Test evaluation of sufficient context."""
        query = "What is Python used for in data science?"
        
        result = await evaluator.evaluate(
            query=query,
            retrieved_docs=sample_docs,
            query_decomposition=None
        )
        
        assert isinstance(result, SufficiencyEvaluation)
        assert result.confidence > 0.0
        assert "term_coverage" in result.metrics
        assert "diversity_score" in result.metrics
        assert result.is_sufficient is not None
    
    @pytest.mark.asyncio
    async def test_evaluate_empty_docs(self, evaluator):
        """Test evaluation with no documents."""
        query = "What is Python?"
        
        result = await evaluator.evaluate(
            query=query,
            retrieved_docs=[],
            query_decomposition=None
        )
        
        assert result.confidence < 0.65
        assert result.is_sufficient is False
    
    def test_calculate_term_coverage(self, evaluator, sample_docs):
        """Test term coverage calculation."""
        query = "python programming data science"
        coverage = evaluator._calculate_term_coverage(query, sample_docs)
        
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0
        assert coverage > 0.5  # Should have good coverage
    
    def test_calculate_term_coverage_no_match(self, evaluator, sample_docs):
        """Test term coverage with no matching terms."""
        query = "quantum computing qubits"
        coverage = evaluator._calculate_term_coverage(query, sample_docs)
        
        assert coverage == 0.0
    
    def test_calculate_diversity(self, evaluator, sample_docs):
        """Test source diversity calculation."""
        diversity = evaluator._calculate_diversity(sample_docs)
        
        assert isinstance(diversity, float)
        assert 0.0 <= diversity <= 1.0
        # Should be 2 sources / 2 docs = 1.0
        assert diversity == 1.0
    
    def test_calculate_diversity_single_source(self, evaluator):
        """Test diversity with all docs from same source."""
        docs = [
            {"doc_id": "d1", "content": "text1", "source": "src1", "score": 0.9},
            {"doc_id": "d2", "content": "text2", "source": "src1", "score": 0.8},
        ]
        diversity = evaluator._calculate_diversity(docs)
        
        # 1 unique source / 2 docs = 0.5
        assert diversity == 0.5
    
    def test_calculate_average_relevance(self, evaluator, sample_docs):
        """Test average relevance calculation."""
        relevance = evaluator._calculate_average_relevance(sample_docs)
        
        assert isinstance(relevance, float)
        assert relevance == pytest.approx((0.9 + 0.85) / 2, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_evaluate_with_decomposition(self, evaluator, sample_docs):
        """Test evaluation with query decomposition."""
        query = "What is Python used for?"
        decomposition = [
            "What is Python?",
            "What are programming languages?",
            "What is data science?",
        ]
        
        result = await evaluator.evaluate(
            query=query,
            retrieved_docs=sample_docs,
            query_decomposition=decomposition
        )
        
        assert "sub_question_coverage" in result.metrics
        assert isinstance(result.metrics["sub_question_coverage"], float)


class TestQueryReformulator:
    """Test QueryReformulator component."""
    
    @pytest.fixture
    def reformulator(self):
        return QueryReformulator(use_llm=False)
    
    @pytest.mark.asyncio
    async def test_reformulate_add_keywords(self, reformulator):
        """Test add_keywords reformulation strategy."""
        query = "What is machine learning?"
        aspects = ["deep_learning", "neural_networks"]
        
        result = await reformulator.reformulate(
            original_query=query,
            missing_aspects=aspects,
            strategy="add_keywords"
        )
        
        assert isinstance(result, QueryReformation)
        assert "deep learning" in result.reformulated_query.lower()
        assert "neural networks" in result.reformulated_query.lower()
        assert result.strategy == "add_keywords"
    
    @pytest.mark.asyncio
    async def test_reformulate_broaden(self, reformulator):
        """Test broaden reformulation strategy."""
        query = "How does TensorFlow 2.0 handle GPU acceleration?"
        
        result = await reformulator.reformulate(
            original_query=query,
            missing_aspects=[],
            strategy="broaden"
        )
        
        assert isinstance(result, QueryReformation)
        assert len(result.reformulated_query) > 0
        assert result.strategy == "broaden"
    
    @pytest.mark.asyncio
    async def test_reformulate_synonyms(self, reformulator):
        """Test synonyms reformulation strategy."""
        query = "What causes climate change?"
        
        result = await reformulator.reformulate(
            original_query=query,
            missing_aspects=[],
            strategy="synonyms"
        )
        
        assert isinstance(result, QueryReformation)
        assert len(result.reformulated_query) > 0
        assert result.strategy == "synonyms"
    
    @pytest.mark.asyncio
    async def test_reformulate_auto_without_llm(self, reformulator):
        """Test auto strategy without LLM (should use add_keywords)."""
        query = "What is Python?"
        aspects = ["typing", "functions"]
        
        result = await reformulator.reformulate(
            original_query=query,
            missing_aspects=aspects,
            strategy="auto"
        )
        assert result.strategy == "add_keywords"
        assert "typing" in result.reformulated_query.lower()
        assert "functions" in result.reformulated_query.lower()

class TestAgenticRetrieverLoop:
    """Test AgenticRetrieverLoop orchestration."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = AsyncMock()
        # Mock retrieve to return (docs, metadata)
        retriever.retrieve.return_value = (
            [{"doc_id": "1", "content": "test content", "source": "src1", "score": 0.8}],
            {"latency": 0.1}
        )
        return retriever

    @pytest.fixture
    def mock_scorer(self):
        scorer = Mock()
        # Mock score_response to return an object with overall_score
        score_obj = MagicMock()
        score_obj.overall_score = 0.8
        scorer.score_response.return_value = score_obj
        return scorer

    @pytest.fixture
    def loop(self, mock_retriever, mock_scorer):
        config = AgenticLoopConfig(
            max_retries=3,
            confidence_threshold=0.7,
            retry_strategy="auto",
            timeout_seconds=10
        )
        return AgenticRetrieverLoop(
            retriever=mock_retriever,
            confidence_scorer=mock_scorer,
            evaluator=ContextSufficiencyEvaluator(mock_scorer),
            reformulator=QueryReformulator(use_llm=False),
            config=config
        )

    @pytest.mark.asyncio
    async def test_loop_success_first_try(self, loop, mock_retriever):
        """Test loop that succeeds on the first retrieval."""
        query = "What is AI?"
        
        # Mock evaluator to return sufficient
        loop.evaluator.evaluate = AsyncMock(return_value=SufficiencyEvaluation(
            confidence=0.8,
            is_sufficient=True,
            metrics={"term_coverage": 0.9},
            missing_aspects=[]
        ))

        result = await loop.retrieve_iteratively(query)
        
        assert result["is_sufficient"] is True
        assert mock_retriever.retrieve.call_count == 1
        assert len(result["documents"]) > 0

    @pytest.mark.asyncio
    async def test_loop_iterative_correction(self, loop, mock_retriever):
        """Test loop that requires reformulation to reach sufficiency."""
        query = "What is AI?"
        
        # First call: insufficient, Second call: sufficient
        loop.evaluator.evaluate = AsyncMock(side_effect=[
            SufficiencyEvaluation(
                confidence=0.4,
                is_sufficient=False,
                metrics={"term_coverage": 0.3},
                missing_aspects=["deep learning"]
            ),
            SufficiencyEvaluation(
                confidence=0.8,
                is_sufficient=True,
                metrics={"term_coverage": 0.9},
                missing_aspects=[]
            )
        ])

        result = await loop.retrieve_iteratively(query)
        
        assert result["is_sufficient"] is True
        assert mock_retriever.retrieve.call_count == 2
        assert result["iterations"] == 2

    @pytest.mark.asyncio
    async def test_loop_max_retries_reached(self, loop, mock_retriever):
        """Test loop that fails to reach sufficiency within max_retries."""
        query = "What is AI?"
        
        # Always return insufficient
        loop.evaluator.evaluate = AsyncMock(return_value=SufficiencyEvaluation(
            confidence=0.3,
            is_sufficient=False,
            metrics={"term_coverage": 0.2},
            missing_aspects=["something"]
        ))

        result = await loop.retrieve_iteratively(query)
        
        assert result["is_sufficient"] is False
        # max_retries is 3, so total attempts = 1 (initial) + 3 (retries) = 4
        assert mock_retriever.retrieve.call_count == 4
        assert result["iterations"] == 4

        
        # Without LLM, should fall back to add_keywords
        assert result.strategy == "add_keywords"
        assert "typing" in result.reformulated_query.lower()
    
    @pytest.mark.asyncio
    async def test_reformulate_empty_aspects(self, reformulator):
        """Test reformulation with empty missing aspects."""
        query = "What is Python?"
        
        result = await reformulator.reformulate(
            original_query=query,
            missing_aspects=[],
            strategy="add_keywords"
        )
        
        # Should return original query when no aspects to add
        assert result.reformulated_query == query


class TestAgenticRetrieverLoop:
    """Test AgenticRetrieverLoop orchestration."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Mock HybridRetriever."""
        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(return_value=[
            Mock(
                doc_id="doc1",
                content="Sample document about Python programming.",
                source="source1",
                score=0.9
            )
        ])
        return retriever
    
    @pytest.fixture
    def mock_scorer(self):
        """Mock ConfidenceScorer."""
        scorer = Mock()
        return scorer
    
    @pytest.fixture
    def mock_evaluator(self):
        """Mock ContextSufficiencyEvaluator."""
        evaluator = AsyncMock()
        evaluator.evaluate = AsyncMock(return_value=SufficiencyEvaluation(
            is_sufficient=True,
            confidence=0.85,
            missing_aspects=[],
            reasoning="Context appears sufficient",
            metrics={"term_coverage": 0.9, "diversity_score": 0.8}
        ))
        return evaluator
    
    @pytest.fixture
    def mock_reformulator(self):
        """Mock QueryReformulator."""
        reformulator = AsyncMock()
        reformulator.reformulate = AsyncMock(return_value=QueryReformation(
            reformulated_query="What is Python used for in data science?",
            strategy="add_keywords",
            confidence=0.7,
            reasoning="Added keywords to target missing aspects"
        ))
        return reformulator
    
    @pytest.fixture
    def agentic_loop(self, mock_retriever, mock_scorer, mock_evaluator, mock_reformulator):
        config = AgenticLoopConfig(
            max_retries=2,
            confidence_threshold=0.7,
            use_query_reformulation=True
        )
        return AgenticRetrieverLoop(
            retriever=mock_retriever,
            confidence_scorer=mock_scorer,
            evaluator=mock_evaluator,
            reformulator=mock_reformulator,
            config=config
        )
    
    @pytest.mark.asyncio
    async def test_retrieve_with_retry_success_on_first_try(self, agentic_loop):
        """Test successful retrieval on first attempt."""
        result = await agentic_loop.retrieve_with_retry(
            query="What is Python?",
            top_k=3
        )
        
        assert result["final_status"] == "success"
        assert result["iterations"] == 1
        assert len(result["documents"]) > 0
        assert result["confidence"] > 0.0
        agentic_loop.retriever.retrieve.assert_awaited_once_with(
            "What is Python?", top_k=3
        )

    @pytest.mark.asyncio
    async def test_retrieve_passes_query_decomposition_to_evaluator(
        self, agentic_loop, mock_evaluator
    ):
        decomposition = ["What is Python?", "What is it used for?"]

        await agentic_loop.retrieve_with_retry(
            query="What is Python used for?",
            query_decomposition=decomposition,
        )

        assert mock_evaluator.evaluate.await_args.kwargs["query_decomposition"] == decomposition

    @pytest.mark.asyncio
    async def test_llm_reformulation_uses_async_model(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = Mock(content="Find Python uses in data science")
        reformulator = QueryReformulator(llm=llm)

        result = await reformulator.reformulate(
            "What is Python?", ["data science"], strategy="auto"
        )

        assert result.reformulated_query == "Find Python uses in data science"
        llm.ainvoke.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_with_retry_needs_reformulation(self, agentic_loop, mock_evaluator):
        """Test retrieval that needs reformulation on second try."""
        # First iteration: low confidence
        # Second iteration: high confidence
        evaluations = [
            SufficiencyEvaluation(
                is_sufficient=False,
                confidence=0.5,
                missing_aspects=["details"],
                reasoning="Need more context",
                metrics={}
            ),
            SufficiencyEvaluation(
                is_sufficient=True,
                confidence=0.85,
                missing_aspects=[],
                reasoning="Context sufficient",
                metrics={}
            )
        ]
        mock_evaluator.evaluate.side_effect = evaluations
        
        result = await agentic_loop.retrieve_with_retry(
            query="What is Python?",
            top_k=5
        )
        
        # Should have retried
        assert result["iterations"] == 2
        assert len(result["queries_tried"]) == 2
        assert result["final_status"] == "success"
    
    @pytest.mark.asyncio
    async def test_retrieve_with_retry_max_retries(self, agentic_loop, mock_evaluator):
        """Test max retries limit."""
        # Always return low confidence
        mock_evaluator.evaluate.return_value = SufficiencyEvaluation(
            is_sufficient=False,
            confidence=0.3,
            missing_aspects=["everything"],
            reasoning="No relevant context",
            metrics={}
        )
        
        result = await agentic_loop.retrieve_with_retry(
            query="Something obscure",
            top_k=5
        )
        
        # Should hit max retries
        assert result["final_status"] == "max_retries"
        assert result["iterations"] == agentic_loop.config.max_retries + 1
    
    @pytest.mark.asyncio
    async def test_retrieve_with_retry_timeout(self, agentic_loop, mock_evaluator):
        """Test timeout protection."""
        # Make evaluation slow
        async def slow_evaluate(*args, **kwargs):
            await asyncio.sleep(2)
            return SufficiencyEvaluation(
                is_sufficient=False,
                confidence=0.5,
                missing_aspects=[],
                reasoning="",
                metrics={}
            )
        
        mock_evaluator.evaluate.side_effect = slow_evaluate
        
        result = await agentic_loop.retrieve_with_retry(
            query="test",
            top_k=5,
            timeout_seconds=1
        )
        
        # Should timeout
        assert result["final_status"] == "timeout"
    
    @pytest.mark.asyncio
    async def test_retrieve_with_retry_empty_docs(self, agentic_loop, mock_retriever):
        """Test handling of empty retrieval results."""
        mock_retriever.retrieve.return_value = []
        
        result = await agentic_loop.retrieve_with_retry(
            query="Something with no docs",
            top_k=5
        )
        
        assert len(result["documents"]) == 0
        assert result["confidence"] < 0.65


class TestAgenticLoopConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AgenticLoopConfig()
        
        assert config.max_retries == 2
        assert config.confidence_threshold == 0.7
        assert config.use_query_reformulation is True
        assert config.retry_strategy == "auto"
        assert config.timeout_seconds == 30
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AgenticLoopConfig(
            max_retries=3,
            confidence_threshold=0.8,
            timeout_seconds=60
        )
        
        assert config.max_retries == 3
        assert config.confidence_threshold == 0.8
        assert config.timeout_seconds == 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
