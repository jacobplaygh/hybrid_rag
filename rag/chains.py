"""LangChain chain definitions for RAG."""

import logging
from typing import Dict, Any, List

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    logging.warning("LangChain not installed")

logger = logging.getLogger(__name__)


class SimpleRAGChain:
    """Simple context-augmented query chain."""
    
    def __init__(self, llm: ChatNVIDIA = None):
        """Initialize SimpleRAGChain."""
        if not HAS_LANGCHAIN:
            raise ImportError("LangChain is required")
        self.llm = llm
        logger.info("Initializing SimpleRAGChain")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Answer questions based on the provided context.
            If the context doesn't contain relevant information, say so clearly.
            Be concise but comprehensive in your answers."""),
            ("human", """Context:
{context}

Question: {query}""")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def invoke(self, query: str, context: str) -> str:
        """Execute simple RAG chain."""
        last_error = None
        for attempt in range(2):
            try:
                response = self.chain.invoke({
                    "query": query,
                    "context": context,
                })
                return response
            except Exception as e:
                last_error = e
                message = str(e)
                if "504" in message or "Gateway Timeout" in message or "timeout" in message.lower():
                    logger.warning(f"SimpleRAGChain transient error on attempt {attempt + 1}: {e}")
                    if attempt < 1:
                        continue
                logger.error(f"SimpleRAGChain error: {e}")
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("SimpleRAGChain failed without an error")

    async def astream(self, query: str, context: str):
        """Stream simple RAG chain response."""
        try:
            async for chunk in self.chain.astream({
                "query": query,
                "context": context,
            }):
                yield chunk
        except Exception as e:
            logger.error(f"SimpleRAGChain streaming error: {e}")
            raise


class MultiTurnChain:
    """Multi-turn conversation chain with memory."""
    
    def __init__(self, llm: ChatNVIDIA = None, memory=None):
        """Initialize MultiTurnChain."""
        if not HAS_LANGCHAIN:
            raise ImportError("LangChain is required")
        self.llm = llm
        self.memory = memory
        logger.info("Initializing MultiTurnChain")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant named Envie.
            You maintain context from the conversation history.
            You provide consistent, helpful responses.
            Use the provided document context when relevant."""),
            ("human", """Conversation History:
{history}

Document Context:
{context}

Current Question: {query}""")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def invoke(self, query: str, history: str, context: str) -> str:
        """Execute multi-turn chain."""
        try:
            response = self.chain.invoke({
                "query": query,
                "history": history,
                "context": context,
            })
            return response
        except Exception as e:
            logger.error(f"MultiTurnChain error: {e}")
            raise

    async def astream(self, query: str, history: str, context: str):
        """Stream multi-turn chain response."""
        try:
            async for chunk in self.chain.astream({
                "query": query,
                "history": history,
                "context": context,
            }):
                yield chunk
        except Exception as e:
            logger.error(f"MultiTurnChain streaming error: {e}")
            raise


class QueryDecomposerChain:
    """Decompose complex queries into sub-questions."""
    
    def __init__(self, llm: ChatNVIDIA = None):
        """Initialize QueryDecomposerChain."""
        if not HAS_LANGCHAIN:
            raise ImportError("LangChain is required")
        self.llm = llm
        logger.info("Initializing QueryDecomposerChain")
        
        self.decompose_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at breaking down complex questions into simpler sub-questions.
            Break the question into 2-4 specific sub-questions that together answer the main question.
            Return ONLY the sub-questions, one per line, numbered.
            Example:
            1. What is X?
            2. How does X relate to Y?
            3. What are the implications of Y on Z?"""),
            ("human", "{query}")
        ])
        
        self.decompose_chain = self.decompose_prompt | self.llm | StrOutputParser()

    async def decompose(self, query: str) -> List[str]:
        """Decompose a complex query into sub-queries."""
        try:
            response = await self.decompose_chain.ainvoke({"query": query})
            # Split by lines and remove empty lines/numbers
            sub_queries = [
                line.strip().lstrip('0123456789. ') 
                for line in response.split('\n') 
                if line.strip()
            ]
            return sub_queries
        except Exception as e:
            logger.error(f"QueryDecomposerChain error: {e}")
            return [query]
    async def invoke(self, query: str, sub_answers: List[str], context: str) -> str:
        """Synthesize final answer from sub-queries and context."""
        try:
            # Use the same LLM as the decomposer for synthesis
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert synthesizer. Combine the following sub-query answers and document context to provide a comprehensive final answer to the original query.
                Ensure the answer is cohesive, accurate, and directly addresses all parts of the original question.
                If the information is contradictory, highlight the discrepancy.
                If the context is insufficient, state what is missing."""),
                ("human", """Original Query: {query}
                
Sub-Query Answers:
{sub_answers}

Document Context:
{context}""")
            ])
            chain = prompt | self.llm | StrOutputParser()
            
            sub_answers_text = "\n".join([f"- {ans}" for ans in sub_answers])
            return await chain.ainvoke({
                "query": query,
                "sub_answers": sub_answers_text,
                "context": context
            })
        except Exception as e:
            logger.error(f"QueryDecomposerChain synthesis error: {e}")
            raise

    async def astream(self, query: str, sub_answers: List[str], context: str):
        """Stream synthesis from sub-queries and context."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert synthesizer. Combine the following sub-query answers and document context to provide a comprehensive final answer to the original query.
                Ensure the answer is cohesive, accurate, and directly addresses all parts of the original question.
                If the information is contradictory, highlight the discrepancy.
                If the context is insufficient, state what is missing."""),
                ("human", """Original Query: {query}
                
Sub-Query Answers:
{sub_answers}

Document Context:
{context}""")
            ])
            chain = prompt | self.llm | StrOutputParser()
            sub_answers_text = "\n".join([f"- {ans}" for ans in sub_answers])
            
            async for chunk in chain.astream({
                "query": query,
                "sub_answers": sub_answers_text,
                "context": context
            }):
                yield chunk
        except Exception as e:
            logger.error(f"QueryDecomposerChain streaming error: {e}")
            raise
def create_chains(llm: ChatNVIDIA, memory=None):
    """Factory function to create all required RAG chains."""
    return {
        "simple": SimpleRAGChain(llm=llm),
        "multi_turn": MultiTurnChain(llm=llm, memory=memory),
        "decomposer": QueryDecomposerChain(llm=llm)
    }
