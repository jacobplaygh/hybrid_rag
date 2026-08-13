import logging
from typing import List, Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class QueryUnderstanding:
    """
    Analyzes user queries to extract intent, entities, and structure.
    This enables the RAG system to choose the best retrieval strategy.
    """

    def __init__(self, llm):
        self.llm = llm
        
        # Intent Classification Prompt
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query analysis expert. Classify the user's query into one of the following intents:
            - FACTUAL: Seeking a specific piece of information, a definition, or a fact.
            - COMPARATIVE: Comparing two or more entities, features, or concepts.
            - NAVIGATIONAL: Looking for a specific section, document, or guide.
            - COMPLEX: A multi-part question that requires decomposition.
            
            Return ONLY the intent label (FACTUAL, COMPARATIVE, NAVIGATIONAL, or COMPLEX)."""),
            ("human", "{query}")
        ])
        self.intent_chain = self.intent_prompt | self.llm | StrOutputParser()

        # Entity Extraction Prompt
        self.entity_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract key entities (technical terms, product names, version numbers, specific components) from the query.
            Return them as a comma-separated list. If no entities are found, return 'None'.
            Example: 'How does the NVIDIA H100 compare to A100?' -> 'NVIDIA H100, A100'"""),
            ("human", "{query}")
        ])
        self.entity_chain = self.entity_prompt | self.llm | StrOutputParser()

        # Query Decomposition Prompt
        self.decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query decomposition expert. Break down a complex user query into a list of simpler, independent sub-queries that can be answered individually to fully address the original request.
            
            Return the sub-queries as a newline-separated list. Each line should be a standalone question.
            Example: 'How does the H100 compare to A100 in terms of memory and power?' -> 
            What is the memory specification of the H100?
            What is the memory specification of the A100?
            What is the power consumption of the H100?
            What is the power consumption of the A100?"""),
            ("human", "{query}")
        ])
        self.decomposition_chain = self.decomposition_prompt | self.llm | StrOutputParser()

    async def analyze(self, query: str) -> Dict[str, Any]:
        """Perform full query analysis."""
        try:
            intent = await self.intent_chain.ainvoke({"query": query})
            entities = await self.entity_chain.ainvoke({"query": query})
            
            sub_queries = []
            if intent.strip().upper() == "COMPLEX":
                decomposition_result = await self.decomposition_chain.ainvoke({"query": query})
                sub_queries = [q.strip() for q in decomposition_result.split("\n") if q.strip()]

            return {
                "intent": intent.strip().upper(),
                "entities": [e.strip() for e in entities.split(",")] if entities != "None" else [],
                "original_query": query,
                "sub_queries": sub_queries
            }
        except Exception as e:
            logger.error(f"Query understanding analysis failed: {e}")
            return {
                "intent": "FACTUAL", 
                "entities": [], 
                "original_query": query
            }
