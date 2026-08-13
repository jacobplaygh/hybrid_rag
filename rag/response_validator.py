import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    score: float
    issues: List[str]
    sanitized_response: Optional[str] = None

class ResponseValidator:
    """
    Handles hallucination detection and response sanitization.
    """
    def __init__(self, llm=None):
        self.llm = llm

    async def validate(self, query: str, response: str, context_docs: List[Any]) -> ValidationResult:
        """
        Validates the response against the provided context to detect hallucinations.
        """
        if not context_docs:
            return ValidationResult(is_valid=True, score=1.0, issues=[])

        # In a production system, this would use a specialized NLI model or a 
        # "judge" LLM to verify each claim in the response against the context.
        # For this implementation, we use a "Judge LLM" approach.
        
        if not self.llm:
            logger.warning("No LLM provided for ResponseValidator; skipping deep validation.")
            return ValidationResult(is_valid=True, score=1.0, issues=[])

        try:
            # Construct a validation prompt
            context_text = "\n\n".join([
                doc.content if hasattr(doc, "content") else doc.get("content", "") 
                for doc in context_docs
            ])
            
            validation_prompt = (
                f"You are a quality assurance judge. Your task is to verify if the following response "
                f"is fully supported by the provided context. Detect any hallucinations (claims not in context).\n\n"
                f"Context:\n{context_text}\n\n"
                f"Query: {query}\n"
                f"Response: {response}\n\n"
                f"Respond in the following format:\n"
                f"SCORE: [0.0 to 1.0]\n"
                f"ISSUES: [List of hallucinations or 'None']\n"
                f"SANITIZED: [The response with hallucinations removed, or 'Same']"
            )

            # Call the judge LLM
            res = await self.llm.ainvoke(validation_prompt)
            content = res.content if hasattr(res, "content") else str(res)
            
            # Parse the response
            score = 1.0
            issues = []
            sanitized = response

            # Use regex to find sections regardless of exact line start
            score_match = re.search(r"SCORE:\s*([\d.]+)", content, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except ValueError:
                    pass

            issues_match = re.search(r"ISSUES:\s*(.*?)(?=\n(?:SCORE|SANITIZED):|$)", content, re.IGNORECASE | re.DOTALL)
            if issues_match:
                issue_text = issues_match.group(1).strip()
                if issue_text and issue_text.lower() != "none":
                    # Split by bullets or newlines if multiple issues are listed
                    issues = [i.strip("- ").strip() for i in re.split(r"\n|•", issue_text) if i.strip()]

            sanitized_match = re.search(r"SANITIZED:\s*(.*?)(?=\n(?:SCORE|ISSUES):|$)", content, re.IGNORECASE | re.DOTALL)
            if sanitized_match:
                sanitized_text = sanitized_match.group(1).strip()
                if sanitized_text and sanitized_text.lower() != "same":
                    sanitized = sanitized_text

            return ValidationResult(
                is_valid=score >= 0.7,
                score=score,
                issues=issues,
                sanitized_response=sanitized if sanitized != response else None
            )

        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return ValidationResult(is_valid=True, score=1.0, issues=[f"Validation error: {str(e)}"])

    def sanitize(self, text: str) -> str:
        """Basic sanitization to remove LLM artifacts or sensitive patterns."""
        # Example: Remove common LLM prefixes
        prefixes = ["As an AI language model,", "Based on the provided context,", "According to the documents,"]
        sanitized = text
        for prefix in prefixes:
            if sanitized.startswith(prefix):
                sanitized = sanitized[len(prefix):].strip()
        
        return sanitized.strip()
