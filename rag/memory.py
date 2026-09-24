"""Conversation memory management."""

import logging
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class Message:
    """Represents a single message in conversation."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


class ConversationMemory:
    """Manages multi-turn conversation history."""
    
    def __init__(self, max_messages: int = 20, max_context_tokens: int = 2000):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum messages to keep in memory
            max_context_tokens: Maximum tokens to use for context
        """
        self.sessions: Dict[str, List[Message]] = defaultdict(list)
        self.max_messages = max_messages
        self.max_context_tokens = max_context_tokens
        logger.info("💾 Initialized ConversationMemory")
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to session."""
        message = Message(role, content)
        self.sessions[session_id].append(message)
        
        # Trim if exceeds max messages
        if len(self.sessions[session_id]) > self.max_messages:
            self.sessions[session_id] = self.sessions[session_id][-self.max_messages:]
    
    def get_context(self, session_id: str) -> str:
        """Get formatted context for LLM."""
        messages = self.sessions.get(session_id, [])
        context = "\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in messages[-10:]  # Last 10 messages
        ])
        return context

    def get_relevant_context(
        self,
        session_id: str,
        query: Optional[str] = None,
        max_messages: int = 10,
    ) -> str:
        """Return recent turns plus older turns relevant to the current query."""
        messages = self.sessions.get(session_id, [])
        if not messages:
            return ""

        candidates = messages[-max_messages:]
        if not query:
            return self._format_messages(candidates)

        query_terms = self._terms(query)
        recent_count = min(2, len(candidates))
        selected_indexes = set(range(len(candidates) - recent_count, len(candidates)))

        scored_indexes = []
        for index, message in enumerate(candidates):
            if index in selected_indexes:
                continue
            overlap = len(query_terms & self._terms(message.content))
            if overlap:
                scored_indexes.append((overlap, index))

        for _, index in sorted(scored_indexes, key=lambda item: (-item[0], -item[1])):
            selected_indexes.add(index)

        selected = [candidates[index] for index in sorted(selected_indexes)]
        return self._fit_message_budget(selected)

    def _fit_message_budget(self, messages: List[Message]) -> str:
        """Keep recent turns verbatim and compress older turns into a summary."""
        if not messages:
            return ""

        recent = messages[-2:]
        older = messages[:-2]
        recent_lines = [f"{message.role.upper()}: {message.content}" for message in recent]
        recent_words = sum(len(line.split()) for line in recent_lines)
        remaining = self.max_context_tokens - recent_words

        full_lines = [
            f"{message.role.upper()}: {message.content}" for message in messages
        ]
        if sum(len(line.split()) for line in full_lines) <= self.max_context_tokens:
            return "\n".join(full_lines)

        if remaining <= 0:
            newest = recent_lines[-1].split()[: self.max_context_tokens]
            return " ".join(newest)

        lines = []
        if older:
            summary = "Earlier conversation summary: " + " ".join(
                f"{message.role}: {message.content}" for message in older
            )
            summary_words = summary.split()
            if len(summary_words) > remaining:
                summary_words = summary_words[:remaining]
            if summary_words:
                lines.append(" ".join(summary_words))

        lines.extend(recent_lines)
        return "\n".join(lines)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _format_messages(messages: List[Message]) -> str:
        return "\n".join(f"{msg.role.upper()}: {msg.content}" for msg in messages)
    
    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages in session."""
        return [msg.to_dict() for msg in self.sessions.get(session_id, [])]
    
    def clear_session(self, session_id: str):
        """Clear session history."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.debug(f"Cleared session {session_id}")
    
    def clear_all(self):
        """Clear all sessions."""
        self.sessions.clear()
