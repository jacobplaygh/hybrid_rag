"""Conversation memory management."""

import logging
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
