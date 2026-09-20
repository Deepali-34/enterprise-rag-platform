from typing import List, Dict
from threading import Lock


class ConversationMemory:
    """
    Stores recent user questions and assistant answers
    for one conversation session.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_turn(self, question: str, answer: str):
        """
        Store one conversation turn.
        """

        self.history.append(
            {
                "question": question,
                "answer": answer,
            }
        )

        # Keep only the most recent conversations
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_history(self) -> List[Dict[str, str]]:
        """
        Return the conversation history.
        """

        return self.history.copy()

    def get_history_text(self) -> str:
        """
        Convert conversation history into text
        for use in the LLM prompt.
        """

        if not self.history:
            return ""

        history_parts = []

        for index, turn in enumerate(self.history, start=1):

            history_parts.append(
                f"""
--- CONVERSATION TURN {index} ---

User:
{turn["question"]}

Assistant:
{turn["answer"]}
"""
            )

        return "\n".join(history_parts).strip()

    def clear(self):
        """
        Clear all conversation history.
        """

        self.history.clear()


class SessionMemoryManager:
    """
    Manages separate ConversationMemory instances
    for different session IDs.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.sessions: Dict[str, ConversationMemory] = {}
        self.lock = Lock()

    def get_memory(self, session_id: str) -> ConversationMemory:
        """
        Get the memory associated with a session.

        Creates a new memory automatically when the
        session does not exist.
        """

        session_id = session_id.strip()

        if not session_id:
            session_id = "default"

        with self.lock:

            if session_id not in self.sessions:

                self.sessions[session_id] = ConversationMemory(
                    max_turns=self.max_turns
                )

            return self.sessions[session_id]

    def clear_session(self, session_id: str):
        """
        Clear one conversation session.
        """

        session_id = session_id.strip()

        with self.lock:

            self.sessions.pop(
                session_id,
                None
            )

    def clear_all(self):
        """
        Clear all conversation sessions.
        """

        with self.lock:
            self.sessions.clear()


# Global session memory manager
session_memory_manager = SessionMemoryManager(
    max_turns=5
)