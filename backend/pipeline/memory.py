from config import MEMORY_WINDOW


class ConversationMemory:
    def __init__(self):
        self.history: list[dict] = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

        # Keep only the last N turns (each turn = user + assistant message)
        max_messages = MEMORY_WINDOW * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_history(self) -> list[dict]:
        return self.history.copy()

    def clear(self):
        self.history = []

    def is_empty(self) -> bool:
        return len(self.history) == 0
