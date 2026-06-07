import numpy as np
from openai import OpenAI
from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_MAX_SIZE,
)

client = OpenAI(api_key=OPENAI_API_KEY)


class SemanticCache:
    def __init__(self):
        self.embeddings: list[np.ndarray] = []
        self.answers:    list[str]        = []
        self.questions:  list[str]        = []

    def _get_embedding(self, text: str) -> np.ndarray:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return np.array(response.data[0].embedding, dtype="float32")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, query: str) -> str | None:
        if not self.embeddings:
            return None

        query_embedding = self._get_embedding(query)

        for i, cached_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity >= CACHE_SIMILARITY_THRESHOLD:
                print(f"[Cache hit] Similar to: '{self.questions[i]}' (similarity: {similarity:.3f})")
                return self.answers[i]

        return None

    def set(self, query: str, answer: str):
        if len(self.embeddings) >= CACHE_MAX_SIZE:
            self.embeddings.pop(0)
            self.answers.pop(0)
            self.questions.pop(0)

        self.embeddings.append(self._get_embedding(query))
        self.answers.append(answer)
        self.questions.append(query)
