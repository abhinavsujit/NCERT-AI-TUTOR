import numpy as np
import faiss
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL, FAISS_INDEX_PATH, TOP_K

client = OpenAI(api_key=OPENAI_API_KEY)

_index = None


def load_index():
    global _index
    if _index is None:
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
    return _index


def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding, dtype="float32")


def semantic_search(query: str, k: int = TOP_K) -> list[tuple[int, float]]:
    index = load_index()
    query_vector = get_embedding(query)

    distances, indices = index.search(np.array([query_vector]), k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            results.append((int(idx), float(dist)))

    return results
