import pickle
from config import BM25_INDEX_PATH, TOP_K

_bm25 = None


def load_bm25():
    global _bm25
    if _bm25 is None:
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25 = pickle.load(f)
    return _bm25


def bm25_search(query: str, k: int = TOP_K) -> list[tuple[int, float]]:
    bm25 = load_bm25()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [(idx, float(scores[idx])) for idx in top_indices]
