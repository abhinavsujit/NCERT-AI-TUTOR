import json
from config import CHUNK_STORE_PATH, TOP_K
from retrieval.semantic import semantic_search
from retrieval.bm25 import bm25_search
from retrieval.reranker import rerank

_chunks = None


def load_chunks():
    global _chunks
    if _chunks is None:
        with open(CHUNK_STORE_PATH, "r", encoding="utf-8") as f:
            _chunks = json.load(f)
    return _chunks


def reciprocal_rank_fusion(
    semantic_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    k: int = 60
) -> list[tuple[int, float]]:
    scores = {}

    for rank, (idx, _) in enumerate(semantic_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    for rank, (idx, _) in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query: str, top_k: int = TOP_K) -> list[dict]:
    semantic_results = semantic_search(query, k=top_k * 2)
    bm25_results     = bm25_search(query,    k=top_k * 2)

    combined = reciprocal_rank_fusion(semantic_results, bm25_results)

    chunks = load_chunks()
    candidates = []
    for idx, score in combined[:top_k * 2]:
        chunk = chunks[idx].copy()
        chunk["score"] = score
        candidates.append(chunk)

    # Rerank candidates and return the best ones
    reranked = rerank(query, candidates, top_k=top_k)

    return reranked
