from sentence_transformers import CrossEncoder

_model = None


def load_model():
    global _model
    if _model is None:
        print("Loading reranker model (first time only)...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    model = load_model()

    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = model.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

    return ranked[:top_k]
