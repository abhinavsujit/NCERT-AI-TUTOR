import json
import pickle

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    BM25_INDEX_PATH,
    CHUNK_STORE_PATH,
    INDEX_DIR,
)

client = OpenAI(api_key=OPENAI_API_KEY)


def _get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def build_index(chunks: list[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building index for {len(chunks)} chunks...")

    # Step 1: embeddings
    embeddings: list[list[float]] = []
    for i, chunk in enumerate(chunks):
        print(f"  Embedding {i + 1}/{len(chunks)}", end="\r")
        embeddings.append(_get_embedding(chunk["text"]))
    print(f"\n  Embeddings created: {len(embeddings)}")

    # Step 2: FAISS index
    embeddings_np = np.array(embeddings, dtype="float32")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"  FAISS index saved → {FAISS_INDEX_PATH}")

    # Step 3: BM25 index
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  BM25  index saved → {BM25_INDEX_PATH}")

    # Step 4: chunk store
    with open(CHUNK_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  Chunk store saved → {CHUNK_STORE_PATH}")

    print("Index building complete.")
