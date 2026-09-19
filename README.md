# NCERT AI Tutor

A retrieval-augmented Q&A system grounded strictly in the NCERT Class 10 Science textbook. Ask a question in natural language and get a cited answer drawn only from the textbook — the system refuses anything it can't support from the source.

<!-- Add a screenshot or GIF of the chat interface here -->

## Why this exists

Most RAG demos stop at "embed the documents, search the top 5, ask the model." That works until it doesn't — follow-up questions retrieve nothing useful, lexical matches get missed by semantic search, and there's no way to tell whether an answer is right beyond reading it.

This project addresses those specifically, and measures whether the fixes worked.

## Architecture

```
PDF textbooks
     ↓  ingestion (offline)
sentence-aware chunking → embeddings → FAISS index + BM25 index
     ↓
query → rewriting → intent classification
     ↓
  ┌──────────────┬──────────────┐
  │ dense (FAISS)│ sparse (BM25)│     parallel retrieval
  └──────────────┴──────────────┘
     ↓  reciprocal rank fusion
     ↓  cross-encoder rerank
top 5 passages → intent-conditioned prompt → streamed answer + citations
```

### Retrieval

Dense and sparse search run in parallel, each returning double the target candidates. The two ranked lists are merged with **reciprocal rank fusion** (k=60), which is score-agnostic — no need to normalise L2 distances against BM25 scores, which aren't comparable. The fused set is then reranked with a cross-encoder for precision on the final five passages.

### Chunking

1000-character chunks with 200-character overlap, where the overlap is built from whole trailing sentences rather than raw character slices. Chunk boundaries never cut mid-sentence. Page numbers are preserved through ingestion so every answer can cite chapter and page.

### Query understanding

- **Intent classification** — a regex rule engine covering five intents (define, explain, practice, summarize, factual). Rules are trusted only when exactly one intent matches; ambiguous queries escalate to an LLM classifier. The common case costs nothing.
- **Conversational rewriting** — follow-ups like "why does that happen?" are rewritten into standalone search queries using chat history, before retrieval runs.
- **Sliding-window memory** — the last five turn pairs, bounding token cost per request.

### Cost and latency

- **Semantic cache** keyed on query embedding rather than exact string. A new query is served from cache at cosine similarity ≥ 0.92, with FIFO eviction at 500 entries.
- **Lazy loading** — indices and models load into module-level singletons on first request, not per query.

### Anti-hallucination

Every prompt enforces two guardrails: mandatory chapter-and-page citation, and a hard scope constraint returning a fixed refusal when the answer isn't in the retrieved context. The model is explicitly instructed not to use outside knowledge.

## Evaluation

The part most RAG projects skip.

`generate_eval.py` samples chunks per chapter and prompts the model to write a question answerable strictly from that passage, along with key terms any correct answer must contain. This produced a 23-question benchmark across all 15 chapters.

`run_eval.py` scores two independent axes:

| Metric | What it measures |
|---|---|
| Retrieval accuracy | Did the correct chapter appear in the retrieved set? |
| Answer quality | What fraction of expected key terms appear in the answer? |

The report enumerates specific failed retrievals and weak answers, so debugging is targeted rather than guesswork.

<!-- Add your actual scores here once you've run it -->

## Stack

**Backend** Python, FastAPI, Uvicorn, OpenAI API (GPT-4o-mini, text-embedding-3-small), FAISS, rank-bm25, sentence-transformers (CrossEncoder), PyMuPDF, NLTK, NumPy

**Frontend** React 18, Vite, react-markdown, Server-Sent Events

## Running it

```bash
git clone <repo-url>
cd ncert-ai-tutor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

echo "OPENAI_API_KEY=sk-..." > .env

# Build the indices (one-off, needs the textbook PDFs in place)
python run_ingestion.py

# Start the API
uvicorn backend.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

There's also a CLI REPL running the identical pipeline, for iterating without the frontend:

```bash
python main.py
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/chat` | POST | Blocking JSON response |
| `/chat/stream` | POST | Token-by-token Server-Sent Events |
| `/clear` | POST | Reset conversation state |

## Project layout

```
backend/
  ingestion/      PDF extraction, chunking, index building
  retrieval/      hybrid search, fusion, reranking
  pipeline/       intent classification, query rewriting, caching
  llm/            intent-conditioned prompting and generation
frontend/         React chat interface
run_ingestion.py  offline index builder
generate_eval.py  golden test set generator
run_eval.py       evaluation harness
main.py           CLI REPL
```

## Notes

Ingestion is offline and only needs re-running when the source PDFs change. The FAISS index, BM25 pickle and chunk store are build artifacts, not source — they're gitignored.
