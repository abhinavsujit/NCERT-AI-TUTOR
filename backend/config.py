import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent        # ai-tutor/
DATA_DIR       = BASE_DIR / "data"
CHAPTERS_DIR   = DATA_DIR / "chapters"
INDEX_DIR      = DATA_DIR / "index"

FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
BM25_INDEX_PATH  = INDEX_DIR / "bm25_index.pkl"
CHUNK_STORE_PATH = INDEX_DIR / "chunk_store.json"

CHAPTER_MAP_PATH = DATA_DIR / "chapter_map.json"

# ── OpenAI ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000   # max characters per chunk
CHUNK_OVERLAP = 200    # characters of overlap between consecutive chunks

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = 5              # number of chunks to retrieve

# ── Intent classifier ──────────────────────────────────────────────────────
INTENT_CONFIDENCE_THRESHOLD = 2   # min rule matches before trusting rule-based

# ── Conversation memory ────────────────────────────────────────────────────
MEMORY_WINDOW = 5      # last N user/assistant turn pairs to keep

# ── Semantic cache ─────────────────────────────────────────────────────────
CACHE_SIMILARITY_THRESHOLD = 0.92  # cosine similarity to count as a cache hit
CACHE_MAX_SIZE             = 500   # max entries before evicting oldest
