import json
import nltk
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHAPTER_MAP_PATH

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# Load chapter metadata once
_chapter_map = {}
if CHAPTER_MAP_PATH.exists():
    with open(CHAPTER_MAP_PATH, "r", encoding="utf-8") as f:
        _chapter_map = json.load(f)


def chunk_text(text: str, metadata: dict) -> list[dict]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_sentences: list[str] = []
    current_size = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_size + sentence_len > CHUNK_SIZE and current_sentences:
            chunks.append({"text": " ".join(current_sentences), **metadata})

            overlap_sentences: list[str] = []
            overlap_size = 0
            for s in reversed(current_sentences):
                if overlap_size + len(s) <= CHUNK_OVERLAP:
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s)
                else:
                    break

            current_sentences = overlap_sentences
            current_size = overlap_size

        current_sentences.append(sentence)
        current_size += sentence_len

    if current_sentences:
        chunks.append({"text": " ".join(current_sentences), **metadata})

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    all_chunks = []

    for page in pages:
        chapter_file = page["chapter"]
        chapter_info = _chapter_map.get(chapter_file, {})

        metadata = {
            "chapter":      chapter_file,
            "page":         page["page"],
            "chapter_name": chapter_info.get("name", chapter_file),
            "subject":      chapter_info.get("subject", "Unknown"),
            "class":        chapter_info.get("class", "Unknown"),
        }
        all_chunks.extend(chunk_text(page["text"], metadata))

    return all_chunks
