import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL, CHUNK_STORE_PATH

client = OpenAI(api_key=OPENAI_API_KEY)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "eval_questions.json")
CHUNKS_PER_CHAPTER = 2


def generate_question_from_chunk(chunk: dict) -> dict | None:
    prompt = f"""You are creating evaluation questions for an NCERT Class 10 Science AI tutor.

Given this passage from the textbook, generate ONE good question that can be answered strictly from this passage.

Passage:
{chunk['text']}

Respond in this exact JSON format:
{{
  "question": "the question",
  "expected_chapter": "{chunk['chapter_name']}",
  "key_terms": ["term1", "term2", "term3", "term4"]
}}

Rules:
- The question must be answerable from the passage alone
- key_terms must be specific words or phrases that MUST appear in a correct answer
- Return only the JSON, nothing else"""

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  Skipping chunk (error: {e})")
        return None


def main():
    print("Loading chunks...")
    with open(CHUNK_STORE_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Group chunks by chapter
    by_chapter = {}
    for chunk in chunks:
        name = chunk.get("chapter_name", chunk["chapter"])
        by_chapter.setdefault(name, []).append(chunk)

    print(f"Found {len(by_chapter)} chapters\n")

    eval_questions = []

    for chapter_name, chapter_chunks in by_chapter.items():
        # Skip the answers chapter
        if "answer" in chapter_name.lower():
            continue

        sampled = random.sample(chapter_chunks, min(CHUNKS_PER_CHAPTER, len(chapter_chunks)))
        print(f"Generating questions for: {chapter_name}")

        for chunk in sampled:
            result = generate_question_from_chunk(chunk)
            if result:
                eval_questions.append(result)
                print(f"  ✓ {result['question'][:80]}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_questions, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(eval_questions)} questions saved to eval_questions.json")


if __name__ == "__main__":
    main()
