import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from retrieval.hybrid import hybrid_search
from pipeline.intent import classify_intent
from pipeline.memory import ConversationMemory
from llm.answerer import generate_answer

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_questions.json")


def check_retrieval(chunks: list[dict], expected_chapter: str) -> bool:
    for chunk in chunks:
        if expected_chapter.lower() in chunk.get("chapter_name", "").lower():
            return True
    return False


def check_answer(answer: str, key_terms: list[str]) -> tuple[int, int]:
    answer_lower = answer.lower()
    matched = sum(1 for term in key_terms if term.lower() in answer_lower)
    return matched, len(key_terms)


def main():
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Running evaluation on {len(questions)} questions...\n")
    print("=" * 70)

    memory = ConversationMemory()

    retrieval_passes = 0
    answer_scores = []
    failed_retrieval = []
    failed_answers = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        expected_chapter = q["expected_chapter"]
        key_terms = q["key_terms"]

        print(f"[{i}/{len(questions)}] {question[:65]}...")

        chunks = hybrid_search(question)
        intent = classify_intent(question)
        answer = generate_answer(question, intent, chunks, memory)

        # Check retrieval
        retrieval_ok = check_retrieval(chunks, expected_chapter)
        if retrieval_ok:
            retrieval_passes += 1
        else:
            failed_retrieval.append(question)

        # Check answer key terms
        matched, total = check_answer(answer, key_terms)
        score = matched / total if total > 0 else 0
        answer_scores.append(score)

        if score < 0.5:
            failed_answers.append({
                "question": question,
                "expected_terms": key_terms,
                "matched": matched,
                "total": total,
            })

        status = "✓" if retrieval_ok and score >= 0.5 else "✗"
        print(f"  {status} Retrieval: {'✓' if retrieval_ok else '✗'}  |  Key terms: {matched}/{total}  |  Intent: {intent}\n")

    # Final report
    total_q = len(questions)
    avg_answer_score = sum(answer_scores) / len(answer_scores) if answer_scores else 0

    print("=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"Total questions:      {total_q}")
    print(f"Retrieval accuracy:   {retrieval_passes}/{total_q} ({100 * retrieval_passes / total_q:.1f}%)")
    print(f"Avg answer score:     {avg_answer_score:.2f} ({avg_answer_score * 100:.1f}%)")

    if failed_retrieval:
        print(f"\nFailed retrieval ({len(failed_retrieval)}):")
        for q in failed_retrieval:
            print(f"  - {q[:70]}")

    if failed_answers:
        print(f"\nWeak answers ({len(failed_answers)}):")
        for fa in failed_answers:
            print(f"  - {fa['question'][:65]}...")
            print(f"    Matched {fa['matched']}/{fa['total']} key terms")

    print("\nDone.")


if __name__ == "__main__":
    main()
