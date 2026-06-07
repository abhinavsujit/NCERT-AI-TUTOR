import re
from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL, INTENT_CONFIDENCE_THRESHOLD

client = OpenAI(api_key=OPENAI_API_KEY)

INTENT_PATTERNS = {
    "define": [
        r"\bwhat is\b", r"\bwhat are\b", r"\bdefine\b",
        r"\bdefinition\b", r"\bmeaning of\b",
    ],
    "explain": [
        r"\bexplain\b", r"\bhow does\b", r"\bhow do\b",
        r"\bwhy does\b", r"\bwhy do\b", r"\bdescribe\b",
        r"\bwhat happens\b",
    ],
    "practice": [
        r"\bpractice\b", r"\bquestions?\b", r"\bquiz\b",
        r"\bmcq\b", r"\btest me\b", r"\bexercise\b",
        r"\bgive me\b", r"\bproblem\b",
    ],
    "summarize": [
        r"\bsummar\b", r"\boverview\b", r"\bkey points\b",
        r"\bmain points\b", r"\bbrief\b", r"\bnotes\b",
    ],
    "factual": [
        r"\bformula\b", r"\bwho\b", r"\bwhen\b", r"\bwhere\b",
        r"\bhow many\b", r"\blist\b", r"\bname\b", r"\bwhich\b",
    ],
}


def classify_rule_based(query: str) -> tuple[str, int]:
    query_lower = query.lower()
    scores = {}

    for intent, patterns in INTENT_PATTERNS.items():
        count = sum(1 for p in patterns if re.search(p, query_lower))
        if count > 0:
            scores[intent] = count

    if not scores:
        return "factual", 0

    best_intent = max(scores, key=scores.get)
    return best_intent, scores[best_intent]


def classify_with_llm(query: str) -> str:
    prompt = f"""Classify this student question into exactly one category:
- define: asking for a definition or meaning
- explain: asking for explanation or how/why something works
- practice: asking for questions, quiz, or MCQs
- summarize: asking for a summary or key points
- factual: asking for a specific fact, formula, name, or list

Question: "{query}"

Reply with only the category name."""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )

    intent = response.choices[0].message.content.strip().lower()
    return intent if intent in INTENT_PATTERNS else "factual"


def classify_intent(query: str) -> str:
    query_lower = query.lower()
    scores = {}

    for intent, patterns in INTENT_PATTERNS.items():
        count = sum(1 for p in patterns if re.search(p, query_lower))
        if count > 0:
            scores[intent] = count

    # Only trust rules if exactly one intent matched
    if len(scores) == 1:
        return list(scores.keys())[0]

    # Multiple intents matched or nothing matched → LLM decides
    return classify_with_llm(query)

