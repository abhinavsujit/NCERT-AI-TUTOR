import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL
from retrieval.hybrid import hybrid_search
from pipeline.intent import classify_intent
from pipeline.memory import ConversationMemory
from pipeline.cache import SemanticCache
from llm.answerer import generate_answer

client = OpenAI(api_key=OPENAI_API_KEY)


def rewrite_query(query: str, memory: ConversationMemory) -> str:
    if memory.is_empty():
        return query

    messages = [
        {"role": "system", "content": "Rewrite the user's question so it makes complete sense without any prior conversation. Keep it short. If it already makes sense on its own, return it unchanged. Reply with only the rewritten question, nothing else."},
    ]
    messages.extend(memory.get_history())
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=100,
    )

    rewritten = response.choices[0].message.content.strip()
    return rewritten


def main():
    memory = ConversationMemory()
    cache  = SemanticCache()

    print("NCERT AI Tutor Ready.")
    print("Commands: 'clear' to reset conversation, 'exit' to quit.\n")

    while True:
        try:
            query = input("Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if query.lower() == "clear":
            memory.clear()
            print("Conversation cleared.\n")
            continue

        # Step 1: cache check
        cached = cache.get(query)
        if cached:
            print(f"\nAnswer:\n{cached}\n")
            print("-" * 50)
            memory.add("user", query)
            memory.add("assistant", cached)
            continue

        # Step 2: intent
        intent = classify_intent(query)
        print(f"[Intent: {intent}]")

        # Step 3: rewrite query for better retrieval
        search_query = rewrite_query(query, memory)
        if search_query != query:
            print(f"[Rewritten: {search_query}]")

        # Step 4: retrieve
        chunks = hybrid_search(search_query)

        # Step 5: generate
        answer = generate_answer(query, intent, chunks, memory)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 50)

        # Step 6: update memory and cache
        memory.add("user", query)
        memory.add("assistant", answer)
        cache.set(query, answer)


if __name__ == "__main__":
    main()
