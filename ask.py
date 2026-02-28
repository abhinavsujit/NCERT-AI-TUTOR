import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# -------------------------
# Load FAISS index
# -------------------------
index = faiss.read_index("../data/faiss_index.bin")

# -------------------------
# Load chunk texts
# -------------------------
with open("../data/chunk_store.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n---\n")

print("NCERT AI Tutor Ready.")
print("Type your question.\n")

while True:
    question = input("Ask: ")

    if question.lower() in ["exit", "quit"]:
        break

    # -------------------------
    # Embed the question
    # -------------------------
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    question_vector = np.array(response.data[0].embedding).astype("float32")

    # -------------------------
    # Search FAISS
    # -------------------------
    D, I = index.search(np.array([question_vector]), k=3)

    retrieved_chunks = [chunks[i] for i in I[0]]

    context = "\n\n".join(retrieved_chunks)

    # -------------------------
    # Ask LLM using context
    # -------------------------
    prompt = f"""
You are a helpful Class 10 science tutor.

Use ONLY the context from the NCERT book below.

Rules:
- If the user asks a factual question → answer from context only.
- If the user asks for important questions, summaries, or explanations → generate them using the context.
- Do NOT use outside knowledge.
- If topic truly not in context → say you don't know.

Context:
{context}

Question:
{question}
"""

    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = chat.choices[0].message.content
    print("\nAnswer:\n", answer)
    print("\n---------------------------\n")