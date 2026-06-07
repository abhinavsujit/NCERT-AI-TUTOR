from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL
from pipeline.memory import ConversationMemory

client = OpenAI(api_key=OPENAI_API_KEY)

CITE_INSTRUCTION = "Always cite the source chapter and page number at the end of your answer (e.g. Source: Chapter Name, Page X)."
SCOPE_INSTRUCTION = "If the answer is NOT in the provided context, respond ONLY with: 'I could not find this in the NCERT textbook. Please ask a question from your Class 10 Science syllabus.' Do NOT use any outside knowledge under any circumstances."

SYSTEM_PROMPTS = {
    "define": f"""You are a Class 10 NCERT science tutor.
The student wants a definition. Give a clear, concise definition in 2-3 sentences.
{CITE_INSTRUCTION}
End with 2 follow-up questions the student might want to ask next, based only on the context provided.
Use ONLY the provided NCERT context. {SCOPE_INSTRUCTION}""",

    "explain": f"""You are a Class 10 NCERT science tutor.
The student wants an explanation. Break it down step by step in simple language.
Use examples from the context if available.
{CITE_INSTRUCTION}
End with 2 follow-up questions the student might want to explore, based only on the context provided.
Use ONLY the provided NCERT context. {SCOPE_INSTRUCTION}""",

    "practice": f"""You are a Class 10 NCERT science tutor.
Generate 5 MCQs based strictly on the context provided.
Format each as:
Q. [question]
A) [option]  B) [option]  C) [option]  D) [option]
Answer: [correct option]

{CITE_INSTRUCTION}
Use ONLY the provided NCERT context. {SCOPE_INSTRUCTION}""",

    "summarize": f"""You are a Class 10 NCERT science tutor.
Provide a structured summary with:
- Main concepts
- Key points (bullet list)
- Important terms to remember
{CITE_INSTRUCTION}
End with 2 topics from the context the student should explore next.
Use ONLY the provided NCERT context. {SCOPE_INSTRUCTION}""",

    "factual": f"""You are a Class 10 NCERT science tutor.
Answer the question directly and concisely.
If the exact answer is in the context, state it clearly.
{CITE_INSTRUCTION}
End with 2 related follow-up questions based only on the context provided.
Use ONLY the provided NCERT context. {SCOPE_INSTRUCTION}""",
}


def format_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        chapter_name = chunk.get("chapter_name", chunk["chapter"])
        parts.append(f"[{chapter_name} — Page {chunk['page']}]\n{chunk['text']}")
    return "\n\n".join(parts)


def generate_answer(
    query: str,
    intent: str,
    chunks: list[dict],
    memory: ConversationMemory,
) -> str:
    context = format_context(chunks)
    system_prompt = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["factual"])

    messages = [{"role": "system", "content": system_prompt}]

    if not memory.is_empty():
        messages.extend(memory.get_history())

    messages.append({
        "role": "user",
        "content": f"Context from NCERT textbook:\n{context}\n\nQuestion: {query}"
    })

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content


def stream_answer(
    query: str,
    intent: str,
    chunks: list[dict],
    memory: ConversationMemory,
):
    context = format_context(chunks)
    system_prompt = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["factual"])

    messages = [{"role": "system", "content": system_prompt}]

    if not memory.is_empty():
        messages.extend(memory.get_history())

    messages.append({
        "role": "user",
        "content": f"Context from NCERT textbook:\n{context}\n\nQuestion: {query}"
    })

    with client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    ) as stream:
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token is not None:
                yield token
