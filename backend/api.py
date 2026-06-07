import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

from config import OPENAI_API_KEY, CHAT_MODEL
from retrieval.hybrid import hybrid_search
from pipeline.intent import classify_intent
from pipeline.memory import ConversationMemory
from pipeline.cache import SemanticCache
from llm.answerer import generate_answer, stream_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)
memory = ConversationMemory()
cache  = SemanticCache()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    intent: str
    rewritten_query: str | None
    cached: bool


def rewrite_query(query: str) -> str:
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
    return response.choices[0].message.content.strip()


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    query = request.query.strip()

    # Step 1: cache check
    cached_answer = cache.get(query)
    if cached_answer:
        memory.add("user", query)
        memory.add("assistant", cached_answer)
        return QueryResponse(
            answer=cached_answer,
            intent="cached",
            rewritten_query=None,
            cached=True
        )

    # Step 2: intent
    intent = classify_intent(query)

    # Step 3: rewrite query
    search_query = rewrite_query(query)
    rewritten = search_query if search_query != query else None

    # Step 4: retrieve
    chunks = hybrid_search(search_query)

    # Step 5: generate
    answer = generate_answer(query, intent, chunks, memory)

    # Step 6: update memory and cache
    memory.add("user", query)
    memory.add("assistant", answer)
    cache.set(query, answer)

    return QueryResponse(
        answer=answer,
        intent=intent,
        rewritten_query=rewritten,
        cached=False
    )



@app.post("/chat/stream")
def chat_stream(request: QueryRequest):
    query = request.query.strip()

    cached_answer = cache.get(query)
    if cached_answer:
        memory.add("user", query)
        memory.add("assistant", cached_answer)

        def cached_gen():
            yield f"data: {json.dumps({'token': cached_answer})}\n\n"
            yield f"data: {json.dumps({'done': True, 'intent': 'cached', 'cached': True})}\n\n"

        return StreamingResponse(cached_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    intent = classify_intent(query)
    search_query = rewrite_query(query)
    chunks = hybrid_search(search_query)
    def generate():
        full_answer = ""
        try:
            for token in stream_answer(query, intent, chunks, memory):
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        memory.add("user", query)
        memory.add("assistant", full_answer)
        cache.set(query, full_answer)
        yield f"data: {json.dumps({'done': True, 'intent': intent, 'cached': False})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})




@app.post("/clear")
def clear():
    memory.clear()
    return {"status": "cleared"}
