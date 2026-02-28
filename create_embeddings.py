import os
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Load chunks
chunks = []
with open("../data/chunks.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Split chunks based on marker
raw_chunks = content.split("===== CHUNK")

for chunk in raw_chunks:
    text = chunk.strip()
    if len(text) > 50:
        chunks.append(text)

print("Total chunks loaded:", len(chunks))

# Generate embeddings
embeddings = []

for i, chunk in enumerate(chunks):
    print(f"Embedding chunk {i+1}/{len(chunks)}")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    vector = response.data[0].embedding
    embeddings.append(vector)

# Convert to numpy
embeddings_np = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# Save index
faiss.write_index(index, "../data/faiss_index.bin")

# Save chunk texts
with open("../data/chunk_store.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n---\n")

print("Embeddings + FAISS index saved.")