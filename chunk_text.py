from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = "../data/science_full.txt"
OUTPUT_FILE = "../data/chunks.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

print("Total characters:", len(full_text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(full_text)

print("Total chunks created:", len(chunks))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"\n\n===== CHUNK {i} =====\n")
        f.write(chunk)

print("Chunks saved to:", OUTPUT_FILE)