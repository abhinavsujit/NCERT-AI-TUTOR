import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ingestion.pdf_reader import extract_all_pdfs
from ingestion.chunker import chunk_pages
from ingestion.embedder import build_index
from config import CHAPTERS_DIR


def main():
    print("=== NCERT AI Tutor — Ingestion Pipeline ===\n")

    print("Step 1: Extracting text from PDFs...")
    pages = extract_all_pdfs(str(CHAPTERS_DIR))

    print("\nStep 2: Chunking text...")
    chunks = chunk_pages(pages)
    print(f"Total chunks created: {len(chunks)}")

    print("\nStep 3: Building search indices...")
    build_index(chunks)

    print("\nDone. You can now run main.py to start the tutor.")


if __name__ == "__main__":
    main()
