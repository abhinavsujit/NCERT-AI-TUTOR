import pymupdf
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    chapter = Path(pdf_path).name
    pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        if text:
            pages.append({
                "text": text,
                "page": page_num + 1,
                "chapter": chapter,
            })

    return pages


def extract_all_pdfs(chapters_dir: str) -> list[dict]:
    all_pages = []
    pdf_files = sorted(Path(chapters_dir).glob("*.pdf"))

    for pdf_path in pdf_files:
        print(f"  Reading: {pdf_path.name}")
        pages = extract_text_from_pdf(str(pdf_path))
        all_pages.extend(pages)

    print(f"Total pages extracted: {len(all_pages)}")
    return all_pages
