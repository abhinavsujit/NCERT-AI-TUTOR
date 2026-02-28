import os
import fitz  # PyMuPDF

DATA_PATH = "../data/chapters"
OUTPUT_FILE = "../data/science_full.txt"

all_text = ""

pdf_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")])

for file in pdf_files:
    pdf_path = os.path.join(DATA_PATH, file)
    print(f"Reading: {file}")

    doc = fitz.open(pdf_path)

    all_text += f"\n\n===== START OF {file} =====\n\n"

    for page in doc:
        text = page.get_text()
        if text:
            all_text += text + "\n"

    all_text += f"\n\n===== END OF {file} =====\n\n"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(all_text)

print("\nText extraction completed.")
print(f"Saved to: {OUTPUT_FILE}")