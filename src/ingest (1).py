import os
import json
import fitz
from docx import Document

RAW_DIR = "data/raw"
OUTPUT_FILE = "data/processed/documents.json"

documents = []

def extract_pdf(path):
    doc = fitz.open(path)
    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages

def extract_docx(path):
    doc = Document(path)
    paragraphs = []

    for number, paragraph in enumerate(doc.paragraphs, start=1):
        text = paragraph.text.strip()

        if text:
            paragraphs.append({
                "paragraph": number,
                "text": text
            })

    return paragraphs

for company in os.listdir(RAW_DIR):

    company_path = os.path.join(RAW_DIR, company)

    if not os.path.isdir(company_path):
        continue

    print(f"\nProcessing: {company.upper()}")

    for filename in os.listdir(company_path):

        path = os.path.join(company_path, filename)

        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(filename)[1].lower()

        try:

            if extension == ".pdf":

                pages = extract_pdf(path)

                full_text = "\n".join(
                    item["text"] for item in pages
                )

                documents.append({
                    "company": company,
                    "filename": filename,
                    "file_type": "PDF",
                    "units": len(pages),
                    "text": full_text
                })

                print(
                    f"  OK: {filename} | "
                    f"Pages: {len(pages)} | "
                    f"Characters: {len(full_text):,}"
                )

            elif extension == ".docx":

                paragraphs = extract_docx(path)

                full_text = "\n".join(
                    item["text"] for item in paragraphs
                )

                documents.append({
                    "company": company,
                    "filename": filename,
                    "file_type": "DOCX",
                    "units": len(paragraphs),
                    "text": full_text
                })

                print(
                    f"  OK: {filename} | "
                    f"Paragraphs: {len(paragraphs)} | "
                    f"Characters: {len(full_text):,}"
                )

            else:
                print(f"  SKIPPED: {filename}")

        except Exception as e:
            print(f"  ERROR: {filename}")
            print(f"  {e}")

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(documents, file, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("KNOWX INGESTION COMPLETE")
print("=" * 60)
print(f"Documents processed: {len(documents)}")
print(f"Output file: {OUTPUT_FILE}")

total_characters = sum(
    len(document["text"])
    for document in documents
)

print(f"Total characters extracted: {total_characters:,}")
