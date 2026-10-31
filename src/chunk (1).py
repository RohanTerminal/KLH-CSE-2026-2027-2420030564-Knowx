import json
import os

INPUT_FILE = "data/processed/documents.json"
OUTPUT_FILE = "data/processed/chunks.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    documents = json.load(file)

chunks = []

for document in documents:

    text = document["text"]

    start = 0
    chunk_number = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk_text = text[start:end].strip()

        if chunk_text:

            chunks.append({
                "chunk_id": len(chunks),
                "company": document["company"],
                "filename": document["filename"],
                "file_type": document["file_type"],
                "chunk_number": chunk_number,
                "text": chunk_text
            })

        chunk_number += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP

print("=" * 60)
print("KNOWX CHUNKING COMPLETE")
print("=" * 60)

print(f"Documents: {len(documents)}")
print(f"Total chunks: {len(chunks):,}")

for company in sorted(set(c["company"] for c in chunks)):
    count = sum(1 for c in chunks if c["company"] == company)
    print(f"{company.upper()}: {count:,} chunks")

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print(f"\nSaved to: {OUTPUT_FILE}")
