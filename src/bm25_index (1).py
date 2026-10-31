import json
import os
import pickle
import re
from rank_bm25 import BM25Okapi

INPUT_FILE = "data/processed/chunks.json"
INDEX_FILE = "models/bm25_index.pkl"

print("=" * 60)
print("KNOWX BM25 INDEX")
print("=" * 60)

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Chunks loaded: {len(chunks):,}")

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

tokenized_chunks = [
    tokenize(chunk["text"])
    for chunk in chunks
]

print("Tokenizing chunks...")

bm25 = BM25Okapi(tokenized_chunks)

os.makedirs("models", exist_ok=True)

with open(INDEX_FILE, "wb") as file:
    pickle.dump(bm25, file)

print("\n" + "=" * 60)
print("KNOWX BM25 INDEX COMPLETE")
print("=" * 60)

print(f"BM25 documents: {len(tokenized_chunks):,}")
print(f"Index saved to: {INDEX_FILE}")
