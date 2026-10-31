import json
import pickle
import re

INDEX_FILE = "models/bm25_index.pkl"
METADATA_FILE = "models/semantic_metadata.json"

with open(INDEX_FILE, "rb") as file:
    bm25 = pickle.load(file)

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

query = input("\nEnter your question: ")

query_tokens = tokenize(query)

scores = bm25.get_scores(query_tokens)

top_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:5]

print("\n" + "=" * 70)
print("KNOWX BM25 SEARCH RESULTS")
print("=" * 70)

for rank, index_id in enumerate(top_indices, start=1):

    chunk = chunks[index_id]

    print(f"\nRESULT {rank}")
    print("-" * 70)
    print(f"Company : {chunk['company'].upper()}")
    print(f"Document: {chunk['filename']}")
    print(f"Score   : {scores[index_id]:.4f}")
    print("\nText:")
    print(chunk["text"][:800])

print("\n" + "=" * 70)
