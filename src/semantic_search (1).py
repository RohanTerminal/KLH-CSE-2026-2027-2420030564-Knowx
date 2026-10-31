import json
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "models/semantic_index.faiss"
METADATA_FILE = "models/semantic_metadata.json"

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

model = SentenceTransformer("all-MiniLM-L6-v2")

query = input("\nEnter your question: ")

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

scores, indices = index.search(query_embedding, 5)

print("\n" + "=" * 70)
print("KNOWX SEMANTIC SEARCH RESULTS")
print("=" * 70)

for rank, (score, index_id) in enumerate(
    zip(scores[0], indices[0]), start=1
):

    chunk = chunks[index_id]

    print(f"\nRESULT {rank}")
    print("-" * 70)
    print(f"Company : {chunk['company'].upper()}")
    print(f"Document: {chunk['filename']}")
    print(f"Score   : {score:.4f}")
    print("\nText:")
    print(chunk["text"][:800])

print("\n" + "=" * 70)
