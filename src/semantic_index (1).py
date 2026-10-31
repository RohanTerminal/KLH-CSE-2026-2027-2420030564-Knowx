import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/processed/chunks.json"
INDEX_FILE = "models/semantic_index.faiss"
METADATA_FILE = "models/semantic_metadata.json"

print("=" * 60)
print("KNOWX SEMANTIC EMBEDDING")
print("=" * 60)

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Chunks loaded: {len(chunks):,}")

print("\nLoading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully.")

texts = [chunk["text"] for chunk in chunks]

print(f"\nCreating embeddings for {len(texts):,} chunks...")
print("This may take a few minutes on CPU.")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings, dtype="float32")

print("\nEmbedding creation complete.")
print(f"Embedding shape: {embeddings.shape}")

dimension = embeddings.shape[1]

print(f"Vector dimension: {dimension}")

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS vectors stored: {index.ntotal:,}")

os.makedirs("models", exist_ok=True)

faiss.write_index(index, INDEX_FILE)

with open(METADATA_FILE, "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("KNOWX SEMANTIC INDEX COMPLETE")
print("=" * 60)

print(f"FAISS index: {INDEX_FILE}")
print(f"Metadata: {METADATA_FILE}")
print(f"Vectors: {index.ntotal:,}")
print(f"Dimensions: {dimension}")
