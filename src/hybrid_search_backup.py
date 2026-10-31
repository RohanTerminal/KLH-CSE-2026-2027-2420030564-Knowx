import json
import pickle
import re
import faiss
from sentence_transformers import SentenceTransformer

SEMANTIC_INDEX = "models/semantic_index.faiss"
METADATA_FILE = "models/semantic_metadata.json"
BM25_INDEX = "models/bm25_index.pkl"

TOP_K = 10
FINAL_K = 20

semantic_index = faiss.read_index(SEMANTIC_INDEX)

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

with open(BM25_INDEX, "rb") as file:
    bm25 = pickle.load(file)

model = SentenceTransformer("all-MiniLM-L6-v2")


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def semantic_search(query):
    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = semantic_index.search(
        embedding,
        TOP_K
    )

    return indices[0].tolist()


def bm25_search(query):
    tokens = tokenize(query)

    scores = bm25.get_scores(tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    return ranked_indices[:TOP_K]


def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
    k=60
):
    fused_scores = {}

    for rank, index_id in enumerate(
        semantic_results,
        start=1
    ):
        fused_scores[index_id] = (
            fused_scores.get(index_id, 0)
            + 1 / (k + rank)
        )

    for rank, index_id in enumerate(
        bm25_results,
        start=1
    ):
        fused_scores[index_id] = (
            fused_scores.get(index_id, 0)
            + 1 / (k + rank)
        )

    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:FINAL_K]


if __name__ == "__main__":

    print("=" * 70)
    print("KNOWX HYBRID RETRIEVER")
    print("=" * 70)

    print(f"Knowledge chunks loaded: {len(chunks):,}")

    query = input("\nEnter your question: ")

    semantic_results = semantic_search(query)
    bm25_results = bm25_search(query)

    hybrid_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results
    )

    print("\n" + "=" * 70)
    print("KNOWX HYBRID SEARCH RESULTS")
    print("=" * 70)

    print("\nSemantic candidates:", len(semantic_results))
    print("BM25 candidates:", len(bm25_results))
    print("Final hybrid results:", len(hybrid_results))

    for rank, (index_id, fusion_score) in enumerate(
        hybrid_results,
        start=1
    ):
        chunk = chunks[index_id]

        print(f"\nRESULT {rank}")
        print("-" * 70)
        print(f"Company      : {chunk['company'].upper()}")
        print(f"Document     : {chunk['filename']}")
        print(f"Chunk        : {chunk['chunk_number']}")
        print(f"Fusion Score : {fusion_score:.6f}")
        print("\nText:")
        print(chunk["text"][:1000])

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL COMPLETE")
    print("=" * 70)
