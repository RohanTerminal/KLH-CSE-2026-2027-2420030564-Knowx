import json
import pickle
import re
import faiss
from sentence_transformers import SentenceTransformer

SEMANTIC_INDEX = "models/semantic_index.faiss"
METADATA_FILE = "models/semantic_metadata.json"
BM25_INDEX = "models/bm25_index.pkl"

TOP_K = 10
FINAL_K = 5


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


print("=" * 70)
print("KNOWX RAG CONTEXT BUILDER")
print("=" * 70)

semantic_index = faiss.read_index(SEMANTIC_INDEX)

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

with open(BM25_INDEX, "rb") as file:
    bm25 = pickle.load(file)

print(f"Knowledge chunks loaded: {len(chunks):,}")

model = SentenceTransformer("all-MiniLM-L6-v2")


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

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    return ranked[:TOP_K]


def hybrid_search(query):

    semantic_results = semantic_search(query)
    bm25_results = bm25_search(query)

    fused_scores = {}

    for rank, index_id in enumerate(
        semantic_results,
        start=1
    ):
        fused_scores[index_id] = (
            fused_scores.get(index_id, 0)
            + 1 / (60 + rank)
        )

    for rank, index_id in enumerate(
        bm25_results,
        start=1
    ):
        fused_scores[index_id] = (
            fused_scores.get(index_id, 0)
            + 1 / (60 + rank)
        )

    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [index_id for index_id, score in ranked[:FINAL_K]]


def build_context(query):

    results = hybrid_search(query)

    context_parts = []

    for rank, index_id in enumerate(results, start=1):

        chunk = chunks[index_id]

        source = (
            f"[Source {rank}]\n"
            f"Company: {chunk['company'].upper()}\n"
            f"Document: {chunk['filename']}\n"
            f"Chunk: {chunk['chunk_number']}\n"
        )

        text = chunk["text"]

        context_parts.append(
            source + "\n" + text
        )

    return "\n\n" + "\n\n".join(context_parts)


question = input("\nEnter your question: ")

context = build_context(question)

print("\n" + "=" * 70)
print("RETRIEVED CONTEXT FOR LLM")
print("=" * 70)

print(context)

print("\n" + "=" * 70)
print("CONTEXT BUILDING COMPLETE")
print("=" * 70)
