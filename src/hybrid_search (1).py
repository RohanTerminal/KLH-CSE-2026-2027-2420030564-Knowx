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


def get_allowed_indices(company):
    """
    Return chunk indices belonging to the authenticated user's company.
    """
    if not company:
        return []

    company = company.strip().upper()

    return [
        index_id
        for index_id, chunk in enumerate(chunks)
        if chunk.get("company", "").strip().upper() == company
    ]


def semantic_search(query, company=None):
    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Retrieve a larger candidate pool first so that
    # company filtering does not eliminate all results.
    search_k = min(
        max(TOP_K * 10, 100),
        len(chunks)
    )

    scores, indices = semantic_index.search(
        embedding,
        search_k
    )

    results = []

    for index_id in indices[0]:
        index_id = int(index_id)

        if index_id < 0:
            continue

        if company is None:
            results.append(index_id)
        elif chunks[index_id].get("company", "").strip().upper() == company.strip().upper():
            results.append(index_id)

        if len(results) >= TOP_K:
            break

    return results


def bm25_search(query, company=None):
    tokens = tokenize(query)

    scores = bm25.get_scores(tokens)

    if company is None:
        allowed_indices = range(len(scores))
    else:
        company = company.strip().upper()

        allowed_indices = [
            index_id
            for index_id, chunk in enumerate(chunks)
            if chunk.get("company", "").strip().upper() == company
        ]

    ranked_indices = sorted(
        allowed_indices,
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
    print("KNOWX MULTI-TENANT HYBRID RETRIEVER")
    print("=" * 70)

    print(f"Knowledge chunks loaded: {len(chunks):,}")

    company = input(
        "\nEnter company "
        "(MICROSOFT / INFOSYS / NVIDIA): "
    ).strip()

    allowed = get_allowed_indices(company)

    if not allowed:
        print(f"\nNo data found for company: {company}")
        exit()

    print(f"Company: {company.upper()}")
    print(f"Accessible chunks: {len(allowed):,}")

    query = input("\nEnter your question: ")

    semantic_results = semantic_search(
        query,
        company
    )

    bm25_results = bm25_search(
        query,
        company
    )

    hybrid_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results
    )

    print("\n" + "=" * 70)
    print("KNOWX TENANT-FILTERED HYBRID SEARCH")
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
    print("TENANT-FILTERED RETRIEVAL COMPLETE")
    print("=" * 70)
