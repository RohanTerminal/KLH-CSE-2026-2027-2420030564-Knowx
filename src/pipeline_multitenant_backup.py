import requests

from src.hybrid_search import (
    chunks,
    semantic_search,
    bm25_search,
    reciprocal_rank_fusion
)

from src.reranker import rerank


# ============================================================
# KNOWX RAG PIPELINE
# Retrieval → RRF → Reranking → Context → LLM
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3:8b"

SEMANTIC_K = 10
BM25_K = 10
RRF_K = 20
FINAL_K = 5


def retrieve_and_rerank(query):
    """
    Retrieve relevant enterprise knowledge using:
    Semantic Search + BM25 + RRF + Cross-Encoder Reranking
    """

    semantic_results = semantic_search(query)[:SEMANTIC_K]

    bm25_results = bm25_search(query)[:BM25_K]

    hybrid_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results
    )

    candidate_ids = [
        index_id
        for index_id, score in hybrid_results[:RRF_K]
    ]

    reranked = rerank(
        query,
        chunks,
        candidate_ids,
        top_k=FINAL_K
    )

    return reranked


def build_context(query, reranked_results):
    """
    Build grounded context for the LLM.
    """

    context_parts = []

    for rank, (index_id, score) in enumerate(
        reranked_results,
        start=1
    ):
        chunk = chunks[index_id]

        source = (
            f"[Source {rank}]\n"
            f"Company: {chunk['company'].upper()}\n"
            f"Document: {chunk['filename']}\n"
            f"Chunk: {chunk['chunk_number']}\n"
        )

        context_parts.append(
            source + "\n" + chunk["text"]
        )

    return "\n\n".join(context_parts)


def generate_answer(query, context):
    """
    Generate a grounded answer using Llama 3 8B.
    """

    prompt = f"""
You are KNOWX, an enterprise knowledge assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information,
say that the information is not available in the
provided enterprise documents.

Do not invent facts.

User question:
{query}

Retrieved context:
{context}

Provide a concise, factual answer.
Mention the relevant source documents when appropriate.
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data.get("response", "").strip()


def run_pipeline(query):
    """
    Complete KNOWX pipeline.

    Returns:
        answer
        retrieved sources
        context
    """

    reranked_results = retrieve_and_rerank(query)

    context = build_context(
        query,
        reranked_results
    )

    try:
        answer = generate_answer(
            query,
            context
        )

    except requests.exceptions.RequestException:
        answer = None

    sources = []

    for rank, (index_id, score) in enumerate(
        reranked_results,
        start=1
    ):
        chunk = chunks[index_id]

        sources.append({
            "rank": rank,
            "company": chunk["company"],
            "document": chunk["filename"],
            "chunk": chunk["chunk_number"],
            "score": float(score)
        })

    return {
        "answer": answer,
        "sources": sources,
        "context": context
    }


if __name__ == "__main__":

    print("=" * 70)
    print("KNOWX HYBRID RAG PIPELINE")
    print("=" * 70)

    print(f"Knowledge chunks loaded: {len(chunks)}")

    query = input("\nEnter your question: ")

    print("\n1. Running semantic retrieval...")
    semantic_results = semantic_search(query)
    print(f"   Semantic results: {len(semantic_results)}")

    print("\n2. Running BM25 retrieval...")
    bm25_results = bm25_search(query)
    print(f"   BM25 results: {len(bm25_results)}")

    print("\n3. Performing RRF hybrid fusion...")
    hybrid_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results
    )
    print(f"   RRF candidates: {len(hybrid_results)}")

    print("\n4. Running Cross-Encoder reranking...")

    candidate_ids = [
        index_id
        for index_id, score in hybrid_results[:RRF_K]
    ]

    reranked_results = rerank(
        query,
        chunks,
        candidate_ids,
        top_k=FINAL_K
    )

    print(f"   Reranked results: {len(reranked_results)}")

    print("\n" + "=" * 70)
    print("FINAL RERANKED SOURCES")
    print("=" * 70)

    for rank, (index_id, score) in enumerate(
        reranked_results,
        start=1
    ):
        chunk = chunks[index_id]

        print(f"\nSOURCE {rank}")
        print("-" * 70)
        print(f"Company: {chunk['company'].upper()}")
        print(f"Document: {chunk['filename']}")
        print(f"Chunk: {chunk['chunk_number']}")
        print(f"Reranker Score: {score:.4f}")

    print("\n5. Building grounded context...")

    context = build_context(
        query,
        reranked_results
    )

    print("\n" + "=" * 70)
    print("LLM GENERATION")
    print("=" * 70)

    try:
        answer = generate_answer(
            query,
            context
        )

        print("\nANSWER")
        print("-" * 70)
        print(answer)

    except requests.exceptions.RequestException:
        print("\nLlama is not connected yet.")
        print("Retrieval + reranking are working correctly.")
        print("The final ranked context is ready for the GPU-based LLM.")
