from hybrid_search import chunks, semantic_search, bm25_search, reciprocal_rank_fusion
from reranker import rerank
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

RRF_CANDIDATES = 20
FINAL_K = 5


def build_context(reranked_results):

    context_parts = []

    for rank, (index_id, rerank_score) in enumerate(
        reranked_results,
        start=1
    ):

        chunk = chunks[index_id]

        source = (
            f"[Source {rank}]\n"
            f"Company: {chunk['company'].upper()}\n"
            f"Document: {chunk['filename']}\n"
            f"Chunk: {chunk['chunk_number']}\n"
            f"Reranker Score: {rerank_score:.4f}\n"
        )

        context_parts.append(
            source + "\n" + chunk["text"]
        )

    return "\n\n".join(context_parts)


def generate_answer(question, context):

    prompt = f"""
You are KNOWX, an enterprise knowledge assistant.

Answer the question using ONLY the retrieved enterprise context.

Do not invent information.

If the context does not contain enough information, say:
"I could not find enough information in the provided enterprise documents."

Cite the retrieved sources using [Source 1], [Source 2], etc.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["response"]


def main():

    print("=" * 60)
    print("KNOWX HYBRID RAG PIPELINE")
    print("=" * 60)

    question = input("\nEnter your question: ")

    print("\nLoading KNOWX indexes...")
    print(f"Metadata chunks: {len(chunks)}")

    print("\n1. Running semantic retrieval...")

    semantic_results = semantic_search(question)

    print(
        f"   Semantic results: {len(semantic_results)}"
    )

    print("\n2. Running BM25 retrieval...")

    bm25_results = bm25_search(question)

    print(
        f"   BM25 results: {len(bm25_results)}"
    )

    print("\n3. Performing RRF hybrid fusion...")

    rrf_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
        k=60
    )

    print(
        f"   RRF candidates: {len(rrf_results)}"
    )

    candidate_ids = [
        index_id
        for index_id, score in rrf_results
    ]

    print("\n4. Running Cross-Encoder reranking...")

    reranked_results = rerank(
        question,
        chunks,
        candidate_ids,
        top_k=FINAL_K
    )

    print(
        f"   Reranked results: "
        f"{len(reranked_results)}"
    )

    print("\n" + "=" * 60)
    print("FINAL RERANKED SOURCES")
    print("=" * 60)

    for rank, (index_id, score) in enumerate(
        reranked_results,
        start=1
    ):

        chunk = chunks[index_id]

        print(f"\nSOURCE {rank}")
        print("-" * 60)
        print(
            f"Company: "
            f"{chunk['company'].upper()}"
        )
        print(
            f"Document: "
            f"{chunk['filename']}"
        )
        print(
            f"Chunk: "
            f"{chunk['chunk_number']}"
        )
        print(
            f"Reranker Score: "
            f"{score:.4f}"
        )

    print("\n5. Building grounded context...")

    context = build_context(
        reranked_results
    )

    print("\n" + "=" * 60)
    print("LLM GENERATION")
    print("=" * 60)

    try:

        answer = generate_answer(
            question,
            context
        )

        print("\nANSWER:\n")
        print(answer)

        print("\nSOURCES:")

        for number, (index_id, score) in enumerate(
            reranked_results,
            start=1
        ):

            chunk = chunks[index_id]

            print(
                f"[Source {number}] "
                f"{chunk['company'].upper()} - "
                f"{chunk['filename']} - "
                f"Chunk {chunk['chunk_number']}"
            )

    except Exception as error:

        print("\nLlama is not connected yet.")
        print("Retrieval + reranking are working correctly.")

        print("\nTechnical message:")
        print(error)

        print(
            "\nThe final ranked context is ready "
            "for the GPU-based LLM."
        )


if __name__ == "__main__":
    main()
