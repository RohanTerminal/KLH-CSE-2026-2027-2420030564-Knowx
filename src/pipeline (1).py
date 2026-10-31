from pathlib import Path
from llama_cpp import Llama

from src.hybrid_search import (
    chunks,
    semantic_search,
    bm25_search,
    reciprocal_rank_fusion
)
from src.reranker import rerank

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "llama3-8b-q4"
    / "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
)

_llm = None

SEMANTIC_K = 10
BM25_K = 10
RRF_K = 20
FINAL_K = 5


def get_llm():
    global _llm
    if _llm is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"GGUF model not found: {MODEL_PATH}")
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_threads=6,
            n_gpu_layers=0,
            verbose=False
        )
    return _llm


def retrieve_and_rerank(query, company):
    """
    Retrieve relevant enterprise knowledge for a specific company.

    Retrieval:
        Semantic Search + BM25 + RRF + Cross-Encoder Reranking

    Security:
        Company filtering is applied during retrieval so that
        documents belonging to other companies are never passed
        to the reranker or LLM.
    """

    if not company:
        raise ValueError("Company is required for tenant-isolated retrieval.")

    company = company.strip().upper()

    semantic_results = semantic_search(
        query,
        company
    )[:SEMANTIC_K]

    bm25_results = bm25_search(
        query,
        company
    )[:BM25_K]

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

    # Final safety check before context generation.
    reranked = [
        (index_id, score)
        for index_id, score in reranked
        if chunks[index_id].get("company", "").strip().upper() == company
    ]

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


def generate_answer(query, context, company):
    """Generate a grounded answer using the local Llama model."""

    if not context.strip():
        return "The information was not found in the retrieved enterprise documents."

    system_prompt = f"""
You are KNOWX, an enterprise knowledge assistant.
The selected company is {company}.
Use ONLY the provided enterprise context. Do not invent facts or use
outside knowledge. Do not disclose information about other companies.
If the context is insufficient, say the information is not available
in the provided enterprise documents.
Do not invent an expansion of KNOWX. Be concise and factual.
"""

    result = get_llm().create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"QUESTION:\\n{query}\\n\\nENTERPRISE CONTEXT:\\n{context}"
            }
        ],
        max_tokens=350,
        temperature=0.1
    )
    return (result["choices"][0]["message"]["content"] or "").strip()


def run_pipeline(query, company):
    """
    Complete multi-tenant KNOWX pipeline.

    Flow:
        User
          ↓
        Company/Tenant
          ↓
        Semantic Search
          +
        BM25
          ↓
        RRF
          ↓
        Reranking
          ↓
        Grounded Context
          ↓
        Llama 3 8B
          ↓
        Answer + Sources
    """

    if not company:
        raise ValueError("Company is required.")

    company = company.strip().upper()

    reranked_results = retrieve_and_rerank(
        query,
        company
    )

    context = build_context(
        query,
        reranked_results
    )

    try:
        answer = generate_answer(
            query,
            context,
            company
        )

    except Exception as e:
        print(f"Local Llama generation error: {e}")
        answer = "The local language model could not generate an answer. Check the terminal logs."

    sources = []

    for rank, (index_id, score) in enumerate(
        reranked_results,
        start=1
    ):
        chunk = chunks[index_id]

        # Final source-level tenant safety check.
        if chunk.get("company", "").strip().upper() != company:
            continue

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
    print("KNOWX MULTI-TENANT HYBRID RAG PIPELINE")
    print("=" * 70)

    print(f"Knowledge chunks loaded: {len(chunks):,}")

    company = input(
        "\nEnter company "
        "(MICROSOFT / INFOSYS / NVIDIA): "
    ).strip().upper()

    query = input("\nEnter your question: ")

    print("\nRunning tenant-filtered RAG pipeline...")

    try:
        result = run_pipeline(
            query,
            company
        )

        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(result["answer"])

        print("\n" + "=" * 70)
        print("SOURCES")
        print("=" * 70)

        for source in result["sources"]:
            print(
                f"\nSOURCE {source['rank']}"
                f"\nCompany: {source['company']}"
                f"\nDocument: {source['document']}"
                f"\nChunk: {source['chunk']}"
                f"\nScore: {source['score']:.4f}"
            )

    except Exception as e:
        print(f"\nKNOWX ERROR: {e}")
