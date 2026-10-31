from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print("Loading KNOWX reranker...")
reranker = CrossEncoder(MODEL_NAME)


def rerank(query, chunks, candidate_ids, top_k=5):

    pairs = []

    for index_id in candidate_ids:

        text = chunks[index_id]["text"]

        pairs.append(
            [query, text]
        )

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]


if __name__ == "__main__":

    print("KNOWX RERANKER READY")
