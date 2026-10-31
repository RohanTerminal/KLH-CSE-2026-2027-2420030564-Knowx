import os
import requests


OLLAMA_URL = os.getenv(
    "KNOWX_OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

MODEL_NAME = os.getenv(
    "KNOWX_LLM_MODEL",
    "llama3:8b"
)


def build_prompt(question, context):

    prompt = f"""
You are KNOWX, an enterprise knowledge assistant.

Answer the question using ONLY the information provided
in the enterprise context.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the context does not contain enough information,
   say that the information was not found in the documents.
4. Give a clear and concise answer.
5. Use the source information to support your answer.

QUESTION:
{question}

ENTERPRISE CONTEXT:
{context}

ANSWER:
"""

    return prompt


def generate_answer(question, context):

    prompt = build_prompt(
        question,
        context
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        },
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    return data.get(
        "response",
        ""
    ).strip()


if __name__ == "__main__":

    print("KNOWX LLM interface ready")
    print("Model:", MODEL_NAME)
    print("URL:", OLLAMA_URL)
