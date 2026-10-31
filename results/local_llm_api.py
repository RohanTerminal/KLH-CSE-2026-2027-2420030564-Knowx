
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama

MODEL_PATH = (
    r"models\llama3-8b-q4"
    r"\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
)

print("Loading KNOWX Llama model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    verbose=False
)

app = FastAPI(title="KNOWX Local LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"status": "KNOWX local model is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    result = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are KNOWX, an enterprise knowledge assistant. "
                    "RAG means Retrieval-Augmented Generation. "
                    "Answer clearly and honestly. Do not invent facts."
                ),
            },
            {"role": "user", "content": request.message},
        ],
        max_tokens=200,
        temperature=0.2,
    )

    answer = result["choices"][0]["message"]["content"]

    return {"answer": answer}
