from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os

# ============================================================
# KNOWX API
# Multi-Tenant Enterprise Knowledge Access
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import run_pipeline


app = FastAPI(
    title="KNOWX",
    description="Hybrid RAG Framework for Enterprise Knowledge Management",
    version="1.0.0"
)


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=os.path.join(
            os.path.dirname(__file__),
            "static"
        )
    ),
    name="static"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str
    company: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return FileResponse(
        os.path.join(
            os.path.dirname(__file__),
            "static",
            "index.html"
        )
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "application": "KNOWX",
        "mode": "multi-tenant"
    }


# ============================================================
# ASK KNOWX
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()
    company = request.company.strip().upper()

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    # --------------------------------------------------------
    # Validate company
    # --------------------------------------------------------

    allowed_companies = {
        "MICROSOFT",
        "INFOSYS",
        "NVIDIA"
    }

    if company not in allowed_companies:
        return {
            "answer": "Invalid company or tenant.",
            "sources": []
        }

    # --------------------------------------------------------
    # Run tenant-filtered RAG pipeline
    # --------------------------------------------------------

    try:

        result = run_pipeline(
            question,
            company
        )

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "context": result.get("context", ""),
            "company": company
        }

    except Exception as e:

        print(f"KNOWX ERROR: {e}")

        return {
            "answer": "KNOWX encountered an error while processing your question.",
            "sources": [],
            "company": company
        }
