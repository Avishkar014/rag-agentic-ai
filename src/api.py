import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent))

from graph import run_rag

SERVICE_NAME = "Agentic AI RAG Chatbot"

app = FastAPI(
    title=SERVICE_NAME,
    description="LangGraph + Pinecone + Hugging Face + Gemini RAG chatbot",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty.",
        )

    try:
        result = run_rag(query)

        required_fields = [
            "final_answer",
            "retrieved_context_chunks",
            "confidence_score",
        ]

        for field in required_fields:
            if field not in result:
                raise ValueError(
                    f"Missing field in RAG result: {field}"
                )

        return ChatResponse(
            query=query,
            final_answer=result["final_answer"],
            retrieved_context_chunks=result[
                "retrieved_context_chunks"
            ],
            confidence_score=float(
                result["confidence_score"]
            ),
        )

    except Exception as e:
        error_message = str(e)

        print("\n" + "=" * 80)
        print("RAG PIPELINE ERROR")
        print("=" * 80)
        print(f"Error Type : {type(e).__name__}")
        print(f"Error      : {error_message}")
        print("=" * 80 + "\n")

        if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota has been exceeded. "
                    "Please wait and retry, or use a model/API "
                    "quota with available capacity."
                ),
            )

        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {error_message}",
        )