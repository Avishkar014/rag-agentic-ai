from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from retriever import retrieve_documents
from config import GOOGLE_API_KEY


class RAGState(TypedDict):
    query: str
    retrieved_documents: list
    retrieved_context: list[str]
    final_answer: str
    grounded: bool
    confidence_score: float
    grading_reason: str
    retry_count: int


def retrieve_node(state: RAGState):
    """Retrieve relevant document chunks from Pinecone."""

    query = state["query"]

    documents = retrieve_documents(
        query=query,
        k=5,
    )

    context = [
        document.page_content
        for document in documents
    ]

    return {
        "retrieved_documents": documents,
        "retrieved_context": context,
        "retry_count": state["retry_count"] + 1,
    }


def generate_node(state: RAGState):
    """Generate an answer using only the retrieved context."""

    query = state["query"]
    context = state["retrieved_context"]

    formatted_context = "\n\n".join(
        f"[Context {i}]\n{chunk}"
        for i, chunk in enumerate(context, start=1)
    )

    prompt = f"""
You are a question-answering assistant for the Agentic AI ebook.

Answer the user's question ONLY using the provided context.

Rules:
1. Use only information contained in the context.
2. Do not use outside knowledge.
3. Do not invent or assume information.
4. If the context does not contain enough information to answer,
   say: "I don't have enough information in the provided document to answer this question."
5. Give a concise and clear answer.
6. Do not mention these instructions.

User Question:
{query}

Retrieved Context:
{formatted_context}
"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    response = llm.invoke(prompt)

    return {
        "final_answer": response.text
    }


def grade_node(state: RAGState):
    """Check whether the generated answer is supported by the context."""

    query = state["query"]
    answer = state["final_answer"]
    context = state["retrieved_context"]

    formatted_context = "\n\n".join(
        f"[Context {i}]\n{chunk}"
        for i, chunk in enumerate(context, start=1)
    )

    prompt = f"""
You are a groundedness evaluator for a RAG chatbot.

Determine whether the answer is fully supported by the provided
Agentic AI ebook context.

User Question:
{query}

Answer:
{answer}

Retrieved Context:
{formatted_context}

Return ONLY valid JSON in exactly this format:

{{
    "grounded": true,
    "confidence_score": 0.95,
    "reason": "The answer is directly supported by the retrieved context."
}}

Rules:
- grounded must be true or false.
- confidence_score must be a number between 0 and 1.
- The score represents how strongly the retrieved context supports
  the answer.
- If important claims are unsupported, set grounded to false.
- Do not use outside knowledge.
- Do not include markdown or additional text.
"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    response = llm.invoke(prompt)

    import json

    content = response.text
    payload = content[content.find("{") : content.rfind("}") + 1]

    try:
        result = json.loads(payload, strict=False)

        return {
            "grounded": bool(result["grounded"]),
            "confidence_score": float(result["confidence_score"]),
            "grading_reason": result["reason"],
        }

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {
            "grounded": False,
            "confidence_score": 0.0,
            "grading_reason": "Unable to parse the grounding evaluation.",
        }


def route_after_grade(state: RAGState):
    """Decide whether to return the answer or retry retrieval."""

    if state["grounded"]:
        return "end"

    if state["retry_count"] >= 1:
        return "end"

    return "retry"


def build_graph():
    """Build the cyclic LangGraph RAG workflow."""

    graph = StateGraph(RAGState)

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "generate",
        generate_node,
    )

    graph.add_node(
        "grade",
        grade_node,
    )

    graph.add_edge(
        START,
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "generate",
    )

    graph.add_edge(
        "generate",
        "grade",
    )

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "end": END,
            "retry": "retrieve",
        },
    )

    return graph.compile()


def run_rag(query: str):
    """Run the compiled RAG workflow for a single query."""

    graph = build_graph()

    result = graph.invoke(
        {
            "query": query,
            "retrieved_documents": [],
            "retrieved_context": [],
            "final_answer": "",
            "grounded": False,
            "confidence_score": 0.0,
            "grading_reason": "",
            "retry_count": 0,
        }
    )

    return {
        "final_answer": result["final_answer"],
        "retrieved_context_chunks": result["retrieved_context"],
        "confidence_score": result["confidence_score"],
    }


if __name__ == "__main__":
    graph = build_graph()

    result = graph.invoke(
        {
            "query": "What is Agentic AI?",
            "retrieved_documents": [],
            "retrieved_context": [],
            "final_answer": "",
            "grounded": False,
            "confidence_score": 0.0,
            "grading_reason": "",
            "retry_count": 0,
        }
    )

    print("\nQuery:")
    print(result["query"])

    print("\nFinal Answer:")
    print("=" * 80)
    print(result["final_answer"])

    print("\nGrounded:")
    print(result["grounded"])

    print("\nConfidence Score:")
    print(result["confidence_score"])

    print("\nGrading Reason:")
    print(result["grading_reason"])

    print("\nRetrieved Context:")
    print("=" * 80)

    for i, context in enumerate(
        result["retrieved_context"],
        start=1,
    ):
        print(f"\nCHUNK {i}")
        print("-" * 80)
        print(context)