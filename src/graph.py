from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from retriever import retrieve_documents


class RAGState(TypedDict):
    query: str
    retrieved_documents: list
    retrieved_context: list[str]


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
    }


def build_graph():
    """Build the LangGraph RAG workflow."""

    graph = StateGraph(RAGState)

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_edge(
        START,
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        END,
    )

    return graph.compile()


if __name__ == "__main__":
    graph = build_graph()

    result = graph.invoke(
        {
            "query": "What is Agentic AI?",
            "retrieved_documents": [],
            "retrieved_context": [],
        }
    )

    print("\nQuery:")
    print(result["query"])

    print("\nRetrieved Context:")
    print("=" * 80)

    for i, context in enumerate(
        result["retrieved_context"],
        start=1,
    ):
        print(f"\nCHUNK {i}")
        print("-" * 80)
        print(context)