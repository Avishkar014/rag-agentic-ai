from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import PINECONE_INDEX_NAME


def get_vector_store():
    """Create the Pinecone vector store using the same embedding model."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )

    return vector_store


def retrieve_documents(query: str, k: int = 5):
    """Retrieve the most relevant document chunks."""

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        query,
        k=k,
    )

    return documents


if __name__ == "__main__":
    query = "What is Agentic AI?"

    documents = retrieve_documents(query, k=5)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(documents)} chunks\n")

    for i, document in enumerate(documents, start=1):
        print("=" * 80)
        print(f"CHUNK {i}")
        print("=" * 80)

        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)