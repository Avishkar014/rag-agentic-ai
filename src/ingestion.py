from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import PINECONE_INDEX_NAME

def load_and_split_pdf(pdf_path: str):
    """Load the PDF and split it into overlapping chunks."""

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def upload_to_pinecone(chunks):
    """Generate embeddings and upload chunks to Pinecone."""

    print("Creating Hugging Face embedding model...")
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Uploading chunks to Pinecone...")

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )

    print("Successfully uploaded chunks to Pinecone.")

    return vector_store


if __name__ == "__main__":
    pdf_path = Path("data/Ebook-Agentic-AI.pdf")

    chunks = load_and_split_pdf(str(pdf_path))

    upload_to_pinecone(chunks)