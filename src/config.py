import os

from dotenv import load_dotenv


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index",
)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env")