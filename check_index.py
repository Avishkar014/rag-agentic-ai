from pinecone import Pinecone

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


pc = Pinecone(api_key=PINECONE_API_KEY)

description = pc.describe_index(PINECONE_INDEX_NAME)

print("Index name:", description.name)
print("Dimension:", description.dimension)
print("Metric:", description.metric)
print("Host:", description.host)
print("Status:", description.status)