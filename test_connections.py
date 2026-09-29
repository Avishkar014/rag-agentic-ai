import time

from pinecone import Pinecone, ServerlessSpec

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


INDEX_DIMENSION = 384
INDEX_METRIC = "cosine"


def main():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [
        index.name for index in pc.list_indexes()
    ]

    if PINECONE_INDEX_NAME in existing_indexes:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")
    else:
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=INDEX_DIMENSION,
            metric=INDEX_METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print("Index creation requested.")

    # Wait until the index is ready
    print("Waiting for index to become ready...")

    while True:
        description = pc.describe_index(PINECONE_INDEX_NAME)

        if description.status.ready:
            break

        print("Index is still initializing...")
        time.sleep(2)

    print("\nPinecone index is ready!")
    print(f"Name: {PINECONE_INDEX_NAME}")
    print(f"Dimension: {INDEX_DIMENSION}")
    print(f"Metric: {INDEX_METRIC}")


if __name__ == "__main__":
    main()