import warnings

# Suppress protobuf version warnings - must be done before other imports
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.runtime_version"
)
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")

import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter
from sentence_transformers import SentenceTransformer
import weaviate.classes.init as wvc

# ====== CONFIG ======


def main(
    WEAVIATE_URL="172.16.30.137",  # replace This is the already correct ip
    COLLECTION_NAME="test_collection",
):
    print("Connecting to weaviate...")
    # Connect to Weaviate with increased timeout
    client = weaviate.connect_to_local(
        WEAVIATE_URL,
        skip_init_checks=True,  # we can't use GRPC
        additional_config=wvc.AdditionalConfig(
            timeout=wvc.Timeout(init=30, query=30, insert=30)
        ),
    )

    # Use your own model for embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightest just for testing

    try:
        # ====== 1. Create a collection/schema ======
        existing_collections = client.collections.list_all()
        if COLLECTION_NAME not in existing_collections:
            client.collections.create(
                COLLECTION_NAME,
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="date", data_type=DataType.DATE),
                ],
                vectorizer_config=None,  # We'll provide vectors manually
            )
            print(f"Created collection '{COLLECTION_NAME}'")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists.")

        collection = client.collections.get(COLLECTION_NAME)

        # ====== 2. Insert sample data ======
        docs = [
            {
                "title": "2023 earnings report",
                "category": "finance",
                "date": "2023-07-01T00:00:00Z",
            },
            {
                "title": "AI in healthcare",
                "category": "health",
                "date": "2023-06-01T00:00:00Z",
            },
        ]

        with collection.batch.dynamic() as batch:
            for d in docs:
                vector = model.encode(d["title"])
                batch.add_object(properties=d, vector=vector)

        print("Inserted test documents.")

        # ====== 3. Run a vector search with filters ======
        query = "earnings report"
        query_vector = model.encode(query)
        result = collection.query.near_vector(
            near_vector=query_vector,
            limit=5,
            filters=(
                Filter.by_property("category").equal("finance")
                & Filter.by_property("date").greater_than("2023-01-01T00:00:00Z")
            ),
        )

        # Print results
        print("Results:")
        for obj in result.objects:
            print(obj.properties)
    finally:
        client.close()


if __name__ == "__main__":
    main()
