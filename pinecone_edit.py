from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pc.delete_index(os.environ["INDEX_NAME"])
pc.create_index(
    name=os.environ["INDEX_NAME"],
    dimension=768,   # ✅ must match embedding model
    metric="cosine"
)