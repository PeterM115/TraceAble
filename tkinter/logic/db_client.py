from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()  # Load values from .env

def get_client():
    return QdrantClient(
        url=os.getenv("QDRANT_DB_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )