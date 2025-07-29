import os
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from dotenv import load_dotenv

load_dotenv()  # Load .env file

class QdrantDatabaseConnector:
    _instance: QdrantClient | None = None

    def __new__(cls, *args, **kwargs) -> QdrantClient:
        if cls._instance is None:
            try:
                use_qdrant_cloud = os.getenv("USE_QDRANT_CLOUD", "False").strip().lower() == "true"

                if use_qdrant_cloud:
                    url = os.getenv("QDRANT_CLOUD_URL", "").strip()
                    api_key = os.getenv("QDRANT_APIKEY", "").strip()

                    cls._instance = QdrantClient(url=url, api_key=api_key)
                    uri = url
                else:
                    host = os.getenv("QDRANT_DATABASE_HOST", "localhost").strip()
                    port = int(os.getenv("QDRANT_DATABASE_PORT", "6333").strip())

                    cls._instance = QdrantClient(host=host, port=port)
                    uri = f"{host}:{port}"

                logger.info(f"✅ Connected to Qdrant DB: {uri}")
            except Exception as e:
                logger.exception("❌ Failed to connect to Qdrant DB")
                raise

        return cls._instance

connection = QdrantDatabaseConnector()
