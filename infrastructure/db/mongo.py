from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from dotenv import load_dotenv
import os

# Load from .env file in current or parent directory
from dotenv import load_dotenv
load_dotenv()


class MongoDatabaseConnector:
    _instance: MongoClient | None = None

    def __new__(cls, *args, **kwargs) -> MongoClient:
        if cls._instance is None:
            try:
                cls._instance = MongoClient(os.getenv("DATABASE_HOST"))
            except ConnectionFailure as e:
                logger.error(f"Couldn't connect to the database: {e!s}")

                raise

        logger.info(f"Connection to MongoDB with URI successful: {os.getenv("DATABASE_HOST")}")

        return cls._instance


connection = MongoDatabaseConnector()
