"""
MongoDB client initialization and connection management.
"""
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
except ImportError:
    # Handle import errors gracefully
    MongoClient = None
    ConnectionFailure = Exception
    ServerSelectionTimeoutError = Exception

from app.api.core.config import settings
from app.api.core.logging import app_logger


class MongoDBClient:
    """
    Singleton class for MongoDB client connections.
    """
    _instance = None

    def __new__(cls):
        """Create a singleton instance of the MongoDBClient."""
        if cls._instance is None:
            cls._instance = super(MongoDBClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize client and database attributes if they don't exist."""
        if not hasattr(self, '_client'):
            self._client = None
            self._tickets_db = None
            self._chat_history_db = None
            self._sessions_db = None
            self.initialize_client()

    def initialize_client(self):
        """Initialize MongoDB client with retry logic."""
        try:
            app_logger.info(f"Connecting to MongoDB at {settings.MONGO_URI}")
            self._client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)

            # Verify connection by executing a command
            self._client.admin.command('ping')

            # Initialize database references
            self._tickets_db = self._client[settings.MONGO_DB_TICKETS]
            self._chat_history_db = self._client[settings.MONGO_DB_CHAT_HISTORY]
            self._sessions_db = self._client[settings.MONGO_DB_SESSIONS]

            app_logger.info("Successfully connected to MongoDB")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            app_logger.error(f"MongoDB connection error: {str(e)}")
            self._client = None
            self._tickets_db = None
            self._chat_history_db = None
            self._sessions_db = None
            raise

    @property
    def client(self):
        """Get the MongoDB client instance."""
        if self._client is None:
            self.initialize_client()
        return self._client

    @property
    def tickets_db(self):
        """Get the tickets database."""
        if self._tickets_db is None:
            self.initialize_client()
        return self._tickets_db

    @property
    def chat_history_db(self):
        """Get the chat history database."""
        if self._chat_history_db is None:
            self.initialize_client()
        return self._chat_history_db

    @property
    def sessions_db(self):
        """Get the sessions database."""
        if self._sessions_db is None:
            self.initialize_client()
        return self._sessions_db

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def check_connection(self):
        """Test connection to MongoDB."""
        try:
            self.client.admin.command('ping')
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            app_logger.error(f"MongoDB connection check failed: {str(e)}")
            return False

    def close(self):
        """Close the MongoDB client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._tickets_db = None
            self._chat_history_db = None
            app_logger.info("MongoDB connection closed")


# Create global instance that will be reused
mongodb_client = MongoDBClient()
