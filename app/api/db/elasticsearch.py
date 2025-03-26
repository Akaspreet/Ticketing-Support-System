"""
Elasticsearch client initialization and connection management.
"""
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import ConnectionError as ESConnectionError, TransportError
except ImportError:
    # Handle import errors gracefully
    Elasticsearch = None
    ESConnectionError = Exception
    TransportError = Exception

from app.api.core.config import settings
from app.api.core.logging import app_logger


class ElasticsearchClient:
    """
    Singleton class for Elasticsearch client connections.
    """
    _instance = None

    def __new__(cls):
        """Create a singleton instance of the ElasticsearchClient."""
        if cls._instance is None:
            cls._instance = super(ElasticsearchClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the client if it hasn't been done already."""
        if not hasattr(self, '_client'):
            self._client = None
            self.initialize_client()

    def initialize_client(self):
        """Initialize Elasticsearch client with retry logic."""
        try:
            app_logger.info(f"Connecting to Elasticsearch at {settings.ES_HOST}")
            self._client = Elasticsearch(
                hosts=[settings.ES_HOST],
                basic_auth=(settings.ES_USERNAME, settings.ES_PASSWORD),
                verify_certs=False  # Use True in production with proper certificates
            )

            # Verify connection by pinging the server
            if not self.ping():
                app_logger.error("Failed to connect to Elasticsearch")
                raise ESConnectionError("Could not connect to Elasticsearch") # pylint: disable=broad-exception-raised

            app_logger.info("Successfully connected to Elasticsearch")
        except ESConnectionError as e:
            app_logger.error(f"Elasticsearch connection error: {str(e)}")
            self._client = None
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def ping(self):
        """Test connection to Elasticsearch."""
        try:
            return self._client.ping()
        except (ESConnectionError, TransportError) as e:
            app_logger.error(f"Elasticsearch ping failed: {str(e)}")
            return False

    @property
    def client(self):
        """Get the Elasticsearch client instance."""
        if self._client is None:
            self.initialize_client()
        return self._client

    def check_index_exists(self, index_name=None):
        """Check if an index exists in Elasticsearch."""
        if not index_name:
            index_name = settings.ES_INDEX_NAME

        try:
            return self.client.indices.exists(index=index_name)
        except (ESConnectionError, TransportError) as e:
            app_logger.error(f"Error checking index existence: {str(e)}")
            return False

    def close(self):
        """Close the Elasticsearch client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            app_logger.info("Elasticsearch connection closed")


# Create global instance that will be reused
es_client = ElasticsearchClient()
