"""
Utilities for generating embeddings using OpenAI.
"""
from typing import Optional, List, Union
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.api.core.config import settings
from app.api.core.logging import app_logger


class EmbeddingGenerator:       # pylint: disable=too-few-public-methods
    """Class for generating embeddings using OpenAI API."""

    def __init__(self):
        """Initialize with OpenAI client."""
        # self.client = OpenAI(api_key=settings.OPENAI_API_KEY,
        # organization=settings.OPENAI_ORG_KEY)
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL
        app_logger.info(f"Initialized embedding generator with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def generate(self, text: Union[str, List[str]]) -> Optional[List[float]]:
        """
        Generate embeddings for input text.

        Args:
            text: Input text to generate embedding for. Can be a single string or a list of strings.

        Returns:
            List of embeddings or None if generation fails.
        """
        try:
            # Ensure text is not empty
            if not text:
                app_logger.warning("Attempted to generate embedding for empty text")
                return None

            # Log operation
            if isinstance(text, list):
                app_logger.debug(f"Generating embeddings for {len(text)} text items")
            else:
                app_logger.debug(f"Generating embedding for text: '{text[:50]}...'")

            # Generate embedding
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )

            # Return the embedding
            if isinstance(text, list):
                return [item.embedding for item in response.data]

            return response.data[0].embedding

        except Exception as e:
            app_logger.error(f"Error generating embedding: {str(e)}")
            raise


# Create a global instance for reuse
embedding_generator = EmbeddingGenerator()
