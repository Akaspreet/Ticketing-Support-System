"""Service for Langfuse integration."""

from langfuse import Langfuse    #pylint: disable=import-error
from app.api.core.config import settings
from app.api.core.logging import app_logger

class LangfuseService:
    """Service for Langfuse integration and observation."""

    def __init__(self):
        """Initialize Langfuse client."""
        try:
            self.client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST
            )
            app_logger.info("Initialized Langfuse Service")
        except Exception as e:              #pylint: disable=broad-exception-caught
            app_logger.error(f"Error initializing Langfuse Service: {str(e)}")
            self.client = None

    def create_trace(self, name: str, user_id: str = None, metadata: dict = None):
        """Create a new trace."""
        if not self.client:
            app_logger.warning("Langfuse client not initialized")
            return None

        try:
            trace = self.client.trace(
                name=name,
                user_id=user_id,
                metadata=metadata or {}
            )
            app_logger.debug(f"Created Langfuse trace: {trace.id}")
            return trace
        except Exception as e:              #pylint: disable=broad-exception-caught
            app_logger.error(f"Error creating Langfuse trace: {str(e)}")
            return None

    def create_span(self, trace_id: str, name: str, metadata: dict = None):
        """Create a new span within a trace."""
        if not self.client:
            app_logger.warning("Langfuse client not initialized")
            return None

        try:
            span = self.client.span(
                name=name,
                trace_id=trace_id,
                metadata=metadata or {}
            )
            app_logger.debug(f"Created Langfuse span: {span.id}")
            return span
        except Exception as e:          #pylint: disable=broad-exception-caught
            app_logger.error(f"Error creating Langfuse span: {str(e)}")
            return None

    #pylint: disable = too-many-positional-arguments
    #pylint: disable = too-many-arguments
    def create_generation(self, trace_id: str, name: str, model: str, prompt: str,
                         completion: str, metadata: dict = None, token_usage: dict = None):
        """Create a generation event."""
        if not self.client:
            app_logger.warning("Langfuse client not initialized")
            return None

        try:
            usage = {}
            if token_usage:
                usage = {
                    "prompt_tokens": token_usage.get("input_tokens", 0),
                    "completion_tokens": token_usage.get("output_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0)
                }

            generation = self.client.generation(
                name=name,
                trace_id=trace_id,
                model=model,
                prompt=prompt,
                completion=completion,
                metadata=metadata or {},
                usage=usage
            )
            app_logger.debug(f"Created Langfuse generation: {generation.id}")
            return generation
        except Exception as e:      #pylint: disable=broad-exception-caught
            app_logger.error(f"Error creating Langfuse generation: {str(e)}")
            return None

    def end_trace(self, trace_id: str, output: dict = None, error: str = None):
        """End a trace with optional output or error."""
        if not self.client:
            app_logger.warning("Langfuse client not initialized")
            return

        try:
            trace = self.client.get_trace(trace_id)
            if trace:
                if output:
                    trace.update(output=output)
                if error:
                    trace.update(error=error, status="error")
                trace.end()
                app_logger.debug(f"Ended Langfuse trace: {trace_id}")
        except Exception as e:          #pylint: disable=broad-exception-caught
            app_logger.error(f"Error ending Langfuse trace: {str(e)}")

# Create a global instance
langfuse_service = LangfuseService()
