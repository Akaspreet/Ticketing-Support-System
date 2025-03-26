# """
# Service for LangSmith integration and monitoring.
# """
# from typing import Dict, Any, Optional
# import os
# from langsmith import Client

# from app.api.core.config import settings
# from app.api.core.logging import app_logger


# class LangSmithService:
#     """Service for LangSmith monitoring and tracking."""

#     def __init__(self):
#         """Initialize LangSmith integration."""
#         self.is_enabled = settings.LANGCHAIN_TRACING and settings.LANGSMITH_API_KEY

#         if self.is_enabled:
#             # Set environment variables for LangSmith
#             os.environ["LANGCHAIN_TRACING"] = "true"
#             os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
#             os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY

#             app_logger.info(f"""LangSmith monitoring
#                             enabled for project: {settings.LANGCHAIN_PROJECT}""")
#         else:
#             app_logger.info("LangSmith monitoring is disabled")

#     def create_run(self, name: str, inputs: Dict[str, Any]) -> Optional[str]:
#         """
#         Create a LangSmith run for tracking.

#         Args:
#             name: Name of the run
#             inputs: Input data for the run

#         Returns:
#             Run ID if successful, None otherwise
#         """
#         app_logger.debug(f"Creating LangSmith run with name: {name} and inputs: {inputs}")
#         if not self.is_enabled:
#             return None

#         try:
#             # from langsmith import Client

#             client = Client()
#              # pylint: disable=assignment-from-none
#             run = client.create_run(
#                 name=name,
#                 inputs=inputs,
#                 run_type="chain",
#                 project_name=settings.LANGCHAIN_PROJECT
#             )

#             if run is None:  #  Ensure run is valid before accessing its attributes
#                 app_logger.error("LangSmith run creation failed.")
#                 return None

#             app_logger.debug(f"Created LangSmith run with ID: {run.id}")
#             return run.id

#         except Exception as e:              # pylint: disable=broad-exception-caught
#             app_logger.error(f"Error creating LangSmith run: {str(e)}")
#             return None

#     def update_run(self, run_id: str, outputs: Dict[str, Any], error: Optional[str] = None) -> bool:
#         """
#         Update a LangSmith run with outputs or error.

#         Args:
#             run_id: ID of the run to update
#             outputs: Output data from the run
#             error: Error message if the run failed

#         Returns:
#             True if update was successful, False otherwise
#         """
#         app_logger.debug(f"""Updating LangSmith run with ID:
#                          {run_id}, outputs: {outputs}, error: {error}""")
#         if not self.is_enabled or not run_id:
#             return False

#         try:
#             # from langsmith import Client

#             client = Client()

#             if error:
#                 client.update_run(
#                     run_id=run_id,
#                     outputs=outputs,
#                     error=error,
#                     end_time=None  # Let LangSmith set the end time
#                 )
#             else:
#                 client.update_run(
#                     run_id=run_id,
#                     outputs=outputs,
#                     end_time=None  # Let LangSmith set the end time
#                 )

#             app_logger.debug(f"Updated LangSmith run: {run_id}")
#             return True

#         except Exception as e:                          # pylint: disable=broad-exception-caught
#             app_logger.error(f"Error updating LangSmith run: {str(e)}")
#             return False


# # Create a global instance
# langsmith_service = LangSmithService()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


"""
Service for LangSmith integration and monitoring.
"""
from typing import Dict, Any, Optional
import os
from langsmith import Client

from app.api.core.config import settings
from app.api.core.logging import app_logger


class LangSmithService:
    """Service for LangSmith monitoring and tracking."""

    def __init__(self, local=False):
        """Initialize LangSmith integration."""
        self.is_enabled = settings.LANGCHAIN_TRACING
        self.is_local = local

        if self.is_enabled:
            # Set environment variables for LangSmith
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

            if self.is_local:
                # Local LangSmith configuration
                os.environ["LANGCHAIN_ENDPOINT"] = "http://localhost:8080"
                os.environ["LANGCHAIN_API_KEY"] = "any-key-works-locally"
                app_logger.info(f"Local LangSmith monitoring enabled for project: {settings.LANGCHAIN_PROJECT}")
            else:
                # Cloud LangSmith configuration
                os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
                app_logger.info(f"Cloud LangSmith monitoring enabled for project: {settings.LANGCHAIN_PROJECT}")
        else:
            app_logger.info("LangSmith monitoring is disabled")

    def create_run(self, name: str, inputs: Dict[str, Any]) -> Optional[str]:
        """
        Create a LangSmith run for tracking.

        Args:
            name: Name of the run
            inputs: Input data for the run

        Returns:
            Run ID if successful, None otherwise
        """
        app_logger.debug(f"Creating LangSmith run with name: {name} and inputs: {inputs}")
        if not self.is_enabled:
            return None

        try:
            # Create client with appropriate endpoint
            if self.is_local:
                client = Client(
                    api_url="http://localhost:8080",
                    api_key="any-key-works-locally"
                )
            else:
                client = Client()  # Uses environment variables

            run = client.create_run(
                name=name,
                inputs=inputs,
                run_type="chain",
                project_name=settings.LANGCHAIN_PROJECT
            )

            if run is None:  # Ensure run is valid before accessing its attributes
                app_logger.error("LangSmith run creation failed.")
                return None

            app_logger.debug(f"Created LangSmith run with ID: {run.id}")
            return run.id

        except Exception as e:  # pylint: disable=broad-exception-caught
            app_logger.error(f"Error creating LangSmith run: {str(e)}")
            return None

    def update_run(self, run_id: str, outputs: Dict[str, Any], error: Optional[str] = None) -> bool:
        """
        Update a LangSmith run with outputs or error.

        Args:
            run_id: ID of the run to update
            outputs: Output data from the run
            error: Error message if the run failed

        Returns:
            True if update was successful, False otherwise
        """
        app_logger.debug(f"Updating LangSmith run with ID: {run_id}, outputs: {outputs}, error: {error}")
        if not self.is_enabled or not run_id:
            return False

        try:
            # Create client with appropriate endpoint
            if self.is_local:
                client = Client(
                    api_url="http://localhost:8080",
                    api_key="any-key-works-locally"
                )
            else:
                client = Client()  # Uses environment variables

            if error:
                client.update_run(
                    run_id=run_id,
                    outputs=outputs,
                    error=error,
                    end_time=None  # Let LangSmith set the end time
                )
            else:
                client.update_run(
                    run_id=run_id,
                    outputs=outputs,
                    end_time=None  # Let LangSmith set the end time
                )

            app_logger.debug(f"Updated LangSmith run: {run_id}")
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            app_logger.error(f"Error updating LangSmith run: {str(e)}")
            return False


# Create global instances
langsmith_service = LangSmithService()  # Cloud instance
local_langsmith_service = LangSmithService(local=True)  # Local instance
