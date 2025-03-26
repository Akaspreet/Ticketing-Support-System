"""
FastAPI Application Setup.

This module initializes and configures the FastAPI application. It includes:
- Setting up logging
- Configuring CORS middleware
- Including API routers for handling different endpoints
- Defining the root endpoint

The application can be started using `uvicorn`.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.wsgi import WSGIMiddleware
from app.api.core.logging import setup_logging
from app.api.routes.query import router as query_router
from app.api.routes.ticket import router as ticket_router
from app.api.routes.session import router as session_router
from app.api.routes import api_router
from app.api.core.config import settings
from app.api.services.session_service import session_service
from app.api.visualizations.dashboard import dash_app  # Import the Dash app from the new directory

# Setup logging
setup_logging()

# Create FastAPI application
app = FastAPI(
    title="Support API",
    description="API for query processing and ticket management",
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None  # Disable default redoc
)

# Configure CORS if needed

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Correct the typo here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query_router, prefix="/query", tags=["query"])
app.include_router(ticket_router, prefix="/ticket", tags=["ticket"])
app.include_router(api_router, prefix="/api")
app.include_router(session_router, prefix="/chat", tags=["session"])

# Mount the Dash app using WSGIMiddleware
app.mount("/dashboard", WSGIMiddleware(dash_app.server))

def custom_openapi():
    """
    Custom OpenAPI schema generator.

    Returns:
        dict: The OpenAPI schema.
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Support API",
        version="1.0.0",
        description="API for query processing and ticket management",
        routes=app.routes,
    )
    openapi_schema["openapi"] = "3.0.0"
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/api/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Custom Swagger UI endpoint.

    Returns:
        HTML: The Swagger UI HTML.
    """
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",  # Ensure this matches the openapi endpoint
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=None,
        swagger_js_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui-bundle.min.js",  # pylint: disable=line-too-long
        swagger_css_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui.min.css",  # pylint: disable=line-too-long
    )

@app.get("/api/redoc", include_in_schema=False)
async def redoc_html():
    """
    Custom ReDoc UI endpoint.

    Returns:
        HTML: The ReDoc UI HTML.
    """
    return get_redoc_html(
        openapi_url="/api/openapi.json",
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

@app.get("/api/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """
    OpenAPI schema endpoint.

    Returns:
        dict: The OpenAPI schema.
    """
    return custom_openapi()

@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint of the API.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the API!"}

async def delete_inactive_sessions_task():
    """
    Background task to delete inactive sessions.

    This task runs every hour to delete sessions that have been inactive
    for a specified period.
    """
    while True:
        session_service.delete_inactive_sessions()
        await asyncio.sleep(3600)  # Run every hour

@asynccontextmanager
async def lifespan(app_context: FastAPI):       #pylint: disable=unused-argument
    """
    Lifespan context manager for the FastAPI application.

    Args:
        app_context (FastAPI): The FastAPI application instance.
    """
    background_tasks = BackgroundTasks()
    background_tasks.add_task(delete_inactive_sessions_task)
    await background_tasks()
    yield
    # Clean-up code if needed

app.lifecycle = lifespan

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
