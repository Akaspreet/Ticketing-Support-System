"""
API routes initialization.
"""
from fastapi import APIRouter

from app.api.routes import query, ticket, session


# Create main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(query.router, prefix="/query", tags=["Query Processing"])
api_router.include_router(ticket.router, prefix="/ticket", tags=["Ticket Management"])
