"""
API routes for handling user queries.
"""

from typing import AsyncIterator
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse

from app.api.models.query import UserQuery, SearchResponse, ChatResponse, InterruptRequest
from app.api.services.langchain_service import langchain_service
from app.api.services.elastic_search_service import es_service
from app.api.core.logging import app_logger

router = APIRouter()

@router.post("/process", response_model=SearchResponse, status_code=200,
             summary="Process a user query",
             description="Process a user query using LangChain and return search results",
             response_description="Search results with metadata")
async def process_query(query: UserQuery):
    """
    Process a user query using LangChain and return search results.

    Args:
        query (UserQuery): The user query to process.

    Returns:
        SearchResponse: The search results with metadata.
    """
    try:
        app_logger.info(f"""Received query process request from user:
                        {query.user_id}, session: {query.session_id}""")
        response = langchain_service.process_query(user_id=query.user_id, query=query.query)
        search_response = SearchResponse(query=query.query,
                                         results=response.get("results", []),
                                         found=response.get("found", False))
        return search_response
    except Exception as e:
        app_logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}") from e

@router.get("/search", response_model=SearchResponse, status_code=200,
            summary="Search for information",
            description="Search for information based on a user query using Elasticsearch",
            response_description="Search results matching the query")
async def search_info(query: str = Query(..., description="The search query"),
                      user_id: str = Query(..., description="User ID for tracking"),
                      top_k: int = Query(5, description="Number of results to return",
                                         ge=1, le=20)):
    """
    Search for information based on a user query using Elasticsearch.

    Args:
        query (str): The search query.
        user_id (str): The user ID for tracking.
        top_k (int): The number of results to return.

    Returns:
        SearchResponse: The search results matching the query.
    """
    try:
        app_logger.info(f"Received search request from user {user_id}: {query}")
        results = es_service.hybrid_search(query, top_k=top_k)
        search_response = SearchResponse(query=query, results=results, found=len(results) > 0)
        return search_response
    except Exception as e:
        app_logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}") from e

@router.post("/chat", response_model=ChatResponse, status_code=200,
             summary="Generate a chat response",
             description="Generate a natural language response to the user's query",
             response_description="Conversational response with supporting search results")
async def chat_response(query: UserQuery):
    """
    Generate a natural language response to the user's query.

    Args:
        query (UserQuery): The user query to generate a response for.

    Returns:
        ChatResponse: The conversational response with supporting search results.
    """
    try:
        app_logger.info(f"""Received chat request from user:
                        {query.user_id}, session: {query.session_id}""")
        response = langchain_service.generate_chat_response(user_id=query.user_id,
                                                            query=query.query,
                                                            session_id=query.session_id)
        chat_response_data = ChatResponse(query=query.query,
                                          results=response.get("results", []),
                                          found=response.get("found", False),
                                          chat_response=response.get("chat_response", ""),
                                          error=response.get("error"))
        return chat_response_data
    except Exception as e:
        app_logger.error(f"Error generating chat response: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"""Error
                            generating chat response:
                            {str(e)}""") from e

@router.post("/chat/stream", summary="Stream a chat response",
             description="Generate a streaming conversational response with interruption support",
             response_description="Event stream containing response tokens")
async def stream_chat_response(query: UserQuery):
    """
    Generate a streaming conversational response with interruption support.

    Args:
        query (UserQuery): The user query to generate a streaming response for.

    Returns:
        StreamingResponse: The event stream containing response tokens.
    """
    try:
        app_logger.info(f"""Received streaming chat request from user:
                        {query.user_id},
                        session: {query.session_id}""")
        stream_id = f"{query.user_id}_{datetime.utcnow().timestamp()}"

        async def generate_stream() -> AsyncIterator[str]:
            async for token in langchain_service.generate_streaming_chat_response(
                user_id=query.user_id, query=query.query, session_id=query.session_id,
                stream_id=stream_id
                ):
                yield f"data: {token}\n\n"

        headers = {"X-Stream-ID": stream_id}
        return StreamingResponse(generate_stream(), media_type="text/event-stream", headers=headers)
    except Exception as e:
        app_logger.error(f"Error generating streaming chat response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"""Error
                            generating streaming chat response:
                            {str(e)}""") from e

@router.post("/chat/interrupt", status_code=200,
             summary="Interrupt a streaming response",
             description="Interrupt an ongoing streaming chat response",
             response_description="Confirmation of interruption request")
async def interrupt_chat_stream(interrupt_request: InterruptRequest):
    """
    Interrupt an ongoing streaming chat response.

    Args:
        interrupt_request (InterruptRequest): The interruption request details.

    Returns:
        JSONResponse: Confirmation of the interruption request.
    """
    try:
        app_logger.info(f"Received interrupt request for stream: {interrupt_request.stream_id}")
        success = langchain_service.interrupt_stream(interrupt_request.stream_id)
        if success:
            return JSONResponse(status_code=200, content={"message": f"""Stream
                                                          {interrupt_request.stream_id}
                                                          interruption requested successfully"""})

        return JSONResponse(status_code=404, content={"message": f"""Stream
                                                      {interrupt_request.stream_id}
                                                      not found or already completed
                                                      """})
    except Exception as e:
        app_logger.error(f"Error interrupting stream: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interrupting stream: {str(e)}") from e

@router.get("/tokens", status_code=200,
            summary="Get token usage statistics",
            description="Get statistics about token usage across all requests",
            response_description="Token usage statistics for the application")
async def get_token_usage():
    """
    Get statistics about token usage across all requests.

    Returns:
        JSONResponse: The token usage statistics.
    """
    try:
        stats = langchain_service.get_token_usage_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        app_logger.error(f"Error fetching token usage stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"""Error
                            fetching token usage
                            stats: {str(e)}""") from e
