"""
API routes for handling chat sessions.
"""
from datetime import datetime
import uuid
import json
import asyncio
from typing import Optional, AsyncIterator, Dict, Any
from datetime import datetime, timedelta
import jwt
import time
from jwt.exceptions import PyJWTError

from fastapi import APIRouter, HTTPException, Depends, Header, Request
import httpx
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from app.api.services.session_service import session_service
from app.api.models.session import ChatMessageRequest, InterruptRequest
from app.api.core.logging import app_logger
from app.api.services.langchain_service import langchain_service, TokenCounter
from app.api.core.config import settings

router = APIRouter()

class MessageRequest(BaseModel):
    """
    Request model for adding a message to a session.
    """
    user_id: str
    message: str
    session_id: Optional[str] = None

class SessionResponse(BaseModel):
    """
    Response model for session creation.
    """
    session_id: str

class LatestSessionRequest(BaseModel):
    """For latest session
    """
    user_id: Optional[str] = None

async def generate_chat_response_stream(
    user_id: Optional[str],
    message: str,
    session_id: str,
    stream_id: str
) -> AsyncIterator[str]:
    """
    Generate a streaming response for a chat message.

    Args:
        user_id (Optional[str]): The user ID, if authenticated
        message (str): The message content
        session_id (str): The session ID
        stream_id (str): The stream ID for the current message

    Yields:
        str: JSON-formatted stream items
    """
    # First, yield the session_id
    session_info = {"type": "data", "value": {"session_id": session_id}}
    yield json.dumps(session_info)

    # Then, yield the stream_id
    stream_info = {"type": "data", "value": {"stream_id": stream_id}}
    yield json.dumps(stream_info)

    # Connect to the LLM service and stream the response
    try:
        async for token in langchain_service.generate_streaming_chat_response(
            user_id=user_id,
            query=message,
            session_id=session_id,
            stream_id=stream_id
        ):
            # Handle special tokens
            #pylint: disable = consider-using-in
            if token == "[DONE]" or token == "[INTERRUPTED]":
                continue

            # Format the token as per the required output format
            response_item = {
                "type": "message",
                "value": {"text": token}
            }
            yield json.dumps(response_item)

            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
    except Exception as e:      #pylint: disable = broad-exception-caught
        # Handle any errors during streaming
        error_item = {
            "type": "error",
            "value": {"message": str(e)}
        }
        yield json.dumps(error_item) + "\n"


# Cache to store user info and reduce Keycloak API calls
# Structure: {token: {"user_info": {...}, "expiry": timestamp}}
token_cache: Dict[str, Dict[str, Any]] = {}

async def validate_and_get_user_info(authorization: str = Header(None)):
    """
    Validate the access token and get user information from Keycloak,
    with caching to reduce API calls and handling token expiration.

    Args:
        authorization (str): The Authorization header containing the access token

    Returns:
        dict: User information including sub (user_id) and email or None if invalid
    """
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        return None

    access_token = authorization.replace("Bearer ", "")

    # Check if we have cached user_info for this token
    current_time = time.time()
    if access_token in token_cache:
        cache_entry = token_cache[access_token]
        # If the cached entry hasn't expired, return it
        if current_time < cache_entry["expiry"]:
            return cache_entry["user_info"]
        else:
            # Remove expired entry from cache
            token_cache.pop(access_token, None)

    # Try to decode token to check expiration without calling Keycloak
    try:
        # Note: This does not verify the token signature, just checks its structure and expiration
        decoded_token = jwt.decode(access_token, options={"verify_signature": False})

        # Check if token is expired
        if 'exp' in decoded_token and decoded_token['exp'] < current_time:
            app_logger.warning("Access token is expired")
            return None

    except PyJWTError as e:
        app_logger.warning(f"Invalid access token format: {str(e)}")
        return None

    # Keycloak userinfo endpoint
    keycloak_url = "https://keycloak-dev.cloudzmall.com/realms/apps/protocol/openid-connect/userinfo"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {access_token}"}
            response = await client.get(keycloak_url, headers=headers)

            if response.status_code == 200:
                user_info = response.json()

                # Cache the user info with expiration (5 minutes before the token expires)
                # Default to 5 minutes if we can't determine expiration from the token
                try:
                    # 300 seconds (5 minutes) before expiration or 5 minutes from now
                    if 'exp' in decoded_token:
                        cache_expiry = min(decoded_token['exp'] - 300, current_time + 300)
                    else:
                        cache_expiry = current_time + 300
                except Exception:
                    cache_expiry = current_time + 300

                token_cache[access_token] = {
                    "user_info": user_info,
                    "expiry": cache_expiry
                }

                return user_info
            elif response.status_code == 401:
                app_logger.warning("Access token rejected by Keycloak")
                return None
            else:
                app_logger.error(f"Failed to get user info from Keycloak: {response.status_code}, {response.text}")
                return None
    except Exception as e:
        app_logger.error(f"Error getting user info from Keycloak: {str(e)}")
        return None

async def cleanup_token_cache():
    """
    Periodically clean up expired tokens from the cache.
    This can be called by a background task scheduler.
    """
    current_time = time.time()
    expired_tokens = [token for token, data in token_cache.items() if data["expiry"] < current_time]
    for token in expired_tokens:
        token_cache.pop(token, None)

    if expired_tokens:
        app_logger.info(f"Cleaned up {len(expired_tokens)} expired tokens from cache")


@router.post("/sessions/sendmessage", summary="Send a message in a chat session")
async def send_chat_message(request: ChatMessageRequest, authorization: Optional[str] = Header(None)):
    """
    Send a message in a chat session with streaming response.

    If no session_id is provided, a new session will be created.

    Args:
        request (ChatMessageRequest): The chat message request

    Returns:
        StreamingResponse: A streaming response with the chat message
    """
    try:
        # Initialize MongoDB connection if not already done
        if not hasattr(session_service, 'collection') or session_service.collection is None:
            session_service.__init__()          #pylint: disable = unnecessary-dunder-call

        token_counter = TokenCounter(settings.OPENAI_MODEL)

        user_info = None
        if authorization:
            user_info = await validate_and_get_user_info(authorization)

        if user_info and 'sub' in user_info:
            user_id = user_info['sub']
            mail_id = user_info.get('email', '')

            # You might want to log successful authentication
            app_logger.info(f"Authenticated user: {mail_id}")
        else:
            # Generate anonymous user_id if not authenticated
            user_id = request.user_id if request.user_id else f"anonymous_{uuid.uuid4()}"
            mail_id = "anonymous@example.com"  # Default email for anonymous users

            if authorization:
                app_logger.warning("Authentication failed or token expired, using anonymous user")

        if not user_id:
            user_id = f"anonymous_{uuid.uuid4()}"

        print("\n\n user id is: \n\n", user_id)
        print("\n\n mail id is: \n\n", mail_id)

        # Generate stream_id
        stream_id = f"{user_id}_{datetime.utcnow().timestamp()}"


        # Generate session_id if not provided
        session_id = request.session_id
        if not session_id:
            session_id = session_service.create_session(user_id, mail_id)

        # Get timestamp
        timestamp = request.time_stamp or datetime.utcnow()     #pylint: disable = unused-variable
        role = request.role if request.role else "user"         #pylint: disable = unused-variable

        # Store the user message in the session
        session_service.add_message(
            session_id=session_id,
            user_id=user_id,
            role=request.role,
            message=request.message,
            token = token_counter.count_tokens(request.message),
            stream_id = stream_id,
            mail_id = mail_id
        )

        # Generate the streaming response
        async def response_generator():
            async for item in generate_chat_response_stream(
                user_id=user_id,
                message=request.message,
                session_id=session_id,
                stream_id=stream_id
            ):
                yield item

            # After the streaming is done, store the assistant's response
            # Note: This assumes that the full response is available in the langchain_service
            # or that you have a way to retrieve it after streaming
            try:
                # Retrieve the full response from wherever it's stored
                full_response = await langchain_service.get_full_response(stream_id)

                # Store the assistant's response in the session
                session_service.add_message(
                    session_id=session_id,
                    user_id=user_id,  # Assistant has no user_id
                    role="assistant",
                    message=full_response,
                    timestamp=datetime.utcnow(),
                    stream_id=stream_id,
                    mail_id = mail_id
                )
            except Exception as e:         #pylint: disable = broad-exception-caught
                # Log the error but don't stop the response
                app_logger.error(f"Error storing assistant response: {str(e)}")

        # Return the streaming response
        return StreamingResponse(
            response_generator(),
            media_type="text/event-stream"
        )
    except Exception as e:
        app_logger.error(f"Error in send_chat_message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat message: {str(e)}"
        ) from e

@router.post("/sessions/latest", summary="Get the latest session of a user")
async def get_latest_session(request: LatestSessionRequest):
    """
    Get the latest session of a user.

    Args:
        request (LatestSessionRequest): The request containing the user ID.

    Returns:
        dict: The latest session ID and last active time.
    """
    user_id = request.user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    # Fetch the latest session
    latest_session = session_service.get_latest_session(user_id)
    if not latest_session:
        raise HTTPException(status_code=404, detail="No sessions found for the user")

    return {
        "session_id": latest_session["session_id"],
        "last_active": latest_session["last_active"].isoformat()
    }

@router.get("/sessions/{session_id}/messages", response_model=Dict[str, Any])
async def get_session_messages(session_id: str):
    """
    Retrieve all messages from a specific chat session.

    Args:
        session_id (str): The unique identifier for the session.

    Returns:
        Dict[str, Any]: A dictionary containing session_id, messages, and last_active timestamp.
    """
    try:
        # Retrieve the session data
        session = session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Format the response
        response = {
            "session_id": session.session_id,
            "messages": [
                {
                    "role": message.role,
                    "message": message.message,
                    "stream_id": message.stream_id
                }
                for message in session.messages
            ],
            "last_active": session.last_active.isoformat()      #pylint: disable = no-member
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session messages: {str(e)}")     #pylint: disable=raise-missing-from


@router.post("/interrupt", status_code=200,
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
