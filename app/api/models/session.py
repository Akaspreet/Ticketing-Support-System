"""
Pydantic models for handling sessions and messages.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4

class Message(BaseModel):
    """Model for a user message within a session."""
    # user_id: Optional[str] = Field(None, description="Unique identifier for the user")
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    message: str = Field(..., description="The message content")
    token: Optional[int] = Field(None, description="No of token for the message")
    stream_id: Optional[str] = Field(None, description="Stream ID for the message")
    tool_call: Optional[dict] = Field(None, description="Tool call data if applicable")
    # tool_call: Optional[dict] = Field(None, description="Tool call data if applicable")

    # timestamp: datetime = Field(
    #     default_factory=datetime.utcnow, description="Timestamp of the message"
    # )

class Session(BaseModel):
    """Model for a chat session."""
    session_id: str = Field(..., description="Unique identifier for the session")
    user_id: str = Field(..., description="Unique identifier for the user")
    mail_id: Optional[str] = Field(None, description="Mail ID of the User")
    # phone: Optional[str] = Field(None, description="Phone number of the user")
    messages: List[Message] = Field(
        default_factory=list, description="List of messages in the session"
    )
    last_active: datetime = Field(
        default_factory=datetime.utcnow, description="Last active time of the session"
    )

class ChatMessageRequest(BaseModel):
    """Request model for sending a chat message."""
    session_id: Optional[str] = Field(None, description="Unique identifier for the session")
    user_id: Optional[str] = Field(None, description="Unique identifier for the user")
    role: str = Field("user", description="Role of the message sender")
    message: str = Field(..., description="The message content")
    time_stamp: Optional[datetime] = Field(None, description="Timestamp of the message")

class ChatMessageResponseItem(BaseModel):
    """Response model for a single item in the chat message response stream."""
    type: str = Field(..., description="Type of the response item")
    value: dict = Field(..., description="Value of the response item")

class InterruptRequest(BaseModel):
    """Model for a request to interrupt a streaming response."""
    stream_id: str = Field(..., description="ID of the stream to interrupt")
    session_id: str = Field(..., description="Session ID for the chat")