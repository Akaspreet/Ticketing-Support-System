"""
Pydantic models for user queries and search results.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class UserQuery(BaseModel):
    """User query model."""
    user_id: str = Field(..., description="Unique identifier for the user")
    query: str = Field(..., description="The question or query from the user")
    session_id: Optional[str] = Field(None, description="Unique identifier for the session")

class SearchResult(BaseModel):
    """Model for a single search result."""
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The answer to the question")
    category: Optional[str] = Field(None, description="Category of the question/answer")
    similarity_score: float = Field(..., description="Similarity score between query and result")


class SearchResponse(BaseModel):
    """Model for the complete search response."""
    query: str = Field(..., description="The original user query")
    results: List[SearchResult] = Field(default_factory=list, description="List of search results")
    found: bool = Field(False, description="Whether relevant results were found")

    @validator('found', always=True)
    def set_found_status(cls, v, values):                   # pylint: disable=no-self-argument, unused-argument
        """Set found status based on whether results exist."""
        return bool(values.get('results', []))


class ChatHistoryEntry(BaseModel):
    """Model for storing chat history in MongoDB."""
    user_id: str = Field(..., description="Unique identifier for the user")
    query: str = Field(..., description="User's question or query")
    response: Optional[Dict[str, Any]] = Field(None, description="System response")
    results_found: bool = Field(False, description="Whether relevant results were found")
    timestamp: datetime = Field(default_factory=datetime.utcnow,
                                description="When the query was made")


class ChatResponse(BaseModel):
    """Chat response model."""
    query: str
    results: List[SearchResult]
    found: bool
    chat_response: str
    error: Optional[str] = None


class StreamToken(BaseModel):
    """Model for streaming token responses."""
    token: str = Field(..., description="A chunk of the streaming response")
    error: Optional[str] = Field(None, description="Error message, if any")


class InterruptRequest(BaseModel):
    """Model for a request to interrupt a streaming response."""
    stream_id: str = Field(..., description="ID of the stream to interrupt")




# Helper for default empty list to avoid mutable default argument issue
def default_list():
    """
    default list
    """
    return []
