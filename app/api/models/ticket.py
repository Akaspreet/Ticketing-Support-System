"""
Pydantic models for ticket handling.
"""
from typing import Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, EmailStr


class TicketStatus(str, Enum):
    """Enum for ticket status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketCreate(BaseModel):
    """Model for creating a new ticket."""
    user_id: str = Field(..., description="Unique identifier for the user")
    query: str = Field(..., description="The question that needs an answer")
    email: EmailStr = Field(..., description="User email address")
    phone_number: str = Field(..., description="User phone number")
    additional_info: Optional[str] = Field(None, description="Any additional information")


class Ticket(BaseModel):
    """Full ticket model including system fields."""
    ticket_id: str = Field(..., description="Unique identifier for the ticket")
    user_id: str = Field(..., description="Unique identifier for the user")
    query: str = Field(..., description="The question that needs an answer")
    email: EmailStr = Field(..., description="User email address")
    phone_number: str = Field(..., description="User phone number")
    additional_info: Optional[str] = Field(None, description="Any additional information")
    status: TicketStatus = Field(default=TicketStatus.OPEN,
                                 description="Current status of the ticket")
    created_at: datetime = Field(default_factory=datetime.utcnow,
                                 description="When the ticket was created")
    updated_at: Optional[datetime] = Field(None, description="When the ticket was last updated")

class TicketResponse(BaseModel):
    """Response model for ticket operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    ticket: Optional[Ticket] = Field(None, description="The ticket data, if applicable")
    message: Optional[str] = Field(None, description="A message about the operation")


class TicketListResponse(BaseModel):
    """Response model for listing tickets."""
    tickets: List[Ticket] = Field(default_factory=list, description="List of tickets")
    total: int = Field(..., description="Total number of tickets")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(10, description="Number of items per page")
