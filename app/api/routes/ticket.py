"""
API routes for ticket management.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import EmailStr
from app.api.models.ticket import TicketCreate, TicketResponse, TicketListResponse, TicketStatus
from app.api.services.ticket_service import ticket_service
from app.api.services.verification_service import verification_service
from app.api.core.logging import app_logger

router = APIRouter()

@router.post("/create", response_model=TicketResponse, status_code=201,
             summary="Create a new support ticket",
             description="Create a new ticket for a query that requires human assistance",
             response_description="Newly created ticket details")
async def create_ticket(ticket: TicketCreate,
                        email: Optional[EmailStr] = None,
                        phone_number: Optional[str] = None,
                        email_otp: Optional[int] = None,
                        phone_otp: Optional[int] = None):
    """
    Create a new support ticket.
    """
    try:
        app_logger.info(f"Received ticket creation request from user: {ticket.user_id}")

        if not ticket.email or not ticket.phone_number:
            if not email:
                email = input("Enter your email: ")
                verification_service.send_otp(ticket.email, "email")
                email_otp = int(input("Enter the OTP sent to your email: "))
                verification_service.verify_otp(ticket.email, email_otp)
            if not phone_number:
                phone_number = input("Enter your phone number: ")
                verification_service.send_otp(ticket.phone_number, "phone")
                phone_otp = int(input("Enter the OTP sent to your phone number: "))
                verification_service.verify_otp(ticket.phone_number, phone_otp)

        created_ticket = ticket_service.create_ticket(
            user_id=ticket.user_id,
            query=ticket.query,
            email=ticket.email,
            phone_number=ticket.phone_number,
            additional_info=ticket.additional_info
        )
        if not created_ticket:
            raise HTTPException(status_code=500, detail="Failed to create ticket")
        response = TicketResponse(
            success=True,
            ticket=created_ticket,
            message="Ticket created successfully"
        )
        return response
    except Exception as e:
        app_logger.error(f"Error creating ticket: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating ticket: {str(e)}") from e

@router.get("/get/{ticket_id}", response_model=TicketResponse, status_code=200,
            summary="Get ticket details",
            description="Retrieve details for a specific ticket by ID",
            response_description="Complete ticket information")
async def get_ticket(
    ticket_id: str = Path(..., description="ID of the ticket to retrieve")
):
    """
    Retrieve details for a specific ticket by ID.
    """
    try:
        app_logger.info(f"Received request to get ticket: {ticket_id}")
        ticket = ticket_service.get_ticket(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail=f"Ticket with ID {ticket_id} not found")
        response = TicketResponse(
            success=True,
            ticket=ticket,
            message="Ticket retrieved successfully"
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error retrieving ticket: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving ticket: {str(e)}") from e

@router.get("/list", response_model=TicketListResponse, status_code=200,
            summary="List user tickets",
            description="Get a paginated list of tickets for a specific user",
            response_description="List of tickets with pagination information")
async def list_tickets(
    user_id: str = Query(..., description="User ID to filter tickets"),
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(10, description="Items per page", ge=1, le=100),
    status: Optional[TicketStatus] = Query(None, description="Filter by ticket status")
):
    """
    Get a paginated list of tickets for a specific user.
    """
    try:
        app_logger.info(f"Received request to list tickets for user: {user_id}")
        skip = (page - 1) * page_size
        filters = {"user_id": user_id}
        if status:
            filters["status"] = status.value
        tickets = ticket_service.get_tickets_by_user(
            user_id=user_id, skip=skip, limit=page_size
        )
        total = ticket_service.count_tickets(filters)
        response = TicketListResponse(
            tickets=tickets,
            total=total,
            page=page,
            page_size=page_size
        )
        return response
    except Exception as e:
        app_logger.error(f"Error listing tickets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing tickets: {str(e)}") from e

@router.put("/update/{ticket_id}/status", response_model=TicketResponse, status_code=200,
            summary="Update ticket status",
            description="Update the status of an existing ticket",
            response_description="Updated ticket information")
async def update_ticket_status(
    status: TicketStatus = Query(..., description="New status for the ticket"),
    ticket_id: str = Path(..., description="ID of the ticket to update")
):
    """
    Update the status of an existing ticket.
    """
    try:
        app_logger.info(f"Received request to update ticket {ticket_id} status to {status}")
        success = ticket_service.update_ticket_status(ticket_id, status)
        if not success:
            raise HTTPException(status_code=404, detail=f"Ticket with ID {ticket_id} not found")
        updated_ticket = ticket_service.get_ticket(ticket_id)
        response = TicketResponse(
            success=True,
            ticket=updated_ticket,
            message=f"Ticket status updated to {status}"
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error updating ticket status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"""Error
                            updating ticket status: {str(e)}""") from e
