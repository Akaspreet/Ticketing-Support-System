"""
Service for managing tickets in MongoDB.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from app.api.core.logging import app_logger
from app.api.db.mongodb import mongodb_client
from app.api.models.ticket import TicketStatus


class TicketService:
    """Service for CRUD operations on tickets."""

    def __init__(self):
        """Initialize with MongoDB connection."""
        self.collection = mongodb_client.tickets_db.tickets
        app_logger.info("Initialized Ticket Service")

    #pylint: disable=too-many-arguments
    #pylint: disable=too-many-positional-arguments
    def create_ticket(self, user_id: str, query: str,email: str, phone_number: str,
                      additional_info: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new ticket in the database.

        Args:
            user_id: ID of the user creating the ticket
            query: The question or query from the user
            additional_info: Any additional information

        Returns:
            The created ticket document
        """
        try:
            # Generate a ticket ID
            ticket_id = str(uuid.uuid4())

            # Create ticket document
            ticket = {
                "ticket_id": ticket_id,
                "user_id": user_id,
                "query": query,
                "email": email,
                "phone_number": phone_number,
                "additional_info": additional_info,
                "status": TicketStatus.OPEN.value,
                "created_at": datetime.utcnow(),
                "updated_at": None
            }

            # Insert into database
            result = self.collection.insert_one(ticket)

            if result.acknowledged:
                app_logger.info(f"Created new ticket with ID: {ticket_id}")
                return ticket

            app_logger.error("Failed to create ticket - not acknowledged")
            return None

        except Exception as e:          # pylint: disable=broad-exception-caught
            app_logger.error(f"Error creating ticket: {str(e)}")
            return None

    def get_ticket(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a ticket by ID.

        Args:
            ticket_id: ID of the ticket to retrieve

        Returns:
            Ticket document or None if not found
        """
        try:
            ticket = self.collection.find_one({"ticket_id": ticket_id})

            if ticket:
                app_logger.info(f"Retrieved ticket with ID: {ticket_id}")
            else:
                app_logger.warning(f"Ticket not found with ID: {ticket_id}")

            return ticket

        except Exception as e:              # pylint: disable=broad-exception-caught
            app_logger.error(f"Error retrieving ticket: {str(e)}")
            return None

    def get_tickets_by_user(self, user_id: str,
                            skip: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get all tickets for a user.

        Args:
            user_id: ID of the user
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return

        Returns:
            List of ticket documents
        """
        try:
            cursor = self.collection.find({"user_id": user_id}).sort(
                "created_at", -1  # Sort by creation date, newest first
            ).skip(skip).limit(limit)

            tickets = list(cursor)
            app_logger.info(f"Retrieved {len(tickets)} tickets for user: {user_id}")

            return tickets

        except Exception as e:              # pylint: disable=broad-exception-caught
            app_logger.error(f"Error retrieving tickets for user: {str(e)}")
            return []


    def update_ticket_status(self, ticket_id: str, status: TicketStatus) -> bool:
        """
        Update the status of a ticket.

        Args:
            ticket_id: ID of the ticket to update
            status: New status for the ticket (TicketStatus enum or string)

        Returns:
            True if update was successful, False otherwise
        """
        try:
            # If status is an enum, get its value; otherwise use it directly
            status_value = status.value if hasattr(status, 'value') else status

            result = self.collection.update_one(
                {"ticket_id": ticket_id},
                {
                    "$set": {
                        "status": status_value,
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            success = result.matched_count > 0

            if success:
                app_logger.info(f"Updated ticket {ticket_id} status to {status_value}")
            else:
                app_logger.warning(f"Failed to update ticket {ticket_id} - not found")

            return success

        except Exception as e:                  # pylint: disable=broad-exception-caught
            app_logger.error(f"Error updating ticket status: {str(e)}")
            return False


    def count_tickets(self, query: Dict[str, Any] = None) -> int:
        """
        Count tickets matching a query.

        Args:
            query: MongoDB query for filtering (optional)

        Returns:
            Number of matching tickets
        """
        try:
            query = query or {}
            count = self.collection.count_documents(query)
            return count

        except Exception as e:                          # pylint: disable=broad-exception-caught
            app_logger.error(f"Error counting tickets: {str(e)}")
            return 0


# Create a global instance
ticket_service = TicketService()
