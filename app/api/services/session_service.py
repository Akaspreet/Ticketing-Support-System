"""
Service module for handling chat sessions and messages.
"""

from datetime import datetime, timedelta
from uuid import uuid4
from typing import Optional
from app.api.db.mongodb import mongodb_client
from app.api.models.session import Session, Message
from app.api.core.config import settings

class SessionService:
    """
    Service class for managing chat sessions and messages.
    """
    def __init__(self):
        """
        Initialize the SessionService with a MongoDB collection.
        """
        self.collection = mongodb_client.sessions_db.sessions

    def create_session(self, user_id, mail_id) -> str:
        """
        Create a new chat session.

        Returns:
            str: The unique identifier for the newly created session.
        """
        session_id = str(uuid4())
        session = Session(session_id=session_id, user_id=user_id, mail_id=mail_id)
        self.collection.insert_one(session.dict())
        return session_id

    #pylint: disable = too-many-arguments
    #pylint: disable = too-many-positional-arguments
    def add_message(
        self,
        session_id: str,
        user_id: Optional[str],
        role: str,
        message: str,
        timestamp: Optional[datetime] = None,
        stream_id: Optional[str] = None,
        mail_id: Optional[str] = None,
        token: Optional[int] = None,
        tool_call: Optional[dict] = None
    ):
        """
        Add a message to an existing chat session.

        Args:
            session_id (str): The unique identifier for the session.
            user_id (Optional[str]): The unique identifier for the user.
            role (str): The role of the message sender (user or assistant).
            message (str): The message content.
            timestamp (Optional[datetime]): The timestamp of the message.
            stream_id (Optional[str]): The stream ID for the message.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        message_obj = Message(
            role=role,
            message=message,
            stream_id=stream_id,
            token = token,
            tool_call = tool_call
        )

        # Check if the session exists, create it if not
        session = self.collection.find_one({"session_id": session_id})
        if not session:
            new_session = Session(session_id=session_id,
                                  user_id=user_id,
                                  mail_id=mail_id,
                                  stream_id=stream_id)
            self.collection.insert_one(new_session.dict())

        self.collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "user_id": user_id,
                    "mail_id": mail_id,
                    "last_active": timestamp
                },
                "$push": {"messages": message_obj.dict()}
            }
        )

    def get_messages(self, session_id: str):
        """
        Retrieve all messages from a specific chat session.

        Args:
            session_id (str): The unique identifier for the session.

        Returns:
            Session: The session object containing all messages, or None if not found.
        """
        session = self.collection.find_one({"session_id": session_id})
        if session:
            return Session(**session)
        return None

    def delete_inactive_sessions(self, hours: int = settings.SESSION_ACTIVE_HOURS):
        """
        Delete chat sessions that have been inactive for a specified number of hours.

        Args:
            hours (int, optional): The number of hours of inactivity before deletion.
            Default is 48 hours.
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        self.collection.delete_many({"last_active": {"$lt": cutoff_time}})

    def get_session(self, session_id: str):
        """
        Retrieve a specific chat session.

        Args:
            session_id (str): The unique identifier for the session.

        Returns:
            Session: The session object, or None if not found.
        """
        session = self.collection.find_one({"session_id": session_id})
        if session:
            return Session(**session)
        return None

    def get_latest_session(self, user_id: str):
        """
        Get the latest session for a given user.

        Args:
            user_id (str): The unique identifier for the user.

        Returns:
            dict: The latest session document, or None if not found.
        """
        latest_session = self.collection.find_one(
            {"user_id": user_id},
            sort=[("last_active", -1)]
        )
        return latest_session

# Create an instance of SessionService
session_service = SessionService()
