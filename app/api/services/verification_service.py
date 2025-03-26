"""
Service module for verifying email and phone number with OTP.
"""

import random
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import HTTPException
from app.api.core.logging import app_logger

class VerificationService:
    """Service for verifying email and phone number with OTP."""

    def __init__(self):
        self.otp_storage = {}
        self.otp_validity = 300

    def send_otp(self, contact: str, contact_type: str):
        """Send OTP to email or phone number."""
        otp = random.randint(100000, 999999)

        if contact_type == "email":
            # Logic to send OTP to email
            app_logger.info(f"Sending OTP {otp} to email: {contact}")
            self.send_email_otp(contact, otp)

        elif contact_type == "phone":
            # Logic to send OTP to phone number
            app_logger.info(f"Sending OTP {otp} to phone number: {contact}")
            if not contact.startswith("+91"):
                contact = "+91" + contact
            self.send_sms_otp(contact, otp)

        else:
            raise HTTPException(status_code=400, detail="Invalid contact type")

        self.otp_storage[contact] = {
            "otp": otp,
            "timestamp": time.time()
        }


    def send_email_otp(self, email: str, otp: int):
        """Send OTP via email using SMTP."""
        sender_email = "akshpreet2002@gmail.com"  # Replace with your email
        sender_password = "crwx leor tter kjes"  # Use an app password (not normal password)

        subject = "Your OTP Code"
        body = f"""Your OTP code is {otp}. Please use it to verify your email.
        This OTP will expire in 5 minutes."""

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())
            server.quit()
            print(f"Email sent successfully to {email}")
        except Exception as e:
            print(f"Error sending email: {e}")
            app_logger.error(f"Failed to send email to {email}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"""Failed to send
                                email: {str(e)}""") from e

    def send_sms_otp(self, phone: str, otp: int):
        """Simulate sending OTP via SMS (replace with Twilio or another SMS API)."""
        # message = f"Your OTP code is {otp}. This code will expire in 5 minutes."
        print(f"Simulating sending OTP {otp} to phone number: {phone}")
        app_logger.info(f"Simulated SMS OTP {otp} to {phone}")

    def verify_otp(self, contact: str, otp: int):
        """Verify the OTP provided by the user."""
        # For phone numbers, ensure consistent format
        if contact.isdigit() or (len(contact) > 10 and contact[1:].isdigit()):
            if not contact.startswith("+91"):
                contact = "+91" + contact

        if contact not in self.otp_storage:
            app_logger.warning(f"OTP verification attempted for {contact} but no OTP was sent")
            raise HTTPException(status_code=400, detail="OTP not sent to this contact")

        otp_data = self.otp_storage[contact]
        current_time = time.time()

        # Check if OTP has expired (current time - timestamp > validity period)
        if current_time - otp_data["timestamp"] > self.otp_validity:
            # Remove expired OTP
            del self.otp_storage[contact]
            app_logger.warning(f"Expired OTP verification attempted for {contact}")
            raise HTTPException(status_code=400, detail="""OTP has expired.
                                Please request a new one.""")

        # Check if OTP matches
        if otp_data["otp"] == otp:
            # Remove verified OTP
            del self.otp_storage[contact]
            app_logger.info(f"OTP verified successfully for {contact}")
            return True

        app_logger.warning(f"Invalid OTP provided for {contact}")
        raise HTTPException(status_code=400, detail="Invalid OTP")

    def clean_expired_otps(self):
        """Clean up expired OTPs from storage to prevent memory leaks."""
        current_time = time.time()
        expired_contacts = []

        for contact, otp_data in self.otp_storage.items():
            if current_time - otp_data["timestamp"] > self.otp_validity:
                expired_contacts.append(contact)

        for contact in expired_contacts:
            del self.otp_storage[contact]

        if expired_contacts:
            app_logger.info(f"Cleaned {len(expired_contacts)} expired OTPs")



# Create a global instance
verification_service = VerificationService()
