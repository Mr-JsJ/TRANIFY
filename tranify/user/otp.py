import pyotp
import smtplib
from email.message import EmailMessage

# Email Configuration
SENDER_EMAIL = "tranifyapp@gmail.com"  # Replace with your email
SENDER_PASSWORD = "oool afxm cerl avjl"  # Replace with your email password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # Use 587 for TLS, 465 for SSL


def generate_otp():
    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    return totp.now()


def send_otp(email, otp):
    """Sends OTP to user's email using SMTP."""
    msg = EmailMessage()
    msg.set_content(f"Your OTP for Trainify registration is: {otp}")
    msg["Subject"] = "Trainify OTP Verification"
    msg["From"] = SENDER_EMAIL
    msg["To"] = email

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False
