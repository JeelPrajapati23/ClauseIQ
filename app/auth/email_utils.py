import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@clauseiq.local")
APP_FRONTEND_URL = os.getenv("APP_FRONTEND_URL", "http://localhost:5173")


def send_reset_email(to_email: str, token: str) -> None:
    reset_url = f"{APP_FRONTEND_URL}?reset_token={token}"
    if not SMTP_HOST or not SMTP_USER:
        print(f"\n{'='*60}\n[PASSWORD RESET] {to_email}\nClick to reset: {reset_url}\n{'='*60}\n", flush=True)
        logger.info("Dev mode: password reset link printed to console for %s", to_email)
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Reset your ClauseIQ password"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    msg.attach(MIMEText(
        f"<html><body style='font-family:sans-serif;padding:32px'>"
        f"<h2>Reset your password</h2>"
        f"<p>Click the button below to set a new password. This link expires in 1 hour.</p>"
        f"<a href='{reset_url}' style='display:inline-block;padding:12px 24px;background:#c6f24a;color:#0c1003;border-radius:8px;text-decoration:none;font-weight:600'>Reset Password</a>"
        f"<p style='margin-top:24px;color:#666'>If you did not request this, ignore this email — your password will not change.</p>"
        f"</body></html>",
        "html",
    ))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(FROM_EMAIL, to_email, msg.as_string())
