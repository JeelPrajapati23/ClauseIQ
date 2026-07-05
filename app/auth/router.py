import logging
import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session
from app.auth.db import get_db
from app.auth.models import User, AuditLog
from app.auth.schemas import RegisterRequest, LoginRequest, UserResponse, ForgotPasswordRequest, ResetPasswordRequest, ChangePasswordRequest
from app.auth.utils import hash_password, verify_password, create_access_token, ENVIRONMENT
from app.auth.email_utils import send_reset_email
from app.auth.dependencies import get_current_user
from app.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_NAME = "access_token"
COOKIE_MAX_AGE = 8 * 3600
MAX_FAILED_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

# Frontend and backend are deployed on different origins (e.g. Vercel + Azure), so the
# cookie must be SameSite=None to survive cross-site fetches; SameSite=None requires
# Secure. Locally, frontend/backend share "localhost" as a site (ports are ignored for
# SameSite purposes) so Lax + non-Secure keeps working over plain HTTP.
COOKIE_SAMESITE = "none" if ENVIRONMENT == "production" else "lax"
COOKIE_SECURE = ENVIRONMENT == "production"


def _log(db: Session, action: str, user_id: str = None, detail: str = None, ip: str = None):
    db.add(AuditLog(user_id=user_id, action=action, detail=detail, ip_address=ip))
    db.commit()


@router.post("/register", status_code=201)
@limiter.limit("5/minute")
def register(body: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(409, "Email already registered")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    is_first_user = db.query(User).count() == 0
    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        role="admin" if is_first_user else "user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    _log(db, "register", user_id=user.id, detail=user.email,
         ip=request.client.host if request.client else None)
    if is_first_user:
        return {"message": "Admin account created. You can log in immediately.", "is_admin": True}
    return {"message": "Account created. You can log in now.", "is_admin": False}


@router.post("/login")
@limiter.limit("10/minute")
def login(body: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()

    if user and user.locked_until and user.locked_until > datetime.utcnow():
        remaining_min = int((user.locked_until - datetime.utcnow()).total_seconds() // 60) + 1
        raise HTTPException(
            403,
            f"Account temporarily locked after repeated failed login attempts. Try again in {remaining_min} minute(s).",
        )

    if not user or not verify_password(body.password, user.hashed_password):
        if user:
            user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
            if user.failed_login_attempts >= MAX_FAILED_LOGIN_ATTEMPTS:
                user.locked_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_MINUTES)
                user.failed_login_attempts = 0
                _log(db, "login_locked", user_id=user.id,
                     ip=request.client.host if request.client else None)
            db.commit()
        raise HTTPException(401, "Invalid email or password")
    if not user.is_active:
        raise HTTPException(403, "Account is deactivated. Contact an administrator.")

    user.failed_login_attempts = 0
    user.locked_until = None
    db.commit()

    token = create_access_token(user.id, user.role)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        max_age=COOKIE_MAX_AGE,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
    )
    _log(db, "login", user_id=user.id, ip=request.client.host if request.client else None)
    return {"message": "Logged in", "role": user.role, "email": user.email, "id": user.id}


@router.post("/logout")
def logout(response: Response, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    response.delete_cookie(COOKIE_NAME, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
    _log(db, "logout", user_id=current_user.id)
    return {"message": "Logged out"}


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.post("/forgot-password")
@limiter.limit("5/minute")
def forgot_password(body: ForgotPasswordRequest, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    # Always return the same message to prevent user enumeration
    if user and user.is_active:
        token = str(uuid.uuid4())
        user.reset_token = token
        user.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)
        db.commit()
        try:
            send_reset_email(user.email, token)
        except Exception:
            # Never let a broken SMTP config surface to the client (still returns the
            # generic message below to avoid user enumeration) — but log it, since this
            # used to fail completely silently and was impossible to debug.
            logger.exception("Failed to send password reset email to %s", user.email)
        _log(db, "forgot_password", user_id=user.id, ip=request.client.host if request.client else None)
    return {"message": "If that email is registered, a reset link has been sent."}


@router.post("/reset-password")
def reset_password(body: ResetPasswordRequest, request: Request, db: Session = Depends(get_db)):
    if len(body.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    user = db.query(User).filter(User.reset_token == body.token).first()
    if not user or not user.reset_token_expiry or user.reset_token_expiry < datetime.utcnow():
        raise HTTPException(400, "Reset link is invalid or has expired.")
    user.hashed_password = hash_password(body.new_password)
    user.reset_token = None
    user.reset_token_expiry = None
    db.commit()
    _log(db, "reset_password", user_id=user.id, ip=request.client.host if request.client else None)
    return {"message": "Password reset successfully. You can now log in."}


@router.post("/change-password")
def change_password(body: ChangePasswordRequest, request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if len(body.new_password) < 8:
        raise HTTPException(400, "New password must be at least 8 characters")
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(401, "Current password is incorrect")
    current_user.hashed_password = hash_password(body.new_password)
    db.commit()
    _log(db, "change_password", user_id=current_user.id, ip=request.client.host if request.client else None)
    return {"message": "Password changed successfully."}
