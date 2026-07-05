import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.auth.models import Base, User

# Default host port is 5433, not Postgres's usual 5432 — many dev machines already
# run a native Postgres service on 5432 (this one included), so this app's own
# container avoids that port entirely rather than fighting over it.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/clauseiq")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _migrate_users_table():
    """Idempotent column additions/removals for schema changes made after the
    initial create_all — safe to run on every startup, including brand-new
    databases."""
    with engine.connect() as conn:
        for ddl in (
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS reset_token VARCHAR",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS reset_token_expiry TIMESTAMP",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER DEFAULT 0",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMP",
            "ALTER TABLE users DROP COLUMN IF EXISTS is_verified",
            "ALTER TABLE users DROP COLUMN IF EXISTS verification_token",
        ):
            conn.execute(text(ddl))
        conn.commit()


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_users_table()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
