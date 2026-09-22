import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.auth.models import Base, User

# Default host port is 5433, not Postgres's usual 5432, to avoid clashing with a
# native Postgres service some dev machines already run on 5432.
# pool_pre_ping issues a lightweight liveness check before handing out a pooled
# connection. Managed Postgres providers often silently close idle connections
# server-side; without this, the pool hands back a dead connection and every
# query on it fails instead of transparently reconnecting.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/clauseiq")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
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
