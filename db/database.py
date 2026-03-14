"""
db/database.py
──────────────
WHY THIS FILE EXISTS:
  Sets up the SQLAlchemy engine and session factory.
  All database connections in the app flow through here.
  Using SQLAlchemy ORM means we write Python — not raw SQL —
  and switching databases later (e.g. SQLite → Postgres) is trivial.

HOW IT WORKS:
  1. Reads DATABASE_URL from .env
  2. Creates an engine (the actual DB connection pool)
  3. Creates a SessionLocal factory (each request gets its own session)
  4. get_db() is a FastAPI dependency — automatically opens and
     closes a DB session per request (with try/finally for safety)
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_vdd_dev.db")

# For SQLite (dev), add connect_args to allow multi-threaded access
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()