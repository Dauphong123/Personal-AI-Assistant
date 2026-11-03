import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()


class Base(DeclarativeBase):
    pass


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL") or os.environ.get("DB_URL")

    if not url:
        raise RuntimeError("Set DATABASE_URL (or DB_URL) in .env to use memories.")

    # A bare "postgresql://" makes SQLAlchemy pick psycopg2, but this project
    # installs psycopg 3.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_database_url())


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    # expire_on_commit=False lets callers read returned rows after the session closes.
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Enable pgvector and create the memory tables if they don't exist."""
    from . import table  # noqa: F401  (registers the models on Base)

    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)
