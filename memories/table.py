from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base

# Output size of BAAI/bge-small-en-v1.5, the embedding model used by rag/rag.py.
EMBEDDING_DIM = 384
CATEGORY_LENGTH = 64  # width of the fact category column


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _embedding_index(table: str) -> Index:
    return Index(
        f"ix_{table}_embedding",
        "embedding",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


class EpisodicMemory(Base):
    """What happened: an event, the task it belonged to and how it ended."""

    __tablename__ = "episodic_memories"
    __table_args__ = (_embedding_index("episodic_memories"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    event: Mapped[str] = mapped_column(Text, nullable=False)
    task: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))


class SemanticMemory(Base):
    """Durable facts: user preferences, project facts, general knowledge."""

    __tablename__ = "semantic_memories"
    __table_args__ = (_embedding_index("semantic_memories"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fact: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(CATEGORY_LENGTH), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))


class ProceduralMemory(Base):
    """How to do a task: ordered steps that worked before."""

    __tablename__ = "procedural_memories"
    __table_args__ = (_embedding_index("procedural_memories"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    steps: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    success_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
