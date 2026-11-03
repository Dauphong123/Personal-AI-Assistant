from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from .database import get_session_factory
from .table import EMBEDDING_DIM, EpisodicMemory, ProceduralMemory, SemanticMemory

# Cosine distance: 0 = identical, 2 = opposite. Calibrated with bge-small-en-v1.5:
# paraphrases are ~0.03, relevant query/fact pairs 0.2-0.3, unrelated text ~0.45-0.5.
# A fact and its negation also land near 0 (~0.004), which is why add_fact overwrites.
FACT_DUPLICATE_DISTANCE = 0.08
PROCEDURE_DUPLICATE_DISTANCE = 0.2


@dataclass
class MemoryHits:
    episodes: list[EpisodicMemory] = field(default_factory=list)
    facts: list[SemanticMemory] = field(default_factory=list)
    procedures: list[ProceduralMemory] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.episodes or self.facts or self.procedures)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _nearest(session: Session, model, vector: list[float], max_distance: float):
    distance = model.embedding.cosine_distance(vector)
    stmt = select(model).where(distance <= max_distance).order_by(distance).limit(1)
    return session.scalars(stmt).first()


class MemoryStore:
    """Persistence and vector search for the three long-term memory types.

    `embeddings` is any object with `embed_query(text) -> list[float]`
    (e.g. a LangChain `HuggingFaceEmbeddings`).
    """

    def __init__(self, embeddings, session_factory: sessionmaker[Session] | None = None):
        self._embeddings = embeddings
        self._session_factory = session_factory or get_session_factory()

    def _embed(self, text: str) -> list[float]:
        vector = list(self._embeddings.embed_query(text))

        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding has {len(vector)} dimensions, tables expect {EMBEDDING_DIM}."
            )

        return vector

    # ---- writes ---------------------------------------------------------

    def add_episode(
        self,
        event: str,
        task: str | None = None,
        outcome: str | None = None,
        importance: float = 0.5,
        thread_id: str | None = None,
    ) -> EpisodicMemory:
        row = EpisodicMemory(
            thread_id=thread_id,
            event=event,
            task=task,
            outcome=outcome,
            importance=importance,
            embedding=self._embed(" ".join(p for p in (event, task) if p)),
        )

        with self._session_factory() as session, session.begin():
            session.add(row)

        return row

    def add_fact(
        self,
        fact: str,
        category: str | None = None,
        confidence: float = 1.0,
    ) -> tuple[SemanticMemory, bool]:
        """Store a fact. Returns (row, created).

        A near-duplicate is updated in place with the NEW wording: a paraphrase is harmless to
        replace, and a contradiction ("uses tabs" -> "uses spaces") embeds almost identically,
        so keeping the old text would silently discard the correction."""
        vector = self._embed(fact)

        with self._session_factory() as session, session.begin():
            existing = _nearest(session, SemanticMemory, vector, FACT_DUPLICATE_DISTANCE)

            if existing is not None:
                existing.fact = fact
                existing.embedding = vector
                existing.category = category or existing.category
                existing.updated_at = _now()
                existing.confidence = max(existing.confidence, confidence)
                return existing, False

            row = SemanticMemory(
                fact=fact, category=category, confidence=confidence, embedding=vector
            )
            session.add(row)

        return row, True

    def add_procedure(
        self,
        task: str,
        steps: list[str],
        notes: str | None = None,
    ) -> tuple[ProceduralMemory, bool]:
        """Store a procedure. A similar task gets its steps replaced and success_count bumped."""
        vector = self._embed(task)

        with self._session_factory() as session, session.begin():
            existing = _nearest(
                session, ProceduralMemory, vector, PROCEDURE_DUPLICATE_DISTANCE
            )

            if existing is not None:
                existing.steps = steps
                existing.notes = notes or existing.notes
                existing.success_count += 1
                existing.last_used_at = _now()
                return existing, False

            row = ProceduralMemory(task=task, steps=steps, notes=notes, embedding=vector)
            session.add(row)

        return row, True

    def delete_facts(self, ids: list[int]) -> int:
        if not ids:
            return 0

        with self._session_factory() as session, session.begin():
            result = session.execute(delete(SemanticMemory).where(SemanticMemory.id.in_(ids)))

        return result.rowcount

    # ---- reads ----------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 3,
        max_distance: float = 0.4,
    ) -> MemoryHits:
        """Return the closest memories of each type (the query is embedded once)."""
        vector = self._embed(query)

        def top(session: Session, model):
            distance = model.embedding.cosine_distance(vector)
            stmt = (
                select(model)
                .where(distance <= max_distance)
                .order_by(distance)
                .limit(limit)
            )
            return list(session.scalars(stmt))

        with self._session_factory() as session:
            return MemoryHits(
                episodes=top(session, EpisodicMemory),
                facts=top(session, SemanticMemory),
                procedures=top(session, ProceduralMemory),
            )
