from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from retrieval import BM25, reciprocal_rank_fusion

from .database import get_session_factory
from .table import EMBEDDING_DIM, EpisodicMemory, MemorySetting, ProceduralMemory, SemanticMemory

# Cosine distance: 0 = identical, 2 = opposite. Calibrated with multilingual-e5-small, which packs
# everything close together (tests/test_memory_eval.py checks these against its pairs):
# paraphrases <= 0.034, a fact and its negation ~0.002 (which is why add_fact overwrites),
# same subject with a new value 0.05-0.14, different subjects >= 0.14; same task <= 0.083,
# different tasks >= 0.18; relevant question/fact 0.13-0.24 (Vietnamese included).
FACT_DUPLICATE_DISTANCE = 0.045
PROCEDURE_DUPLICATE_DISTANCE = 0.12
# Recall cutoff: 40/40 eval cases at 0.20 with 0.3 unrequested facts a question (0.22: 0.6).
RECALL_DISTANCE = 0.20
# Facts offered to the extractor as "known", so it can mark them superseded. Kept tight: at 0.24
# a Vietnamese turn pulled in unrelated facts (0.21-0.23) and qwen3:8b then missed the new fact.
# Outdated facts the question does not reach ("Mình chuyển sang dùng Vim rồi" -> "The user's
# code editor is VS Code" is 0.229) are caught by the review of facts near the new one
# (manager.REVIEW_DISTANCE, fact to fact 0.054).
KNOWN_FACT_DISTANCE = 0.18

EMBEDDING_MODEL_KEY = "embedding_model"

# Recalled on every turn whatever the question. About 12 tokens a fact, so the cap keeps the
# block near 150 tokens of the MAX_HISTORY_TOKENS budget.
PROFILE_CATEGORIES = ("user_profile", "user_preference", "environment")
PROFILE_LIMIT = 12


@dataclass
class MemoryHits:
    episodes: list[EpisodicMemory] = field(default_factory=list)
    facts: list[SemanticMemory] = field(default_factory=list)
    procedures: list[ProceduralMemory] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.episodes or self.facts or self.procedures)


def _now() -> datetime:
    return datetime.now(UTC)


def _episode_text(event: str, task: str | None) -> str:
    """What an episode is embedded from (also when re-embedding)."""
    return " ".join(p for p in (event, task) if p)


def rank_facts(
    query: str, facts: list[tuple[int, str]], by_meaning: list[int], limit: int
) -> list[int]:
    """Fact ids, best first: the vector ranking `by_meaning` fused with BM25 over `facts`.

    Words catch what embeddings blur: ports, names, model tags ("15432", "Miu", "qwen3:8b").
    A fact matched neither way is left out."""
    bm25 = BM25([text for _, text in facts])
    by_words = [facts[i][0] for i in bm25.top(query, limit * 3)]

    return [fact_id for fact_id, _ in reciprocal_rank_fusion([by_meaning, by_words])[:limit]]


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
            embedding=self._embed(_episode_text(event, task)),
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

    # ---- embedding model -----------------------------------------------

    def embedding_model(self) -> str | None:
        """Name of the model the stored rows were embedded with (None before it was recorded)."""
        with self._session_factory() as session:
            setting = session.get(MemorySetting, EMBEDDING_MODEL_KEY)
            return setting.value if setting else None

    def ensure_embedding_model(self, name: str) -> int:
        """Re-embed every stored row if they were embedded by another model; record `name`.

        Vectors from two models are not comparable, so a model change without this would make
        every search silently miss. Returns the number of rows re-embedded."""
        if self.embedding_model() == name:
            return 0

        count = 0
        with self._session_factory() as session, session.begin():
            for episode in session.scalars(select(EpisodicMemory)):
                episode.embedding = self._embed(_episode_text(episode.event, episode.task))
                count += 1
            for fact in session.scalars(select(SemanticMemory)):
                fact.embedding = self._embed(fact.fact)
                count += 1
            for procedure in session.scalars(select(ProceduralMemory)):
                procedure.embedding = self._embed(procedure.task)
                count += 1
            session.merge(MemorySetting(key=EMBEDDING_MODEL_KEY, value=name))

        return count

    # ---- reads ----------------------------------------------------------

    def profile_facts(self, limit: int = PROFILE_LIMIT) -> list[SemanticMemory]:
        """Facts about the user (PROFILE_CATEGORIES), most recently updated first."""
        stmt = (
            select(SemanticMemory)
            .where(SemanticMemory.category.in_(PROFILE_CATEGORIES))
            .order_by(SemanticMemory.updated_at.desc(), SemanticMemory.id.desc())
            .limit(limit)
        )

        with self._session_factory() as session:
            return list(session.scalars(stmt))

    def search(
        self,
        query: str,
        limit: int = 3,
        max_distance: float = RECALL_DISTANCE,
        keywords: bool = True,
    ) -> MemoryHits:
        """Return the closest memories of each type (the query is embedded once).

        With `keywords`, facts are also matched by words (`rank_facts`); `max_distance` then
        limits only the vector side."""
        vector = self._embed(query)

        def top(session: Session, model, count: int = limit):
            distance = model.embedding.cosine_distance(vector)
            stmt = (
                select(model)
                .where(distance <= max_distance)
                .order_by(distance)
                .limit(count)
            )
            return list(session.scalars(stmt))

        def facts(session: Session) -> list[SemanticMemory]:
            if not keywords:
                return top(session, SemanticMemory)

            by_meaning = [f.id for f in top(session, SemanticMemory, limit * 3)]
            # The facts table stays small (one short row per durable fact), so BM25 is built
            # over all of it per search instead of keeping an index in step with writes.
            texts = [(row.id, row.fact) for row in session.execute(select(SemanticMemory.id, SemanticMemory.fact))]
            ids = rank_facts(query, texts, by_meaning, limit)
            rows = {f.id: f for f in session.scalars(select(SemanticMemory).where(SemanticMemory.id.in_(ids)))}
            return [rows[i] for i in ids if i in rows]

        with self._session_factory() as session:
            return MemoryHits(
                episodes=top(session, EpisodicMemory),
                facts=facts(session),
                procedures=top(session, ProceduralMemory),
            )
