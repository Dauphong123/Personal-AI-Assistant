# The manager tests pass small duck-typed fakes (stores, extractors, rows) where the code
# expects MemoryStore / MemoryExtractor / ORM rows; the tests themselves check the behavior.
# pyright: reportArgumentType=false

import hashlib
import os
import random
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from sqlalchemy import delete, select
from sqlalchemy.exc import DBAPIError

from memories.extractor import (
    Episode,
    Fact,
    MemoryExtraction,
    MemoryExtractor,
    Procedure,
    Superseded,
    format_turn,
    is_trivial_turn,
)
from memories.manager import REVIEW_DISTANCE, MemoryManager, format_memories
from memories.store import MemoryHits, MemoryStore
from memories.table import EMBEDDING_DIM


class FakeEmbeddings:
    """Deterministic: same text -> same vector, different text -> unrelated vector."""

    def embed_query(self, text):
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        rng = random.Random(seed)
        return [rng.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]


def finished_turn():
    return [
        HumanMessage(content="add a hello file", id="1"),
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"path": "hello.txt"}, "id": "c1"}]),
        ToolMessage(content="Successful wrote hello.txt", tool_call_id="c1"),
        AIMessage(content="Created hello.txt"),
    ]


class FormatTurnTests(unittest.TestCase):
    def test_includes_user_tool_calls_and_answer(self):
        text = format_turn(finished_turn())
        self.assertIn("User: add a hello file", text)
        self.assertIn("write_file", text)
        self.assertIn("Tool result: Successful wrote", text)
        self.assertIn("Assistant: Created hello.txt", text)

    def test_only_latest_turn(self):
        messages = [HumanMessage(content="old question"), AIMessage(content="old answer")] + finished_turn()
        self.assertNotIn("old question", format_turn(messages))

    def test_unfinished_or_empty_turn_is_skipped(self):
        self.assertEqual(format_turn([]), "")
        self.assertEqual(format_turn([AIMessage(content="no human")]), "")
        pending = finished_turn()[:2]  # ends on an AI message with tool calls
        self.assertEqual(format_turn(pending), "")

    def test_long_tool_output_is_clipped(self):
        messages = finished_turn()
        messages[2] = ToolMessage(content="x" * 5000, tool_call_id="c1")
        self.assertLess(len(format_turn(messages)), 1000)


class ExtractorTests(unittest.TestCase):
    def test_returns_structured_result(self):
        expected = MemoryExtraction(facts=[Fact(fact="User prefers tabs")])
        model = SimpleNamespace(with_structured_output=lambda schema: RunnableLambda(lambda _: expected))
        self.assertEqual(MemoryExtractor(model).extract(finished_turn()), expected)

    def test_skips_model_call_when_nothing_to_learn(self):
        def boom(_):
            raise AssertionError("model should not be called")

        model = SimpleNamespace(with_structured_output=lambda schema: RunnableLambda(boom))
        self.assertEqual(MemoryExtractor(model).extract([]), MemoryExtraction())

    def test_runs_the_model_at_temperature_zero(self):
        from langchain_ollama import ChatOllama

        chat = ChatOllama(model="qwen3:8b", temperature=0.7)
        used = []
        real = ChatOllama.with_structured_output

        def spy(self, schema, **kwargs):
            used.append(self.temperature)
            return real(self, schema, **kwargs)

        with patch.object(ChatOllama, "with_structured_output", spy):
            MemoryExtractor(chat)

        self.assertEqual(used, [0, 0])
        self.assertEqual(chat.temperature, 0.7)  # the chat model itself is untouched


class SecretFilterTests(unittest.TestCase):
    def test_key_like_values_are_secret(self):
        from memories.manager import looks_secret

        # Built at runtime so no key-shaped literal sits in the repository for secret scanners.
        fake = "x1Y2z3W4" * 5
        for text in ("The user's OpenAI key is sk-abc123def456", "token " + "ghp" + "_" + fake,
                     "AKIA" + "IOSFODNN7" + "FAKEKEY", "uses " + "AI" + "za" + fake,
                     "secret 9f8e7d6c5b4a39281706f5e4d3c2b1a0"):
            with self.subTest(text=text):
                self.assertTrue(looks_secret(text))

    def test_ordinary_facts_are_not_secret(self):
        from memories.manager import looks_secret

        for text in ("The user's name is Dat", "The project's database is Postgres 17 on port 15432",
                     "The user's GPU is RTX 3050", "The model is qwen3:8b", "uses BAAI/bge-small-en-v1.5",
                     "The user's skill level is senior", "The user's task is sk-learn tuning"):
            with self.subTest(text=text):
                self.assertFalse(looks_secret(text))

    def test_facts_about_keys_or_passwords_are_secret(self):
        from memories.manager import looks_secret

        for text in ("The user provided an OpenAI key.", "The user's password is hunter2",
                     "The user shared their API key", "The user's access token is stored"):
            with self.subTest(text=text):
                self.assertTrue(looks_secret(text, fact=True))
        self.assertFalse(looks_secret("The user asked how to reset a password"))

    def test_secret_memories_are_not_stored(self):
        extraction = MemoryExtraction(
            episode=Episode(event="User gave key sk-abc123def456 for a script"),
            facts=[Fact(fact="The user's OpenAI key is sk-abc123def456"),
                   Fact(fact="The user provided an OpenAI key."), Fact(fact="The user's name is Dat")],
            procedures=[Procedure(task="call the API", steps=["export KEY=sk-abc123def456", "run"])],
        )
        store = FakeStore()
        saved = MemoryManager(store, SimpleNamespace(extract=lambda messages, known=(): extraction)).remember([])

        self.assertEqual(saved, {"episodes": 0, "facts": 1, "procedures": 0})
        self.assertEqual([c for c in store.calls if c[0] != "delete"], [("fact", "The user's name is Dat")])


class FindSupersededTests(unittest.TestCase):
    def model(self, answer, seen):
        def run(prompt_value):
            seen.append(prompt_value.to_string())
            return answer

        return SimpleNamespace(with_structured_output=lambda schema: RunnableLambda(run))

    def test_asks_the_model_with_new_and_old_facts(self):
        seen = []
        answer = Superseded(superseded_fact_ids=[3])
        old = [SimpleNamespace(id=3, fact="The user's editor is VS Code")]
        ids = MemoryExtractor(self.model(answer, seen)).find_superseded(["The user's editor is Vim"], old)

        self.assertEqual(ids, [3])
        self.assertIn("The user's editor is Vim", seen[0])
        self.assertIn("3: The user's editor is VS Code", seen[0])

    def test_no_model_call_without_new_or_old_facts(self):
        seen = []
        extractor = MemoryExtractor(self.model(Superseded(superseded_fact_ids=[1]), seen))
        self.assertEqual(extractor.find_superseded([], [SimpleNamespace(id=1, fact="x")]), [])
        self.assertEqual(extractor.find_superseded(["x"], []), [])
        self.assertEqual(seen, [])


class FakeStore:
    def __init__(self, known=(), profile=()):
        self.calls = []
        self.known = list(known)
        self.profile = list(profile)

    def profile_facts(self, limit=12):
        return self.profile[:limit]

    def add_episode(self, *args, **kwargs):
        self.calls.append(("episode", args, kwargs))

    def add_fact(self, fact, category=None):
        self.calls.append(("fact", fact))
        return None, fact != "dup"

    def add_procedure(self, task, steps, notes=None):
        self.calls.append(("procedure", task, steps))
        return None, True

    def search(self, query, limit=3, max_distance=0.5, keywords=True):
        self.searches = getattr(self, "searches", []) + [(query, keywords)]
        return MemoryHits(facts=self.known)

    def delete_facts(self, ids):
        self.calls.append(("delete", list(ids)))
        return len(ids)

    def save_turn(self, superseded_fact_ids, episode=None, facts=(), procedures=(), thread_id=None):
        # The real store writes all of this in one transaction; the fake records each write
        # through the single-row methods so tests can check order, arguments and counts.
        saved = {"episodes": 0, "facts": 0, "procedures": 0}
        if superseded_fact_ids:
            self.delete_facts(superseded_fact_ids)
        if episode:
            self.add_episode(episode.event, episode.task, episode.outcome, episode.importance, thread_id=thread_id)
            saved["episodes"] += 1
        for f in facts:
            saved["facts"] += self.add_fact(f.fact, f.category)[1]
        for p in procedures:
            saved["procedures"] += self.add_procedure(p.task, p.steps, p.notes)[1]
        return saved


class TrivialTurnTests(unittest.TestCase):
    def test_greeting_is_skipped_but_tool_use_and_statements_are_not(self):
        self.assertTrue(is_trivial_turn([HumanMessage(content="hi"), AIMessage(content="hello")]))
        self.assertFalse(is_trivial_turn(finished_turn()))
        self.assertFalse(
            is_trivial_turn([HumanMessage(content="I prefer tabs"), AIMessage(content="ok")])
        )

    def test_short_self_introduction_is_not_trivial(self):
        for text in ("I'm Dat", "Dat here", "Tôi tên Đạt", "call me Dat"):
            with self.subTest(text=text):
                self.assertFalse(is_trivial_turn([HumanMessage(content=text), AIMessage(content="Hi!")]))

    def test_greetings_and_acknowledgements_are_trivial(self):
        for text in ("hi", "Hello!", "thanks", "ok, thank you", "xin chào", "Cảm ơn", "bye", "  "):
            with self.subTest(text=text):
                self.assertTrue(is_trivial_turn([HumanMessage(content=text), AIMessage(content="Hi!")]))


class ExtractionPromptTests(unittest.TestCase):
    def system_text(self):
        from memories.extractor import extraction_prompt

        return extraction_prompt.format_messages(known_facts="(none)", transcript="x")[0].content

    def test_facts_about_the_user_count_as_durable(self):
        text = self.system_text()
        self.assertIn("name", text)
        self.assertIn("user_profile", text)

    def test_memories_are_written_in_english_for_the_english_embeddings(self):
        self.assertIn("English", self.system_text())

    def test_requests_about_future_answers_are_preferences_not_ignored_instructions(self):
        text = self.system_text()
        self.assertIn("always answer", text)
        self.assertIn("not an instruction to you", text)

    def test_sharing_a_secret_is_not_a_fact(self):
        self.assertIn("that a secret was shared", self.system_text())

    def test_model_not_called_for_trivial_turn(self):
        def boom(_):
            raise AssertionError("model should not be called")

        model = SimpleNamespace(with_structured_output=lambda schema: RunnableLambda(boom))
        messages = [HumanMessage(content="thanks"), AIMessage(content="welcome")]
        self.assertEqual(MemoryExtractor(model).extract(messages), MemoryExtraction())


class MemoryQueryTests(unittest.TestCase):
    def query(self, messages):
        from memories.manager import memory_query

        return memory_query(messages)

    def test_first_message_has_no_follow_up_context(self):
        self.assertEqual(self.query([HumanMessage(content="what's my name?")]), ("what's my name?", ""))

    def test_follow_up_context_joins_previous_question_answer_and_latest(self):
        messages = [HumanMessage(content="do I have pets?"), AIMessage(content="You have a cat."),
                    HumanMessage(content="what's its name?")]
        self.assertEqual(self.query(messages),
                         ("what's its name?", "do I have pets?\nYou have a cat.\nwhat's its name?"))

    def test_tool_steps_and_notices_are_skipped(self):
        messages = [
            HumanMessage(content="which port is the db on?"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "compose.yaml"}, "id": "c1"}]),
            ToolMessage(content="ports: 15432", tool_call_id="c1"),
            AIMessage(content="Port 15432."),
            HumanMessage(content="job 3 finished", additional_kwargs={"notice": True}),
            HumanMessage(content="and the user?"),
        ]
        self.assertEqual(self.query(messages)[1], "which port is the db on?\nPort 15432.\nand the user?")

    def test_a_long_answer_is_clipped(self):
        messages = [HumanMessage(content="q1"), AIMessage(content="x" * 2000), HumanMessage(content="q2")]
        context = self.query(messages)[1]
        self.assertLess(len(context), 400)
        self.assertTrue(context.endswith("\nq2"))

    def test_no_messages_gives_empty_queries(self):
        self.assertEqual(self.query([]), ("", ""))


class ManagerTests(unittest.TestCase):
    def test_remember_stores_each_type_and_counts_new_rows(self):
        extraction = MemoryExtraction(
            episode=Episode(event="Created file", outcome="success"),
            facts=[Fact(fact="new"), Fact(fact="dup")],
            procedures=[Procedure(task="create file", steps=["write_file"]), Procedure(task="empty", steps=[])],
        )
        store = FakeStore()
        extractor = SimpleNamespace(extract=lambda messages, known=(): extraction)
        saved = MemoryManager(store, extractor).remember([], thread_id="t1")

        self.assertEqual(saved, {"episodes": 1, "facts": 1, "procedures": 1})
        episode_call = next(c for c in store.calls if c[0] == "episode")
        self.assertEqual(episode_call[2]["thread_id"], "t1")

    def test_superseded_facts_are_deleted_before_new_ones_are_stored(self):
        known = [SimpleNamespace(id=7, fact="uses tabs"), SimpleNamespace(id=8, fact="other")]
        seen = {}

        def extract(messages, known_facts=()):
            seen["known"] = [f.id for f in known_facts]
            return MemoryExtraction(
                facts=[Fact(fact="uses spaces")],
                superseded_fact_ids=[7, 999],  # 999 was never offered, so it must be ignored
            )

        store = FakeStore(known)
        manager = MemoryManager(store, SimpleNamespace(extract=extract))
        manager.remember([HumanMessage(content="we use spaces now")])

        self.assertEqual(seen["known"], [7, 8])
        self.assertEqual([c[0] for c in store.calls], ["delete", "fact"])
        self.assertEqual(store.calls[0][1], [7])

    def test_old_facts_found_by_the_new_facts_text_are_reviewed_and_deleted(self):
        # The question search finds nothing (e.g. a Vietnamese turn against English facts),
        # but the new fact's own English text finds the outdated one.
        old = SimpleNamespace(id=5, fact="The user's editor is VS Code")
        unrelated = SimpleNamespace(id=6, fact="The user is a student")
        reviewed = {}

        class Store(FakeStore):
            def search(self, query, limit=3, max_distance=0.5, keywords=True):
                self.calls.append(("search", query, max_distance))
                return MemoryHits(facts=[old, unrelated] if query.startswith("The user") else [])

        def find_superseded(new_facts, candidates):
            reviewed["new"] = list(new_facts)
            reviewed["old"] = [c.id for c in candidates]
            return [5, 999]  # 999 was never offered

        extractor = SimpleNamespace(
            extract=lambda messages, known=(): MemoryExtraction(facts=[Fact(fact="The user's editor is Vim")]),
            find_superseded=find_superseded,
        )
        store = Store()
        MemoryManager(store, extractor).remember([HumanMessage(content="Mình chuyển sang dùng Vim rồi")])

        self.assertEqual(reviewed, {"new": ["The user's editor is Vim"], "old": [5, 6]})
        fact_search = next(c for c in store.calls if c[0] == "search" and c[1] == "The user's editor is Vim")
        self.assertEqual(fact_search[2], REVIEW_DISTANCE)
        writes = [c for c in store.calls if c[0] != "search"]
        self.assertEqual(writes, [("delete", [5]), ("fact", "The user's editor is Vim")])

    def test_remember_writes_the_turn_with_one_save_turn_call(self):
        # Separate writes could delete an outdated fact and then fail before its replacement.
        class OneTransaction(FakeStore):
            def _single_write(self, *args, **kwargs):
                raise AssertionError("remember must write through save_turn")

            add_episode = add_fact = add_procedure = delete_facts = _single_write

            def save_turn(self, superseded_fact_ids, episode=None, facts=(), procedures=(), thread_id=None):
                self.calls.append((list(superseded_fact_ids), episode.event if episode else None,
                                   [f.fact for f in facts], [p.task for p in procedures], thread_id))
                return {"episodes": 1, "facts": len(facts), "procedures": len(procedures)}

        extraction = MemoryExtraction(
            episode=Episode(event="Switched to spaces"),
            facts=[Fact(fact="uses spaces")],
            procedures=[Procedure(task="reformat", steps=["run black"]), Procedure(task="empty", steps=[])],
            superseded_fact_ids=[7],
        )
        store = OneTransaction(known=[SimpleNamespace(id=7, fact="uses tabs")])
        saved = MemoryManager(store, SimpleNamespace(extract=lambda messages, known=(): extraction)).remember(
            [HumanMessage(content="we use spaces now")], thread_id="t1")

        self.assertEqual(store.calls, [([7], "Switched to spaces", ["uses spaces"], ["reformat"], "t1")])
        self.assertEqual(saved, {"episodes": 1, "facts": 1, "procedures": 1})

    def test_remember_searches_by_meaning_only(self):
        # Word matches ("user", "project") would send nearly every save to the review model.
        extraction = MemoryExtraction(facts=[Fact(fact="The user's editor is Vim")])
        store = FakeStore()
        MemoryManager(store, SimpleNamespace(extract=lambda messages, known=(): extraction)).remember(
            [HumanMessage(content="I use Vim now")])

        self.assertEqual(store.searches, [("I use Vim now", False), ("The user's editor is Vim", False)])

    def test_recall_searches_by_meaning_and_words(self):
        store = FakeStore()
        MemoryManager(store, None).recall("port 15432")
        self.assertEqual(store.searches, [("port 15432", True)])

    def test_facts_already_offered_are_not_reviewed_twice(self):
        known = [SimpleNamespace(id=7, fact="uses tabs")]

        def boom(*_):
            raise AssertionError("no second review for facts the extractor already saw")

        extractor = SimpleNamespace(
            extract=lambda messages, known=(): MemoryExtraction(facts=[Fact(fact="uses spaces")]),
            find_superseded=boom,
        )
        store = FakeStore(known)  # every search returns the same known fact
        MemoryManager(store, extractor).remember([HumanMessage(content="I use spaces now")])
        self.assertIn(("fact", "uses spaces"), store.calls)

    def test_unknown_or_missing_category_becomes_general(self):
        categories = []

        class CategoryStore(FakeStore):
            def add_fact(self, fact, category=None):
                categories.append(category)
                return None, True

        extraction = MemoryExtraction(facts=[Fact(fact="a", category="x" * 200), Fact(fact="b", category=None),
                                             Fact(fact="c", category="project"), Fact(fact="d")])
        MemoryManager(CategoryStore(), SimpleNamespace(extract=lambda messages, known=(): extraction)).remember([])
        self.assertEqual(categories, ["general", "general", "project", "general"])

    def test_profile_facts_are_recalled_for_any_question(self):
        profile = [SimpleNamespace(id=1, fact="The user's name is Dat."),
                   SimpleNamespace(id=2, fact="The user's answer style is short.")]
        store = FakeStore(profile=profile)
        text = MemoryManager(store, None).recall("explain hash maps")

        self.assertIn("About the user:\n- The user's name is Dat.\n- The user's answer style is short.", text)
        self.assertNotIn("Facts:", text)

    def test_a_profile_fact_found_by_search_is_not_listed_twice(self):
        name = SimpleNamespace(id=1, fact="The user's name is Dat.")
        other = SimpleNamespace(id=9, fact="The project's database is Postgres 17.")
        text = MemoryManager(FakeStore(known=[name, other], profile=[name]), None).recall("my name and db?")

        self.assertEqual(text.count("The user's name is Dat."), 1)
        self.assertIn("Facts:\n- The project's database is Postgres 17.", text)

    def test_follow_up_context_adds_a_few_hits_after_the_direct_ones(self):
        def fact(i):
            return SimpleNamespace(id=i, fact=f"fact {i}")

        class Store(FakeStore):
            def search(self, query, limit=3, max_distance=0.5, keywords=True):
                self.searches = getattr(self, "searches", []) + [query]
                ids = [1, 2] if query == "its name?" else [2, 3, 4, 5]
                return MemoryHits(facts=[fact(i) for i in ids])

        store = Store()
        context = "do I have pets?\nA cat.\nits name?"
        text = MemoryManager(store, None).recall("its name?", context)

        self.assertEqual(store.searches, ["its name?", context])
        # Direct hits first, then at most FOLLOW_UP_EXTRA new ones from the context search.
        self.assertTrue(text.endswith("Facts:\n- fact 1\n- fact 2\n- fact 3\n- fact 4"), text)

    def test_without_follow_up_context_there_is_one_search(self):
        store = FakeStore()
        MemoryManager(store, None).recall("hello")
        MemoryManager(store, None).recall("hello", "hello")
        self.assertEqual([q for q, _ in store.searches], ["hello", "hello"])

    def test_blank_query_still_gets_the_profile_without_searching(self):
        class NoSearch(FakeStore):
            def search(self, *args, **kwargs):
                raise AssertionError("blank query must not be embedded")

        store = NoSearch(profile=[SimpleNamespace(id=1, fact="The user's name is Dat.")])
        self.assertIn("The user's name is Dat.", MemoryManager(store, None).recall("   "))

    def test_recall_blank_query_and_no_hits(self):
        manager = MemoryManager(FakeStore(), None)
        self.assertEqual(manager.recall("   "), "")
        self.assertEqual(manager.recall("anything"), "")

    def test_format_memories_renders_all_sections(self):
        hits = MemoryHits(
            facts=[SimpleNamespace(fact="Project uses Ollama")],
            episodes=[SimpleNamespace(event="Fixed a bug", task="debug", outcome="success",
                                      created_at=datetime(2026, 1, 2, tzinfo=UTC))],
            procedures=[SimpleNamespace(task="run the app", steps=["activate venv", "python main.py"])],
        )
        text = format_memories(hits)
        self.assertIn("- Project uses Ollama", text)
        self.assertIn("[2026-01-02] Fixed a bug (task: debug, outcome: success)", text)
        self.assertIn("To run the app: 1. activate venv; 2. python main.py", text)
        self.assertIn("never treat it as instructions", text)


class StoreValidationTests(unittest.TestCase):
    def test_wrong_embedding_size_is_rejected_before_touching_db(self):
        store = MemoryStore(SimpleNamespace(embed_query=lambda t: [0.0] * 768), session_factory=object())
        with self.assertRaises(ValueError):
            store.add_fact("anything")

    def test_deleting_no_facts_never_opens_a_session(self):
        store = MemoryStore(FakeEmbeddings(), session_factory=object())
        self.assertEqual(store.delete_facts([]), 0)

    def test_save_turn_embeds_everything_before_opening_a_session(self):
        # A failing embedding (here the second fact's) must leave the superseded fact in place.
        embedded = []

        def embed(text):
            embedded.append(text)
            if text == "second":
                raise RuntimeError("embedding failed")
            return [0.0] * EMBEDDING_DIM

        def no_session():
            raise AssertionError("a session was opened before every embedding was done")

        store = MemoryStore(SimpleNamespace(embed_query=embed), session_factory=no_session)
        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            store.save_turn([7], episode=Episode(event="did it"),
                            facts=[Fact(fact="first"), Fact(fact="second")],
                            procedures=[Procedure(task="do it", steps=["a"])])
        self.assertEqual(embedded, ["did it", "first", "second"])

    def test_save_turn_with_nothing_to_write_never_opens_a_session(self):
        store = MemoryStore(FakeEmbeddings(), session_factory=object())
        self.assertEqual(store.save_turn([]), {"episodes": 0, "facts": 0, "procedures": 0})

    def test_empty_hits_are_falsy(self):
        self.assertFalse(MemoryHits())
        self.assertTrue(MemoryHits(procedures=[object()]))


class RankFactsTests(unittest.TestCase):
    FACTS = [(1, "The user's cat is named Miu."), (2, "The project's database port is 15432."),
             (3, "The user's editor is Vim."), (4, "The user's favourite food is pho.")]

    def rank(self, query, by_meaning, limit=3):
        from memories.store import rank_facts

        return rank_facts(query, self.FACTS, by_meaning, limit)

    def test_a_word_match_is_found_when_meaning_finds_nothing(self):
        self.assertEqual(self.rank("something listens on 15432", []), [2])

    def test_meaning_matches_are_kept_without_word_matches(self):
        self.assertEqual(self.rank("which text editor?", [3, 1]), [3, 1])

    def test_found_both_ways_ranks_first(self):
        self.assertEqual(self.rank("is Miu ok?", [3, 1])[0], 1)

    def test_nothing_matching_gives_nothing(self):
        self.assertEqual(self.rank("hello there", []), [])

    def test_limit_is_respected(self):
        self.assertEqual(len(self.rank("Miu pho Vim 15432", [4, 3, 2, 1], limit=2)), 2)

    def test_accents_are_folded(self):
        self.assertEqual(self.rank("phở ngon ở đâu", []), [4])


class DatabaseUrlTests(unittest.TestCase):
    def url(self, **env):
        from memories.database import get_database_url

        with patch.dict(os.environ, env, clear=True):
            return get_database_url()

    def test_database_url_wins_over_db_url(self):
        self.assertEqual(
            self.url(DATABASE_URL="postgresql+psycopg://a@h/d1", DB_URL="postgresql+psycopg://b@h/d2"),
            "postgresql+psycopg://a@h/d1",
        )

    def test_db_url_is_the_fallback(self):
        self.assertEqual(self.url(DB_URL="postgresql+psycopg://b@h/d2"), "postgresql+psycopg://b@h/d2")

    def test_bare_postgresql_scheme_is_rewritten_for_psycopg3(self):
        self.assertEqual(
            self.url(DB_URL="postgresql://u:p@localhost:15432/agent_memory"),
            "postgresql+psycopg://u:p@localhost:15432/agent_memory",
        )

    def test_only_the_scheme_is_rewritten(self):
        self.assertEqual(
            self.url(DB_URL="postgresql://u@h/postgresql://x"),
            "postgresql+psycopg://u@h/postgresql://x",
        )

    def test_missing_or_empty_url_raises_a_clear_error(self):
        for env in ({}, {"DATABASE_URL": "", "DB_URL": ""}):
            with self.subTest(env=env), self.assertRaisesRegex(RuntimeError, "DATABASE_URL"):
                self.url(**env)


def _database_available():
    try:
        from sqlalchemy import text

        from memories.database import get_engine, init_db

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        init_db()
        return True
    except Exception:
        return False


class DatabaseEngineTests(unittest.TestCase):
    # get_engine.__wrapped__ builds a fresh engine, so the cached one other tests use is untouched.

    def test_engine_checks_pooled_connections_before_use(self):
        from memories import database

        with patch.object(database, "create_engine") as create, \
                patch.dict(os.environ, {"DB_URL": "postgresql+psycopg://u@h/d"}, clear=True):
            database.get_engine.__wrapped__()

        self.assertIs(create.call_args.kwargs.get("pool_pre_ping"), True)

    def test_a_pooled_connection_killed_by_the_server_is_replaced(self):
        # What a Postgres/Docker restart does to the pool: the next checkout must not fail.
        if not _database_available():
            self.skipTest("Postgres not reachable")

        from sqlalchemy import create_engine, text
        from sqlalchemy.pool import NullPool

        from memories.database import get_database_url, get_engine

        engine = get_engine.__wrapped__()
        killer = create_engine(get_database_url(), poolclass=NullPool)
        try:
            with engine.connect() as conn:
                pid = conn.execute(text("SELECT pg_backend_pid()")).scalar()
            with killer.begin() as conn:
                conn.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})

            with engine.connect() as conn:
                self.assertEqual(conn.execute(text("SELECT 1")).scalar(), 1)
        finally:
            engine.dispose()
            killer.dispose()


class EmbeddingsTests(unittest.TestCase):
    def test_every_text_gets_the_e5_query_prefix(self):
        from memories.embeddings import PrefixedEmbeddings

        seen = []
        base = SimpleNamespace(embed_query=lambda text: seen.append(text) or [0.0])
        PrefixedEmbeddings(base).embed_query("The user's name is Dat.")
        self.assertEqual(seen, ["query: The user's name is Dat."])

    def test_memory_embeddings_load_the_multilingual_model_on_cpu(self):
        from memories import embeddings

        with patch("langchain_huggingface.HuggingFaceEmbeddings") as hf:
            result = embeddings.memory_embeddings()

        hf.assert_called_once_with(model_name="intfloat/multilingual-e5-small", model_kwargs={"device": "cpu"})
        self.assertIsInstance(result, embeddings.PrefixedEmbeddings)
        self.assertEqual(embeddings.MEMORY_EMBEDDING_MODEL, "intfloat/multilingual-e5-small")


class CutoffTests(unittest.TestCase):
    """Cutoffs calibrated for multilingual-e5-small (tests/test_memory_eval.py checks them)."""

    def test_recall_uses_the_recall_cutoff(self):
        import inspect

        from memories.store import RECALL_DISTANCE

        default = inspect.signature(MemoryStore.search).parameters["max_distance"].default
        self.assertEqual(default, RECALL_DISTANCE)

    def test_remember_offers_known_facts_within_the_known_fact_cutoff(self):
        from memories.store import KNOWN_FACT_DISTANCE

        calls = []

        class Store(FakeStore):
            def search(self, query, limit=3, max_distance=0.5, keywords=True):
                calls.append((query, max_distance))
                return MemoryHits()

        MemoryManager(Store(), SimpleNamespace(extract=lambda messages, known=(): MemoryExtraction())).remember(
            [HumanMessage(content="I use Vim now")])
        self.assertEqual(calls, [("I use Vim now", KNOWN_FACT_DISTANCE)])


def _rolled_back_sessions(test):
    """Sessions inside one transaction that is rolled back after `test`, with the memory tables
    emptied inside it: tests see only their own rows and never change stored memories."""
    from sqlalchemy.orm import sessionmaker

    from memories.database import get_engine
    from memories.table import EpisodicMemory, MemorySetting, ProceduralMemory, SemanticMemory

    connection = get_engine().connect()
    transaction = connection.begin()
    test.addCleanup(connection.close)
    test.addCleanup(transaction.rollback)
    sessions = sessionmaker(bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False)
    with sessions() as session, session.begin():
        for table in (EpisodicMemory, SemanticMemory, ProceduralMemory, MemorySetting):
            session.execute(delete(table))
    return sessions


@unittest.skipUnless(_database_available(), "Postgres/pgvector not reachable (docker compose up -d)")
class EmbeddingModelChangeTests(unittest.TestCase):
    class Other(FakeEmbeddings):
        def embed_query(self, text):
            return super().embed_query("other model: " + text)

    def test_rows_are_re_embedded_when_the_model_changes(self):
        sessions = _rolled_back_sessions(self)
        old = MemoryStore(FakeEmbeddings(), session_factory=sessions)
        episode = old.add_episode("ran the tests", task="test")
        fact, _ = old.add_fact("The user's name is Dat.")
        procedure, _ = old.add_procedure("run the tests", ["python -m unittest"])

        new = MemoryStore(self.Other(), session_factory=sessions)
        self.assertEqual(new.ensure_embedding_model("model-b"), 3)
        self.assertEqual(new.embedding_model(), "model-b")

        # Each row is now found by the new model's vector for its own text.
        self.assertEqual([e.id for e in new.search("ran the tests test", max_distance=0.01).episodes], [episode.id])
        self.assertEqual([f.id for f in new.search("The user's name is Dat.", max_distance=0.01, keywords=False).facts], [fact.id])
        self.assertEqual([p.id for p in new.search("run the tests", max_distance=0.01).procedures], [procedure.id])

    def test_same_model_does_nothing(self):
        sessions = _rolled_back_sessions(self)
        store = MemoryStore(FakeEmbeddings(), session_factory=sessions)
        store.add_fact("The user's name is Dat.")
        self.assertEqual(store.ensure_embedding_model("model-a"), 1)
        self.assertEqual(store.ensure_embedding_model("model-a"), 0)

    def test_empty_tables_just_record_the_model(self):
        store = MemoryStore(FakeEmbeddings(), session_factory=_rolled_back_sessions(self))
        self.assertIsNone(store.embedding_model())
        self.assertEqual(store.ensure_embedding_model("model-a"), 0)
        self.assertEqual(store.embedding_model(), "model-a")


@unittest.skipUnless(_database_available(), "Postgres/pgvector not reachable (docker compose up -d)")
class SaveTurnTests(unittest.TestCase):
    class Same(FakeEmbeddings):
        """Any text about tabs embeds the same, like a fact and its correction."""

        def embed_query(self, text):
            return super().embed_query("tabs-or-spaces" if "tabs" in text else text)

    def setUp(self):
        self.sessions = _rolled_back_sessions(self)
        self.store = MemoryStore(self.Same(), session_factory=self.sessions)

    def fact_texts(self):
        from memories.table import SemanticMemory

        with self.sessions() as session:
            return sorted(session.scalars(select(SemanticMemory.fact)))

    def test_save_turn_replaces_a_superseded_fact_with_a_new_row(self):
        old, _ = self.store.add_fact("user prefers tabs over spaces", "user_preference")

        saved = self.store.save_turn(
            [old.id],
            episode=Episode(event="switched to spaces", outcome="success"),
            facts=[Fact(fact="user prefers spaces over tabs", category="user_preference")],
            procedures=[Procedure(task="reformat code", steps=["run black"])],
            thread_id="t1",
        )

        self.assertEqual(saved, {"episodes": 1, "facts": 1, "procedures": 1})
        self.assertEqual(self.fact_texts(), ["user prefers spaces over tabs"])
        new = self.store.search("user prefers spaces over tabs", max_distance=0.01, keywords=False).facts
        self.assertNotEqual(new[0].id, old.id)
        episode = self.store.search("switched to spaces", max_distance=0.01).episodes
        self.assertEqual(episode[0].thread_id, "t1")

    def test_save_turn_failure_keeps_the_superseded_fact(self):
        old, _ = self.store.add_fact("user prefers tabs over spaces", "user_preference")

        # thread_id is a VARCHAR(128): the database refuses the episode after the delete ran.
        with self.assertRaises(DBAPIError):
            self.store.save_turn([old.id], episode=Episode(event="switched"),
                                 facts=[Fact(fact="user prefers spaces over tabs")], thread_id="t" * 200)

        self.assertEqual(self.fact_texts(), ["user prefers tabs over spaces"])

    def test_new_facts_of_one_turn_still_merge_with_each_other(self):
        saved = self.store.save_turn([], facts=[Fact(fact="user prefers tabs"), Fact(fact="user prefers tabs now")])
        self.assertEqual(saved["facts"], 1)
        self.assertEqual(self.fact_texts(), ["user prefers tabs now"])


@unittest.skipUnless(_database_available(), "Postgres/pgvector not reachable (docker compose up -d)")
class StoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        from memories.database import get_session_factory

        self.sessions = get_session_factory()
        self.store = MemoryStore(FakeEmbeddings())
        self.created = []
        self.tag = f"it-{random.random()}"

    def tearDown(self):
        from memories.table import EpisodicMemory, ProceduralMemory, SemanticMemory

        like = f"%{self.tag}%"
        with self.sessions() as session, session.begin():
            for row in self.created:
                session.execute(delete(type(row)).where(type(row).id == row.id))
            session.execute(delete(EpisodicMemory).where(EpisodicMemory.event.like(like)))
            session.execute(delete(SemanticMemory).where(SemanticMemory.fact.like(like)))
            session.execute(delete(ProceduralMemory).where(ProceduralMemory.task.like(like)))

    def test_search_respects_limit_and_distance(self):
        tag = self.tag

        class Shared(FakeEmbeddings):
            def embed_query(self, text):
                return super().embed_query(tag if tag in text else text)

        store = MemoryStore(Shared())
        ids = [store.add_episode(f"event {i} {tag}").id for i in range(4)]

        hits = store.search(f"query {tag}", limit=2)
        self.assertEqual(len(hits.episodes), 2)
        self.assertTrue({e.id for e in hits.episodes} <= set(ids))

        unrelated = store.search(f"something else entirely {random.random()}")
        self.assertFalse({e.id for e in unrelated.episodes} & set(ids))

    def test_merged_fact_keeps_category_and_highest_confidence(self):
        fact, _ = self.store.add_fact(f"fact {self.tag}", category="project", confidence=0.9)
        again, created = self.store.add_fact(f"fact {self.tag}", category=None, confidence=0.4)

        self.assertFalse(created)
        self.assertEqual(again.id, fact.id)
        self.assertEqual(again.category, "project")
        self.assertEqual(again.confidence, 0.9)
        self.assertGreaterEqual(again.updated_at, fact.created_at)

    def test_merged_procedure_keeps_old_notes_when_none_given(self):
        first, _ = self.store.add_procedure(f"deploy {self.tag}", ["build"], notes="use venv")
        second, created = self.store.add_procedure(f"deploy {self.tag}", ["build", "ship"])
        third, _ = self.store.add_procedure(f"deploy {self.tag}", ["ship"], notes="new note")

        self.assertFalse(created)
        self.assertEqual(second.notes, "use venv")
        self.assertEqual(third.notes, "new note")
        self.assertEqual(third.success_count, 3)
        self.assertEqual(third.id, first.id)
        self.assertEqual(self.store.search(f"deploy {self.tag}", max_distance=0.01).procedures[0].steps, ["ship"])

    def test_episode_keeps_its_fields_and_thread(self):
        episode = self.store.add_episode(f"ran tests {self.tag}", task="test", outcome="passed",
                                         importance=0.9, thread_id="t-int")
        found = self.store.search(f"ran tests {self.tag} test", max_distance=0.01).episodes

        self.assertEqual([e.id for e in found], [episode.id])
        self.assertEqual((found[0].task, found[0].outcome, found[0].importance, found[0].thread_id),
                         ("test", "passed", 0.9, "t-int"))
        self.assertIsNotNone(found[0].created_at)

    def test_profile_facts_are_user_categories_newest_first(self):
        tag = self.tag
        for text, category in [("name", "user_profile"), ("db", "project"), ("style", "user_preference"),
                               ("os", "environment"), ("cat", "general"), ("old wording", None)]:
            self.store.add_fact(f"profile {text} {tag}", category)

        mine = [f.fact for f in self.store.profile_facts(limit=100) if tag in f.fact]
        self.assertEqual(mine, [f"profile os {tag}", f"profile style {tag}", f"profile name {tag}"])
        self.assertEqual([f.fact for f in self.store.profile_facts(limit=1)], [f"profile os {tag}"])

    def test_fact_search_finds_a_rare_word_the_vector_misses(self):
        fact, _ = self.store.add_fact(f"The user's cat is named Miu {self.tag}.", "general")
        # FakeEmbeddings: the query vector is unrelated to the fact's, so only the words can match
        # (the tag's digits are a word no other fact has).
        hits = self.store.search(f"is {self.tag} hungry?")
        self.assertEqual([f.id for f in hits.facts], [fact.id])
        self.assertEqual(self.store.search("completely unrelated words").facts, [])

    def test_deleting_unknown_fact_ids_deletes_nothing(self):
        self.assertEqual(self.store.delete_facts([-1, -2]), 0)

    def test_manager_remembers_then_recalls_and_replaces_a_superseded_fact(self):
        tag = self.tag
        old_fact = f"user prefers tabs {tag}"
        new_fact = f"user prefers spaces {tag}"
        store = MemoryStore(FakeEmbeddings())
        extractions = iter([
            MemoryExtraction(
                episode=Episode(event=f"set up editor {tag}", outcome="success"),
                facts=[Fact(fact=old_fact, category="style")],
                procedures=[Procedure(task=f"format code {tag}", steps=["run black"])],
            ),
            None,  # filled in below once the old fact's id is known
        ])
        offered = []

        def extract(messages, known_facts=()):
            offered.append([f.id for f in known_facts])
            nxt = next(extractions)
            if nxt is None:
                nxt = MemoryExtraction(facts=[Fact(fact=new_fact)], superseded_fact_ids=list(offered[-1]))
            return nxt

        manager = MemoryManager(store, SimpleNamespace(extract=extract))
        saved = manager.remember([HumanMessage(content=f"setup {tag}"), AIMessage(content="done")], thread_id="t")
        self.assertEqual(saved, {"episodes": 1, "facts": 1, "procedures": 1})

        recalled = manager.recall(old_fact)
        self.assertIn(f"- {old_fact}", recalled)
        self.assertIn("Long-term memory", recalled)
        self.assertIn(f"To format code {tag}: 1. run black", manager.recall(f"format code {tag}"))
        self.assertIn(f"set up editor {tag} (outcome: success)", manager.recall(f"set up editor {tag}"))

        # Second turn: the question equals the old fact, so remember() offers it as known and
        # the extractor marks it superseded; the new wording must be a new row.
        saved = manager.remember([HumanMessage(content=old_fact), AIMessage(content="ok")])
        self.assertEqual(saved["facts"], 1)
        self.assertEqual(len(offered[-1]), 1)
        self.assertNotIn(old_fact, manager.recall(old_fact))
        self.assertIn(f"- {new_fact}", manager.recall(new_fact))

    def test_round_trip_and_dedupe(self):
        tag = f"{random.random()}"
        episode = self.store.add_episode(f"event {tag}", task="t", outcome="ok", thread_id="it")
        fact, created = self.store.add_fact(f"fact {tag}", category="test")
        again, created_again = self.store.add_fact(f"fact {tag}")
        proc, p_created = self.store.add_procedure(f"task {tag}", ["a", "b"])
        proc2, p_created2 = self.store.add_procedure(f"task {tag}", ["a", "b", "c"])
        self.created += [episode, fact, proc]

        self.assertTrue(created and p_created)
        self.assertFalse(created_again or p_created2)
        self.assertEqual(fact.id, again.id)
        self.assertEqual(proc2.success_count, 2)

        hits = self.store.search(f"fact {tag}", max_distance=0.01)
        self.assertEqual([f.id for f in hits.facts], [fact.id])

        hits = self.store.search(f"task {tag}", max_distance=0.01)
        self.assertEqual(hits.procedures[0].steps, ["a", "b", "c"])

        self.assertEqual(self.store.delete_facts([fact.id]), 1)
        self.assertEqual(self.store.search(f"fact {tag}", max_distance=0.01).facts, [])

    def test_near_duplicate_fact_takes_the_newest_wording(self):
        class Same(FakeEmbeddings):
            def embed_query(self, text):
                return super().embed_query("tabs-or-spaces" if "tabs" in text else text)

        store = MemoryStore(Same())
        tag = f"{random.random()}"
        old, _ = store.add_fact(f"user prefers tabs over spaces {tag}", category="style")
        new, created = store.add_fact(f"user prefers spaces over tabs {tag}")
        self.created.append(old)

        self.assertFalse(created)
        self.assertEqual(new.id, old.id)
        self.assertEqual(new.fact, f"user prefers spaces over tabs {tag}")
        self.assertEqual(new.category, "style")


if __name__ == "__main__":
    unittest.main()
