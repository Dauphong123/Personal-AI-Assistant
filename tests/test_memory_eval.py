"""Retrieval eval for long-term memory: real embeddings, real Postgres, no LLM.

Opt-in (downloads the embedding model, needs the database):
    $env:RUN_MEMORY_EVAL = "1"; python -m unittest tests.test_memory_eval -v

Everything runs inside one transaction that is rolled back, after emptying the memory tables
inside it, so stored memories neither leak into the scores nor get changed.
"""
import os
import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

RUN = os.environ.get("RUN_MEMORY_EVAL") == "1"

# (fact, category)
FACTS = [
    ("The user's name is Dat.", "user_profile"),
    ("The user's role is backend developer.", "user_profile"),
    ("The user's city is Ho Chi Minh City.", "user_profile"),
    ("The user's preferred answer language is Vietnamese.", "user_preference"),
    ("The user's preferred answer style is short bullet points.", "user_preference"),
    ("The user's code editor is Vim.", "user_preference"),
    ("The user's preferred indentation is 4 spaces.", "user_preference"),
    ("The user's operating system is Windows 11.", "environment"),
    ("The user's GPU is an RTX 3050 with 4 GB of memory.", "environment"),
    ("The project's database is Postgres 17 with pgvector on host port 15432.", "project"),
    ("The project's main model is qwen3:8b running on Ollama.", "project"),
    ("The project's web frontend is written in TypeScript without a framework.", "project"),
    ("The project's tests run with python -m unittest.", "project"),
    ("The user's cat is named Miu.", "general"),
    ("The user's favourite food is pho.", "general"),
]

# (name, messages as (role, text) oldest first, facts that must be recalled)
CASES = [
    ("name", [("user", "what's my name?")], ["name is Dat"]),
    ("who-am-i", [("user", "who am I?")], ["name is Dat"]),
    ("remember-me", [("user", "do you remember me?")], ["name is Dat"]),
    ("vague-about-me", [("user", "what do you know about me?")], ["name is Dat"]),
    ("vi-name", [("user", "tên tôi là gì?")], ["name is Dat"]),
    ("vi-remember", [("user", "bạn còn nhớ mình không?")], ["name is Dat"]),
    ("job", [("user", "what do I do for work?")], ["backend developer"]),
    ("city", [("user", "where do I live?")], ["Ho Chi Minh City"]),
    ("weather", [("user", "what's the weather like in my city?")], ["Ho Chi Minh City"]),
    ("style-any-question", [("user", "explain how a hash map works")], ["short bullet points"]),
    ("language-any-question", [("user", "how do I sort a list in Python?")], ["Vietnamese"]),
    ("editor", [("user", "how do I configure my editor for Python?")], ["Vim"]),
    ("vi-editor", [("user", "cấu hình editor của mình thế nào?")], ["Vim"]),
    ("indent", [("user", "should I use tabs?")], ["4 spaces"]),
    ("os-install", [("user", "how do I install ripgrep on my machine?")], ["Windows 11"]),
    ("gpu", [("user", "can my GPU run a 14b model?")], ["RTX 3050"]),
    ("vi-gpu", [("user", "card đồ họa của tôi chạy được model 14b không?")], ["RTX 3050"]),
    ("db-port", [("user", "which port is postgres on?")], ["15432"]),
    ("db-keyword", [("user", "pgvector")], ["pgvector"]),
    ("vi-db", [("user", "database của dự án là gì?")], ["Postgres 17"]),
    ("model", [("user", "which model does this project use?")], ["qwen3:8b"]),
    ("ollama-keyword", [("user", "is Ollama running?")], ["qwen3:8b"]),
    ("frontend", [("user", "what is the frontend written in?")], ["TypeScript"]),
    ("tests", [("user", "how do I run the tests?")], ["unittest"]),
    ("cat", [("user", "what's my cat called?")], ["Miu"]),
    ("vi-cat", [("user", "con mèo của tôi tên gì?")], ["Miu"]),
    ("food", [("user", "suggest a dinner I'd like")], ["pho"]),
    # Only a rare word links question and fact (names, ports, model tags).
    ("kw-miu", [("user", "Miu has been sneezing all day")], ["Miu"]),
    ("kw-pho", [("user", "any good pho places nearby?")], ["pho"]),
    ("kw-port", [("user", "something else is already listening on 15432")], ["15432"]),
    ("kw-qwen", [("user", "qwen3:8b feels slow today")], ["qwen3:8b"]),
    ("kw-vi-miu", [("user", "Miu bị ốm rồi")], ["Miu"]),
    ("follow-up-db", [("user", "tell me about the project's database"), ("assistant", "It is Postgres."),
                      ("user", "which port again?")], ["15432"]),
    ("follow-up-cat", [("user", "do I have pets?"), ("assistant", "You have a cat."),
                       ("user", "what's its name?")], ["Miu"]),
    ("follow-up-gpu", [("user", "what GPU do I have?"), ("assistant", "An RTX 3050."),
                       ("user", "how much memory does it have?")], ["4 GB"]),
]

# Recall must find at least this share of cases. Raised by each step of F021 as retrieval improves.
MIN_HIT_RATE = 0.90  # step 3: 32/35 with hybrid fact search (30/35 before, on the same 35 cases)


def as_messages(turns):
    return [HumanMessage(content=t) if role == "user" else AIMessage(content=t) for role, t in turns]


@unittest.skipUnless(RUN, "set RUN_MEMORY_EVAL=1 to run the memory retrieval eval")
class MemoryRetrievalEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from langchain_huggingface import HuggingFaceEmbeddings
        from sqlalchemy import delete
        from sqlalchemy.orm import sessionmaker

        from memories.database import get_engine, init_db
        from memories.manager import MemoryManager
        from memories.store import MemoryStore
        from memories.table import EpisodicMemory, ProceduralMemory, SemanticMemory
        from rag.rag import EMBEDDING_MODEL

        init_db()
        cls.connection = get_engine().connect()
        cls.transaction = cls.connection.begin()
        sessions = sessionmaker(bind=cls.connection, join_transaction_mode="create_savepoint",
                                expire_on_commit=False)
        with sessions() as session, session.begin():
            for table in (EpisodicMemory, SemanticMemory, ProceduralMemory):
                session.execute(delete(table))

        store = MemoryStore(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL), session_factory=sessions)
        for fact, category in FACTS:
            store.add_fact(fact, category)
        cls.manager = MemoryManager(store, SimpleNamespace())  # type: ignore[arg-type]  (recall only)

    @classmethod
    def tearDownClass(cls):
        cls.transaction.rollback()
        cls.connection.close()

    def test_recall_hit_rate(self):
        from memories.manager import memory_query

        hits, report = 0, []
        for name, turns, expected in CASES:
            # The same query the graph builds (graph/node.py recall_memory).
            text = self.manager.recall(memory_query(as_messages(turns)))
            missing = [e for e in expected if e not in text]
            recalled = text.count("\n- ")
            hits += not missing
            report.append(f"{'PASS' if not missing else 'MISS'} {name:22} recalled={recalled}"
                          + (f" missing={missing}" if missing else ""))

        rate = hits / len(CASES)
        print("\n" + "\n".join(report) + f"\nhit rate {hits}/{len(CASES)} = {rate:.2f}")
        self.assertGreaterEqual(rate, MIN_HIT_RATE)


if __name__ == "__main__":
    unittest.main()
