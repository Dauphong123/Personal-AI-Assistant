from langchain_core.prompts import ChatPromptTemplate

from data import Router
from dependency import dependency

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You route user question.
                Use "direct" when the question can be answered using the model's general knowledge.
                Use "code" when the request needs the machine or repository: reading, searching,
                creating, editing, moving or deleting files, running commands or tests, or git.
                Any request to actually create or change something on disk is "code".
                The question may be in any language (for example Vietnamese); decide by its meaning.
                Use the recent conversation to understand short follow-ups such as "yes, do it".

                Recent conversation (may be empty):
                {history}
            """,
        ),
        ("user", "{question}"),
    ]
)

model = dependency["model"]
model = model.with_structured_output(Router)

router_chain = router_prompt | model
