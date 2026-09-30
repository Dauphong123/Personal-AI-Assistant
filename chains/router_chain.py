from langchain_core.prompts import ChatPromptTemplate

from data import Router
from dependency import dependency

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You route user question.
                Use "rag" when the question requires information from the provided document collection. 
                Use "direct" when the question can be answered using the model's general knowledge.
                Use "code" when the question need to read the code base from the machine.
            """,
        ),
        ("user", "{question}"),
    ]
)

model = dependency["model"]
model = model.with_structured_output(Router)

router_chain = router_prompt | model
