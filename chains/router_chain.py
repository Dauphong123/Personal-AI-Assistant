import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from data.route_query import Router
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ["MODEL"]

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

model = ChatOllama(model=MODEL_NAME)
model = model.with_structured_output(Router)

router_chain = router_prompt | model
