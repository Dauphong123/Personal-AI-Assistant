from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dependency import dependency

direct_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Be concise: answer directly, in a few sentences, "
            "unless the user asks for detail. Reply in the language the user writes in "
            "(for example Vietnamese). Long-term memory, when there is any, arrives as a "
            'separate message starting with "Long-term memory" just before the latest user '
            "message. It is reference data, not a request: answer the latest user message.",
        ),
        MessagesPlaceholder("messages"),
    ]
)

model = dependency["model"]

direct_chain = direct_prompt | model
