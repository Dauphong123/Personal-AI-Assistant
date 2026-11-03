from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dependency import dependency

direct_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Be concise: answer directly, in a few sentences, "
            "unless the user asks for detail. Reply in the language the user writes in "
            "(for example Vietnamese).\n\n{memory}",
        ),
        MessagesPlaceholder("messages"),
    ]
)

model = dependency["model"]

direct_chain = direct_prompt | model
