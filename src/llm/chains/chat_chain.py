from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a virtual assistant who answers users questions based on the previous messages."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = AzureChatOpenAI()
chat_chain = chat_prompt | llm
