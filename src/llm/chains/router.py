from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["search", "chat"] = Field(
        ...,
        description="Given a user prompt choose to route it to search or chat based on previous messages.",
    )


llm = AzureChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user prompt to a search or chat.
Route to search node if the user is asking a new question about the documents.
And route to chat node if the user is asking a follow up question about previous messages or not sending domain specific messages like Hello."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
