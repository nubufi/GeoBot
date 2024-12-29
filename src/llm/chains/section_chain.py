from langchain_core.prompts import ChatPromptTemplate
from llm.output_parsers.list_output_parser import ListOutputParser
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from typing import List
def get_content():
    with open("data/contents.md", "r") as f:
        content = f.read()
    return content

class ContentList(BaseModel):
    """Return list of most related main section to the user query."""

    sections: List[str] = Field(
        ...,
        description="Given a user prompt select the most suitable main sections numbers from content and return them in list of strings such as ['2','4','16A'].",
    )

prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                (
                    "You are an assistant that finds the most relevant sections numbers from the file content" 
                    "based on the user query. "
                    "Return the main section numbers in a list. For example if you think 2.2, 4 and 4.5 are the relevant sections you should return"
                    "['2','4']. The only exception is 16A which should be returned as is."
                    "Return empty list if no relevant sections are found."
                )
            ),
            ("user", "<content>{contents}</content>\n\n{input}"),
        ]
    )

llm = AzureChatOpenAI(temperature=0)
structured_llm = llm.with_structured_output(ContentList)
section_chain = prompt_template  | structured_llm