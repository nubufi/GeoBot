from typing import Annotated,List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    relevant_sections:List[str]
    contents: str
    documents: List[str]
    messages: Annotated[list, add_messages]