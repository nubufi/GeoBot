from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from llm.chains.section_chain import get_content
from langchain_openai import AzureChatOpenAI
from llm.state import State
from llm.nodes import find_relevant_sections,chat,route_question,answer_question
from llm.consts import SEARCH,CHAT,ANSWER
import os

memory = MemorySaver()

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"], 
    model="gpt-4o-mini"
)

graph_builder = StateGraph(State)
graph_builder.add_node(SEARCH, find_relevant_sections)
graph_builder.add_node(CHAT, chat)
graph_builder.add_node(ANSWER, answer_question)

graph_builder.set_conditional_entry_point(
    route_question,
    {
        SEARCH: SEARCH,
        CHAT: CHAT,
    }
)
graph_builder.add_edge(SEARCH, ANSWER)
graph_builder.add_edge(ANSWER, END)

graph = graph_builder.compile(checkpointer=memory)

#Adjust later
config = {"configurable": {"thread_id": "5"}}

contents = get_content()
def stream_event(user_query):
    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {
            "messages": [("user", user_query)],
            "contents": contents,
            "relevant_sections": [],
        }, config, stream_mode="values"
    )
    events_list = list(events)
    last_event = events_list[-1]
 
    return last_event["messages"][-1].content

