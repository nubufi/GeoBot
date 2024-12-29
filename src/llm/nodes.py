from llm.chains.section_chain import section_chain
from llm.state import State
from llm.chains.router import RouteQuery, question_router
from llm.chains.chat_chain import chat_chain
from llm.chains.answer_chain import get_answer_chain
from llm.consts import SEARCH,CHAT

def find_relevant_sections(state:State):
    contents = state["contents"]
    user_prompt = state["messages"][-1].content
    
    sections =  section_chain.invoke({"input": user_prompt, "contents": contents}).sections

    return {"relevant_sections": sections}

def chat(state:State):
    output = chat_chain.invoke(state["messages"])

    return {"messages":output.content}

def answer_question(state:State):
    sections = state["relevant_sections"]
    answer_chain = get_answer_chain(sections)
    output = answer_chain(state["messages"][-1].content)

    return {"messages":output.content}

def route_question(state: State) -> str:

    question = state["messages"][-1].content
    source: RouteQuery = question_router.invoke({"question": question})

    if source.datasource == SEARCH:
        return SEARCH
    elif source.datasource == CHAT:
        return CHAT
