from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List
import os

embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
        api_version="2023-05-15"
)
llm = AzureChatOpenAI()

prompt_template = ChatPromptTemplate(
    messages=[
        (
            "system",
            "Sen bir uzman inşaat mühendisisin. Verilen bilgilere göre kullanıcının sorusunu cevapla."
        ),
        (
            "user",
            "Verilen bilgiyi kullanarak, kullanıcının sorusunu cevapla:\n\n{context}\n\nSoru: {user_input}"
        ),
    ]
)

def get_retrievers(section_name:List[str]) -> List[FAISS]:
    retrievers = []
    for name in section_name:
        retriever = FAISS.load_local(
            f"data/vector_stores/tbdy/section{name}.faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )
        retrievers.append(retriever.as_retriever())

    return retrievers

def search_multiple_vectorstores(retrievers,user_input) -> List[Document]:
    # Retrieve documents from multiple vector stores
    documents = []
    for retriever in retrievers:
        documents += retriever.invoke(user_input)
    
    return documents

def get_answer_chain(section_names:List[str]):
    retrievers = get_retrievers(section_names)

    def answer_user_query(user_input: str) -> str:
        # Step 3: Search across multiple vector stores
        documents = search_multiple_vectorstores(retrievers, user_input)

        # Step 4: Prepare the context for the LLM
        context = "\n".join([doc.page_content for doc in documents])  # Assuming each Document has a 'page_content' attribute

        # Step 5: Create the prompt using the context and user input
        prompt = prompt_template.invoke({
            "context": context,
            "user_input": user_input
        })

        # Step 6: Generate a response using the LLM
        response = llm.invoke(prompt)

        return response

    return answer_user_query

