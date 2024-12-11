from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""), model="gpt-4o-mini"
)


def answer_question(question: str) -> str:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                "Act like a Turkish civil engineer expert. You are working as a geotechnical engineering consultant who answers questions based on the given context. Make sure you give reference the section name such as 14.5.1",
            ),
            ("user", "<context>{context}</context>\n\n{input}"),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result = retrival_chain.invoke(input={"input": question})

    return result["answer"]
