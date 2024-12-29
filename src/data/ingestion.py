from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
        api_version="2023-05-15"
)

def read_file(file_name):
    with open("markdowns/"+file_name, 'r') as file:
        content= file.read()
        document = Document(page_content=content, metadata={"source": file_name})

    return [document]

text_splitter = RecursiveCharacterTextSplitter.from_language(
    Language.MARKDOWN, chunk_size=5000, chunk_overlap=1000
)

def save_vector(file_head):
    document = read_file(file_head+".md")
    docs = text_splitter.split_documents(document)
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(f"vector_stores/tbdy/{file_head}.faiss")

if __name__ == "__main__":
    for f in os.listdir("markdowns"):
        if f.endswith(".md"):
            save_vector(f[:-3])
