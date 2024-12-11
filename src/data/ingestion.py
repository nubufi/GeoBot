import pymupdf4llm as pm
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    md_text = pm.to_markdown("src/data/tbdy2018.pdf")
    document = Document(page_content=md_text, metadata={"source": "tbdy2018.pdf"})

    print("splitting...")
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=10000, chunk_overlap=1000
    )
    texts = text_splitter.split_documents([document])
    print(f"created {len(texts)} chunks")

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    print("ingesting...")
    PineconeVectorStore.from_documents(texts, hf, index_name=os.environ["INDEX_NAME"])
    print("finish")
