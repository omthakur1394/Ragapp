import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import Chroma_DB,EMBEDDING_MODEL
from ingest import load_doc

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(Chroma_DB):
        return Chroma(
            persist_directory=Chroma_DB,
            embedding_function=embeddings     
        )
    documents = load_doc()
    vectorstore = Chroma.from_documents(
        embedding=embeddings,
        documents=documents,
        persist_directory=Chroma_DB
    )
    return vectorstore

