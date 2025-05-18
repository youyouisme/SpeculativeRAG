import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

from dotenv import load_dotenv
load_dotenv()
# Load the API key from the environment variable
gpt_api_key = os.getenv("OPENAI_API_KEY")

def load_documents(data_dir: str):
    """Load and chunk text documents from a specified directory."""
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(data_dir, file)
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

def split_documents(docs: List[Document], chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for indexing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def create_vector_store(data_dir: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=gpt_api_key)
    raw_docs = load_documents(data_dir)
    documents = split_documents(raw_docs)
    print(f"Split new articles into {len(documents)} sub-documents.")
    faiss_vector_store = FAISS.from_documents(documents, embeddings)
    faiss_vector_store.save_local("faiss_index_textbook")
    return faiss_vector_store

def load_vector_store():
    ## Change the path to the path of the vector store
    faiss_index_path = "./faiss_index_textbook"
    if os.path.exists(faiss_index_path):
        print(f"Loading existing vector store {faiss_index_path}")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        faiss_vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded.")
    else:
        print("Vector store not found. Creating new vector store.")
        ## Change the path to the path of the raw textbooks
        faiss_vector_store = create_vector_store("./textbooks")
    return faiss_vector_store