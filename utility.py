from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
#from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
import os

def load_api_key():
    load_dotenv()
    #print(f"[API KEY]\n{os.environ['OPENAI_API_KEY']}")

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
    
def build_sample_db():
    loader = WebBaseLoader("https://teddylee777.github.io/openai/openai-assistant-tutorial/", encoding="utf-8")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = loader.load_and_split(text_splitter)
    openai_embedding = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, openai_embedding)
    return db

def initialize_retriever(db):
    retriever = db.as_retriever()
    return retriever

def initialize_multiquery_retriever(db):
    llm = ChatOpenAI(temperature=0)
    multiquery_retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
    return multiquery_retriever

def search_documents(retriever, query):
    relevant_docs = retriever.get_relevant_documents(query)
    return relevant_docs