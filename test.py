## Build a sample vectorDB
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# Load data
loader = TextLoader("./state_of_the_union.txt")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(data)

embedding =  HuggingFaceEmbeddings()
retriever = FAISS.from_documents(texts, embedding).as_retriever()

docs = retriever.get_relevant_documents(
    "What did the president say about Ketanji Brown Jackson"
)
pretty_print_docs(docs)




