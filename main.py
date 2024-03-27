import logging
from utility import *
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    load_api_key()
    db = build_sample_db()

    # 임베딩 모델을 오픈소스(허깅페이스)모델로 이용할 수 있습니다(OpenAI 없이 로컬에서 동작가능)
    # 또한 데이터셋도 로컬 텍스트 파일로 이용가능합니다.
    #db = build_sample_db(
        #loader = TextLoader("./local_datasets/state_of_the_union.txt"),
        #embedding= HuggingFaceEmbeddings()
        #                 )

    #기본 retriever
    retriever = initialize_retriever(db)
    
    #Test for compression
    #docs = retriever.get_relevant_documents(
    #"What did the president say about Ketanji Brown Jackson"
    #)
    #pretty_print_docs(docs)

    #multiquery retriever
    #multiquery_retriever = initialize_multiquery_retriever(db)

    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    question = "OpenAI Assistant API의 Functions 사용법에 대해 알려주세요."
    relevant_docs = search_documents(retriever, question)

    print(f"===============\n검색된 문서 개수: {len(relevant_docs)}", end="\n===============\n")
    print(relevant_docs[0].page_content)

if __name__ == "__main__":
    main()
