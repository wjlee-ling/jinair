import os

from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_teddynote.community.pinecone import (
    create_sparse_encoder,
    fit_save_sparse_encoder,
    init_pinecone_index,
    preprocess_documents,
    PineconeKiwiHybridRetriever,
    upsert_documents,
)
from langchain_teddynote.korean import stopwords
from pinecone import Pinecone


load_dotenv(find_dotenv())
INDEX_NAME = "jinair"
NAMESPACE = "0712"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def get_pinecone_index(index_name: str):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pc_index = pc.Index(index_name)
    return pc_index


def get_pinecone_kiwi_retriever(encoder_path: str):
    pinecone_params = init_pinecone_index(
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
        api_key=PINECONE_API_KEY,
        sparse_encoder_pkl_path=encoder_path,
        stopwords=stopwords(),
        tokenizer="kiwi",
        embeddings=embeddings,
        top_k=2,
        alpha=0.5,  # ratio of dense embeddings to sparse embeddings
    )

    retriever = PineconeKiwiHybridRetriever(**pinecone_params)
    return retriever


def upsert_korean_spare_embeddings_pinecone(
    csv_path, csv_loader_params: dict, pinecone_params: dict
):
    """ref: https://github.com/teddylee777/langchain-kr/blob/main/10-VectorStore/05-Pinecone.ipynb
    In order to get sparse embeddings specified for Korean preprocessed with Kiwi,
    1. remove Korean stopwords
    2. document-ify data
    3. upsert the given Pinecone Index with the new documents
    """

    assert "metadata_columns" in csv_loader_params
    assert "index" in pinecone_params and "namespace" in pinecone_params

    loader = CSVLoader(file_path=csv_path, **csv_loader_params)
    docs = loader.load()
    contents, metadatas = preprocess_documents(
        split_docs=docs, metadata_keys=csv_loader_params["metadata_columns"]
    )
    sparse_encoder = create_sparse_encoder(stopwords(), mode="kiwi")
    saved_path = fit_save_sparse_encoder(
        sparse_encoder=sparse_encoder,
        contents=contents,
        save_path="./sparse_encoder.pkl",
    )

    # upsert Pinecone Index
    upsert_documents(
        **pinecone_params,
        contents=contents,
        metadatas=metadatas,
        sparse_encoder=sparse_encoder,
        embedder=embeddings,
    )

    return saved_path


## 🚨 upsert twice -> create a new test.py and run there
# pc = Pinecone(api_key=PINECONE_API_KEY)
# pc_index = get_pinecone_index("jinair")
# saved_sparse_encoder = upsert_korean_spare_embeddings_pinecone(
#     csv_path="qna_0708.csv",
#     csv_loader_params={"metadata_columns": ["대주제", "소주제", "action"]},
#     pinecone_params={
#         "index": pc_index,
#         "namespace": NAMESPACE,
#     },
# )
