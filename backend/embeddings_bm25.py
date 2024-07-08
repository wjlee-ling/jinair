"""TODO
The current pinecone_text.sparse.BM25Encoder cannot process Korean. Modifiy the BM25Encoder to process Korean.
"""

import pandas as pd


def _get_train_data(csv_path, columns) -> list:
    df = pd.read_csv(csv_path, header=0)[columns]
    df = df.map(lambda cell: cell.strip())
    df["corpus"] = df.apply(
        lambda row: "\n".join([f"{col}: {row[col]}" for col in columns]), axis=1
    )
    return df["corpus"].to_list()


def get_trained_pinecone_bm25_encoder(csv_path, columns, model_name):
    from pinecone_text.sparse import BM25Encoder

    """Fit BM25 encoder to a csv file of text data. Each row is joined into a single document."""
    corpus = _get_train_data(csv_path, columns)

    bm25_encoder = BM25Encoder().default()
    bm25_encoder.fit(corpus)
    bm25_encoder.dump(f"{model_name}.json")
    bm25_encoder = BM25Encoder().load(f"{model_name}.json")
    return bm25_encoder


def get_trained_kiwi_retriever(csv_path, columns):
    from langchain_teddynote.retrievers import KiwiBM25Retriever

    corpus = _get_train_data(csv_path, columns)
    kiwi = KiwiBM25Retriever.from_texts(corpus)
    kiwi.k = 2
    return kiwi
