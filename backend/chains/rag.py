from ..embeddings_bm25 import get_trained_kiwi_retriever

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "jinair"

CORPUS_PATH = "qna_0708.csv"

template = """Answer the question based only on the following context:

{context}

Question: {question}
A: 
"""


def _format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def _strip(string):
    return string.lstrip("A:").strip()


prompt = ChatPromptTemplate.from_template(template)
retriever = get_trained_kiwi_retriever(CORPUS_PATH, ["Q", "A"])


def get_QnA_chain(llm):
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | _strip
    ).with_config(run_name="RAG_chain_for_QnA")
    return rag_chain
