from ..embeddings.embeddings_bm25 import get_trained_kiwi_retriever

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


CORPUS_PATH = "qna_0708.csv"

template = """Answer the question based only on the following context. \
If you need more information to better answer the question, please kindly ask for it. \
Make sure to answer in the lanuage of the Question. 
Make sure to include useful and relevant information in the context as much as possible.

## context
{context}

##
Question: {question}
A: 
"""


def _format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def _strip(string):
    return string.lstrip("A:").strip()


def get_QnA_chain(llm, retriever=None):
    prompt = ChatPromptTemplate.from_template(template)
    retriever = retriever or get_trained_kiwi_retriever(CORPUS_PATH, ["Q", "A"])
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | _format_docs,
            "question": itemgetter("input"),
        }
        | prompt
        | llm
        | StrOutputParser()
        | _strip
    ).with_config(run_name="RAG_chain_for_QnA")
    return rag_chain
