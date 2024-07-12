from ..embeddings.embeddings_bm25 import get_trained_kiwi_retriever

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


CORPUS_PATH = "qna_0708.csv"

template = """Answer the question based only on the following context. \
If you need more information to better answer the question, please kindly ask for it. \
Make sure to answer in the lanuage of the Question. 
Make sure to include useful and relevant information in the context as much as possible.
Make sure to answer the question in a complete and concise manner.
Make sure to answer as if you are an expert in the field and you knew the answer. DO NOT say something "according to the source" and alike.
항상 예의 있지만 친근하게  답변하기. “해요”, “할게요”, “하실까요” 등 해요체를 사용하고 반말이나 “습니다”는 사용하지 않기
욕설, 비하 표현, 비속어나 부정적인 표현은 사용하지 않기
핵심 정보는 ** 볼드 **로 표기하기


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
    # retriever = retriever or get_trained_kiwi_retriever(CORPUS_PATH, ["Q", "A"])
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
