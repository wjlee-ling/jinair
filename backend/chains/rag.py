from ..embeddings_bm25 import get_trained_kiwi_retriever

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "jinair"

MODEL_NAME = "gpt-3.5-turbo-0125"
CORPUS_PATH = "qna_0708.csv"

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""


def _format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, verbose=True)
retriever = get_trained_kiwi_retriever(
    "/Users/lwj/workspace/jinair/qna_0708.csv", ["Q", "A"]
)

chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

resp = chain.invoke("기내식 예약할 수 있나요?")
print(resp)
