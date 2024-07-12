from enum import Enum
from operator import itemgetter

from langchain.output_parsers.enum import EnumOutputParser
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


class Intent(Enum):
    search_flights = "search_flights"
    ask_QnA = "ask_QnA"
    chitchat = "chitchat"


_TEMPLATE = """Given the chat history, query and output format, you are to classify the intent of the query.
Make sure to classify the intent of the query based on the chat history and query given. But refer to relevant information only from the chat history.
Make sure to return only the intent given in the output format without any prefix or suffix.

## 설명
- search_flights: 조건에 맞는 항공편 및 항공 스케쥴 검색
- ask_QnA : 예매, 결제, 할인, 체크인, 수속, 수하물 및 기타 기내 서비스에 관한 질문에 답변
- chitchat : 기타 질의. 분류할 수 없는 질문은 "chitchat"으로 분류

## output format
{output_format}

## chat history
{chat_history}

## query
{query}

## intent
"""


def get_intent_classifier(llm):
    parser = EnumOutputParser(enum=Intent)
    prompt = PromptTemplate(
        template=_TEMPLATE,
        input_variables=["chat_history", "query"],
        partial_variables={"output_format": parser.get_format_instructions()},
    )

    intent_classifier_chain = (
        {
            "chat_history": itemgetter("chat_history")
            | RunnableLambda(lambda x: get_buffer_string(x)),
            "query": itemgetter("query"),
        }
        | prompt
        | llm
        | parser
    ).with_config(run_name="intent_classification")

    return intent_classifier_chain
