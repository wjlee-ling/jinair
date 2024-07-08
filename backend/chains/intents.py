from enum import Enum
from operator import itemgetter

from langchain.output_parsers.enum import EnumOutputParser
from langchain_core.prompts import PromptTemplate


class Intent(Enum):
    search_flights = "search_flights"
    ask_QnA = "ask_QnA"


_TEMPLATE = """Given the chat history, query and output format, you are to classify the intent of the query.
Make sure to classify the intent of the query based on the chat history and query given. But refer to relevant information only from the chat history.
Make sure to return only the intent given in the output format.

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
        {"chat_history": itemgetter("chat_history"), "query": itemgetter("query")}
        | prompt
        | llm
        | parser
    ).with_config(run_name="intent_classification")

    return intent_classifier_chain
