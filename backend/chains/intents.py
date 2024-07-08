from enum import Enum

from langchain.output_parsers.enum import EnumOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


class Intent(Enum):
    search_flights = "search_flights"
    ask_QnA = "ask_QnA"


_TEMPLATE = """Given the chat history, query and output format, you are to classify the intent of the query.
Make sure to return only the intent given in the output format.

## chat history
{chat_history}

## output format
{output_format}

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
        {"chat_history": RunnablePassthrough(), "query": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    ).with_config(run_name="intent_classification")

    return intent_classifier_chain
