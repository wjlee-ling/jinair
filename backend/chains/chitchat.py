from chains.prompts import CHITCHAT_DIALOGUE_PROMPT

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv(find_dotenv())

MODEL_NAME = "gpt-4o"

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            CHITCHAT_DIALOGUE_PROMPT,  # Make sure to contextualize the user input with the chat history.
        ),
        # MessagesPlaceholder(variable_name="chat_history"),
        ("user", "input: {input}"),
    ]
)


def get_chitchat_chain(llm):
    chain = (
        # {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
        # |
        prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="chitchat_chain")

    return chain
