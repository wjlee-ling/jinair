from .flights import (
    FlightFinder,
    get_flights_chain,
    get_flights_SQL_chain,
)
from .rag import get_QnA_chain
from .intents import get_intent_classifier
from ..callbacks import FlightConditionCallbackHandler

import os
from dotenv import find_dotenv, load_dotenv

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(find_dotenv())


MODEL_NAME = "gpt-3.5-turbo-0125"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, verbose=True)
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# response = chain_entity_flights.invoke(
#     {"query": "1월 후에 인천에서 도쿄 가는 비행기", "state_entities": {}}
# )
# print(response.dict())
# sql_command = chain_sql_flights.invoke({"question": str(response.dict())})
# print(sql_command)

## agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for 진에어, South Korean Airline. You are to find flights that satisfy the needs and conditions of the user query.

Make sure to contextualize or augment the raw_input with the chat history and use the new raw_input when calling a function or tool.
Make sure to ask for more information when you need to figure out required entities like flight origin, destination, and/or date. \
But DO NOT ask for information about optional entities like the number of passengers or flight number.

## examples

Human: "11월 12일에 인천-도쿄 비행기"
AI: 원하시는 조건의 항공편을 찾았어요.:
- **비행 번호**: LJ201
- **출발지**: 인천
- **도착지**: 나리타
- **출발일**: 7월 5일
- **출발 시간**: 06:40
- **도착 시간**: 09:20
Human: 다른 건, 다른 비행기는 없어?
AI: Invoking `FlightFinder` with `11월 12일 인천-도쿄 flight_number != LJ201`

""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "raw_input: {raw_input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def get_flight_search_agent(llm):
    callbacks = [FlightConditionCallbackHandler()]
    chain_entity_flights = get_flights_chain(llm=llm)
    chain_sql_flights = get_flights_SQL_chain(llm=llm)
    tool = FlightFinder(
        llm=llm,
        slot_filler=chain_entity_flights,
        sql_runner=chain_sql_flights,
        callbacks=callbacks,
    )
    tools = [tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
    )

    return agent_executor


# states = {}
# user_input = {
#     "input": "인천에서 도쿄 가는 비행기",
#     "raw_input": "인천에서 도쿄 가는 비행기",
# }
# # for step in agent_executor.iter(user_input):
# #     print(step)
# resp = agent_executor.invoke(user_input)
# print("🤖", resp["output"])
# query = input()
# while query != "exit":
#     resp = agent_executor.invoke({"input": query, "raw_input": query})
#     print("🤖", resp["output"])
#     query = input()

__all__ = [
    "get_intent_classifier",
    "get_QnA_chain",
    "get_flight_search_agent",
]
