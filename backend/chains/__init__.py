from .flights import (
    get_flights_chain,
    get_flights_SQL_chain,
    FlightScheduleTool,
    FlightCondition,
)

import os
from dotenv import find_dotenv, load_dotenv

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "jinair"


MODEL_NAME = "gpt-3.5-turbo-0125"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, verbose=True)
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# chain_entity_flights = get_flights_chain(llm=llm)
# chain_sql_flights = get_flights_SQL_chain(llm=llm)

# response = chain_entity_flights.invoke(
#     {"query": "1월 후에 인천에서 도쿄 가는 비행기", "state_entities": {}}
# )
# print(response.dict())
# sql_command = chain_sql_flights.invoke({"question": str(response.dict())})
# print(sql_command)

## agent
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant. Remember that the current year is 2024. Do not make up infomation.",
#         ),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )
# tools = [FlightScheduleTool(llm=llm)]
# agent = create_tool_calling_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# user_input = {"input": "인천에서 도쿄 가는 비행기"}
# for step in agent_executor.iter(user_input):
#     print(step)
