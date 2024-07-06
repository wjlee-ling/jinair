from .flights import (
    get_flights_chain,
    get_flights_SQL_chain,
    FlightScheduleTool,
)

import os
from dotenv import find_dotenv, load_dotenv

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "jinair"


MODEL_NAME = "gpt-3.5-turbo-0125"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, verbose=True)
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# chain_entity_flights = get_flights_chain(llm=llm)
# chain_sql_flights = get_flights_SQL_chain(llm=llm)

# response = chain_entity_flights.invoke(
#     {"query": "도쿄행 8월 1일 이후 출발", "state_entities": {"origin": "인천"}}
# )
# sql_command = chain_sql_flights.invoke({"question": str(response.dict())})
# print(response.dict())
# print(sql_command)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Always use the full input when calling any function or tool.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
tools = [FlightScheduleTool(llm=llm)]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


user_input = {"input": "인천에서 도쿄 가는 8월 3일 비행기"}
# resp = agent_executor.invoke({"input": "도쿄 가는 비행기"})
for step in agent_executor.iter(user_input):
    print(step)
