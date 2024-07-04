from .flights import get_flights_chain, get_flights_SQL_chain

import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "jinair"


MODEL_NAME = "gpt-3.5-turbo-0125"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, verbose=True)
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

chain_entity_flights = get_flights_chain(llm=llm)
chain_sql_flights = get_flights_SQL_chain(llm=llm)

response = chain_entity_flights.invoke(
    {"query": "7월 12일에 인천에서 나리타 성인 1명", "state_entities": ""}
)
print(response.dict())

sql_command = chain_sql_flights.invoke({"question": str(response.dict())})
print(sql_command)
