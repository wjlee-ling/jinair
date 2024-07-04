from .flights import get_flights_chain

from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo-0125"
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, verbose=True)

chain_entity_flights = get_flights_chain(llm=llm)
