from backend.chains import (
    get_flight_search_agent,
)
import os

from typing import Any, List, Union

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langserve.pydantic_v1 import BaseModel, Field

load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "jinair-api"


class FlightSearchInput(BaseModel):
    raw_input: str
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = Field(
        ...,
        extra={
            "widget": {"type": "chat", "raw_input": "raw_input", "output": "output"}
        },
    )


class FlightSearchOutput(BaseModel):
    output: str


openai_4o = ChatOpenAI(model_name="gpt-4o", temperature=0, verbose=True)
openai = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0, verbose=True)
flight_search_agent = (
    get_flight_search_agent(agent_llm=openai_4o, chain_llm=openai)
    .with_types(input_type=FlightSearchInput, output_type=FlightSearchOutput)
    .with_config({"run_name": "flight_search_agent"})
)
