# from backend.chains import (
#     get_QnA_chain,
#     get_flight_search_agent,
# )
from api.flight_search import run_flight_search
import os

from typing import Any, List, Union
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "jinair-api"

load_dotenv(find_dotenv())


# openai_4o = ChatOpenAI(model_name="gpt-4o", temperature=0, verbose=True)
# openai = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0, verbose=True)

app = FastAPI(
    title="Jaid API Server",
    version="1.0",
    description="API server for Jaid",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class Input(BaseModel):
    raw_input: str
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = Field(
        ...,
        extra={
            "widget": {"type": "chat", "raw_input": "raw_input", "output": "output"}
        },
    )


class Output(BaseModel):
    output: str


add_routes(
    app,
    RunnableLambda(run_flight_search)
    .with_types(input_type=Input, output_type=Output)
    .with_config({"run_name": "run_flight_search"}),
    path="/flight_search",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
