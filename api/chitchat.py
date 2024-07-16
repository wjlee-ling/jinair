from backend.chains import get_chitchat_chain

import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langserve.pydantic_v1 import BaseModel

load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "jinair-api"


class Input(BaseModel):
    input: str


# class Output(BaseModel):
#     output: str


chitchat_chain = (
    get_chitchat_chain(
        llm=ChatOpenAI(model_name="gpt-4o", temperature=0.2, verbose=True),
    ).with_types(input_type=Input)
    # .with_types(input_type=Input, output_type=Output)
    .with_config({"run_name": "run_chitchat"})
)
