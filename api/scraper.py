from backend.chains import run_web_scraping

import os
from functools import partial
from dotenv import find_dotenv, load_dotenv
from langchain_core.runnables import RunnableLambda
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

_fn = partial(
    run_web_scraping,
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
    root_url="https://help.jinair.com/hc/ko/categories/4408759363353-%EC%9E%90%EC%A3%BC-%EB%AC%BB%EB%8A%94-%EC%A7%88%EB%AC%B8-FAQ",
)

scraper_chain = (
    RunnableLambda(lambda inputs: _fn(inputs["input"])).with_types(input_type=Input)
    # .with_types(input_type=Input, output_type=Output)
    .with_config({"run_name": "run_scraper"})
)
