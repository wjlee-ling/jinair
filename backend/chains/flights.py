import os
import re

from dotenv import find_dotenv, load_dotenv

from langchain.chains import create_sql_query_chain
from langchain.output_parsers import PydanticOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.runnables import (
    RunnableLambda,
)

load_dotenv(find_dotenv())
AWS_RDS_PASSWORD = os.getenv("AWS_RDS_PASSWORD")
AWS_RDS_HOST = os.getenv("AWS_RDS_HOST")

MAP_AIRPORTS = {
    "인천": "서울/인천",
    "서울": "서울/인천",
    "서울 인천": "서울/인천",
    "나리타": "도쿄/나리타",
    "도쿄": "도쿄/나리타",
}


class Flight(BaseModel):
    origin: str = Field(description="origin city or airport of the flight")
    destination: str = Field(description="destination city or aiport of the flight")
    date: str = Field(description="date of the flight")
    persons: int = Field(1, description="number of persons for booking")
    follow_up: str = Field(description="follow up question for necessary entities")
    # price: Optional[int] = Field(None, description="price of the flight")

    @validator("origin", allow_reuse=True)
    def postprocess_origin(cls, field):
        field = re.sub("공항", "", field)
        field = re.sub("국제", "", field).strip()
        if field in MAP_AIRPORTS:
            field = MAP_AIRPORTS[field]

        return field

    @validator("destination", allow_reuse=True)
    def postprocess_destination(cls, field):
        field = re.sub("공항", "", field)
        field = re.sub("국제", "", field).strip()
        if field in MAP_AIRPORTS:
            field = MAP_AIRPORTS[field]

        return field


_template = """Given the user [query], [entities] and [output format] below, you are to:
1. extract entities from the query \
2. fill in and update the [entities] with newly extracted entities and \
3. as an airline chatbot ask **follow_up** questions to fill in empty slots for **necessary** entities.
Make sure not to make up information.
Make sure to extract & answer the follow-up question in Korean.
Make sure to leave 'follow-up' empty when all the rest entities are extracted.

## [output format]
{output_format}

## [entities]
{state_entities}

## [query]
{query}

## [output]
"""

_Text2SQL_template = """You are a PostgreSQL expert. Given an input, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question. \
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database. \
Wrap each column name in double quotes (") to denote them as delimited identifiers.

Make sure to use `ILIKE` rather than `=` for 'origin' and 'destination'.
Make sure to use only the columns you can see in the tables below. Be careful to not query for columns that do not exist.
Make sure to format the values of [input] according to the corresponding data types given in the [table info].
Make sure to set the year of the date to 2024 even if it is not mentioned.

## table info:
{table_info}

## examples
input:
{{'origin': '인천', 'destination': '나리타', 'date': '7월 12일', 'persons': 1}}

SQLQuery: 
SELECT * FROM flights WHERE origin ILIKE '인천' AND destination ILIKE '나리타' AND departure_date = '2024-07-12' AND persons = 1 LIMIT 3

## [input]
{input}

"""
Text2SQL_PROMPT = PromptTemplate.from_template(template=_Text2SQL_template)


def get_flights_chain(llm):

    parser = PydanticOutputParser(pydantic_object=Flight)
    prompt = PromptTemplate(
        template=_template,
        input_variables=["query", "state_entities"],
        partial_variables={"output_format": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    return chain


def get_flights_SQL_chain(llm):

    db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://root:{AWS_RDS_PASSWORD}@{AWS_RDS_HOST}:5432/textnet",
        include_tables=["flights"],
        sample_rows_in_table_info=2,
    )

    write_query = create_sql_query_chain(
        llm,
        db,
        k=3,
        prompt=Text2SQL_PROMPT,
    ).with_config(
        run_name="write_sql_query"
    )  # https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/sql_database/query.py

    # # `write_query`에서 만들어진 쿼리에 임베딩할 단어가 있거나, 오류가 있을 시 쿼리를 수정합니다.
    # rewrite_query = RunnableLambda(lambda sql_query: _get_query(sql_query)).with_config(
    #     run_name="rewrite_query"
    # )

    execute_query = QuerySQLDataBaseTool(db=db).with_config(
        run_time="execute_sql_query"
    )

    return_no_result_messages = RunnableLambda(
        lambda result: (
            "해당 조건을 만족하는 항공편이 없습니다. 새로 검색하시겠습니까?"
            if result == ""
            else result
        )
    ).with_config(run_name="return_no_result_messages")

    chain = (write_query | execute_query | return_no_result_messages).with_config(
        run_name="search_flights_from_db"
    )
    return chain
