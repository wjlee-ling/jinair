from chains.prompts import FLIGHT_DIALOGUE_PROMPT

import os
import re

from dotenv import find_dotenv, load_dotenv
from operator import itemgetter
from typing import Optional


from langchain.chains import create_sql_query_chain
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import BaseTool
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
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


class FlightCondition(BaseModel):
    origin: str = Field(description="origin city or airport the user departs from")
    destination: str = Field(description="destination city or aiport the user flies to")
    date: str = Field(description="date the user wants to book a flight for")
    persons: int = Field(1, description="number of persons for booking")
    flight_number: Optional[str] = Field("", description="flight number of the flight")
    # follow_up: str = Field(description="follow up question for necessary entities")
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


_template = """Given the query, entities and output format below, you are to:
1. extract entities from the query \
2. fill in and update the entities with newly extracted entities.

Make sure NOT TO make up information or infer. You should only **extract** entities from the given query.
Make sure the date is in the format 'YYYY-MM-DD'. And the year ('YYYY') is 2024 if not specified in the query.
Make sure to include the comparison operators (eq, gt, gte, lt, lte, or, not) if there are such words.
Make sure to calculate the total number of persons mentioned in the query and the history.

## output format
{output_format}

---

## entities
{{}}

## query
8월 이후 인천행 비행기 예약

## output
{{ "origin": "", "destination": "인천", "date": "gte 8월", "persons": 1, "flight_number": "" }}

---

## entities
{{}}

## query
1월 8일 런던-파리 성인 하나 아이 둘

## output
{{ "origin": "런던", "destination": "파리", "date": "2024-01-08", "persons": 3, "flight_number": ""  }}

---

## entities
{{ 'date': '2024-08-19' }}

## query
제주 가는 비행기

## output
{{ "origin": "", "destination": "제주", "date": "2024-08-19", "persons": 1, "flight_number": ""  }}

---

## entities
{{ 'origin': '서울/인천', 'destination': '제주', 'date': 'gte 2024-07-05', 'persons': 1, "flight_number": ""  }}

## query
김포에서 가는 건 없어?

## output
{{ "origin": "김포", "destination": "제주", "date": "gte 2024-07-05", "persons": 1, "flight_number": "" }}

---

## entities
{{ "origin": "서울/인천", 'destination': '제주', 'date': '2024-07-05', 'persons': 1, 'flight_number': 'LJ123' }}

## query
다른 비행기 편은 없어?

## output
{{ "origin": "서울/인천", "destination": "제주", "date": "2024-07-05", "persons": 1, "flight_number": "not LJ123" }}

Now Begin! Make sure to add or update the entities with infomation extracted from the query.

## entities
{state_entities}

## query
{query}

## output
"""

_Text2SQL_template = """You are a PostgreSQL expert. Given an input, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question. \
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database. \
Wrap each column name in double quotes (") to denote them as delimited identifiers.

Make sure to use `ILIKE` rather than `=` for 'origin' and 'destination'.
Make sure to use only the columns you can see in the tables below. Be careful to not query for columns that do not exist.
Make sure to format the values of input according to the corresponding data types given in the table info.
Make sure to set the year of the date to 2024 even if it is not mentioned.
Make sure to use the flight_number in the input if it is given and is not empty.

## table info:
{table_info}

## examples
input:
{{'origin': '출발지', 'destination': '도착지', 'date': 'YYYY-MM-DD', 'persons': '사람수', 'flight_number': ''}}

SQLQuery: 
SELECT * FROM flights WHERE origin ILIKE '출발지' AND destination ILIKE '도착지' AND departure_date = 'YYYY-MM-DD' AND persons = '사람수' LIMIT 3

---

input:
{{'origin': '출발지', 'destination': '도착지', 'date': 'YYYY-MM-DD', 'persons': '사람수', 'flight_number': '비행기명'}}

SQLQuery: 
SELECT * FROM flights WHERE origin ILIKE '출발지' AND destination ILIKE '도착지' AND departure_date = 'YYYY-MM-DD' AND persons = '사람수' AND flight_number ='비행기명' LIMIT 3

## begin!
input:
{input}

SQLQuery:
"""
Text2SQL_PROMPT = PromptTemplate.from_template(template=_Text2SQL_template)


def get_flights_chain(llm):

    parser = PydanticOutputParser(pydantic_object=FlightCondition)
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

    def _handle_query_error(query):
        if not query.startswith("SELECT"):
            query = query.strip("`")
            query = query.strip("sql")
            query = query.strip("\n")
            return query
        return query

    def _rewrite_query(prev_dict) -> dict:
        sql_query = prev_dict["sql_query"]
        departure_part = re.search(
            r"departure_date [=><]+ ['\-0-9 ]+", sql_query
        )  # -> departure_date = '2024-07-05'
        (start_idx, end_idx) = departure_part.span()
        departure_date = (
            re.search(r"['\-0-9]+", departure_part.group()).group().strip()
        )  #  -> '2024-07-05'

        new_sql_query = (
            sql_query[:start_idx]
            + f"departure_date BETWEEN (DATE {departure_date} - INTERVAL '1 month') AND (DATE {departure_date} + INTERVAL '1 month')"
            + sql_query[end_idx:]
        )
        # print(new_sql_query)
        new_results = db.run(new_sql_query)
        if new_results:
            response = (
                "## 원래 **고객이 원하는 날짜에 항공편이 없어** 출발일을 조정하여 새로 검색한 결과라는 것을 강조하기##\n"
                + new_results
            )
        else:
            response = "원래 고객이 원하는 항공편이 없으니 죄송함을 표현 후 새로운 항공편 검색 원하는지 물어보기"

        return {"sql_query": new_sql_query, "results": response}

    write_query = create_sql_query_chain(
        llm,
        db,
        k=3,
        prompt=Text2SQL_PROMPT,
    ).with_config(
        run_name="write_sql_query"
    )  # https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/sql_database/query.py

    # `write_query`에서 만들어진 쿼리에 임베딩할 단어가 있거나, 오류가 있을 시 쿼리를 수정합니다.
    handle_query_error = RunnableLambda(
        lambda sql_query: _handle_query_error(sql_query)
    ).with_config(run_name="rewrite_query")

    execute_query = QuerySQLDataBaseTool(db=db).with_config(
        run_time="execute_sql_query"
    )

    write_text2sql = (
        write_query
        | handle_query_error
        | {
            "sql_query": RunnablePassthrough(),
            "results": execute_query,
        }
    )

    retry_if_no_results = RunnableLambda(
        lambda prev_dict: (
            _rewrite_query(prev_dict) if prev_dict["results"] == "" else prev_dict
        )
    ).with_config(run_name="retry_if_no_results")

    chain = (write_text2sql | retry_if_no_results).with_config(
        run_name="search_flights_from_db"
    )

    return chain


class FlightFinder(BaseTool):
    llm: BaseChatModel
    slot_filler: Runnable
    sql_runner: Runnable
    callbacks: list[BaseCallbackHandler]
    description = "useful when the user wants to searches for flight schedules that satisfy conditions. Before searching, the tool extracts entities from the query."
    name = "FlightFinder"

    def _is_slot_empty(self, entities: dict):
        empty_slots = [
            entity
            for entity, value in entities.items()
            if (entity != "flight_number" and value == "")
        ]
        if empty_slots == []:
            return False
        return empty_slots

    def _ask_follow_up(self, empty_slots: list):
        reply = "항공권 검색을 위해 " + ", ".join(empty_slots) + " 정보가 필요합니다."
        return reply

    def _run(self, query: str):
        states = self.callbacks[0].entities
        entities = self.slot_filler.invoke(
            {"query": query, "state_entities": states},
            config={"callbacks": self.callbacks, "run_name": "fill_slots"},
        ).dict()

        if empty_slots := self._is_slot_empty(entities):
            return self._ask_follow_up(empty_slots)

        response = self.sql_runner.invoke({"question": str(entities)})
        return response


_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", FLIGHT_DIALOGUE_PROMPT),
        ("user", "raw_input: {input}"),
    ]
)


def _format_input(dict):
    def _postprocess_city(city):
        city = re.sub("공항", "", city)
        city = re.sub("국제", "", city).strip()
        if city in MAP_AIRPORTS:
            city = MAP_AIRPORTS[city]
        return city

    old = dict["input"]
    dict["question"] = {
        "origin": _postprocess_city(old["origin"]),
        "destination": _postprocess_city(old["destination"]),
        "date": old["date"],
        "persons": old["persons"] if "persons" in old else 1,
        "flight_number": old["flight_number"] if "flight_number" in old else "",
    }
    dict["question"] = str(dict["question"])
    dict.pop("input")
    return dict


def get_flight_search_API_chain(agent_llm, chain_llm):
    chain_sql_flights = (
        RunnableLambda(lambda x: _format_input(x))
        | get_flights_SQL_chain(llm=chain_llm)
        | {"input": itemgetter("results")}
        | _prompt
        | agent_llm
        | StrOutputParser()
    )

    return chain_sql_flights
