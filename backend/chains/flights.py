import re

from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI


class Flight(BaseModel):
    origin: str = Field(description="origin city or airport of the flight")
    destination: str = Field(description="destination city or aiport of the flight")
    date: str = Field(description="date of the flight")
    persons: int = Field(description="number of persons for booking")

    @validator("origin")
    def question_ends_with_question_mark(cls, field):
        if field.endswith("공항"):
            return re.sub("공항", "", field).strip()

        return field.strip()

    @validator("destination")
    def question_ends_with_question_mark(cls, field):
        if field.endswith("공항"):
            return re.sub("공항", "", field).strip()

        return field.strip()


_template = """Given the user [query], [entities] and [output format] below, you are to:
1. extract entities from the query \
2. fill in and update the [entities] with newly extracted entities \
3. and ask follow-up questions to fill in empty slots for **necessary** entities.
Make sure you do not make up information.

## [output format]
{output_format}

## [entities]
{state_entities}

## [query]
{query}

## [output]
"""

parser = PydanticOutputParser(pydantic_object=Flight)
prompt = PromptTemplate(
    template=_template,
    input_variables=["query", "state_entities"],
    partial_variables={"output_format": parser.get_format_instructions()},
)
