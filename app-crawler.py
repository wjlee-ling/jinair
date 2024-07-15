from backend.chains import run_web_scraping

import os
import streamlit as st

from dotenv import find_dotenv, load_dotenv
from streamlit import session_state as sst

# from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "jinair-test"


st.set_page_config(layout="wide")
MODEL_NAME = "gpt-3.5-turbo-0125"
GREETING = "안녕하세요. 🤖 Jaid입니다! 항공편 조회나 자주묻는질문 검색이 가능해요😀"


# @st.cache_resource
def load_chains(model_name, temp=0.0):
    # anthropic = ChatAnthropic(
    #     model_name="claude-3-5-sonnet-20240620", temperature=temp, verbose=True
    # )
    sst.openai_4o = ChatOpenAI(model_name="gpt-4o", temperature=temp, verbose=True)
    print("🚒 Chains have been newly created.")


if "messages" not in sst:
    sst.messages = []
    sst.steps = []
    load_chains(model_name=MODEL_NAME)

# 출력되는 이미지 크기 조정

st.title("JinAir Jaid 크롤링 기반 QnA 🤖")

with st.chat_message("ai"):
    st.markdown(GREETING)

# 대화 내역을 출력합니다.
for i, message in enumerate(sst.messages):
    role = "human" if isinstance(message, HumanMessage) else "ai"
    with st.chat_message(role):
        st.markdown(message.content)


if prompt := st.chat_input(""):
    sst.steps.append(None)
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        answer = run_web_scraping(
            prompt,
            sst.openai_4o,
            root_url="https://help.jinair.com/hc/ko/categories/4408759363353-%EC%9E%90%EC%A3%BC-%EB%AC%BB%EB%8A%94-%EC%A7%88%EB%AC%B8-FAQ",
        )

        sst.reply_placeholder = st.empty()
        sst.reply_placeholder.markdown(answer)
        final_answer = answer

    sst.messages.append(HumanMessage(content=prompt))
    sst.messages.append(AIMessage(content=final_answer))
