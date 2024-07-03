from backend.chains import get_conversational_agent
from backend.tools import get_tools

import os
import streamlit as st

from dotenv import load_dotenv
from streamlit import session_state as sst
from langchain import callbacks, hub
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langsmith import Client

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "jinair"

st.set_page_config(layout="wide")
MODEL_NAME = "gpt-4o"  # "gpt-3.5-turbo-0125"
GREETING = ""
SYSTEM_MESSAGE = "🤖 세일즈봇이 준비되었습니다."


@st.cache_resource
def get_langsmith_client():
    """
    log 분석을 위해 langsmith client를 생성합니다.
    """
    return Client(api_key=os.environ["LANGCHAIN_API_KEY"])


def refresh():
    st.rerun()


@st.cache_resource
def create_conversational_agent(
    model_name="gpt-4o",
    temp=0.0,
):
    """
    유저가 지정한 프롬프트(`agent_instruct_template`)를 사용하여 GPT 기반의 agent를 생성합니다. 이 agent는 멀티턴 대화를 처리하고, 툴을 사용합니다. 프롬프트에 중괄호('{}')가 쓰이는 경우가 있어 `template_format="jinja2"`로 설정합니다.
    """
    sst.llm = ChatOpenAI(model_name=model_name, temperature=temp, verbose=True)
    sst.tools = get_tools()

    chain = get_conversational_agent(
        llm=sst.llm,
        tools=sst.tools,
        prompts=sst.templates,
    )

    print("🚒 Chains have been newly created.")
    return chain


if "messages" not in sst:
    sst.agent = create_conversational_agent(model_name=MODEL_NAME)
    sst.messages = []
    sst.steps = []
    sst.langsmith_client = get_langsmith_client()
    sst.templates = {}

# 출력되는 이미지 크기 조정
st.markdown(
    """
<style>
img {
    max-height: 150px;
}
</style>
""",
    unsafe_allow_html=True,
)
st.title("JinAir 세일즈봇 🤖")

if "agent" in sst:
    sst.messages.append(SystemMessage(content=SYSTEM_MESSAGE))

    with st.chat_message("ai"):
        st.markdown(GREETING)

# 대화 내역을 출력합니다.
for i, message in enumerate(sst.messages):
    role = "human" if isinstance(message, HumanMessage) else "ai"
    with st.chat_message(role):
        st.markdown(message.content)

        if role == "ai":
            with st.expander("🤖 내부 단계", expanded=False):
                st.write(sst.steps[i])

if prompt := st.chat_input(""):
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        sst.reply_placeholder = st.empty("")
        outputs = sst.agent.stream(
            {
                "user_query": prompt,
                "chat_history": sst.messages,
            },
            config={
                "callbacks": [st_callback],
            },
        )
        answer = ""
        for output in outputs:
            # pprint(output)
            for msg in output["messages"]:
                # tool 사용 전, 후의 AI 메시지 통합
                if isinstance(msg, (AIMessageChunk, AIMessage)):
                    msg_chunk = msg.content
                    if not answer.endswith(msg_chunk):
                        answer += "\n" + (msg.content)
                        sst.reply_placeholder.markdown(answer)

            if "intermediate_steps" in output:
                intermediate = output["intermediate_steps"]

        final_answer = answer

    sst.messages.append(HumanMessage(content=prompt))
    sst.messages.append(AIMessage(content=final_answer))
    sst.steps.append(None)
    sst.steps.append(intermediate)
