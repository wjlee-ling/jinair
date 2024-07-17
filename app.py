from backend.api import request_LLM_API
from backend.embeddings import get_pinecone_kiwi_retriever
from backend.callbacks import CustomStreamlitCallbackHandler
from backend.chains import (
    get_chitchat_chain,
    get_intent_classifier,
    get_QnA_chain,
    get_flight_search_agent,
)

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
os.environ["LANGCHAIN_PROJECT"] = "jinair"


st.set_page_config(layout="wide")
MODEL_NAME = "gpt-3.5-turbo-0125"
GREETING = "안녕하세요. 🤖 Jaid입니다! 항공편 조회나 자주묻는질문 검색이 가능해요😀"


# @st.cache_resource
def load_chains(model_name, temp=0.0):
    # anthropic = ChatAnthropic(
    #     model_name="claude-3-5-sonnet-20240620", temperature=temp, verbose=True
    # )
    openai_4o = ChatOpenAI(model_name="gpt-4o", temperature=temp, verbose=True)
    openai = ChatOpenAI(model_name=model_name, temperature=temp, verbose=True)
    sst.intent_classifier = get_intent_classifier(openai)
    sst.flight_search_agent = get_flight_search_agent(
        agent_llm=openai_4o, chain_llm=openai
    )
    sst.hybrid_retriever = get_pinecone_kiwi_retriever("sparse_encoder.pkl")
    sst.QnA_chain = get_QnA_chain(openai_4o, sst.hybrid_retriever)
    sst.chitchat_chain = get_chitchat_chain(openai)
    print("🚒 Chains have been newly created.")


if "messages" not in sst:
    sst.messages = []
    sst.steps = []
    load_chains(model_name=MODEL_NAME)

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
st.title("JinAir Jaid 🤖")

with st.chat_message("ai"):
    st.markdown(GREETING)

# 대화 내역을 출력합니다.
for i, message in enumerate(sst.messages):
    role = "human" if isinstance(message, HumanMessage) else "ai"
    with st.chat_message(role):
        st.markdown(message.content)

        # if role == "ai" and sst.steps[i]:
        #     with st.expander("🤖 내부 단계", expanded=False):
        #         st.write(sst.steps[i])

if prompt := st.chat_input(""):
    sst.steps.append(None)
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st_callback = CustomStreamlitCallbackHandler(st.container())
        # st_callback = StreamlitCallbackHandler(st.container())
        sst.reply_placeholder = st.empty()

        intent = sst.intent_classifier.invoke(
            {"chat_history": sst.messages, "query": prompt}
        )

        if intent.name == "search_flights":
            chat_history = sst.messages if len(sst.messages) > 1 else []
            answer = request_LLM_API(
                chain=sst.flight_search_agent,
                callbacks=[st_callback],
                inputs={"chat_history": chat_history, "raw_input": prompt},
            )
            sst.reply_placeholder.markdown(answer)

            final_answer = answer

        elif intent.name == "ask_QnA":
            outputs = sst.QnA_chain.stream(
                {"input": prompt},  # , "chat_history": sst.messages
                config={"callbacks": [st_callback]},
            )
            with sst.reply_placeholder:
                final_answer = st.write_stream(outputs)

            sst.reply_placeholder.markdown(final_answer)
            sst.steps.append(None)

        else:
            # chitchat or fallback
            final_answer = sst.chitchat_chain.invoke(
                {"input": prompt},
                # {"input": prompt, "chat_history": sst.messages},
                config={"callbacks": [st_callback]},
            )
            sst.reply_placeholder.markdown(final_answer)

    sst.messages.append(HumanMessage(content=prompt))
    sst.messages.append(AIMessage(content=final_answer))
