from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
)


def request_LLM_API(chain, callbacks, inputs):
    answer = ""
    for chunk in chain.stream(
        inputs,
        config={"callbacks": callbacks},
    ):
        print(chunk)
        for msg in chunk["messages"]:
            if isinstance(msg.content, str) and isinstance(eval(msg.content), list):
                # Anthropic-specific
                answer += eval(msg.content)[0]["text"]
            elif isinstance(msg, AIMessageChunk):
                # just before using the tool
                answer += msg.content
            elif isinstance(msg, AIMessage):
                # after using the tool
                answer += msg.content

    return answer
