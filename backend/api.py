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
        # print(chunk)
        for msg in chunk["messages"]:
            # if isinstance(msg.content, str) and isinstance(eval(msg.content), list):
            #     # Anthropic-specific
            #     answer += eval(msg.content)[0]["text"]
            if isinstance(msg, AIMessageChunk) or isinstance(msg, AIMessage):
                # just before using the tool or after using the tool.
                try:
                    answer += msg.content
                except TypeError:
                    answer += msg.content[0]["text"]

    return answer
