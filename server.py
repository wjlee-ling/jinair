from fastapi import FastAPI

from langserve import add_routes

app = FastAPI(
    title="Jaid API Server",
    version="1.0",
    description="API server for Jaid",
)

# add_routes(
#     app,
#     ChatOpenAI(model="gpt-3.5-turbo-0125"),
#     path="/openai",
# )

# add_routes(
#     app,
#     ChatAnthropic(model="claude-3-haiku-20240307"),
#     path="/anthropic",
# )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
