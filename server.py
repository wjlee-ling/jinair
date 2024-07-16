from api import chitchat_chain, flight_search_agent, QnA_chain, scraper_chain

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableLambda
from langserve import add_routes


app = FastAPI(
    title="Jaid API Server",
    version="1.0",
    description="API server for Jaid",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


add_routes(
    app,
    flight_search_agent | RunnableLambda(lambda resp: resp["output"]),
    path="/flight_search",
)

add_routes(
    app,
    QnA_chain,
    path="/QnA",
)

add_routes(
    app,
    scraper_chain,
    path="/scraper",
)

add_routes(
    app,
    chitchat_chain,
    path="/chitchat",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8503)
    # uvicorn.run(app, host="localhost", port=8000)
