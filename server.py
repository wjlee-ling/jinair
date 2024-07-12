from api import run_flight_search, FlightSearchInput, FlightSearchOutput, QnA_chain

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
    RunnableLambda(run_flight_search)
    .with_types(input_type=FlightSearchInput, output_type=FlightSearchOutput)
    .with_config({"run_name": "run_flight_search"}),
    path="/flight_search",
)


add_routes(
    app,
    QnA_chain,
    path="/QnA",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
