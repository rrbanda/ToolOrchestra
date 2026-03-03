from fastapi import FastAPI
from pydantic import BaseModel

from retrieval import __version__

app = FastAPI(
    title="Retrieval Service",
    version=__version__,
)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "version": __version__}


@app.post("/v1/retrieve")
async def retrieve(request: RetrieveRequest) -> list:
    # Placeholder: returns empty list of documents
    return []
