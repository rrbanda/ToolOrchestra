from fastapi import FastAPI
from pydantic import BaseModel

from sandbox import __version__

app = FastAPI(
    title="Code Sandbox",
    version=__version__,
)


class ExecuteRequest(BaseModel):
    code: str
    timeout: int = 60


class ExecuteResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "version": __version__}


@app.post("/v1/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    # TODO: Implement actual code execution in isolated sandbox
    return ExecuteResponse(
        stdout="",
        stderr="TODO: Implement sandbox execution",
        exit_code=-1,
    )
