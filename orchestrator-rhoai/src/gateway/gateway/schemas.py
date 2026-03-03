from pydantic import BaseModel, Field


class OrchestrationRequest(BaseModel):
    question: str
    context: str | None = None
    max_turns: int | None = None


class ToolCallRecord(BaseModel):
    tool_name: str
    model_selected: str
    latency_ms: float
    result_preview: str = ""


class OrchestrationResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    total_turns: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: int
