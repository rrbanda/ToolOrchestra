import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from gateway import __version__
from gateway.config import GatewayConfig, load_model_mapping, load_tool_definitions
from gateway.llm_client import chat_completion
from gateway.schemas import (
    HealthResponse,
    OrchestrationRequest,
    OrchestrationResponse,
    ToolCallRecord,
)
from gateway.tools import TOOL_HANDLERS

logger = logging.getLogger(__name__)

config: GatewayConfig | None = None
model_mapping: dict = {}
tool_definitions: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, model_mapping, tool_definitions
    config = GatewayConfig()
    model_mapping = load_model_mapping(config.model_mapping_path)
    tool_definitions = load_tool_definitions(config.tool_definitions_path)
    logging.basicConfig(level=getattr(logging, config.log_level.upper()))
    logger.info(
        "Gateway started: orchestrator=%s, %d model mappings, %d tools",
        config.orchestrator_endpoint,
        len(model_mapping),
        len(tool_definitions),
    )
    yield


app = FastAPI(
    title="Orchestrator Gateway",
    description="Multi-turn orchestration gateway for ToolOrchestra on RHOAI",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version=__version__,
        models_loaded=len(model_mapping),
    )


@app.get("/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/v1/orchestrate", response_model=OrchestrationResponse)
async def orchestrate(request: OrchestrationRequest) -> OrchestrationResponse:
    """Run the full multi-turn orchestration loop.

    1. Send the problem to the Orchestrator-8B with tool definitions
    2. Parse tool_calls from the response
    3. Execute the selected tool (search, enhance_reasoning, or answer)
    4. Accumulate context and repeat until 'answer' is called
    """
    assert config is not None

    max_turns = request.max_turns or config.max_turns
    orchestrator_model = config.orchestrator_model
    orchestrator_endpoint = config.orchestrator_endpoint

    doc_list: list[str] = []
    reasoning_list: list[str] = []
    tool_call_records: list[ToolCallRecord] = []
    used_tools: list[str] = []
    final_answer = ""
    total_start = time.monotonic()

    for turn in range(max_turns):
        context_str = _build_context(doc_list, reasoning_list)

        user_content = f"Problem: {request.question}"
        if context_str:
            user_content += f"\n\n{context_str}"
        user_content += "\n\nChoose an appropriate tool."

        messages = [
            {"role": "system", "content": "You are good at using tools."},
            {"role": "user", "content": user_content},
        ]

        current_tools = tool_definitions
        if len(used_tools) > 1 and used_tools[-1] == used_tools[-2]:
            repeated = used_tools[-1]
            current_tools = [t for t in tool_definitions if t["function"]["name"] != repeated]
            logger.info("Turn %d: removed repeated tool '%s' from options", turn, repeated)

        try:
            orch_response = await chat_completion(
                endpoint=orchestrator_endpoint,
                model=orchestrator_model,
                messages=messages,
                tools=current_tools,
                max_tokens=2048,
                temperature=1.0,
                timeout=config.request_timeout,
            )
        except Exception:
            logger.exception("Orchestrator call failed on turn %d", turn)
            break

        if not orch_response.tool_calls:
            logger.warning("Turn %d: no tool calls returned, ending loop", turn)
            if orch_response.content:
                final_answer = _extract_content_after_think(orch_response.content)
            break

        tc = orch_response.tool_calls[0]
        tool_name = tc.name
        tool_model_id = tc.arguments.get("model", "")

        logger.info("Turn %d: tool=%s, model=%s", turn, tool_name, tool_model_id)

        used_tools.append(tool_name)

        if tool_name not in TOOL_HANDLERS:
            logger.warning("Unknown tool: %s", tool_name)
            break

        mapping = model_mapping.get(tool_model_id, {})
        specialist_endpoint = mapping.get("endpoint", config.specialist_endpoint)
        specialist_model = mapping.get("model", "llama-32-3b-instruct")

        handler = TOOL_HANDLERS[tool_name]
        try:
            result = await handler(
                model=tool_model_id,
                problem=request.question,
                context=context_str,
                specialist_endpoint=specialist_endpoint,
                specialist_model=specialist_model,
            )
        except Exception:
            logger.exception("Tool %s failed on turn %d", tool_name, turn)
            break

        tool_call_records.append(
            ToolCallRecord(
                tool_name=result.tool_name,
                model_selected=result.model_selected,
                latency_ms=result.latency_ms,
                result_preview=result.content[:500] if result.content else "",
            )
        )

        if result.is_final_answer:
            final_answer = result.content
            break

        if result.tool_name == "search":
            for doc in result.documents:
                if doc not in doc_list:
                    doc_list.append(doc)

        if result.tool_name == "enhance_reasoning" and result.content:
            reasoning_list.append(result.content)

    total_latency = (time.monotonic() - total_start) * 1000

    if not final_answer and tool_call_records:
        final_answer = "(Orchestrator did not produce a final answer within the allowed turns)"

    return OrchestrationResponse(
        answer=final_answer,
        tool_calls=tool_call_records,
        total_turns=len(tool_call_records),
        total_latency_ms=total_latency,
    )


def _build_context(doc_list: list[str], reasoning_list: list[str]) -> str:
    """Build the accumulated context string from documents and reasoning."""
    parts = []
    if doc_list:
        doc_str = "\n\n".join(
            f"Doc {i + 1}: {doc[:1200]}" for i, doc in enumerate(doc_list)
        )
        parts.append(f"Documents:\n{doc_str}")
    if reasoning_list:
        reasoning_str = "\n\n".join(reasoning_list)
        parts.append(f"Previous analysis:\n{reasoning_str}")
    return "\n\n".join(parts)


def _extract_content_after_think(content: str) -> str:
    """Strip <think>...</think> tags from model output."""
    if "</think>" in content:
        return content.split("</think>")[-1].strip()
    return content.strip()
