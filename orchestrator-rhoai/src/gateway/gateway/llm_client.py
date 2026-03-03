import json
import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


async def chat_completion(
    endpoint: str,
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
    max_tokens: int = 2048,
    temperature: float = 1.0,
    timeout: float = 180.0,
) -> LLMResponse:
    """Call an OpenAI-compatible /v1/chat/completions endpoint (KServe vLLM)."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage", {})

    parsed_tool_calls = []
    for tc in msg.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}
        parsed_tool_calls.append(
            ToolCall(id=tc.get("id", ""), name=tc["function"]["name"], arguments=args)
        )

    return LLMResponse(
        content=msg.get("content"),
        tool_calls=parsed_tool_calls,
        finish_reason=choice.get("finish_reason", ""),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )
