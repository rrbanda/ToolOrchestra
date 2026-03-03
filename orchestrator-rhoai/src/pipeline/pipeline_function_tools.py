"""Pipeline 4: Custom Function Tools.

Demonstrates the orchestrator selecting from custom-defined function tools.
The Responses API returns the tool call; this pipeline shows how the
orchestrator reasons about which tool to use.

Step 1: Present function tools (enhance_reasoning, answer, search) to orchestrator
Step 2: Show the orchestrator's tool selection and arguments
"""

import json

from kfp import dsl

DEFAULT_TOOL_DEFINITIONS = json.dumps([
    {
        "type": "function",
        "name": "enhance_reasoning",
        "description": "Analyze the problem step by step with detailed reasoning. "
        "Use for complex problems that need careful analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": 'Specialist model. Choices: ["reasoner-1", "reasoner-2", "reasoner-3"]',
                }
            },
            "required": ["model"],
        },
    },
    {
        "type": "function",
        "name": "answer",
        "description": "Give the final answer directly. Use when you have enough information.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": 'Answer model. Choices: ["answer-1", "answer-math-1", "answer-math-2"]',
                }
            },
            "required": ["model"],
        },
    },
    {
        "type": "function",
        "name": "search",
        "description": "Search for missing information needed to solve the problem.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": 'Search model. Choices: ["search-1", "search-2"]',
                }
            },
            "required": ["model"],
        },
    },
])


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["httpx>=0.28.0"],
)
def orchestrator_tool_selection(
    question: str,
    llamastack_url: str,
    model: str,
    tool_definitions_json: str,
    max_output_tokens: int,
    instructions: str,
) -> str:
    """Orchestrator selects which tool to call for the given question."""
    import json
    import time

    import httpx

    tools = json.loads(tool_definitions_json)
    start = time.monotonic()

    payload = {
        "model": model,
        "input": question.strip(),
        "tools": tools,
        "max_output_tokens": max_output_tokens,
        "instructions": instructions,
    }
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(f"{llamastack_url.rstrip('/')}/v1/responses", json=payload)
        resp.raise_for_status()
        data = resp.json()

    tool_calls = []
    text_output = ""
    for item in data.get("output", []):
        if item.get("type") == "function_call":
            tool_calls.append({
                "tool": item.get("name", ""),
                "arguments": item.get("arguments", ""),
            })
        elif item.get("type") == "message":
            for c in item.get("content", []):
                t = c.get("text", "")
                if "</think>" in t:
                    t = t.split("</think>")[-1].strip()
                text_output = t

    return json.dumps({
        "question": question.strip(),
        "tool_calls": tool_calls,
        "text_output": text_output,
        "latency_ms": round((time.monotonic() - start) * 1000),
        "tokens": data.get("usage", {}),
    })


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
)
def format_tool_selection(result_json: str) -> str:
    """Format the orchestrator's tool selection."""
    import json

    data = json.loads(result_json)

    lines = [
        "=== PIPELINE: Function Tool Selection ===",
        "",
        f"Question: {data['question']}",
        f"Latency: {data['latency_ms']}ms",
        "",
        "Orchestrator's tool selection:",
    ]
    if data["tool_calls"]:
        for tc in data["tool_calls"]:
            lines.append(f"  Tool: {tc['tool']}")
            lines.append(f"  Arguments: {tc['arguments']}")
    else:
        lines.append("  (No tool called -- direct answer)")
        lines.append(f"  Response: {data['text_output'][:300]}")

    tokens = data.get("tokens", {})
    lines.append("")
    lines.append(f"Tokens: {tokens.get('input_tokens', 0)} in / {tokens.get('output_tokens', 0)} out")

    output = "\n".join(lines)
    print(output)
    return output


@dsl.pipeline(
    name="toolorchestra-function-tools",
    description="Demonstrates the orchestrator selecting from custom function tool "
    "definitions (enhance_reasoning, answer, search). Shows how the RL-trained "
    "orchestrator reasons about which tool to use.",
)
def function_tools_pipeline(
    question: str = "What is 12 factorial?",
    llamastack_url: str = "http://llamastack-service.llamastack.svc.cluster.local:8321",
    model: str = "vllm-orchestrator/orchestrator-8b",
    tool_definitions_json: str = DEFAULT_TOOL_DEFINITIONS,
    max_output_tokens: int = 1024,
    instructions: str = "You are good at using tools.",
):
    result = orchestrator_tool_selection(
        question=question,
        llamastack_url=llamastack_url,
        model=model,
        tool_definitions_json=tool_definitions_json,
        max_output_tokens=max_output_tokens,
        instructions=instructions,
    )
    format_tool_selection(result_json=result.output)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=function_tools_pipeline,
        package_path="pipeline_function_tools.yaml",
    )
    print("Compiled: pipeline_function_tools.yaml")
