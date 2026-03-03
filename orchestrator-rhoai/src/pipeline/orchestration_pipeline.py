"""ToolOrchestra orchestration pipeline for RHOAI.

Uses LlamaStack Responses API for server-side agentic orchestration.
The Responses API handles the full agentic loop automatically: tool discovery,
execution planning, multi-step tool chaining, and response synthesis -- all
in a single API call.
"""

from kfp import dsl


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["httpx>=0.28.0"],
)
def orchestrate_via_responses(
    question: str,
    llamastack_url: str,
    model: str,
    mcp_server_url: str,
    mcp_server_label: str,
    max_output_tokens: int,
    instructions: str,
) -> str:
    """Run the ToolOrchestra agentic loop via LlamaStack Responses API.

    A single POST /v1/responses call triggers the full orchestration:
    the orchestrator model auto-discovers MCP tools, decides which to call,
    the server executes them, feeds results back, and repeats until the
    model produces a final answer.
    """
    import json
    import time

    import httpx

    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string")

    start = time.monotonic()

    payload = {
        "model": model,
        "input": question.strip(),
        "tools": [
            {
                "type": "mcp",
                "server_label": mcp_server_label,
                "server_url": mcp_server_url,
                "require_approval": "never",
            }
        ],
        "max_output_tokens": max_output_tokens,
        "instructions": instructions,
    }

    with httpx.Client(timeout=300.0) as client:
        resp = client.post(f"{llamastack_url.rstrip('/')}/v1/responses", json=payload)
        resp.raise_for_status()
        data = resp.json()

    latency_ms = round((time.monotonic() - start) * 1000)

    tool_trace = []
    answer_text = ""
    for item in data.get("output", []):
        item_type = item.get("type", "")
        if item_type == "mcp_list_tools":
            tool_trace.append({
                "step": "discover",
                "tools": [t["name"] for t in item.get("tools", [])],
            })
        elif item_type == "mcp_call":
            tool_trace.append({
                "step": "call",
                "tool": item.get("name", ""),
                "arguments": item.get("arguments", ""),
            })
        elif item_type == "mcp_call_output":
            tool_trace.append({
                "step": "result",
                "output_length": len(str(item.get("output", ""))),
            })
        elif item_type == "message":
            for content_block in item.get("content", []):
                text = content_block.get("text", "")
                if "</think>" in text:
                    text = text.split("</think>")[-1].strip()
                answer_text = text

    usage = data.get("usage", {})

    return json.dumps({
        "question": question.strip(),
        "answer": answer_text or "(No answer produced)",
        "response_id": data.get("id", ""),
        "status": data.get("status", ""),
        "tool_trace": tool_trace,
        "latency_ms": latency_ms,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    })


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
)
def format_output(result_json: str) -> str:
    """Format the orchestration result for display."""
    import json

    data = json.loads(result_json)

    lines = [
        f"Question: {data['question']}",
        f"Status: {data['status']}",
        f"Latency: {data['latency_ms']}ms",
        f"Tokens: {data['input_tokens']} in / {data['output_tokens']} out",
        f"Response ID: {data['response_id']}",
        "",
        "Tool trace:",
    ]
    for t in data.get("tool_trace", []):
        step = t.get("step", "")
        if step == "discover":
            lines.append(f"  Discovered: {t['tools']}")
        elif step == "call":
            lines.append(f"  Called: {t['tool']}({t.get('arguments', '')[:100]})")
        elif step == "result":
            lines.append(f"  Result: {t['output_length']} chars")

    lines.append("")
    lines.append(f"Answer: {data['answer']}")

    summary = "\n".join(lines)
    print(summary)
    return summary


@dsl.pipeline(
    name="toolorchestra-responses-api",
    description="ToolOrchestra orchestration via LlamaStack Responses API. "
    "Uses Nemotron-Orchestrator-8B with MCP tools for server-side agentic "
    "orchestration -- tool discovery, execution, and synthesis in one call.",
)
def orchestration_pipeline(
    question: str = "What are the steps to deploy a model on OpenShift AI?",
    llamastack_url: str = "http://llamastack-service.llamastack.svc.cluster.local:8321",
    model: str = "vllm-orchestrator/orchestrator-8b",
    mcp_server_url: str = "http://mcp-server.rhokp.svc.cluster.local:8010/mcp",
    mcp_server_label: str = "rhokp",
    max_output_tokens: int = 1024,
    instructions: str = "You are good at using tools. Search Red Hat documentation when you need information to answer the question accurately.",
):
    result = orchestrate_via_responses(
        question=question,
        llamastack_url=llamastack_url,
        model=model,
        mcp_server_url=mcp_server_url,
        mcp_server_label=mcp_server_label,
        max_output_tokens=max_output_tokens,
        instructions=instructions,
    )
    format_output(result_json=result.output)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=orchestration_pipeline,
        package_path="orchestration_pipeline.yaml",
    )
    print("Pipeline compiled to orchestration_pipeline.yaml")
