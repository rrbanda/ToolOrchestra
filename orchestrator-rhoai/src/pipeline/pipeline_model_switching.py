"""Pipeline 2: Dynamic Model Switching.

Demonstrates switching between the orchestrator (8B) and specialist (3B)
models mid-conversation using previous_response_id for context continuity.

Step 1: Orchestrator-8B answers with MCP doc search
Step 2: Specialist-3B summarizes the answer using conversation context
"""

from kfp import dsl


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["httpx>=0.28.0"],
)
def orchestrator_search(
    question: str,
    llamastack_url: str,
    model: str,
    mcp_server_url: str,
    mcp_server_label: str,
    max_output_tokens: int,
    instructions: str,
) -> str:
    """Step 1: Orchestrator uses MCP tools to research the question."""
    import json
    import time

    import httpx

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

    answer = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                text = c.get("text", "")
                if "</think>" in text:
                    text = text.split("</think>")[-1].strip()
                answer = text

    return json.dumps({
        "response_id": data.get("id", ""),
        "answer": answer,
        "latency_ms": round((time.monotonic() - start) * 1000),
        "model": model,
        "tokens": data.get("usage", {}),
    })


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["httpx>=0.28.0"],
)
def specialist_summarize(
    step1_result: str,
    llamastack_url: str,
    specialist_model: str,
    summarize_prompt: str,
    max_output_tokens: int,
) -> str:
    """Step 2: Specialist model summarizes using conversation context."""
    import json
    import time

    import httpx

    step1 = json.loads(step1_result)
    prev_id = step1["response_id"]

    start = time.monotonic()
    payload = {
        "model": specialist_model,
        "input": summarize_prompt,
        "previous_response_id": prev_id,
        "max_output_tokens": max_output_tokens,
    }
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(f"{llamastack_url.rstrip('/')}/v1/responses", json=payload)
        resp.raise_for_status()
        data = resp.json()

    summary = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                text = c.get("text", "")
                if "</think>" in text:
                    text = text.split("</think>")[-1].strip()
                summary = text

    return json.dumps({
        "response_id": data.get("id", ""),
        "summary": summary,
        "latency_ms": round((time.monotonic() - start) * 1000),
        "model": specialist_model,
        "tokens": data.get("usage", {}),
    })


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
)
def format_model_switch(step1_result: str, step2_result: str) -> str:
    """Format both steps showing the model switch."""
    import json

    s1 = json.loads(step1_result)
    s2 = json.loads(step2_result)

    lines = [
        "=== PIPELINE: Dynamic Model Switching ===",
        "",
        f"Step 1 - Orchestrator ({s1['model']}):",
        f"  Latency: {s1['latency_ms']}ms",
        f"  Answer: {s1['answer'][:500]}",
        "",
        f"Step 2 - Specialist ({s2['model']}):",
        f"  Latency: {s2['latency_ms']}ms",
        f"  Summary: {s2['summary']}",
        "",
        f"Total latency: {s1['latency_ms'] + s2['latency_ms']}ms",
        f"Model switch: {s1['model']} -> {s2['model']}",
    ]
    output = "\n".join(lines)
    print(output)
    return output


@dsl.pipeline(
    name="toolorchestra-model-switching",
    description="Demonstrates dynamic model switching: Orchestrator-8B answers "
    "with MCP doc search, then Specialist-3B summarizes via previous_response_id.",
)
def model_switching_pipeline(
    question: str = "What is model serving in OpenShift AI?",
    llamastack_url: str = "http://llamastack-service.llamastack.svc.cluster.local:8321",
    orchestrator_model: str = "vllm-orchestrator/orchestrator-8b",
    specialist_model: str = "vllm-inference/llama-32-3b-instruct",
    mcp_server_url: str = "http://mcp-server.rhokp.svc.cluster.local:8010/mcp",
    mcp_server_label: str = "rhokp",
    max_output_tokens: int = 1024,
    summarize_max_tokens: int = 512,
    instructions: str = "You are good at using tools. Search Red Hat documentation to answer accurately.",
    summarize_prompt: str = "Summarize the above answer in exactly 3 concise bullet points.",
):
    step1 = orchestrator_search(
        question=question,
        llamastack_url=llamastack_url,
        model=orchestrator_model,
        mcp_server_url=mcp_server_url,
        mcp_server_label=mcp_server_label,
        max_output_tokens=max_output_tokens,
        instructions=instructions,
    )
    step2 = specialist_summarize(
        step1_result=step1.output,
        llamastack_url=llamastack_url,
        specialist_model=specialist_model,
        summarize_prompt=summarize_prompt,
        max_output_tokens=summarize_max_tokens,
    )
    format_model_switch(
        step1_result=step1.output,
        step2_result=step2.output,
    )


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=model_switching_pipeline,
        package_path="pipeline_model_switching.yaml",
    )
    print("Compiled: pipeline_model_switching.yaml")
