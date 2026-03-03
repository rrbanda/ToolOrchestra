"""Pipeline 3: Conversation Branching.

Demonstrates branching from a previous response to explore different angles
of the same topic, using previous_response_id to share conversation context.

Step 1: Ask about a topic with MCP doc search
Step 2: Branch from step 1 to explore a different angle
Step 3: Compare both branches
"""

from kfp import dsl


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["httpx>=0.28.0"],
)
def initial_query(
    question: str,
    llamastack_url: str,
    model: str,
    mcp_server_url: str,
    mcp_server_label: str,
    max_output_tokens: int,
    instructions: str,
) -> str:
    """Step 1: Initial question with MCP doc search."""
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
    })


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["httpx>=0.28.0"],
)
def branch_query(
    branch_question: str,
    step1_result: str,
    llamastack_url: str,
    model: str,
    mcp_server_url: str,
    mcp_server_label: str,
    max_output_tokens: int,
    instructions: str,
) -> str:
    """Step 2: Branch from step 1 with a different angle."""
    import json
    import time

    import httpx

    step1 = json.loads(step1_result)
    prev_id = step1["response_id"]

    start = time.monotonic()
    payload = {
        "model": model,
        "input": branch_question.strip(),
        "tools": [
            {
                "type": "mcp",
                "server_label": mcp_server_label,
                "server_url": mcp_server_url,
                "require_approval": "never",
            }
        ],
        "max_output_tokens": max_output_tokens,
        "previous_response_id": prev_id,
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
        "previous_response_id": prev_id,
        "answer": answer,
        "latency_ms": round((time.monotonic() - start) * 1000),
    })


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
)
def compare_branches(
    question: str,
    branch_question: str,
    step1_result: str,
    branch_result: str,
) -> str:
    """Step 3: Compare the original and branched responses."""
    import json

    s1 = json.loads(step1_result)
    br = json.loads(branch_result)

    lines = [
        "=== PIPELINE: Conversation Branching ===",
        "",
        f"Original question: {question}",
        f"  Latency: {s1['latency_ms']}ms",
        f"  Answer: {s1['answer'][:400]}",
        "",
        f"Branch question: {branch_question}",
        f"  Branched from: {br['previous_response_id'][:30]}...",
        f"  Latency: {br['latency_ms']}ms",
        f"  Answer: {br['answer'][:400]}",
        "",
        f"Total latency: {s1['latency_ms'] + br['latency_ms']}ms",
    ]
    output = "\n".join(lines)
    print(output)
    return output


@dsl.pipeline(
    name="toolorchestra-branching",
    description="Demonstrates conversation branching: ask an initial question, "
    "then branch from that response to explore a different angle via previous_response_id.",
)
def branching_pipeline(
    question: str = "How does OpenShift AI handle model serving?",
    branch_question: str = "Instead, tell me about GPU requirements for model serving.",
    llamastack_url: str = "http://llamastack-service.llamastack.svc.cluster.local:8321",
    model: str = "vllm-orchestrator/orchestrator-8b",
    mcp_server_url: str = "http://mcp-server.rhokp.svc.cluster.local:8010/mcp",
    mcp_server_label: str = "rhokp",
    max_output_tokens: int = 1024,
    instructions: str = "You are good at using tools. Search Red Hat documentation to answer accurately.",
):
    step1 = initial_query(
        question=question,
        llamastack_url=llamastack_url,
        model=model,
        mcp_server_url=mcp_server_url,
        mcp_server_label=mcp_server_label,
        max_output_tokens=max_output_tokens,
        instructions=instructions,
    )
    branch = branch_query(
        branch_question=branch_question,
        step1_result=step1.output,
        llamastack_url=llamastack_url,
        model=model,
        mcp_server_url=mcp_server_url,
        mcp_server_label=mcp_server_label,
        max_output_tokens=max_output_tokens,
        instructions=instructions,
    )
    compare_branches(
        question=question,
        branch_question=branch_question,
        step1_result=step1.output,
        branch_result=branch.output,
    )


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=branching_pipeline,
        package_path="pipeline_branching.yaml",
    )
    print("Compiled: pipeline_branching.yaml")
