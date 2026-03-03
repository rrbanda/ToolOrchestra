"""Pipeline 5: RAG via File Search.

Demonstrates retrieval-augmented generation using LlamaStack's built-in
file_search tool with vector stores. Uploaded documents are embedded via
granite-embedding and searched at query time.

Step 1: Search the vector store for relevant document chunks
Step 2: Generate an answer grounded in retrieved content
"""

from kfp import dsl


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["httpx>=0.28.0"],
)
def rag_query(
    question: str,
    llamastack_url: str,
    model: str,
    vector_store_id: str,
    max_output_tokens: int,
    instructions: str,
) -> str:
    """Query the LLM with file_search RAG augmentation."""
    import json
    import time

    import httpx

    start = time.monotonic()
    payload = {
        "model": model,
        "input": question.strip(),
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
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
    search_queries = []
    for item in data.get("output", []):
        if item.get("type") == "file_search_call":
            search_queries = item.get("queries", [])
        elif item.get("type") == "message":
            for c in item.get("content", []):
                text = c.get("text", "")
                if "</think>" in text:
                    text = text.split("</think>")[-1].strip()
                answer = text

    usage = data.get("usage", {})

    return json.dumps({
        "question": question.strip(),
        "answer": answer or "(No answer produced)",
        "search_queries": search_queries,
        "response_id": data.get("id", ""),
        "latency_ms": round((time.monotonic() - start) * 1000),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    })


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
)
def format_rag_result(result_json: str) -> str:
    """Format the RAG result for display."""
    import json

    data = json.loads(result_json)

    lines = [
        "=== PIPELINE: RAG File Search ===",
        "",
        f"Question: {data['question']}",
        f"Latency: {data['latency_ms']}ms",
        f"Tokens: {data['input_tokens']} in / {data['output_tokens']} out",
        "",
        f"Search queries: {data['search_queries']}",
        "",
        f"Answer: {data['answer']}",
    ]

    output = "\n".join(lines)
    print(output)
    return output


@dsl.pipeline(
    name="toolorchestra-rag-filesearch",
    description="Demonstrates RAG using LlamaStack file_search with vector stores. "
    "Documents are embedded via granite-embedding and searched at query time "
    "to ground answers in project documentation.",
)
def rag_filesearch_pipeline(
    question: str = "What results does ToolOrchestra achieve on the HLE benchmark?",
    llamastack_url: str = "http://llamastack-service.llamastack.svc.cluster.local:8321",
    model: str = "vllm-inference/llama-32-3b-instruct",
    vector_store_id: str = "vs_4b92c181-1043-4bc8-8111-bfaba954886a",
    max_output_tokens: int = 1024,
    instructions: str = "Search the uploaded files first, then answer based on what you find.",
):
    result = rag_query(
        question=question,
        llamastack_url=llamastack_url,
        model=model,
        vector_store_id=vector_store_id,
        max_output_tokens=max_output_tokens,
        instructions=instructions,
    )
    format_rag_result(result_json=result.output)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=rag_filesearch_pipeline,
        package_path="pipeline_rag_filesearch.yaml",
    )
    print("Compiled: pipeline_rag_filesearch.yaml")
