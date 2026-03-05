"""ToolOrchestra comparison pipeline for RHOAI.

Runs the same question through two paths side-by-side:
1. Direct: specialist model answers alone (no orchestration)
2. Orchestrated: Orchestrator-8B coordinates specialist models

Demonstrates the value of orchestration vs. a single model.
"""

import json

from kfp import dsl

ORCHESTRATOR_DEFAULT = "http://orchestrator-8b-predictor.orchestrator-rhoai.svc.cluster.local:8080/v1"
SPECIALIST_DEFAULT = "http://llama-32-3b-instruct-predictor.my-first-model.svc.cluster.local:8080/v1"

DEFAULT_MODEL_MAPPING = json.dumps({
    "search-1": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
    "search-2": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
    "search-3": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
    "reasoner-1": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
    "reasoner-2": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
    "reasoner-3": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
    "answer-1": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
    "answer-2": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
    "answer-3": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
    "answer-4": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
    "answer-math-1": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
    "answer-math-2": {"endpoint": SPECIALIST_DEFAULT, "model": "llama-32-3b-instruct"},
})


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["httpx>=0.28.0"],
)
def direct_answer(
    question: str,
    specialist_endpoint: str,
    specialist_model: str,
    max_output_tokens: int,
) -> str:
    """Ask the specialist model directly -- no orchestration, no tools."""
    import json
    import time

    import httpx

    start = time.monotonic()
    payload = {
        "model": specialist_model,
        "messages": [
            {"role": "system", "content": "Please reason step by step and give a clear, concise final answer."},
            {"role": "user", "content": question.strip()},
        ],
        "max_tokens": max_output_tokens,
        "temperature": 1.0,
    }
    url = f"{specialist_endpoint.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"].get("content", "")
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    latency = round((time.monotonic() - start) * 1000)

    return json.dumps({
        "question": question.strip(),
        "answer": content,
        "method": "direct (single model, no orchestration)",
        "model": specialist_model,
        "latency_ms": latency,
    })


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["httpx>=0.28.0", "duckduckgo-search>=7.0.0"],
)
def orchestrated_answer(
    question: str,
    orchestrator_endpoint: str,
    orchestrator_model: str,
    model_mapping_json: str,
    tool_definitions_json: str,
    max_turns: int,
    max_output_tokens: int,
) -> str:
    """Run the full ToolOrchestra orchestration loop."""
    import json
    import os
    import subprocess
    import tempfile
    import time

    import httpx

    tools = json.loads(tool_definitions_json)
    model_mapping = json.loads(model_mapping_json)

    TOOL_PRICING = {
        "reasoner-1": {"input": 1.25, "output": 10.0},
        "reasoner-2": {"input": 0.25, "output": 2.0},
        "reasoner-3": {"input": 0.8, "output": 0.8},
        "answer-1": {"input": 1.25, "output": 10.0},
        "answer-2": {"input": 0.25, "output": 2.0},
        "answer-3": {"input": 0.9, "output": 0.9},
        "answer-4": {"input": 0.8, "output": 0.8},
        "answer-math-1": {"input": 0.9, "output": 0.9},
        "answer-math-2": {"input": 0.2, "output": 0.2},
        "search-1": {"input": 1.25, "output": 10.0},
        "search-2": {"input": 0.25, "output": 2.0},
        "search-3": {"input": 0.8, "output": 0.8},
    }
    total_cost = 0.0

    def chat_completion(endpoint, model, messages, tools_list=None, max_tokens=2048):
        url = f"{endpoint.rstrip('/')}/chat/completions"
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 1.0}
        if tools_list:
            payload["tools"] = tools_list
            payload["tool_choice"] = "auto"
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    def call_specialist(endpoint, model, messages, max_tokens=1024):
        return chat_completion(endpoint, model, messages, max_tokens=max_tokens)

    def handle_search(problem, context_str, spec_endpoint, spec_model):
        prompt = f"{context_str}\n\nProblem: {problem}\n\nGenerate a concise web search query to find the missing information. Output only the query, nothing else."
        msgs = [{"role": "system", "content": "You generate concise search queries."}, {"role": "user", "content": prompt}]
        resp = call_specialist(spec_endpoint, spec_model, msgs, max_tokens=128)
        query = resp["choices"][0]["message"].get("content", problem).strip().strip('"').strip("'")
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=3):
                    results.append(f"Title: {r.get('title','')}\n{r.get('body','')}\nURL: {r.get('href','')}")
            return "\n\n".join(results) if results else f"No results for: {query}", query
        except Exception:
            return f"Search unavailable for: {query}", query

    def handle_reasoning(problem, context_str, spec_endpoint, spec_model):
        prompt = (f"{context_str}\n\nQuestion: {problem}\n\n"
                  "Instead of directly answering, write Python code that gives intermediate results. "
                  "Wrap code within ```python and ```. Make it self-contained.")
        msgs = [{"role": "system", "content": "You are an expert analyst. Write executable Python code."}, {"role": "user", "content": prompt}]
        resp = call_specialist(spec_endpoint, spec_model, msgs, max_tokens=1024)
        content = resp["choices"][0]["message"].get("content", "")
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        generated_code = ""
        exec_result = ""
        if "```python" in content:
            try:
                generated_code = content.split("```python")[-1].split("```")[0].strip()
            except (IndexError, ValueError):
                pass
        if generated_code:
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
                    f.write(generated_code)
                    code_path = f.name
                result = subprocess.run(["python3", code_path], timeout=30, capture_output=True, text=True)
                exec_result = result.stdout.strip()
                os.unlink(code_path)
            except Exception:
                exec_result = "[Code execution failed]"
        parts = []
        if generated_code:
            parts.append(f"Code:\n```python\n{generated_code}\n```")
        if exec_result:
            parts.append(f"Output: {exec_result}")
        if not parts:
            parts.append(content)
        return "\n".join(parts)

    def handle_answer(problem, context_str, spec_endpoint, spec_model):
        prompt = f"{context_str}\n\nProblem:\n{problem}"
        msgs = [{"role": "system", "content": "Please reason step by step and give a clear, concise final answer."}, {"role": "user", "content": prompt}]
        resp = call_specialist(spec_endpoint, spec_model, msgs, max_tokens=1024)
        content = resp["choices"][0]["message"].get("content", "")
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        return content

    doc_list = []
    reasoning_list = []
    tool_trace = []
    used_tools = []
    final_answer = ""
    total_start = time.monotonic()

    for turn in range(max_turns):
        ctx_parts = []
        if doc_list:
            ctx_parts.append("Documents:\n" + "\n\n".join(f"Doc {i+1}: {d[:1200]}" for i, d in enumerate(doc_list)))
        if reasoning_list:
            ctx_parts.append("Previous analysis:\n" + "\n\n".join(reasoning_list))
        context_str = "\n\n".join(ctx_parts)

        user_content = f"Problem: {question}"
        if context_str:
            user_content += f"\n\n{context_str}"
        user_content += "\n\nChoose an appropriate tool."

        current_tools = list(tools)
        if turn == max_turns - 1 and (doc_list or reasoning_list):
            current_tools = [t for t in tools if t["function"]["name"] == "answer"]
        elif len(used_tools) > 1 and used_tools[-1] == used_tools[-2]:
            current_tools = [t for t in tools if t["function"]["name"] != used_tools[-1]]

        try:
            orch_resp = chat_completion(
                orchestrator_endpoint, orchestrator_model,
                [{"role": "system", "content": "You are good at using tools."}, {"role": "user", "content": user_content}],
                tools_list=current_tools, max_tokens=max_output_tokens,
            )
        except Exception as e:
            tool_trace.append({"turn": turn, "tool": "error", "error": str(e)})
            break

        msg = orch_resp["choices"][0]["message"]
        tc_list = msg.get("tool_calls", [])
        if not tc_list:
            content = msg.get("content", "")
            if content and "</think>" in content:
                final_answer = content.split("</think>")[-1].strip()
            elif content:
                final_answer = content
            break

        tc = tc_list[0]
        tool_name = tc["function"]["name"]
        try:
            tool_args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            tool_args = {}
        model_id = tool_args.get("model", "")
        used_tools.append(tool_name)
        spec_info = model_mapping.get(model_id, {})
        spec_endpoint = spec_info.get("endpoint", orchestrator_endpoint)
        spec_model = spec_info.get("model", orchestrator_model)
        start = time.monotonic()

        pricing = TOOL_PRICING.get(model_id, {"input": 0.0, "output": 0.0})
        est_tokens = len(question.split()) * 2 + len(context_str.split()) * 2 + 300
        step_cost = (est_tokens * pricing["input"]) / 1_000_000

        if tool_name == "search":
            out, query = handle_search(question, context_str, spec_endpoint, spec_model)
            doc_list.append(out)
            total_cost += step_cost
            tool_trace.append({"turn": turn, "tool": "search", "specialist": model_id, "query": query, "latency_ms": round((time.monotonic() - start) * 1000), "est_cost_usd": round(step_cost, 6)})
        elif tool_name == "enhance_reasoning":
            reasoning = handle_reasoning(question, context_str, spec_endpoint, spec_model)
            if reasoning:
                reasoning_list.append(reasoning)
            total_cost += step_cost
            tool_trace.append({"turn": turn, "tool": "enhance_reasoning", "specialist": model_id, "latency_ms": round((time.monotonic() - start) * 1000), "est_cost_usd": round(step_cost, 6)})
        elif tool_name == "answer":
            final_answer = handle_answer(question, context_str, spec_endpoint, spec_model)
            total_cost += step_cost
            tool_trace.append({"turn": turn, "tool": "answer", "specialist": model_id, "latency_ms": round((time.monotonic() - start) * 1000), "est_cost_usd": round(step_cost, 6)})
            break

    total_latency = round((time.monotonic() - total_start) * 1000)
    if not final_answer:
        final_answer = "(No final answer within allowed turns)"

    return json.dumps({
        "question": question.strip(),
        "answer": final_answer,
        "method": "orchestrated (Orchestrator-8B + specialists)",
        "tool_trace": tool_trace,
        "total_turns": len(tool_trace),
        "total_latency_ms": total_latency,
        "est_total_cost_usd": round(total_cost, 6),
    })


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
)
def compare_results(direct_json: str, orchestrated_json: str) -> str:
    """Compare direct vs orchestrated results side by side."""
    import json

    direct = json.loads(direct_json)
    orch = json.loads(orchestrated_json)

    lines = [
        "=" * 60,
        "  ToolOrchestra: Direct vs Orchestrated Comparison",
        "=" * 60,
        "",
        f"Question: {direct['question']}",
        "",
        "-" * 60,
        "  PATH 1: Direct (Single Model, No Orchestration)",
        "-" * 60,
        f"  Model: {direct.get('model', 'N/A')}",
        f"  Latency: {direct['latency_ms']}ms",
        f"  Answer: {direct['answer'][:500]}",
        "",
        "-" * 60,
        "  PATH 2: Orchestrated (Orchestrator-8B + Specialists)",
        "-" * 60,
        f"  Turns: {orch['total_turns']}",
        f"  Latency: {orch['total_latency_ms']}ms",
        f"  Est. cost: ${orch.get('est_total_cost_usd', 0):.6f}",
        "",
        "  Tool trace:",
    ]
    for t in orch.get("tool_trace", []):
        lines.append(
            f"    Turn {t['turn']}: {t['tool']} -> {t.get('specialist', '')} "
            f"[{t.get('latency_ms', '')}ms, ${t.get('est_cost_usd', 0):.6f}]"
        )
    lines.extend([
        "",
        f"  Answer: {orch['answer'][:500]}",
        "",
        "=" * 60,
        "  KEY DIFFERENCES",
        "=" * 60,
        f"  Direct: 1 model call, {direct['latency_ms']}ms",
        f"  Orchestrated: {orch['total_turns']} specialist calls, {orch['total_latency_ms']}ms",
        f"  Orchestrated used: {', '.join(t['tool'] for t in orch.get('tool_trace', []))}",
        "",
        "  The orchestrator searched for information, reasoned about it,",
        "  and then had a specialist produce the final answer -- rather",
        "  than relying on a single model's knowledge alone.",
        "=" * 60,
    ])

    summary = "\n".join(lines)
    print(summary)
    return summary


_FALLBACK_TOOLS = json.dumps([
    {"type": "function", "function": {"name": "enhance_reasoning", "description": "tool to enhance answer model reasoning. analyze the problem, write code, execute it and return intermidiate results that will help solve the problem", "parameters": {"properties": {"model": {"description": "The model used to reason. Choices: ['reasoner-1', 'reasoner-2', 'reasoner-3']. reasoner-1 demonstrates strong understanding and reasoning capabilities. reasoner-2 can analyze some problems. reasoner-3 can reason over context.\nModel | price per million input tokens | price per million output tokens | average latency\nreasoner-1 | $1.25 | $10 | 31s\nreasoner-2 | $0.25 | $2 | 25s\nreasoner-3 | $0.8 | $0.8 | 9s", "type": "string"}}, "required": ["model"], "type": "object"}}},
    {"type": "function", "function": {"name": "answer", "description": "give the final answer. Not allowed to call if documents is empty.", "parameters": {"properties": {"model": {"description": "The model used to answer. Choices: ['answer-1', 'answer-2', 'answer-3', 'answer-4', 'answer-math-1', 'answer-math-2']. answer-1 exhibits strong functional calling abilities. answer-math-1 can solve moderate math problems. answer-math-2 handles basic math.\nModel | price/M input | price/M output | avg latency\nanswer-1 | $1.25 | $10 | 96s\nanswer-2 | $0.25 | $2 | 27s\nanswer-3 | $0.9 | $0.9 | 15s\nanswer-4 | $0.8 | $0.8 | 11s\nanswer-math-1 | $0.9 | $0.9 | 13s\nanswer-math-2 | $0.2 | $0.2 | 9s", "type": "string"}}, "required": ["model"], "type": "object"}}},
    {"type": "function", "function": {"name": "search", "description": "Search for missing information", "parameters": {"properties": {"model": {"description": "The model used to search. Choices: ['search-1', 'search-2', 'search-3']. search-1 identifies missing info and writes concise queries. search-2 reasons over context. search-3 writes queries.\nModel | price/M input | price/M output | avg latency\nsearch-1 | $1.25 | $10 | 22s\nsearch-2 | $0.25 | $2 | 16s\nsearch-3 | $0.8 | $0.8 | 8s", "type": "string"}}, "required": ["model"], "type": "object"}}},
])


@dsl.pipeline(
    name="toolorchestra-comparison",
    description="Side-by-side comparison: Direct specialist answer vs "
    "Orchestrator-8B coordinated answer. Shows the value of orchestration.",
)
def comparison_pipeline(
    question: str = "What is the approximate ratio of the population of Tokyo to the population of Paris, and which city has higher population density?",
    orchestrator_endpoint: str = ORCHESTRATOR_DEFAULT,
    orchestrator_model: str = "orchestrator-8b",
    specialist_endpoint: str = SPECIALIST_DEFAULT,
    specialist_model: str = "llama-32-3b-instruct",
    model_mapping_json: str = DEFAULT_MODEL_MAPPING,
    tool_definitions_json: str = _FALLBACK_TOOLS,
    max_turns: int = 10,
    max_output_tokens: int = 2048,
):
    direct = direct_answer(
        question=question,
        specialist_endpoint=specialist_endpoint,
        specialist_model=specialist_model,
        max_output_tokens=max_output_tokens,
    )
    orchestrated = orchestrated_answer(
        question=question,
        orchestrator_endpoint=orchestrator_endpoint,
        orchestrator_model=orchestrator_model,
        model_mapping_json=model_mapping_json,
        tool_definitions_json=tool_definitions_json,
        max_turns=max_turns,
        max_output_tokens=max_output_tokens,
    )
    compare_results(
        direct_json=direct.output,
        orchestrated_json=orchestrated.output,
    )


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=comparison_pipeline,
        package_path="comparison_pipeline.yaml",
    )
    print("Comparison pipeline compiled to comparison_pipeline.yaml")
