"""ToolOrchestra orchestration pipeline for RHOAI.

Implements the paper's true orchestration loop: Orchestrator-8B (the agent)
calls vLLM chat completions with tool definitions, receives tool_calls,
routes them to specialist models for execution, and repeats until done.

No LlamaStack, no Responses API -- direct vLLM calls matching the paper.
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

_FALLBACK_TOOLS = json.dumps([
    {
        "type": "function",
        "function": {
            "name": "enhance_reasoning",
            "description": "tool to enhance answer model reasoning. analyze the problem, write code, execute it and return intermidiate results that will help solve the problem",
            "parameters": {
                "properties": {
                    "model": {
                        "description": "The model used to reason. Choices: ['reasoner-1', 'reasoner-2', 'reasoner-3']. reasoner-1 demonstrates strong understanding and reasoning capabilities, which usually provides reliable insights. reasoner-2 can analyze some problems, but could hallucinate and make mistakes in difficult scenarios. reasoner-3 can reason over the context and reveal the logic. \nModel | price per million input tokens | price per million output tokens | average latency\nreasoner-1 | $1.25 | $10 | 31s\nreasoner-2 | $0.25 | $2 | 25s\nreasoner-3 | $0.8 | $0.8 | 9s",
                        "type": "string",
                    }
                },
                "required": ["model"],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "give the final answer. Not allowed to call if documents is empty.",
            "parameters": {
                "properties": {
                    "model": {
                        "description": "The model used to answer. Choices: ['answer-1', 'answer-2', 'answer-3', 'answer-4', 'answer-math-1', 'answer-math-2']. answer-1 exhibits strong functional calling abilities and performs excellent in most domains. answer-math-1 can solve moderate math problems. answer-math-2 handles basic math.\nModel | price per million input tokens | price per million output tokens | average latency\nanswer-1 | $1.25 | $10 | 96s\nanswer-2 | $0.25 | $2 | 27s\nanswer-3 | $0.9 | $0.9 | 15s\nanswer-4 | $0.8 | $0.8 | 11s\nanswer-math-1 | $0.9 | $0.9 | 13s\nanswer-math-2 | $0.2 | $0.2 | 9s",
                        "type": "string",
                    }
                },
                "required": ["model"],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for missing information",
            "parameters": {
                "properties": {
                    "model": {
                        "description": "The model used to search for missing information. Choices: ['search-1', 'search-2', 'search-3']. search-1 usually identifies the missing information and can write concise queries. search-2 can reason over context and write queries. search-3 can also write queries.\nModel | price per million input tokens | price per million output tokens | average latency\nsearch-1 | $1.25 | $10 | 22s\nsearch-2 | $0.25 | $2 | 16s\nsearch-3 | $0.8 | $0.8 | 8s",
                        "type": "string",
                    }
                },
                "required": ["model"],
                "type": "object",
            },
        },
    },
])



@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["httpx>=0.28.0", "duckduckgo-search>=7.0.0"],
)
def orchestration_loop(
    question: str,
    orchestrator_endpoint: str,
    orchestrator_model: str,
    model_mapping_json: str,
    tool_definitions_json: str,
    max_turns: int,
    max_output_tokens: int,
) -> str:
    """Run the ToolOrchestra multi-turn orchestration loop.

    Matches the paper's architecture:
    1. Call Orchestrator-8B via vLLM /v1/chat/completions with tool definitions
    2. Parse tool_calls from response
    3. Route to the specialist model specified by MODEL_MAPPING
    4. Feed specialist result back to orchestrator
    5. Repeat until orchestrator calls 'answer' or max_turns reached
    """
    import json
    import os
    import subprocess
    import tempfile
    import time

    import httpx

    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string")

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
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 1.0,
        }
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
        msgs = [
            {"role": "system", "content": "You generate concise search queries."},
            {"role": "user", "content": prompt},
        ]
        resp = call_specialist(spec_endpoint, spec_model, msgs, max_tokens=128)
        query = resp["choices"][0]["message"].get("content", problem).strip().strip('"').strip("'")

        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=3):
                    results.append(f"Title: {r.get('title','')}\n{r.get('body','')}\nURL: {r.get('href','')}")
            search_output = "\n\n".join(results) if results else f"No results for: {query}"
        except Exception:
            search_output = f"Search unavailable for: {query}"

        return search_output, query

    def handle_reasoning(problem, context_str, spec_endpoint, spec_model):
        prompt = (
            f"{context_str}\n\nQuestion: {problem}\n\n"
            "Instead of directly answering the question, please write additional "
            "Python code that will give intermediate results after execution. "
            "Wrap the code within ```python and ```. The code should be "
            "self-contained with all the import and initialization."
        )
        msgs = [
            {"role": "system", "content": "You are an expert analyst. Write executable Python code to solve problems step by step."},
            {"role": "user", "content": prompt},
        ]
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
                generated_code = ""

        if generated_code:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False, dir="/tmp"
                ) as f:
                    f.write(generated_code)
                    code_path = f.name
                result = subprocess.run(
                    ["python3", code_path],
                    timeout=30,
                    capture_output=True,
                    text=True,
                )
                exec_result = result.stdout.strip()
                if result.returncode != 0 and result.stderr:
                    exec_result += f"\n[stderr]: {result.stderr[:500]}"
                os.unlink(code_path)
            except subprocess.TimeoutExpired:
                exec_result = "[Code execution timed out after 30s]"
            except Exception as e:
                exec_result = f"[Code execution error: {e}]"

        parts = []
        if generated_code:
            parts.append(f"Generated code:\n```python\n{generated_code}\n```")
        if exec_result:
            parts.append(f"Execution output:\n{exec_result}")
        if not parts:
            parts.append(content)

        return "\n\n".join(parts)

    def handle_answer(problem, context_str, spec_endpoint, spec_model):
        prompt = f"{context_str}\n\nProblem:\n{problem}"
        msgs = [
            {"role": "system", "content": "Please reason step by step and give a clear, concise final answer."},
            {"role": "user", "content": prompt},
        ]
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
            ctx_parts.append("Documents:\n" + "\n\n".join(
                f"Doc {i+1}: {d[:1200]}" for i, d in enumerate(doc_list)
            ))
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
            repeated = used_tools[-1]
            current_tools = [t for t in tools if t["function"]["name"] != repeated]

        try:
            orch_resp = chat_completion(
                orchestrator_endpoint,
                orchestrator_model,
                [
                    {"role": "system", "content": "You are good at using tools."},
                    {"role": "user", "content": user_content},
                ],
                tools_list=current_tools,
                max_tokens=max_output_tokens,
            )
        except Exception as e:
            tool_trace.append({"turn": turn, "tool": "error", "error": str(e)})
            break

        choice = orch_resp["choices"][0]
        msg = choice["message"]
        tc_list = msg.get("tool_calls", [])

        if not tc_list:
            content = msg.get("content", "")
            if content and "</think>" in content:
                final_answer = content.split("</think>")[-1].strip()
            elif content:
                final_answer = content
            tool_trace.append({"turn": turn, "tool": "(direct)", "note": "No tool called"})
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
        est_input_tokens = len(question.split()) * 2 + len(context_str.split()) * 2
        est_output_tokens = 300
        step_cost = (est_input_tokens * pricing["input"] + est_output_tokens * pricing["output"]) / 1_000_000

        if tool_name == "search":
            search_output, query = handle_search(question, context_str, spec_endpoint, spec_model)
            doc_list.append(search_output)
            latency = round((time.monotonic() - start) * 1000)
            total_cost += step_cost
            tool_trace.append({
                "turn": turn,
                "tool": "search",
                "specialist": f"{spec_model}@{model_id}",
                "query": query,
                "latency_ms": latency,
                "est_cost_usd": round(step_cost, 6),
            })

        elif tool_name == "enhance_reasoning":
            reasoning = handle_reasoning(question, context_str, spec_endpoint, spec_model)
            if reasoning:
                reasoning_list.append(reasoning)
            latency = round((time.monotonic() - start) * 1000)
            est_output_tokens = 500
            step_cost = (est_input_tokens * pricing["input"] + est_output_tokens * pricing["output"]) / 1_000_000
            total_cost += step_cost
            has_code = "Generated code:" in reasoning if reasoning else False
            tool_trace.append({
                "turn": turn,
                "tool": "enhance_reasoning",
                "specialist": f"{spec_model}@{model_id}",
                "latency_ms": latency,
                "code_executed": has_code,
                "est_cost_usd": round(step_cost, 6),
            })

        elif tool_name == "answer":
            final_answer = handle_answer(question, context_str, spec_endpoint, spec_model)
            latency = round((time.monotonic() - start) * 1000)
            total_cost += step_cost
            tool_trace.append({
                "turn": turn,
                "tool": "answer",
                "specialist": f"{spec_model}@{model_id}",
                "latency_ms": latency,
                "est_cost_usd": round(step_cost, 6),
            })
            break

        else:
            tool_trace.append({"turn": turn, "tool": tool_name, "note": "Unknown tool"})

    total_latency = round((time.monotonic() - total_start) * 1000)

    if not final_answer and tool_trace:
        final_answer = "(Orchestrator did not produce a final answer within the allowed turns)"

    return json.dumps({
        "question": question.strip(),
        "answer": final_answer,
        "tool_trace": tool_trace,
        "total_turns": len(tool_trace),
        "total_latency_ms": total_latency,
        "est_total_cost_usd": round(total_cost, 6),
    })


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
)
def format_output(result_json: str) -> str:
    """Format the orchestration result for display."""
    import json

    data = json.loads(result_json)

    lines = [
        "=== ToolOrchestra Orchestration Result ===",
        "",
        f"Question: {data['question']}",
        f"Total turns: {data['total_turns']}",
        f"Total latency: {data['total_latency_ms']}ms",
        f"Estimated cost: ${data.get('est_total_cost_usd', 0):.6f}",
        "",
        "Tool trace (Orchestrator-8B decisions):",
    ]
    for t in data.get("tool_trace", []):
        spec = t.get("specialist", "")
        latency = t.get("latency_ms", "")
        cost = t.get("est_cost_usd", 0)
        query = t.get("query", "")
        note = t.get("note", "")
        code_exec = t.get("code_executed", None)
        detail = f" query={query}" if query else (f" {note}" if note else "")
        if code_exec is not None:
            detail += f" code_executed={code_exec}"
        lines.append(
            f"  Turn {t['turn']}: {t['tool']}  specialist={spec}  "
            f"[{latency}ms, ${cost:.6f}]{detail}"
        )

    lines.append("")
    lines.append(f"Answer (from specialist model):\n{data['answer']}")

    summary = "\n".join(lines)
    print(summary)
    return summary


@dsl.pipeline(
    name="toolorchestra-true-orchestration",
    description="ToolOrchestra paper-faithful orchestration: Orchestrator-8B (agent) "
    "calls specialist models via vLLM chat completions. Multi-turn loop with "
    "search, reasoning, and answer tools routed to configurable specialists.",
)
def orchestration_pipeline(
    question: str = "What is 7 factorial?",
    orchestrator_endpoint: str = ORCHESTRATOR_DEFAULT,
    orchestrator_model: str = "orchestrator-8b",
    model_mapping_json: str = DEFAULT_MODEL_MAPPING,
    tool_definitions_json: str = _FALLBACK_TOOLS,
    max_turns: int = 10,
    max_output_tokens: int = 2048,
):
    result = orchestration_loop(
        question=question,
        orchestrator_endpoint=orchestrator_endpoint,
        orchestrator_model=orchestrator_model,
        model_mapping_json=model_mapping_json,
        tool_definitions_json=tool_definitions_json,
        max_turns=max_turns,
        max_output_tokens=max_output_tokens,
    )
    format_output(result_json=result.output)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=orchestration_pipeline,
        package_path="orchestration_pipeline.yaml",
    )
    print("Pipeline compiled to orchestration_pipeline.yaml")
