"""ToolOrchestra — FastAPI backend with SSE streaming.

Serves the React frontend and exposes /api/orchestrate as an SSE endpoint
that streams typed events as the orchestration loop runs.
"""

import json
import os
import subprocess
import tempfile
import time

import httpx
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Load configuration from YAML
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_config() -> dict:
    """Load config.yaml from CONFIG_PATH env var or same directory as script."""
    paths = [
        os.getenv("CONFIG_PATH", ""),
        os.path.join(_SCRIPT_DIR, "config.yaml"),
    ]
    for p in paths:
        if p and os.path.isfile(p):
            with open(p) as f:
                cfg = yaml.safe_load(f) or {}
            print(f"[config] Loaded from {p}")
            return cfg
    print("[config] No config.yaml found — using built-in defaults")
    return {}


def _load_tools_json(cfg: dict) -> list[dict]:
    """Load tool definitions from the paper's tools.json."""
    rel = cfg.get("tools_json_path", "tools.json")
    config_dir = os.path.dirname(os.getenv("CONFIG_PATH", ""))
    paths = [
        os.path.join(config_dir, rel) if config_dir else "",
        os.path.join(_SCRIPT_DIR, rel),
        os.path.join(os.getcwd(), rel),
    ]
    paths = [p for p in paths if p]
    for p in paths:
        if os.path.isfile(p):
            with open(p) as f:
                tools = json.load(f)
            print(f"[config] Loaded tools from {p}")
            return tools
    raise FileNotFoundError(f"tools.json not found at: {paths}")


CFG = _load_config()

# --- Orchestrator settings ---
_orch = CFG.get("orchestrator", {})
ORCHESTRATOR_ENDPOINT = _orch.get("endpoint", "http://orchestrator-8b-predictor.orchestrator-rhoai.svc.cluster.local:8080/v1")
ORCHESTRATOR_MODEL = _orch.get("model", "orchestrator-8b")
ORCH_MAX_TOKENS = _orch.get("max_tokens", 12000)
ORCH_TEMPERATURE = _orch.get("temperature", 1.0)
ORCH_SYSTEM_PROMPT = _orch.get("system_prompt", "You are good at using tools.")
MAX_TURNS_DEFAULT = _orch.get("max_turns", 30)

# --- Specialist model mapping ---
MODEL_MAPPING: dict[str, dict] = CFG.get("specialists", {})

# --- Tool definitions (loaded from paper's tools.json) ---
TOOL_DEFINITIONS = _load_tools_json(CFG)

# --- Pricing ---
TOOL_PRICING: dict[str, dict] = CFG.get("pricing", {})

# --- Context limits ---
SPECIALIST_CONTEXT_LIMITS: dict[str, dict] = CFG.get("context_limits", {})

# --- Prompt templates ---
_prompts = CFG.get("prompts", {})
PROMPT_SEARCH_QUERY = _prompts.get(
    "search_query",
    "Instead of directly answering the question, please write a query "
    "to search for a piece of relevant and missing information. The "
    "query should be a few key words about the information to search "
    "or a short sentence. Wrap the query within <query> and </query>.",
)
PROMPT_REASONING_CODE = _prompts.get(
    "reasoning_code",
    "Instead of directly answering the question, please write "
    "additional python code that will give intermidiate results "
    "after execution. Wrap the code within ```python and ```. "
    "The code should be self-contained with all the import and "
    "initialization.",
)
PROMPT_ANSWER_QWEN_SYSTEM = _prompts.get(
    "answer_qwen_system",
    "Please reason step by step, and put your final answer within \\boxed{}.",
)
PROMPT_ANSWER_GENERAL = _prompts.get(
    "answer_general",
    "Take a deep breath and think hard with high reasoning, "
    "wrap the thoughts within <think> and </think>, and wrap only the exact answer "
    "without any explanation within <answer> and </answer>."
    "Output using the following format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>",
)
PROMPT_ANSWER_LLAMA = _prompts.get(
    "answer_llama",
    "Wrap the thinking process and explanation between <think> and </think> "
    "and wrap only the exact answer without any explanation within <answer> "
    "and </answer>.",
)

# --- Specialist fallback defaults (used only when model_id has no config entry) ---
_spec_defs = CFG.get("specialist_defaults", {})
SEARCH_MAX_TOKENS_DEFAULT = _spec_defs.get("search_max_tokens", 8000)
REASONING_MAX_TOKENS_DEFAULT = _spec_defs.get("reasoning_max_tokens", 8000)
ANSWER_MAX_TOKENS_DEFAULT = _spec_defs.get("answer_max_tokens", 2000)

# --- UI presets ---
PRESET_QUESTIONS: list[str] = CFG.get("preset_questions", [
    "What is 17 factorial divided by 13 factorial, and is the result a prime number?",
    "What is the population of Tokyo compared to Paris?",
])

# ---------------------------------------------------------------------------
# Token estimation / truncation (matches paper)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) if text else 0


def cut_to_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    if not text:
        return text, 0
    words = text.split()
    est = int(len(words) * 1.3)
    if est <= max_tokens:
        return text, est
    keep = int(max_tokens / 1.3)
    return " ".join(words[-keep:]), max_tokens


# ---------------------------------------------------------------------------
# vLLM / LLM helpers
# ---------------------------------------------------------------------------

_http_client = httpx.Client(timeout=300.0, verify=False)


def chat_completion(endpoint, model, messages, *, tools_list=None,
                    max_tokens=512, temperature=1.0, _retries=2):
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    if tools_list:
        payload["tools"] = tools_list
        payload["tool_choice"] = "auto"
    last_err = None
    for attempt in range(_retries + 1):
        try:
            resp = _http_client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPStatusError, httpx.TimeoutException,
                httpx.NetworkError, httpx.ProtocolError,
                ValueError) as exc:
            last_err = exc
            if attempt < _retries:
                time.sleep(2 ** attempt)
                continue
    return {
        "choices": [{"message": {"content": None, "role": "assistant"}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "_error": str(last_err),
    }


def call_specialist(endpoint, model, messages, max_tokens=1024, temperature=0.0,
                    model_id=""):
    spec_cfg = MODEL_MAPPING.get(model_id, {})
    if "temperature" in spec_cfg:
        temperature = spec_cfg["temperature"]
    if "max_tokens" in spec_cfg:
        max_tokens = spec_cfg["max_tokens"]
    return chat_completion(
        endpoint, model, messages, max_tokens=max_tokens, temperature=temperature,
    )


def stream_orchestrator(endpoint, model, messages, *, tools_list=None,
                        max_tokens=512, temperature=1.0):
    url = f"{endpoint.rstrip('/')}/chat/completions"
    body = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature, "stream": True,
    }
    if tools_list:
        body["tools"] = tools_list
        body["tool_choice"] = "auto"
    body["stream_options"] = {"include_usage": True}

    content_acc: list[str] = []
    tool_calls_acc: list[dict] = []
    finish_reason = None
    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with _http_client.stream("POST", url, json=body) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if chunk.get("usage"):
                u = chunk["usage"]
                usage_data = {
                    "prompt_tokens": u.get("prompt_tokens", 0),
                    "completion_tokens": u.get("completion_tokens", 0),
                    "total_tokens": u.get("total_tokens", 0),
                }

            if not chunk.get("choices"):
                continue
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            tok = delta.get("content") or ""
            if tok:
                content_acc.append(tok)
                yield "token", "".join(content_acc)

            if "tool_calls" in delta:
                for tc_d in delta["tool_calls"]:
                    idx = tc_d.get("index", 0)
                    while len(tool_calls_acc) <= idx:
                        tool_calls_acc.append(
                            {"id": "", "type": "function",
                             "function": {"name": "", "arguments": ""}})
                    if "id" in tc_d:
                        tool_calls_acc[idx]["id"] = tc_d["id"]
                    if "function" in tc_d:
                        fn = tc_d["function"]
                        if fn.get("name"):
                            tool_calls_acc[idx]["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]

    full_content = "".join(content_acc) or None
    msg = {"role": "assistant", "content": full_content}
    if tool_calls_acc:
        msg["tool_calls"] = tool_calls_acc
    yield "done", {
        "choices": [{"message": msg, "finish_reason": finish_reason or "stop"}],
        "usage": usage_data,
    }


def _extract_thinking(raw: str) -> str:
    if "<think>" not in raw:
        return ""
    after = raw.split("<think>", 1)[1]
    if "</think>" in after:
        return after.split("</think>", 1)[0].strip()
    return after.strip()


def strip_think(content: str) -> str:
    if not content:
        return ""
    if "</think>" in content:
        return content.split("</think>")[-1].strip()
    return content.strip()


# ---------------------------------------------------------------------------
# Tool handlers (paper-aligned)
# ---------------------------------------------------------------------------


def _ddg_search_with_retry(query: str, max_results: int = 5, max_retries: int = 3) -> list[str]:
    from duckduckgo_search import DDGS

    for attempt in range(max_retries):
        try:
            doc_contents: list[str] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query[:390], max_results=max_results):
                    doc_contents.append(
                        f"Title: {r.get('title', '')}\n"
                        f"{r.get('body', '')}\n"
                        f"URL: {r.get('href', '')}"
                    )
            if doc_contents:
                return doc_contents
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
    return []


def _usage(resp):
    u = resp.get("usage", {})
    prompt = u.get("prompt_tokens", 0)
    completion = u.get("completion_tokens", 0)
    total = u.get("total_tokens", 0)
    if completion == 0 and total > prompt:
        completion = total - prompt
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total or (prompt + completion),
    }


def handle_search(problem, context_str, spec_endpoint, spec_model, model_id=""):
    prompt = f"{context_str}\n\nQuestion: {problem}\n{PROMPT_SEARCH_QUERY}"
    msgs = [{"role": "user", "content": prompt}]
    resp = call_specialist(spec_endpoint, spec_model, msgs,
                           max_tokens=SEARCH_MAX_TOKENS_DEFAULT, model_id=model_id)
    usage = _usage(resp)
    content = strip_think(resp["choices"][0]["message"].get("content") or "")
    query = None
    if "<query>" in content and "</query>" in content:
        query = content.split("<query>")[-1].split("</query>")[0].strip()
    if not query or len(query) < 5:
        query = problem

    doc_contents = _ddg_search_with_retry(query)
    if not doc_contents:
        simplified = " ".join(query.split()[:8])
        doc_contents = _ddg_search_with_retry(simplified)
    return doc_contents, query, usage


def handle_reasoning(problem, context_str, spec_endpoint, spec_model, model_id=""):
    prompt = f"{context_str.strip()}\n\nQuestion: {problem}\n{PROMPT_REASONING_CODE}"
    msgs = [{"role": "user", "content": prompt}]
    resp = call_specialist(spec_endpoint, spec_model, msgs,
                           max_tokens=REASONING_MAX_TOKENS_DEFAULT, model_id=model_id)
    usage = _usage(resp)
    content = strip_think(resp["choices"][0]["message"].get("content") or "")

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
                ["python3", code_path], timeout=60,
                capture_output=True, text=True,
            )
            exec_result = result.stdout.strip()
            if result.returncode != 0 and result.stderr:
                exec_result += f"\n[stderr]: {result.stderr[:500]}"
            os.unlink(code_path)
        except subprocess.TimeoutExpired:
            exec_result = "[Code execution timed out after 60s]"
        except Exception as e:
            exec_result = f"[Code execution error: {e}]"
    return generated_code, exec_result, usage


def _is_garbage(text: str, threshold: float = 0.35) -> bool:
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) > threshold


def handle_answer(problem, context_str, spec_endpoint, spec_model, model_id=""):
    is_qwen = "qwen" in spec_model.lower() or model_id.startswith("answer-math")
    if is_qwen:
        user_content = context_str.strip() + "\n\nProblem:\n" + problem
        msgs = [
            {"role": "system", "content": PROMPT_ANSWER_QWEN_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        resp = call_specialist(spec_endpoint, spec_model, msgs,
                               max_tokens=ANSWER_MAX_TOKENS_DEFAULT, model_id=model_id)
        usage = _usage(resp)
        content = strip_think(resp["choices"][0]["message"].get("content") or "")
        pred = ""
        if "\\boxed{" in content:
            try:
                parts = content.split("\\boxed{")[-1].split("}")[:-1]
                pred = "}".join(parts).strip()
            except (IndexError, ValueError):
                pass
        if not pred:
            pred = content.strip()
    else:
        model_family = _specialist_model_key(spec_model, model_id)
        if model_family == "llama":
            answer_prompt = PROMPT_ANSWER_LLAMA
        else:
            answer_prompt = PROMPT_ANSWER_GENERAL
        prompt = context_str.strip() + "\n\nProblem:\n" + problem + "\n\n" + answer_prompt
        msgs = [{"role": "user", "content": prompt}]
        resp = call_specialist(spec_endpoint, spec_model, msgs,
                               max_tokens=ANSWER_MAX_TOKENS_DEFAULT, model_id=model_id)
        usage = _usage(resp)
        content = strip_think(resp["choices"][0]["message"].get("content") or "")
        pred = ""
        if "<answer>" in content and "</answer>" in content:
            pred = content.split("<answer>")[-1].split("</answer>")[0].strip()
        if not pred:
            pred = content.strip()

    garbage = _is_garbage(pred)
    if garbage:
        pred = "[Model produced invalid output]"
    if _is_garbage(content):
        content = pred
    return content, pred, usage, garbage


# ---------------------------------------------------------------------------
# Context builder (matches paper's eval_hle.py)
# ---------------------------------------------------------------------------


def build_context(question, doc_list, code_list, attempt_list):
    doc_str = ""
    for doc_idx, doc in enumerate(doc_list):
        doc_str += f"Doc {doc_idx + 1}: {doc[:1200]} ...\n\n"

    code_str = ""
    for code_piece in code_list:
        code_str += (
            f"```python\n{code_piece['code']}\n```\n\n"
            f"```output\n{code_piece['output']}\n```\n\n"
        )

    attempt_str = ""
    for attempt_idx, attempt in enumerate(attempt_list):
        attempt_str += (
            f"Attempt{attempt_idx + 1} answer by "
            f"{attempt['model']}: {attempt['answer']}\n"
        )

    attempt_str, _ = cut_to_tokens(attempt_str, 8000)
    if attempt_str and not attempt_str.startswith("Attempt"):
        attempt_str = "Attempt answer: " + attempt_str

    code_attempt_str, code_attempt_len = cut_to_tokens(
        code_str + attempt_str, 12000
    )
    if code_attempt_str and not code_attempt_str.startswith("```"):
        code_attempt_str = "```\n" + code_attempt_str

    problem_length = estimate_tokens(question)
    max_ctx_budget = 27000 - problem_length

    if code_attempt_len < max_ctx_budget:
        if code_attempt_str:
            raw = (
                doc_str
                + "\npython code and execution outputs:\n"
                + code_attempt_str
            )
            context_str, _ = cut_to_tokens(raw, max_ctx_budget)
        else:
            context_str, _ = cut_to_tokens(doc_str, max_ctx_budget)
        if doc_str:
            context_str = "Documents:\n" + context_str
    else:
        context_str = code_attempt_str
    return context_str



def _specialist_model_key(spec_model: str, model_id: str) -> str:
    m = spec_model.lower()
    if "qwen2.5-math" in m or "qwen-math" in m or model_id.startswith("answer-math"):
        return "qwen-math"
    if "gemini" in m:
        return "gemini"
    if "llama" in m:
        return "llama"
    return "default"


def build_specialist_context(
    tool_name: str, question: str, doc_list: list[str],
    code_list: list[dict], spec_model: str, model_id: str,
) -> str:
    """Build context tailored to a specialist's context window, matching the paper."""
    _default_limits = {"max_code": 12000, "max_ctx": 24000, "doc_trunc": 1000}
    key = _specialist_model_key(spec_model, model_id)
    tool_limits = SPECIALIST_CONTEXT_LIMITS.get(tool_name, {})
    limits = tool_limits.get(key, tool_limits.get("default", _default_limits))

    max_code = limits["max_code"]
    max_ctx = limits["max_ctx"]
    doc_trunc = limits["doc_trunc"]

    doc_str = ""
    for doc_idx, doc in enumerate(doc_list):
        if doc_trunc:
            doc_str += f"Doc {doc_idx + 1}: {doc[:doc_trunc]}\n\n"
        else:
            doc_str += f"Doc {doc_idx + 1}: {doc}\n\n"

    code_str = ""
    for code_piece in code_list:
        code_str += (
            f"```python\n{code_piece['code']}\n```\n\n"
            f"```output\n{code_piece['output']}\n```\n\n"
        )
    code_str, code_len = cut_to_tokens(code_str, max_code)
    if code_str and not code_str.startswith("```"):
        code_str = "```\n" + code_str

    problem_len = estimate_tokens(question)
    ctx, _ = cut_to_tokens(doc_str + code_str, max_ctx - problem_len)
    if doc_str:
        ctx = "Documents:\n" + ctx
    return ctx


def resolve_display_model(model_id: str) -> tuple[str, str]:
    info = MODEL_MAPPING.get(model_id, {})
    display = info.get("display", "")
    model_name = info.get("model", "")
    if "qwen" in display.lower() or "math" in model_id.lower():
        return "qwen", display or "Qwen2.5-Math-7B"
    if "gemini" in display.lower() or "gemini" in model_name.lower():
        return "gemini", display or "Gemini"
    return "llama", display or "Llama-3.2-3B"


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


VALID_MODEL_IDS = {
    "enhance_reasoning": {"reasoner-1", "reasoner-2", "reasoner-3"},
    "answer": {"answer-1", "answer-2", "answer-3", "answer-4", "answer-math-1", "answer-math-2"},
    "search": {"search-1", "search-2", "search-3"},
}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


import re

_MATH_RE = re.compile(
    r"(?:\d+\s*[!*/+\-^])|"           # digits followed by operator
    r"(?:factorial|permut|combin)|"     # math keywords
    r"(?:prime|divisib|gcd|lcm)|"       # number theory
    r"(?:solv|integr|deriv|f\(x\))|"   # calculus / algebra
    r"(?:compound.{0,10}interest)|"     # finance math
    r"(?:sqrt|log|sin|cos|tan)|"        # functions
    r"(?:\d+\s*%)|"                     # percentages
    r"(?:calculat|comput|evaluat)",     # action words
    re.IGNORECASE,
)


def _looks_like_math(question: str) -> bool:
    return bool(_MATH_RE.search(question))


# ---------------------------------------------------------------------------
# Orchestration SSE generator
# ---------------------------------------------------------------------------


def orchestrate_sse(question: str, max_turns: int):
    """Generator that yields SSE-formatted strings for each orchestration event."""

    yield _sse("status", {"message": "Orchestrator-8B is analyzing your question..."})

    doc_list: list[str] = []
    code_list: list[dict] = []
    attempt_list: list[dict] = []
    tool_trace: list[dict] = []
    used_tools: list[str] = []
    final_answer = ""
    total_cost = 0.0
    total_tokens = {"prompt": 0, "completion": 0}
    is_math = _looks_like_math(question)
    reasoning_done = False

    for turn in range(max_turns):
        context_str = build_context(question, doc_list, code_list, attempt_list)
        user_content = f"Problem: {question}\n\n{context_str}\n\nChoose an appropriate tool."

        current_tools = list(TOOL_DEFINITIONS)
        if len(used_tools) > 1 and used_tools[-1] == used_tools[-2]:
            repeated = used_tools[-1]
            current_tools = [t for t in current_tools if t["function"]["name"] != repeated]

        orch_start = time.monotonic()
        orch_resp = None
        last_think_yield = 0

        try:
            for ev_type, ev_data in stream_orchestrator(
                ORCHESTRATOR_ENDPOINT, ORCHESTRATOR_MODEL,
                [
                    {"role": "system", "content": ORCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                tools_list=current_tools, max_tokens=ORCH_MAX_TOKENS, temperature=ORCH_TEMPERATURE,
            ):
                if ev_type == "done":
                    orch_resp = ev_data
                elif ev_type == "token":
                    now = time.monotonic()
                    if now - last_think_yield >= 0.2:
                        last_think_yield = now
                        thinking = _extract_thinking(ev_data)
                        if thinking:
                            yield _sse("thinking", {
                                "content": thinking[-600:],
                                "elapsed_s": round(now - orch_start, 1),
                                "turn": turn,
                            })
        except Exception as e:
            yield _sse("error", {"message": str(e)})
            return

        orch_usage = orch_resp.get("usage", {})
        total_tokens["prompt"] += orch_usage.get("prompt_tokens", 0)
        total_tokens["completion"] += orch_usage.get("completion_tokens", 0)

        choice = orch_resp["choices"][0]
        msg = choice["message"]
        tc_list = msg.get("tool_calls", [])

        if not tc_list:
            content = strip_think(msg.get("content", ""))
            if content and turn == max_turns - 1:
                final_answer = content
                tool_trace.append({"turn": turn, "tool": "(direct)", "note": "No tool called — final turn"})
                break
            tool_trace.append({"turn": turn, "tool": "(retry)", "note": "No tool call — retrying"})
            continue

        tool_name = None
        tool_args = {}
        model_id = ""
        for tc in tc_list:
            _fn_name = tc.get("function", {}).get("name", "")
            valid_ids = VALID_MODEL_IDS.get(_fn_name)
            if not valid_ids:
                continue
            try:
                _fn_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                _fn_args = {}
            _mid = _fn_args.get("model", "")
            if _mid not in valid_ids:
                continue
            tool_name = _fn_name
            tool_args = _fn_args
            model_id = _mid
            break

        if not tool_name:
            tool_trace.append({
                "turn": turn, "tool": "(invalid)",
                "note": "No valid tool call found in response — retrying",
            })
            continue

        if is_math and tool_name == "answer" and not reasoning_done:
            tool_name = "enhance_reasoning"
            model_id = "reasoner-3"
            tool_args = {"model": model_id}

        if tool_name == "enhance_reasoning":
            reasoning_done = True

        used_tools.append(tool_name)

        spec_info = MODEL_MAPPING.get(model_id, {})
        spec_endpoint = spec_info.get("endpoint", ORCHESTRATOR_ENDPOINT)
        spec_model = spec_info.get("model", ORCHESTRATOR_MODEL)
        diagram_key, display_name = resolve_display_model(model_id)
        orch_latency = round((time.monotonic() - orch_start) * 1000)

        yield _sse("tool_call", {
            "turn": turn,
            "tool": tool_name,
            "model_id": model_id,
            "display_name": display_name,
            "diagram_key": diagram_key,
            "orch_latency_ms": orch_latency,
        })

        start = time.monotonic()
        pricing = TOOL_PRICING.get(model_id, {"input": 0.0, "output": 0.0})

        if tool_name == "search":
            spec_ctx = build_specialist_context("search", question, doc_list, code_list, spec_model, model_id)
            doc_contents, query, spec_usage = handle_search(question, spec_ctx, spec_endpoint, spec_model, model_id=model_id)
            n_real = len(doc_contents)
            for doc in doc_contents[::-1]:
                if doc.strip() and doc not in doc_list:
                    doc_list.append(doc)
            latency = round((time.monotonic() - start) * 1000)
            in_tok = spec_usage["prompt_tokens"]
            out_tok = spec_usage["completion_tokens"]
            total_tokens["prompt"] += in_tok
            total_tokens["completion"] += out_tok
            step_cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000
            total_cost += step_cost
            trace_entry = {
                "turn": turn, "tool": "search",
                "specialist": f"{display_name} ({model_id})",
                "query": query, "latency_ms": latency,
                "in_tokens": in_tok, "out_tokens": out_tok,
                "est_cost_usd": round(step_cost, 6),
            }
            tool_trace.append(trace_entry)

            yield _sse("search_result", {
                **trace_entry, "count": n_real,
                "diagram_key": diagram_key, "display_name": display_name,
                "total_cost": round(total_cost, 6),
            })

        elif tool_name == "enhance_reasoning":
            spec_ctx = build_specialist_context("enhance_reasoning", question, doc_list, code_list, spec_model, model_id)
            generated_code, exec_result, spec_usage = handle_reasoning(question, spec_ctx, spec_endpoint, spec_model, model_id=model_id)
            if exec_result.strip():
                code_list.append({"code": generated_code, "output": exec_result})
            latency = round((time.monotonic() - start) * 1000)
            in_tok = spec_usage["prompt_tokens"]
            out_tok = spec_usage["completion_tokens"]
            total_tokens["prompt"] += in_tok
            total_tokens["completion"] += out_tok
            step_cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000
            total_cost += step_cost
            trace_entry = {
                "turn": turn, "tool": "enhance_reasoning",
                "specialist": f"{display_name} ({model_id})",
                "latency_ms": latency,
                "in_tokens": in_tok, "out_tokens": out_tok,
                "code_executed": bool(generated_code and exec_result),
                "est_cost_usd": round(step_cost, 6),
            }
            tool_trace.append(trace_entry)

            yield _sse("reasoning_result", {
                **trace_entry,
                "code_preview": generated_code[:400] if generated_code else "",
                "exec_output": exec_result[:300] if exec_result else "",
                "diagram_key": diagram_key, "display_name": display_name,
                "total_cost": round(total_cost, 6),
            })

        elif tool_name == "answer":
            spec_ctx = build_specialist_context("answer", question, doc_list, code_list, spec_model, model_id)
            full_response, pred, spec_usage, garbage = handle_answer(
                question, spec_ctx, spec_endpoint, spec_model, model_id=model_id
            )
            latency = round((time.monotonic() - start) * 1000)
            in_tok = spec_usage["prompt_tokens"]
            out_tok = spec_usage["completion_tokens"]
            total_tokens["prompt"] += in_tok
            total_tokens["completion"] += out_tok
            step_cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000
            total_cost += step_cost
            trace_entry = {
                "turn": turn, "tool": "answer",
                "specialist": f"{display_name} ({model_id})",
                "latency_ms": latency,
                "in_tokens": in_tok, "out_tokens": out_tok,
                "est_cost_usd": round(step_cost, 6),
            }
            tool_trace.append(trace_entry)

            if garbage:
                escalate_id = "answer-1"
                esc_info = MODEL_MAPPING.get(escalate_id, {})
                esc_endpoint = esc_info.get("endpoint", ORCHESTRATOR_ENDPOINT)
                esc_model = esc_info.get("model", ORCHESTRATOR_MODEL)
                esc_dk, esc_dn = resolve_display_model(escalate_id)

                yield _sse("answer_retry", {
                    **trace_entry,
                    "prediction": pred[:500],
                    "is_final": False,
                    "diagram_key": diagram_key, "display_name": display_name,
                    "total_cost": round(total_cost, 6),
                    "reason": "Model produced invalid output — escalating to stronger model",
                    "escalate_to": esc_dn,
                })

                esc_start = time.monotonic()
                esc_ctx = build_specialist_context("answer", question, doc_list, code_list, esc_model, escalate_id)
                esc_response, esc_pred, esc_usage, _ = handle_answer(
                    question, esc_ctx, esc_endpoint, esc_model, model_id=escalate_id
                )
                esc_latency = round((time.monotonic() - esc_start) * 1000)
                esc_in = esc_usage["prompt_tokens"]
                esc_out = esc_usage["completion_tokens"]
                total_tokens["prompt"] += esc_in
                total_tokens["completion"] += esc_out
                esc_pricing = TOOL_PRICING.get(escalate_id, {"input": 0.0, "output": 0.0})
                esc_cost = (esc_in * esc_pricing["input"] + esc_out * esc_pricing["output"]) / 1_000_000
                total_cost += esc_cost

                esc_trace = {
                    "turn": turn, "tool": "answer (escalated)",
                    "specialist": f"{esc_dn} ({escalate_id})",
                    "latency_ms": esc_latency,
                    "in_tokens": esc_in, "out_tokens": esc_out,
                    "est_cost_usd": round(esc_cost, 6),
                }
                tool_trace.append(esc_trace)
                final_answer = esc_pred

                yield _sse("answer_final", {
                    **esc_trace,
                    "prediction": esc_pred[:500],
                    "is_final": True,
                    "diagram_key": esc_dk, "display_name": esc_dn,
                    "total_cost": round(total_cost, 6),
                })
                break

            if attempt_list:
                final_answer = pred
                yield _sse("answer_final", {
                    **trace_entry,
                    "prediction": pred[:500],
                    "is_final": True,
                    "diagram_key": diagram_key, "display_name": display_name,
                    "total_cost": round(total_cost, 6),
                })
                break
            else:
                attempt_list.append({"model": model_id, "answer": pred})
                yield _sse("answer_attempt", {
                    **trace_entry,
                    "prediction": pred[:500],
                    "is_final": False,
                    "diagram_key": diagram_key, "display_name": display_name,
                    "total_cost": round(total_cost, 6),
                    "note": "First attempt stored — orchestrator will refine",
                })
        else:
            tool_trace.append({"turn": turn, "tool": tool_name, "note": "Unknown tool"})

    if not final_answer:
        final_answer = "(Orchestrator did not produce a final answer within the allowed turns)"

    total_tok = total_tokens["prompt"] + total_tokens["completion"]
    yield _sse("done", {
        "final_answer": final_answer,
        "total_turns": len(tool_trace),
        "total_tokens": total_tok,
        "prompt_tokens": total_tokens["prompt"],
        "completion_tokens": total_tokens["completion"],
        "total_cost": round(total_cost, 6),
        "tool_trace": tool_trace,
    })


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ToolOrchestra API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class OrchestrateRequest(BaseModel):
    question: str
    max_turns: int = MAX_TURNS_DEFAULT


@app.post("/api/orchestrate")
async def api_orchestrate(req: OrchestrateRequest):
    return StreamingResponse(
        orchestrate_sse(req.question, req.max_turns),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/models")
async def api_models():
    return {
        "mapping": MODEL_MAPPING,
        "presets": PRESET_QUESTIONS,
    }


@app.get("/api/health")
async def api_health():
    return {"status": "ok"}


INDEX_HTML = os.path.join(os.path.dirname(__file__), "index.html")


@app.get("/")
async def serve_index():
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html")
    return {"message": "Frontend not built yet. Run: cd frontend && npm run build"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
