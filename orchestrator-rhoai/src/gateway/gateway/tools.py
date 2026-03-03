import logging
import time
from dataclasses import dataclass, field

from gateway.llm_client import chat_completion

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    tool_name: str
    model_selected: str
    latency_ms: float
    content: str = ""
    documents: list[str] = field(default_factory=list)
    is_final_answer: bool = False


async def handle_search(
    model: str,
    problem: str,
    context: str,
    specialist_endpoint: str,
    specialist_model: str,
) -> ToolResult:
    """Execute the search tool: generate a query and search the web via DuckDuckGo."""
    start = time.monotonic()

    prompt = f"{context}\n\nProblem: {problem}\n\nGenerate a concise web search query to find the missing information needed to solve this problem. Output only the search query, nothing else."
    messages = [
        {"role": "system", "content": "You generate concise search queries."},
        {"role": "user", "content": prompt},
    ]

    query_resp = await chat_completion(
        endpoint=specialist_endpoint,
        model=specialist_model,
        messages=messages,
        max_tokens=128,
        temperature=0.3,
    )
    query = (query_resp.content or problem).strip().strip('"').strip("'")
    logger.info("Search query: %s", query)

    documents = await _duckduckgo_search(query)
    latency = (time.monotonic() - start) * 1000

    return ToolResult(
        tool_name="search",
        model_selected=model,
        latency_ms=latency,
        content=f"Search results for: {query}",
        documents=documents,
    )


async def handle_enhance_reasoning(
    model: str,
    problem: str,
    context: str,
    specialist_endpoint: str,
    specialist_model: str,
) -> ToolResult:
    """Execute the enhance_reasoning tool: call the specialist for analysis."""
    start = time.monotonic()

    prompt = (
        f"{context}\n\nQuestion: {problem}\n\n"
        "Instead of directly answering the question, analyze it step by step. "
        "Identify key components, provide intermediate reasoning, and outline "
        "the approach to solve it."
    )
    messages = [
        {"role": "system", "content": "You are an expert analyst. Provide detailed intermediate reasoning."},
        {"role": "user", "content": prompt},
    ]

    resp = await chat_completion(
        endpoint=specialist_endpoint,
        model=specialist_model,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
    )
    reasoning = resp.content or ""
    latency = (time.monotonic() - start) * 1000

    return ToolResult(
        tool_name="enhance_reasoning",
        model_selected=model,
        latency_ms=latency,
        content=reasoning,
    )


async def handle_answer(
    model: str,
    problem: str,
    context: str,
    specialist_endpoint: str,
    specialist_model: str,
) -> ToolResult:
    """Execute the answer tool: call the specialist for the final answer."""
    start = time.monotonic()

    prompt = f"{context}\n\nProblem:\n{problem}"
    messages = [
        {
            "role": "system",
            "content": "Please reason step by step and give a clear, concise final answer.",
        },
        {"role": "user", "content": prompt},
    ]

    resp = await chat_completion(
        endpoint=specialist_endpoint,
        model=specialist_model,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
    )
    answer_text = resp.content or ""
    latency = (time.monotonic() - start) * 1000

    return ToolResult(
        tool_name="answer",
        model_selected=model,
        latency_ms=latency,
        content=answer_text,
        is_final_answer=True,
    )


TOOL_HANDLERS = {
    "search": handle_search,
    "enhance_reasoning": handle_enhance_reasoning,
    "answer": handle_answer,
}


async def _duckduckgo_search(query: str, max_results: int = 3) -> list[str]:
    """Search the web using DuckDuckGo and return result snippets."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                snippet = f"Title: {r.get('title', '')}\n{r.get('body', '')}\nURL: {r.get('href', '')}"
                results.append(snippet)
        if results:
            logger.info("DDG returned %d results for: %s", len(results), query)
            return results
    except Exception:
        logger.warning("DuckDuckGo search failed for: %s, using specialist fallback", query, exc_info=True)

    return [f"No web search results available for: {query}"]
