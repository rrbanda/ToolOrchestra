import pytest
from pydantic import ValidationError

from gateway.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
    ToolCallRecord,
)


@pytest.mark.unit
def test_orchestration_request_valid():
    req = OrchestrationRequest(question="What is 2+2?")
    assert req.question == "What is 2+2?"
    assert req.context is None
    assert req.max_turns is None

    req_with_optional = OrchestrationRequest(
        question="Hello",
        context="Some context",
        max_turns=5,
    )
    assert req_with_optional.context == "Some context"
    assert req_with_optional.max_turns == 5


@pytest.mark.unit
def test_orchestration_request_missing_question():
    with pytest.raises(ValidationError):
        OrchestrationRequest()


@pytest.mark.unit
def test_orchestration_response_serialization():
    resp = OrchestrationResponse(
        answer="42",
        tool_calls=[
            ToolCallRecord(
                tool_name="calculator",
                model_selected="gpt-4",
                latency_ms=100.5,
                token_count=10,
            ),
        ],
        total_turns=1,
        total_latency_ms=100.5,
    )
    serialized = resp.model_dump()
    deserialized = OrchestrationResponse.model_validate(serialized)
    assert deserialized.answer == resp.answer
    assert len(deserialized.tool_calls) == len(resp.tool_calls)
    assert deserialized.tool_calls[0].tool_name == "calculator"
    assert deserialized.total_turns == resp.total_turns
