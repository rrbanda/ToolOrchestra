import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ready_endpoint(client):
    response = await client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_orchestrate_placeholder(client):
    response = await client.post(
        "/v1/orchestrate",
        json={"question": "What is 2+2?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "tool_calls" in data
    assert "total_turns" in data
    assert "total_latency_ms" in data
    assert isinstance(data["tool_calls"], list)
