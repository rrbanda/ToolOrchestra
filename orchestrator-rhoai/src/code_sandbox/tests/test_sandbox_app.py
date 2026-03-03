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
async def test_execute_endpoint(client):
    response = await client.post(
        "/v1/execute",
        json={"code": "print(1+1)", "timeout": 60},
    )
    assert response.status_code == 200
    data = response.json()
    assert "stdout" in data
    assert "stderr" in data
    assert "exit_code" in data
