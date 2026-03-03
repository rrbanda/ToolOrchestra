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
async def test_retrieve_endpoint(client):
    response = await client.post(
        "/v1/retrieve",
        json={"query": "test query", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
