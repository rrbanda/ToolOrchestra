import pytest
from httpx import ASGITransport, AsyncClient

from gateway.app import app
from gateway.config import GatewayConfig


@pytest.fixture
def sample_config() -> GatewayConfig:
    return GatewayConfig()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
