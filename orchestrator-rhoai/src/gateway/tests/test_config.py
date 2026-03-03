import json

import pytest

from gateway.config import GatewayConfig, load_model_mapping


@pytest.mark.unit
def test_default_config():
    config = GatewayConfig()
    assert config.orchestrator_endpoint == "http://orchestrator-predictor:8080/v1"
    assert config.model_mapping_path == "/config/mapping.json"
    assert config.max_turns == 30
    assert config.request_timeout == 120.0
    assert config.log_level == "INFO"


@pytest.mark.unit
def test_load_model_mapping_missing_file():
    result = load_model_mapping("/nonexistent/path/mapping.json")
    assert result == {}


@pytest.mark.unit
def test_load_model_mapping_valid_file(tmp_path):
    mapping = {"tool_a": "model-1", "tool_b": "model-2"}
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping))
    result = load_model_mapping(str(path))
    assert result == mapping
