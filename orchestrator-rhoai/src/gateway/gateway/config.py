import json
import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class GatewayConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_")

    orchestrator_endpoint: str = "http://orchestrator-8b-predictor.orchestrator-rhoai.svc.cluster.local:8080/v1"
    orchestrator_model: str = "orchestrator-8b"
    specialist_endpoint: str = "http://llama-32-3b-instruct-predictor.my-first-model.svc.cluster.local:8080/v1"
    model_mapping_path: str = "/config/mapping.json"
    tool_definitions_path: str = "/config/tools.json"
    max_turns: int = 10
    request_timeout: float = 180.0
    log_level: str = "INFO"


def load_model_mapping(path: str) -> dict:
    """Load model mapping from JSON file."""
    p = Path(path)
    if not p.exists():
        logger.warning("Model mapping file not found: %s, using defaults", path)
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load model mapping from %s: %s", path, e)
        return {}


def load_tool_definitions(path: str) -> list[dict]:
    """Load tool definitions from JSON file."""
    p = Path(path)
    if not p.exists():
        logger.warning("Tool definitions not found: %s, using defaults", path)
        return _default_tool_definitions()
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load tool definitions from %s: %s", path, e)
        return _default_tool_definitions()


def _default_tool_definitions() -> list[dict]:
    """Fallback tool definitions matching ToolOrchestra's format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "enhance_reasoning",
                "description": "tool to enhance answer model reasoning. analyze the problem, write code, execute it and return intermediate results that will help solve the problem",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": 'Choices: ["reasoner-1", "reasoner-2", "reasoner-3"]. reasoner-1 demonstrates strong understanding and reasoning. reasoner-2 can analyze some problems. reasoner-3 can reason over context.',
                        }
                    },
                    "required": ["model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "answer",
                "description": "give the final answer. Not allowed to call if documents is empty.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": 'Choices: ["answer-1", "answer-2", "answer-3", "answer-4", "answer-math-1", "answer-math-2"]. answer-1 is excellent in most domains. answer-math-1 solves moderate math. answer-math-2 handles basic math.',
                        }
                    },
                    "required": ["model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for missing information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": 'Choices: ["search-1", "search-2", "search-3"]. search-1 identifies missing info and writes concise queries. search-2 can reason and find missing content. search-3 writes queries to find info.',
                        }
                    },
                    "required": ["model"],
                },
            },
        },
    ]
