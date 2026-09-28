from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMProfile:
    model_name: str
    provider: str = "ollama"
    temperature: float = 0.0
    num_ctx: int = 4096
    num_predict: Optional[int] = None
    keep_alive: Optional[str] = None
    reasoning: bool | str = True
    sdk_max_retries: Optional[int] = None
    openrouter_provider: Optional[Dict[str, Any]] = None


# model TA in future: qwen3:8b, it support tool calling natively
@dataclass
class LLMConfig:
    profiles: Dict[str, LLMProfile] = field(default_factory=lambda: {
        "graph": LLMProfile(
            model_name="gpt-oss:120b-cloud",
            temperature=0.2, 
            num_ctx=8192,
            num_predict=8192,
            reasoning="low",
        ),
        "ta": LLMProfile(
            model_name="nvidia/nemotron-3-super-120b-a12b:free",
            provider="openrouter",
            temperature=0.67,
            num_ctx=4096,
            num_predict=4096,
            keep_alive="1h",
            openrouter_provider={"allow_fallbacks": True, "require_parameters": True},
        ),
        "evaluator": LLMProfile(
            model_name="qwen3:8b", 
            temperature=0.3, 
            num_ctx=2048,
        ),
        "generator": LLMProfile(
            model_name="nvidia/nemotron-3-super-120b-a12b:free",
            provider="openrouter",
            temperature=0.4, 
            num_ctx=4096,
            openrouter_provider={"allow_fallbacks": True, "require_parameters": True},
        ),
        "rag": LLMProfile(
            model_name="qwen3:8b",
            temperature=0.0,
            num_ctx=8192,
            num_predict=512,
            keep_alive="30m",
        ),
        "retrieval_aggregator": LLMProfile(
            model_name="nvidia/nemotron-3-super-120b-a12b:free",
            provider="openrouter",
            temperature=0.0,
            num_ctx=65536,
            num_predict=1024,
            openrouter_provider={"allow_fallbacks": True, "require_parameters": True},
        ),
        "retrieval_planner": LLMProfile(
            model_name="nvidia/nemotron-3-super-120b-a12b:free",
            provider="openrouter",
            temperature=0.0,
            num_ctx=65536,
            num_predict=3072,  ## CoT enabled, needs headroom past the hidden reasoning budget
            openrouter_provider={"allow_fallbacks": True, "require_parameters": True},
        ),
        "retrieval_answerer": LLMProfile(
            model_name="nvidia/nemotron-3-super-120b-a12b:free",
            provider="openrouter",
            temperature=0.0,
            num_ctx=65536,
            num_predict=256,
            openrouter_provider={"allow_fallbacks": True, "require_parameters": True},
        ),
        "worker": LLMProfile(
            model_name="gpt-oss:120b-cloud",
            temperature=0.0,
            num_ctx=768,
            num_predict=2048,
            keep_alive="5m",
            reasoning="low",
        )
    })

    default_profile: str = "graph"

config_instance = LLMConfig()
