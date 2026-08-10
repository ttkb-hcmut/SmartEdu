from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LLMProfile:
    model_name: str
    temperature: float = 0.0
    num_ctx: int = 4096
    num_predict: Optional[int] = None
    keep_alive: Optional[str] = None


# model TA in future: qwen3:8b, it support tool calling natively
@dataclass
class LLMConfig:
    profiles: Dict[str, LLMProfile] = field(default_factory=lambda: {
        "graph": LLMProfile(
            model_name="gpt-oss:120b-cloud", 
            temperature=0.2, 
            num_ctx=8192,
        ),
        "ta": LLMProfile(
            model_name="gpt-oss:120b-cloud", 
            temperature=0.67, 
            num_ctx=4096,
            keep_alive="1h",
        ),
        "evaluator": LLMProfile(
            model_name="qwen3:8b", 
            temperature=0.3, 
            num_ctx=2048,
        ),
        "generator": LLMProfile(
            model_name="gpt-oss:120b-cloud", 
            temperature=0.4, 
            num_ctx=4096,
        ),
        "rag": LLMProfile(
            model_name="qwen3:8b",
            temperature=0.0,
            num_ctx=8192,
            num_predict=512,
            keep_alive="30m",
        ),
        "retrieval_aggregator": LLMProfile(
            model_name="gpt-oss:120b-cloud",
            temperature=0.0,
            num_ctx=65536,
            num_predict=1024,
        ),
        "retrieval_answerer": LLMProfile(
            model_name="gpt-oss:120b-cloud",
            temperature=0.0,
            num_ctx=65536,
            num_predict=256,
        ),
        "worker": LLMProfile(
            model_name="gemma3:1b", 
            temperature=0.0,
            num_ctx=768,
            num_predict=256,
            keep_alive="5m",
        )
    })

    default_profile: str = "graph"

config_instance = LLMConfig()
