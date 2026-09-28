from typing import Any, Dict
import os
from langchain_ollama import ChatOllama
from langchain_core.exceptions import OutputParserException
from core.llm.config import LLMConfig


class CoreLLMEngine:
    def __init__(self, config: LLMConfig = LLMConfig()):
        self.config = config
        self._instances: Dict[str, Any] = {}

    def _get_llm(self, profile_name: str) -> Any:
        profile_name = profile_name.lower()
        if profile_name not in self._instances:
            if profile_name not in self.config.profiles:
                raise ValueError(f"Profile '{profile_name}' not found in configuration.")
            
            profile = self.config.profiles[profile_name]

            if profile.provider in {"google_genai", "openrouter"} and profile.sdk_max_retries not in (None, 0):
                raise ValueError("external model profiles require sdk_max_retries=0")

            if profile.provider == "ollama":
                kwargs: Dict[str, Any] = {
                    "model": profile.model_name,
                    "temperature": profile.temperature,
                    "num_ctx": profile.num_ctx,
                    "reasoning": profile.reasoning,
                }
                if profile.num_predict is not None:
                    kwargs["num_predict"] = profile.num_predict
                if profile.keep_alive is not None:
                    kwargs["keep_alive"] = profile.keep_alive
                self._instances[profile_name] = ChatOllama(**kwargs)
            elif profile.provider == "google_genai":
                if not os.getenv("GOOGLE_API_KEY"):
                    raise RuntimeError("GOOGLE_API_KEY is required for Google Gemini profiles")
                from langchain_google_genai import ChatGoogleGenerativeAI

                kwargs = {
                    "model": profile.model_name,
                    "temperature": profile.temperature,
                    "max_retries": profile.sdk_max_retries if profile.sdk_max_retries is not None else 0,
                }
                if profile.num_predict is not None:
                    kwargs["max_tokens"] = profile.num_predict
                self._instances[profile_name] = ChatGoogleGenerativeAI(**kwargs)
            elif profile.provider == "openrouter":
                if not os.getenv("OPENROUTER_API_KEY"):
                    raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter profiles")
                provider_options = profile.openrouter_provider or {}
                if (
                    not isinstance(provider_options.get("allow_fallbacks"), bool)
                    or provider_options.get("require_parameters") is not True
                ):
                    raise ValueError(
                        "OpenRouter profiles require boolean allow_fallbacks and require_parameters=true"
                    )
                from langchain_openrouter import ChatOpenRouter

                kwargs = {
                    "model": profile.model_name,
                    "temperature": profile.temperature,
                    "max_retries": profile.sdk_max_retries if profile.sdk_max_retries is not None else 0,
                    "openrouter_provider": provider_options,
                }
                if profile.num_predict is not None:
                    kwargs["max_tokens"] = profile.num_predict
                self._instances[profile_name] = ChatOpenRouter(**kwargs)
            else:
                raise ValueError(f"Unsupported LLM provider '{profile.provider}'.")
        return self._instances[profile_name]
    
    def invoke_with_retry(self, prompt_template, parser, input_data: dict, profile_name: str = None, max_retries: int = None):
        target_profile = (profile_name or self.config.default_profile).lower()
        llm = self._get_llm(target_profile)
        
        profile_config = self.config.profiles[target_profile]
        retries = max_retries if max_retries is not None else getattr(profile_config, 'max_retries', 3)
        
        chain = prompt_template | llm | parser
        
        for attempt in range(retries):
            try:
                return chain.invoke(input_data)
            except OutputParserException:
                if attempt == retries - 1:
                    return None
            except Exception as e:
                print(f"LLM Error: {str(e)}")
                if attempt == retries - 1:
                    return None
        return None

    async def ainvoke_with_retry(self, prompt_template, parser, input_data: dict, profile_name: str = None, max_retries: int = None):
        target_profile = (profile_name or self.config.default_profile).lower()
        llm = self._get_llm(target_profile)
        
        profile_config = self.config.profiles[target_profile]
        retries = max_retries if max_retries is not None else getattr(profile_config, 'max_retries', 3)
        
        chain = prompt_template | llm | parser
        
        for attempt in range(retries):
            try:
                return await chain.ainvoke(input_data)
            except OutputParserException:
                if attempt == retries - 1:
                    return None
            except Exception as e:
                print(f"LLM Error: {str(e)}")
                if attempt == retries - 1:
                    return None
        return None
