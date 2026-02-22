"""Provider implementations for Aegis.

This package contains provider implementations for various LLM backends:
- CloudProviderBase: Abstract base for cloud API providers
- OpenAIProvider: OpenAI API (GPT-4, GPT-3.5)
- AnthropicProvider: Anthropic API (Claude)
- GoogleProvider: Google Generative AI (Gemini)
- HFLocalProvider: HuggingFace local models
- ToolProvider: Classic ML tool adapters

Usage:
    from aegis.providers import OpenAIProvider, PROVIDER_REGISTRY

    # Direct instantiation
    provider = OpenAIProvider(model_name="gpt-4")

    # Or via registry
    provider_cls = PROVIDER_REGISTRY.get("openai")
    provider = provider_cls(model_name="gpt-4", api_key="...")
"""

from typing import Dict, Type, Any

from aegis.providers.base import CloudProviderBase
from aegis.providers.openai_provider import OpenAIProvider
from aegis.providers.anthropic_provider import AnthropicProvider
from aegis.providers.google_provider import GoogleProvider
from aegis.providers.hf_local import HFLocalProvider, create_hf_provider
from aegis.providers.tool_provider import ToolProvider
from aegis.providers.claude_code_security import ClaudeCodeSecurityProvider

# Provider registry for plugin-style discovery
# Maps provider type names to their implementation classes
PROVIDER_REGISTRY: Dict[str, Type[Any]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "hf_local": HFLocalProvider,
    "tool": ToolProvider,
    "claude_code_security": ClaudeCodeSecurityProvider,
}


def register_provider(name: str, provider_cls: Type[Any]) -> None:
    """
    Register a custom provider implementation.

    Args:
        name: Provider type name (e.g., 'custom_llm')
        provider_cls: Provider class to register

    Example:
        from aegis.providers import register_provider

        class MyCustomProvider:
            def __init__(self, model_name: str, **kwargs):
                ...
            async def generate(self, prompt: str, **kwargs) -> str:
                ...

        register_provider("my_custom", MyCustomProvider)
    """
    PROVIDER_REGISTRY[name] = provider_cls


def get_provider(name: str) -> Type[Any]:
    """
    Get a provider class by name.

    Args:
        name: Provider type name

    Returns:
        Provider class

    Raises:
        KeyError: If provider not found
    """
    if name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise KeyError(f"Provider '{name}' not found. Available: {available}")
    return PROVIDER_REGISTRY[name]


__all__ = [
    # Base class
    "CloudProviderBase",
    # Cloud providers
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    # Local providers
    "HFLocalProvider",
    "create_hf_provider",
    # Tool adapter
    "ToolProvider",
    # Agentic
    "ClaudeCodeSecurityProvider",
    # Registry
    "PROVIDER_REGISTRY",
    "register_provider",
    "get_provider",
]
