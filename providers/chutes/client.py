"""Chutes AI provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

CHUTES_BASE_URL = "https://llm.chutes.ai/v1"


class ChutesProvider(OpenAICompatibleProvider):
    """Chutes AI provider using OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="CHUTES",
            base_url=config.base_url or CHUTES_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)
