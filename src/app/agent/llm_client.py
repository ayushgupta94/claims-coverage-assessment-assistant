"""LLM client for the agent's tool-calling loop.

Two real, production-grade providers -- OpenAI directly, or Azure OpenAI
(selected via LLM_PROVIDER) -- both using real function-calling. No stub,
no offline mode: the same code path runs locally and in production.

This is intentionally the only non-deterministic moving part in the whole
system -- everything downstream of a tool call (rule checks, fraud
scoring, persistence) is plain, deterministic Python.
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from app.config import Settings
from app.core.exceptions import LLMProviderError


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # tool name, set on role="tool" messages


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = []


async def _call_chat_completions(client, model: str, messages: list[ChatMessage], tools: list[dict]) -> LLMResponse:
    """Shared request/response handling for both OpenAI and Azure OpenAI --
    both expose the same chat.completions.create() interface via the
    openai SDK, so only client construction differs between them."""
    payload = []
    for m in messages:
        entry: dict = {"role": m.role}
        if m.content is not None:
            entry["content"] = m.content
        if m.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        payload.append(entry)

    try:
        response = await client.chat.completions.create(model=model, messages=payload, tools=tools or None)
    except Exception as exc:  # pragma: no cover - network failure path
        raise LLMProviderError(f"LLM provider call failed: {exc}") from exc

    choice = response.choices[0].message
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
        for tc in (choice.tool_calls or [])
    ]
    return LLMResponse(content=choice.content, tool_calls=tool_calls)


class OpenAIChatLLMClient:
    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI

        if not settings.openai_api_key:
            raise LLMProviderError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    async def generate(self, messages: list[ChatMessage], tools: list[dict]) -> LLMResponse:
        return await _call_chat_completions(self._client, self._model, messages, tools)


class AzureOpenAIChatLLMClient:
    def __init__(self, settings: Settings) -> None:
        from openai import AsyncAzureOpenAI

        if not (settings.azure_openai_endpoint and settings.azure_openai_api_key and settings.azure_openai_deployment):
            raise LLMProviderError(
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT are "
                "required for LLM_PROVIDER=azure_openai"
            )
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self._model = settings.azure_openai_deployment

    async def generate(self, messages: list[ChatMessage], tools: list[dict]) -> LLMResponse:
        return await _call_chat_completions(self._client, self._model, messages, tools)


def get_llm_client(settings: Settings):
    if settings.llm_provider == "azure_openai":
        return AzureOpenAIChatLLMClient(settings)
    return OpenAIChatLLMClient(settings)
