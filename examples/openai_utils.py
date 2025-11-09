from __future__ import annotations

from typing import Any, List, Optional

from openai import AsyncOpenAI

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


def _extract_text_from_content(block: Any) -> Optional[str]:
    if block is None:
        return None
    text = getattr(block, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    if isinstance(block, dict):
        value = block.get("text")
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(block, str) and block.strip():
        return block
    return None


def _collect_response_text(response: Any) -> List[str]:
    chunks: List[str] = []
    output = getattr(response, "output", None)
    if output:
        for item in output:
            contents = getattr(item, "content", None)
            if contents:
                for content in contents:
                    text = _extract_text_from_content(content)
                    if text:
                        chunks.append(text)
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        chunks.append(output_text)
    choices = getattr(response, "choices", None)
    if choices:
        for choice in choices:
            message = getattr(choice, "message", None)
            content_list = getattr(message, "content", None) if message else None
            if content_list:
                for content in content_list:
                    if isinstance(content, dict) and content.get("type") == "text":
                        text = content.get("text")
                        if isinstance(text, str) and text.strip():
                            chunks.append(text)
    return chunks


def _normalize_text(chunks: List[str]) -> str:
    combined = "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    return combined.strip()


async def generate_summary_text(model: str, prompt: str) -> str:
    """Call the OpenAI Responses API and return the textual output."""
    client = _get_client()
    response = await client.responses.create(model=model, input=prompt)
    chunks = _collect_response_text(response)
    text = _normalize_text(chunks)
    return text or str(response)
