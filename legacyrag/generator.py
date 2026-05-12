"""
Generator module for LegacyRAG.

Calls the llama.cpp server's OpenAI-compatible /v1/chat/completions endpoint
on port 8080 (phi3-mini via Vulkan backend, dual K4200 split). Uses streaming
internally to detect generation stalls: if no token arrives within
STALL_TIMEOUT_S seconds, the request is cancelled and retried with a reduced
context (half the chunks on first retry, one chunk on second).
"""

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from legacyrag.vram_scheduler import GenerationContext

logger = logging.getLogger(__name__)

LLAMA_SERVER_BASE = "http://localhost:8080"
_CONNECT_TIMEOUT = 10.0
_POOL_TIMEOUT = 10.0
# Per-token read deadline — stall declared if no token for this long
# Time to wait for the FIRST token — must cover queue wait on a 1-slot server.
# With 5 concurrent requests each taking ~120s, last-in-queue waits ~480s.
FIRST_TOKEN_TIMEOUT_S = 600.0
# Inter-token stall: if generation has STARTED but goes silent for this long → stall
STALL_TIMEOUT_S = 45.0
_MAX_ATTEMPT_S = 1200.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
MAX_RETRIES = 2


class GenerationStallError(Exception):
    """Raised when the llama.cpp server stops producing tokens."""


def _build_messages(context_chunks: list[str], question: str) -> list[dict[str, str]]:
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful research assistant. Answer questions using only the provided context. "
        "If the context does not contain enough information, say so clearly. "
        "Be concise and cite relevant passages when possible."
    )
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


async def _stream_with_stall_detection(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
) -> tuple[str, int, int]:
    """
    Stream a completion with two-phase stall detection:
      Phase 1 (queue wait): wait up to FIRST_TOKEN_TIMEOUT_S for the first token.
                            Covers queuing behind other requests on the 1-slot server.
      Phase 2 (generation): once tokens are flowing, raise GenerationStallError
                            if any inter-token gap exceeds STALL_TIMEOUT_S.

    Returns (text, prompt_tokens, completion_tokens).
    """
    tokens: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    first_token_received = False

    async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()

        async def _next_line():
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    return line
            return None

        while True:
            timeout = STALL_TIMEOUT_S if first_token_received else FIRST_TOKEN_TIMEOUT_S
            try:
                line = await asyncio.wait_for(_next_line(), timeout=timeout)
            except asyncio.TimeoutError:
                if not first_token_received:
                    raise GenerationStallError(
                        f"No first token within {FIRST_TOKEN_TIMEOUT_S}s — server unresponsive or overloaded"
                    )
                raise GenerationStallError(
                    f"Inter-token gap exceeded {STALL_TIMEOUT_S}s — generation stalled mid-sequence"
                )

            if line is None:
                break

            payload_str = line[6:].strip()
            if payload_str == "[DONE]":
                break

            try:
                chunk = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            if delta:
                tokens.append(delta)
                first_token_received = True

            usage = chunk.get("usage") or {}
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", len(tokens))

    if not completion_tokens:
        completion_tokens = len(tokens)

    return "".join(tokens), prompt_tokens, completion_tokens


async def generate_async(
    context_chunks: list[str],
    question: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a response from llama.cpp server with stall detection and retry.

    On stall: retries with progressively reduced context:
      attempt 1: all chunks
      attempt 2: top half of chunks
      attempt 3: top 1 chunk only
    """
    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT,
        read=_MAX_ATTEMPT_S,
        write=30.0,
        pool=_POOL_TIMEOUT,
    )

    last_exc: Exception | None = None
    chunks_for_attempt = context_chunks

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            # Reduce context on retry
            n = max(1, len(context_chunks) // (2 ** attempt))
            chunks_for_attempt = context_chunks[:n]
            logger.warning(
                "Retry %d/%d after stall — reducing context to %d chunk(s)",
                attempt, MAX_RETRIES, n,
            )

        messages = _build_messages(chunks_for_attempt, question)
        payload: dict[str, Any] = {
            "model": "phi3-mini",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if extra_params:
            payload.update({k: v for k, v in extra_params.items() if k != "stream"})

        t0 = time.perf_counter()
        try:
            with GenerationContext():
                async with httpx.AsyncClient(
                    base_url=LLAMA_SERVER_BASE, timeout=timeout
                ) as client:
                    text, prompt_tokens, completion_tokens = await _stream_with_stall_detection(
                        client, payload
                    )

            latency_s = time.perf_counter() - t0
            tokens_per_sec = completion_tokens / latency_s if latency_s > 0 else 0.0

            if attempt > 0:
                logger.info(
                    "generate (retry %d): %.2fs | %d tokens | %.2f tok/s",
                    attempt, latency_s, completion_tokens, tokens_per_sec,
                )
            else:
                logger.info(
                    "generate: %.2fs | %d tokens | %.1f tok/s",
                    latency_s, completion_tokens, tokens_per_sec,
                )

            return {
                "text": text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency_s": latency_s,
                "tokens_per_sec": tokens_per_sec,
                "finish_reason": "stop",
                "retries": attempt,
                "context_chunks_used": len(chunks_for_attempt),
            }

        except GenerationStallError as exc:
            last_exc = exc
            logger.error("Stall on attempt %d: %s", attempt + 1, exc)
            if attempt == MAX_RETRIES:
                break

    raise RuntimeError(
        f"Generation failed after {MAX_RETRIES + 1} attempts: {last_exc}"
    )
