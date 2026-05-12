"""
Embedding module for LegacyRAG.

Uses nomic-embed-text via the Ollama REST API. Respects VRAM scheduler decisions:
  - GPU device → pass num_gpu=-1 (Ollama uses all available GPU layers)
  - CPU device → pass num_gpu=0 (forces CPU inference)

nomic-embed-text produces 768-dimensional float32 vectors.
"""

import logging
import time
from typing import Any

import httpx
import numpy as np

from legacyrag.vram_scheduler import decide_device

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
_HTTP_TIMEOUT = 120.0


def _build_options(device: str) -> dict[str, Any]:
    """Map scheduler device decision to Ollama model options."""
    return {"num_gpu": -1 if device == "gpu" else 0}


async def embed_texts_async(
    texts: list[str],
    generation_active: bool = False,
) -> tuple[np.ndarray, str]:
    """
    Embed a list of texts asynchronously.

    Returns:
        embeddings: float32 array of shape (len(texts), EMBED_DIM)
        device: 'gpu' or 'cpu' — the device actually used
    """
    device = decide_device(generation_active=generation_active)
    options = _build_options(device)
    embeddings: list[list[float]] = []

    async with httpx.AsyncClient(base_url=OLLAMA_BASE, timeout=_HTTP_TIMEOUT) as client:
        for text in texts:
            t0 = time.perf_counter()
            response = await client.post(
                "/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text, "options": options},
            )
            response.raise_for_status()
            data = response.json()
            elapsed = time.perf_counter() - t0
            logger.debug("embed [%s] %.3fs | len=%d", device, elapsed, len(text))
            embeddings.append(data["embedding"])

    arr = np.array(embeddings, dtype=np.float32)
    return arr, device


def embed_texts(
    texts: list[str],
    generation_active: bool = False,
) -> tuple[np.ndarray, str]:
    """Synchronous wrapper around embed_texts_async."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Called from within an async context (e.g. FastAPI handler via run_in_executor)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, embed_texts_async(texts, generation_active))
            return future.result()
    else:
        return asyncio.run(embed_texts_async(texts, generation_active))
