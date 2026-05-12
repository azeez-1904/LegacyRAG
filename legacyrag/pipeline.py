"""
Orchestration pipeline for LegacyRAG.

Ties together VRAM-aware embedding, vector retrieval, and generation.
Exposes two coroutines:
  - ingest(text, doc_id)  — chunk → embed → store
  - query(question, ...)  — embed query → retrieve → generate → benchmark
"""

import logging
import time
from dataclasses import dataclass, field

from legacyrag.benchmark import BenchmarkLogger
from legacyrag.embedder import embed_texts_async
from legacyrag.generator import generate_async
from legacyrag.retriever import VectorStore, chunk_text
from legacyrag.vram_scheduler import query_vram

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    benchmark: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        max_tokens: int = 512,
        temperature: float = 0.2,
        auto_persist: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.auto_persist = auto_persist

        self.store = VectorStore()
        self.benchmarker = BenchmarkLogger()

        # Try to restore a previous session
        if self.store.load():
            logger.info("Restored vector store with %d chunks", self.store.size)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    async def ingest(self, text: str, doc_id: str = "doc") -> dict:
        """
        Chunk text, embed chunks (VRAM-aware), and add to the vector store.

        Returns a summary dict suitable for an HTTP response.
        """
        t_start = time.perf_counter()

        chunks = chunk_text(
            text,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            doc_id=doc_id,
        )
        if not chunks:
            return {"doc_id": doc_id, "chunks": 0, "status": "no text extracted"}

        texts = [c.text for c in chunks]

        t_embed_start = time.perf_counter()
        vram_before = _snapshot_vram()
        embeddings, embed_device = await embed_texts_async(texts, generation_active=False)
        embed_latency = time.perf_counter() - t_embed_start
        vram_after = _snapshot_vram()

        self.store.add(chunks, embeddings)

        if self.auto_persist:
            self.store.save()

        total_latency = time.perf_counter() - t_start
        logger.info(
            "ingest: doc_id=%s chunks=%d embed_device=%s embed_latency=%.3fs",
            doc_id,
            len(chunks),
            embed_device,
            embed_latency,
        )

        return {
            "doc_id": doc_id,
            "chunks": len(chunks),
            "embed_device": embed_device,
            "embed_latency_s": round(embed_latency, 4),
            "total_latency_s": round(total_latency, 4),
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "store_size": self.store.size,
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(
        self,
        question: str,
        top_k: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> QueryResult:
        """
        Full RAG query: embed → retrieve → generate.

        VRAM scheduler is consulted for the query embedding; the GenerationContext
        inside generator.py automatically blocks GPU embedding during generation.
        """
        top_k = top_k or self.top_k
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        t_start = time.perf_counter()

        # 1. Embed the query (VRAM check happens inside embed_texts_async)
        t_embed = time.perf_counter()
        vram_before = _snapshot_vram()
        q_emb, embed_device = await embed_texts_async([question], generation_active=False)
        embed_latency = time.perf_counter() - t_embed

        # 2. Retrieve top-k chunks
        t_retrieve = time.perf_counter()
        results = self.store.query(q_emb[0], top_k=top_k)
        retrieve_latency = time.perf_counter() - t_retrieve

        context_chunks = [chunk.text for chunk, _ in results]
        sources = [
            {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "score": round(score, 4),
                "text_preview": chunk.text[:120],
            }
            for chunk, score in results
        ]

        # 3. Generate answer (GenerationContext blocks GPU embedding internally)
        t_gen = time.perf_counter()
        gen_result = await generate_async(
            context_chunks,
            question,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        gen_latency = time.perf_counter() - t_gen
        vram_after = _snapshot_vram()

        total_latency = time.perf_counter() - t_start

        benchmark_entry = self.benchmarker.record(
            question=question,
            embed_latency_s=embed_latency,
            retrieve_latency_s=retrieve_latency,
            gen_latency_s=gen_latency,
            total_latency_s=total_latency,
            tokens_per_sec=gen_result["tokens_per_sec"],
            prompt_tokens=gen_result["prompt_tokens"],
            completion_tokens=gen_result["completion_tokens"],
            embed_device=embed_device,
            vram_before_mb=vram_before,
            vram_after_mb=vram_after,
            top_k=top_k,
            chunks_retrieved=len(results),
        )

        return QueryResult(
            answer=gen_result["text"],
            sources=sources,
            benchmark=benchmark_entry,
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _snapshot_vram() -> list[dict]:
    """Return a compact VRAM snapshot for benchmark logging."""
    return [
        {"gpu": g.index, "free_mb": g.free_mb, "used_mb": g.used_mb}
        for g in query_vram()
    ]
