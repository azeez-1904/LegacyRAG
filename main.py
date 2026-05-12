"""
LegacyRAG — FastAPI entry point.

Endpoints:
  POST /ingest   — ingest a document (plain text or JSON body)
  POST /query    — query the RAG pipeline
  GET  /health   — liveness check + GPU stats
  GET  /benchmark/summary — aggregate benchmark statistics
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from legacyrag.benchmark import BenchmarkLogger
from legacyrag.pipeline import RAGPipeline
from legacyrag.vram_scheduler import query_vram

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("legacyrag.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
pipeline: RAGPipeline | None = None
benchmarker: BenchmarkLogger | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, benchmarker
    logger.info("LegacyRAG starting up")
    pipeline = RAGPipeline(
        chunk_size=512,
        chunk_overlap=64,
        top_k=5,
        max_tokens=512,
        temperature=0.2,
        auto_persist=True,
    )
    benchmarker = pipeline.benchmarker
    logger.info("Pipeline ready — vector store has %d chunks", pipeline.store.size)
    yield
    logger.info("LegacyRAG shutting down")


app = FastAPI(
    title="LegacyRAG",
    description="VRAM-aware RAG pipeline for legacy GPU inference research",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    text: str = Field(..., description="Raw document text to ingest", min_length=1)
    doc_id: str = Field(default="doc", description="Unique document identifier")


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    embed_device: str
    embed_latency_s: float
    total_latency_s: float
    store_size: int
    vram_before_mb: list
    vram_after_mb: list


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    max_tokens: int = Field(default=512, ge=16, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class QueryResponse(BaseModel):
    answer: str
    sources: list
    benchmark: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    gpus = query_vram()
    return {
        "status": "ok",
        "store_chunks": pipeline.store.size if pipeline else 0,
        "gpus": [
            {
                "index": g.index,
                "name": g.name,
                "free_mb": g.free_mb,
                "used_mb": g.used_mb,
                "total_mb": g.total_mb,
            }
            for g in gpus
        ],
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    try:
        result = await pipeline.ingest(request.text, doc_id=request.doc_id)
    except Exception as exc:
        logger.exception("Ingest failed for doc_id=%s", request.doc_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(content=result)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    if pipeline.store.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Vector store is empty — ingest documents first via POST /ingest",
        )
    try:
        result = await pipeline.query(
            question=request.question,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as exc:
        logger.exception("Query failed: %s", request.question)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        benchmark=result.benchmark,
    )


@app.get("/benchmark/summary")
async def benchmark_summary():
    if benchmarker is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return benchmarker.summary()


@app.post("/stress-test")
async def run_stress_test(n_concurrent: int = 5):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    if pipeline.store.size == 0:
        raise HTTPException(status_code=400, detail="Ingest documents first")
    from legacyrag.benchmark import stress_test
    try:
        result = await stress_test(pipeline, n_concurrent=n_concurrent)
    except Exception as exc:
        logger.exception("Stress test failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.delete("/store")
async def clear_store():
    """Clear the in-memory vector store and delete persisted files (dev/test use)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    import os

    pipeline.store.clear()
    for f in ("vector_store.npz", "vector_store_meta.json"):
        if os.path.exists(f):
            os.remove(f)
    return {"status": "cleared", "store_size": 0}
