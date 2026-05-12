"""
Benchmark logger for LegacyRAG.

Appends one JSON record per request to benchmark_results.json so the
research paper can report:
  - Per-request latency breakdown (embed / retrieve / generate / total)
  - Tokens per second
  - VRAM free/used before and after each request
  - Whether embeddings ran on GPU or CPU
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from legacyrag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

BENCHMARK_FILE = "benchmark_results.json"
STRESS_TEST_FILE = "stress_test_results.json"


class BenchmarkLogger:
    def __init__(self, output_file: str = BENCHMARK_FILE) -> None:
        self.output_file = output_file
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not os.path.exists(self.output_file):
            with open(self.output_file, "w") as fh:
                json.dump([], fh)

    def record(
        self,
        *,
        question: str,
        embed_latency_s: float,
        retrieve_latency_s: float,
        gen_latency_s: float,
        total_latency_s: float,
        tokens_per_sec: float,
        prompt_tokens: int,
        completion_tokens: int,
        embed_device: str,
        vram_before_mb: list[dict],
        vram_after_mb: list[dict],
        top_k: int,
        chunks_retrieved: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_preview": question[:80],
            "latency": {
                "embed_s": round(embed_latency_s, 4),
                "retrieve_s": round(retrieve_latency_s, 4),
                "generate_s": round(gen_latency_s, 4),
                "total_s": round(total_latency_s, 4),
            },
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "per_sec": round(tokens_per_sec, 2),
            },
            "embed_device": embed_device,
            "retrieval": {
                "top_k": top_k,
                "chunks_retrieved": chunks_retrieved,
            },
            "vram_before": vram_before_mb,
            "vram_after": vram_after_mb,
        }
        if extra:
            entry["extra"] = extra

        self._append(entry)
        logger.debug(
            "benchmark recorded: total=%.3fs toks/s=%.1f device=%s",
            total_latency_s, tokens_per_sec, embed_device,
        )
        return entry

    def _append(self, entry: dict[str, Any]) -> None:
        try:
            with open(self.output_file, "r") as fh:
                records: list = json.load(fh)
        except (json.JSONDecodeError, OSError):
            records = []
        records.append(entry)
        with open(self.output_file, "w") as fh:
            json.dump(records, fh, indent=2)

    def summary(self) -> dict[str, Any]:
        try:
            with open(self.output_file) as fh:
                records: list[dict] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}

        if not records:
            return {"total_requests": 0}

        total_s = [r["latency"]["total_s"] for r in records]
        tps = [r["tokens"]["per_sec"] for r in records]
        gpu_count = sum(1 for r in records if r.get("embed_device") == "gpu")

        return {
            "total_requests": len(records),
            "latency_s": {
                "mean": round(_mean(total_s), 4),
                "min": round(min(total_s), 4),
                "max": round(max(total_s), 4),
                "p50": round(_percentile(total_s, 50), 4),
                "p95": round(_percentile(total_s, 95), 4),
            },
            "tokens_per_sec": {
                "mean": round(_mean(tps), 2),
                "max": round(max(tps), 2),
            },
            "embed_device_distribution": {
                "gpu": gpu_count,
                "cpu": len(records) - gpu_count,
            },
        }


# ------------------------------------------------------------------
# Stress test
# ------------------------------------------------------------------

STRESS_QUESTIONS = [
    "How many days does Newark have to respond to an OPRA request?",
    "What records are exempt from OPRA disclosure in Newark?",
    "How many OPRA requests did Newark receive in fiscal year 2023?",
    "What are the fees for copying records under OPRA?",
    "What happens if a Newark OPRA request is denied?",
]


async def stress_test(pipeline: "RAGPipeline", n_concurrent: int = 5) -> dict[str, Any]:
    """
    Fire n_concurrent RAG queries simultaneously and measure:
      - Per-request latency and tok/s
      - Per-GPU VRAM delta (before vs after the whole batch)
      - Scheduler CPU fallback rate
      - Aggregate throughput

    Results are written to stress_test_results.json and returned.
    """
    from legacyrag.vram_scheduler import query_vram

    logger.info("stress_test: starting %d concurrent requests", n_concurrent)

    questions = (STRESS_QUESTIONS * ((n_concurrent // len(STRESS_QUESTIONS)) + 1))[:n_concurrent]

    # Snapshot VRAM before burst
    vram_before = {g.index: g.free_mb for g in query_vram()}

    # Count scheduler decisions during the test by tailing the log file
    import os
    log_pos_before = _tail_jsonl_count("schedule_decisions.jsonl")

    t_batch_start = time.perf_counter()

    tasks = [pipeline.query(q, max_tokens=128) for q in questions]
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    t_batch_end = time.perf_counter()
    batch_wall_s = t_batch_end - t_batch_start

    # Snapshot VRAM after burst
    vram_after = {g.index: g.free_mb for g in query_vram()}

    # Parse scheduler decisions made during the test
    new_decisions = _read_jsonl_from("schedule_decisions.jsonl", log_pos_before)
    cpu_decisions = sum(1 for d in new_decisions if d.get("decision") == "cpu")
    gpu_decisions = sum(1 for d in new_decisions if d.get("decision") == "gpu")

    # Collate per-request results
    per_request = []
    total_completion_tokens = 0
    errors = 0

    for i, (q, result) in enumerate(zip(questions, results_raw)):
        if isinstance(result, Exception):
            errors += 1
            per_request.append({
                "request_id": i + 1,
                "question_preview": q[:60],
                "error": str(result),
            })
            logger.error("stress request %d failed: %s", i + 1, result)
        else:
            bench = result.benchmark
            completion_tokens = bench.get("tokens", {}).get("completion", 0)
            total_completion_tokens += completion_tokens
            per_request.append({
                "request_id": i + 1,
                "question_preview": q[:60],
                "total_latency_s": bench["latency"]["total_s"],
                "generate_latency_s": bench["latency"]["generate_s"],
                "embed_latency_s": bench["latency"]["embed_s"],
                "tokens_per_sec": bench["tokens"]["per_sec"],
                "completion_tokens": completion_tokens,
                "embed_device": bench["embed_device"],
            })

    vram_delta = {
        f"gpu{idx}_delta_mb": round(vram_before.get(idx, 0) - vram_after.get(idx, 0), 1)
        for idx in set(list(vram_before.keys()) + list(vram_after.keys()))
    }

    successful = [r for r in per_request if "error" not in r]
    tps_values = [r["tokens_per_sec"] for r in successful]
    latency_values = [r["total_latency_s"] for r in successful]

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_concurrent": n_concurrent,
            "max_tokens_per_request": 128,
        },
        "batch_wall_time_s": round(batch_wall_s, 2),
        "aggregate_throughput_tok_s": round(
            total_completion_tokens / batch_wall_s if batch_wall_s > 0 else 0, 2
        ),
        "requests": {
            "total": n_concurrent,
            "succeeded": n_concurrent - errors,
            "failed": errors,
        },
        "tokens_per_sec": {
            "mean": round(_mean(tps_values), 2),
            "min": round(min(tps_values), 2) if tps_values else 0,
            "max": round(max(tps_values), 2) if tps_values else 0,
        },
        "latency_s": {
            "mean": round(_mean(latency_values), 2),
            "min": round(min(latency_values), 2) if latency_values else 0,
            "max": round(max(latency_values), 2) if latency_values else 0,
        },
        "scheduler": {
            "total_decisions": len(new_decisions),
            "gpu_decisions": gpu_decisions,
            "cpu_decisions": cpu_decisions,
            "cpu_fallback_rate_pct": round(
                cpu_decisions / len(new_decisions) * 100 if new_decisions else 0, 1
            ),
        },
        "vram_delta_mb": vram_delta,
        "vram_before_mb": {f"gpu{k}": v for k, v in vram_before.items()},
        "vram_after_mb": {f"gpu{k}": v for k, v in vram_after.items()},
        "per_request": per_request,
    }

    with open(STRESS_TEST_FILE, "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        "stress_test done: %ds wall | %.2f tok/s aggregate | cpu_fallback=%.0f%%",
        batch_wall_s,
        summary["aggregate_throughput_tok_s"],
        summary["scheduler"]["cpu_fallback_rate_pct"],
    )
    return summary


# ------------------------------------------------------------------
# Stats helpers
# ------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _tail_jsonl_count(path: str) -> int:
    """Return current line count of a JSONL file."""
    try:
        with open(path) as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _read_jsonl_from(path: str, start_line: int) -> list[dict]:
    """Read JSONL entries starting from start_line index."""
    entries = []
    try:
        with open(path) as fh:
            for i, line in enumerate(fh):
                if i >= start_line and line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return entries
