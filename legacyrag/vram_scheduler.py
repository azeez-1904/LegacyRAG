"""
VRAM-aware scheduler for LegacyRAG.

Monitors both Quadro K4200 GPUs independently. Before every embedding
operation, queries nvidia-smi for per-GPU free VRAM. Routing decisions:
  - CPU if generation is active (GenerationContext held)
  - CPU if ANY GPU falls below VRAM_THRESHOLD_MB
  - GPU otherwise, with the specific available GPU indices logged

All decisions are appended to schedule_decisions.jsonl with full
per-GPU stats for research paper analysis.
"""

import subprocess
import json
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import NamedTuple

logger = logging.getLogger(__name__)

VRAM_THRESHOLD_MB = 800
SCHEDULE_LOG_FILE = "schedule_decisions.jsonl"

_generation_lock = threading.Lock()
_generation_active_count = 0


@dataclass
class GPUInfo:
    index: int
    free_mb: float
    used_mb: float
    total_mb: float
    name: str = ""

    @property
    def utilization_pct(self) -> float:
        if self.total_mb == 0:
            return 0.0
        return round(self.used_mb / self.total_mb * 100, 1)

    @property
    def above_threshold(self) -> bool:
        return self.free_mb >= VRAM_THRESHOLD_MB


class ScheduleDecision(NamedTuple):
    device: str           # "gpu" or "cpu"
    reason: str
    available_gpus: list  # indices of GPUs above threshold
    gpu_stats: list       # list of GPUInfo


def query_vram() -> list[GPUInfo]:
    """Query nvidia-smi for VRAM stats on all available GPUs."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi returned non-zero: %s", result.stderr.strip())
            return []

        gpus: list[GPUInfo] = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            idx, free, used, total = parts[0], parts[1], parts[2], parts[3]
            name = parts[4] if len(parts) > 4 else "unknown"
            gpus.append(
                GPUInfo(
                    index=int(idx),
                    free_mb=float(free),
                    used_mb=float(used),
                    total_mb=float(total),
                    name=name,
                )
            )
        return gpus

    except FileNotFoundError:
        logger.warning("nvidia-smi not found — running CPU-only mode")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
        return []
    except Exception as exc:
        logger.warning("nvidia-smi query failed: %s", exc)
        return []


def _log_decision(decision: ScheduleDecision, generation_active: bool) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision.device,
        "reason": decision.reason,
        "generation_active": generation_active,
        "available_gpus": decision.available_gpus,
        "gpu_stats": [asdict(g) for g in decision.gpu_stats],
        "per_gpu_threshold_status": [
            {
                "gpu": g.index,
                "free_mb": g.free_mb,
                "above_threshold": g.above_threshold,
                "utilization_pct": g.utilization_pct,
            }
            for g in decision.gpu_stats
        ],
    }
    try:
        with open(SCHEDULE_LOG_FILE, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.error("Failed to write schedule log: %s", exc)

    per_gpu_str = " | ".join(
        f"GPU{g.index}={'OK' if g.above_threshold else 'LOW'} ({g.free_mb:.0f}MB free)"
        for g in decision.gpu_stats
    )
    logger.info("[SCHEDULER] device=%s | %s | %s", decision.device, decision.reason, per_gpu_str)


def decide_device(generation_active: bool | None = None) -> str:
    """
    Return 'gpu' or 'cpu' for the upcoming embedding operation.

    Logic (per-GPU independent checks):
      1. nvidia-smi unavailable → CPU
      2. Generation active → CPU (protect VRAM for ongoing inference)
      3. Any GPU below VRAM_THRESHOLD_MB → CPU
      4. All GPUs above threshold → GPU
    """
    decision = _make_decision(generation_active)
    _log_decision(decision, generation_active or False)
    return decision.device


def _make_decision(generation_active: bool | None) -> ScheduleDecision:
    global _generation_active_count

    with _generation_lock:
        gen_active = _generation_active_count > 0 if generation_active is None else generation_active

    gpus = query_vram()

    if not gpus:
        return ScheduleDecision(
            device="cpu",
            reason="nvidia-smi unavailable",
            available_gpus=[],
            gpu_stats=[],
        )

    if gen_active:
        return ScheduleDecision(
            device="cpu",
            reason="generation in progress — protecting VRAM",
            available_gpus=[],
            gpu_stats=gpus,
        )

    # Per-GPU independent check
    constrained = [g for g in gpus if not g.above_threshold]
    available = [g.index for g in gpus if g.above_threshold]

    if constrained:
        names = ", ".join(
            f"GPU{g.index} ({g.free_mb:.0f} MB < {VRAM_THRESHOLD_MB} MB threshold)"
            for g in constrained
        )
        return ScheduleDecision(
            device="cpu",
            reason=f"VRAM below threshold: {names}",
            available_gpus=available,
            gpu_stats=gpus,
        )

    min_free = min(g.free_mb for g in gpus)
    return ScheduleDecision(
        device="gpu",
        reason=f"all {len(gpus)} GPUs above threshold — min free {min_free:.0f} MB",
        available_gpus=available,
        gpu_stats=gpus,
    )


def get_gpu_status() -> list[dict]:
    """Return current per-GPU status snapshot for health checks and benchmarks."""
    return [
        {
            "index": g.index,
            "name": g.name,
            "free_mb": g.free_mb,
            "used_mb": g.used_mb,
            "total_mb": g.total_mb,
            "utilization_pct": g.utilization_pct,
            "above_threshold": g.above_threshold,
        }
        for g in query_vram()
    ]


class GenerationContext:
    """Context manager that marks generation as active for the scheduler."""

    def __enter__(self):
        global _generation_active_count
        with _generation_lock:
            _generation_active_count += 1
        return self

    def __exit__(self, *_):
        global _generation_active_count
        with _generation_lock:
            _generation_active_count = max(0, _generation_active_count - 1)
