"""Fine-grained profiling instrumentation for retrieval pipelines.

Measures wall time, CPU time, GPU time (CUDA events), model initialization,
corpus loading, BM25, dense retrieval, cross-encoder reranking, serialization,
and peak VRAM usage.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional


@dataclass
class StageTiming:
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    peak_vram_bytes: Optional[int] = None
    peak_vram_reserved_bytes: Optional[int] = None


class RetrievalProfiler:
    """Thread-safe, sub-stage profiler supporting CPU and CUDA event timing."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.has_cuda = (device == "cuda")
        self._torch = None
        if self.has_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    self._torch = torch
                else:
                    self.has_cuda = False
            except ImportError:
                self.has_cuda = False

        self.stages: Dict[str, StageTiming] = {}
        self.total_wall_start: float = 0.0
        self.total_wall_end: float = 0.0
        self.total_cpu_start: float = 0.0
        self.total_cpu_end: float = 0.0
        self.initial_vram_allocated: Optional[int] = None
        self.initial_vram_reserved: Optional[int] = None
        self.peak_vram_allocated: Optional[int] = None
        self.peak_vram_reserved: Optional[int] = None
        self.total_system_vram: Optional[int] = None

    def start_claim(self):
        """Mark start of claim-level retrieval."""
        if self.has_cuda and self._torch:
            self._torch.cuda.synchronize()
            self._torch.cuda.reset_peak_memory_stats()
            self.initial_vram_allocated = self._torch.cuda.memory_allocated()
            self.initial_vram_reserved = self._torch.cuda.memory_reserved()
            try:
                self.total_system_vram = self._torch.cuda.get_device_properties(0).total_memory
            except Exception:
                self.total_system_vram = None
        self.total_wall_start = time.perf_counter()
        self.total_cpu_start = time.process_time()
        self.stages.clear()

    def end_claim(self) -> Dict[str, Any]:
        """Mark end of claim-level retrieval and compile metrics dictionary."""
        self.total_wall_end = time.perf_counter()
        self.total_cpu_end = time.process_time()

        if self.has_cuda and self._torch:
            self._torch.cuda.synchronize()
            self.peak_vram_allocated = self._torch.cuda.max_memory_allocated()
            self.peak_vram_reserved = self._torch.cuda.max_memory_reserved()
            try:
                self.total_system_vram = self._torch.cuda.get_device_properties(0).total_memory
            except Exception:
                pass

        total_wall = max(0.0, self.total_wall_end - self.total_wall_start)
        total_cpu = max(0.0, self.total_cpu_end - self.total_cpu_start)
        total_gpu = sum(s.gpu_seconds for s in self.stages.values())

        report = {
            "wall_time_seconds": round(total_wall, 6),
            "cpu_time_seconds": round(total_cpu, 6),
            "gpu_time_seconds": round(total_gpu, 6),
            "peak_vram_bytes": self.peak_vram_allocated,
            "peak_vram_allocated_bytes": self.peak_vram_allocated,
            "peak_vram_reserved_bytes": self.peak_vram_reserved,
            "total_system_vram_bytes": self.total_system_vram,
            "stages": {
                name: {
                    "wall_seconds": round(st.wall_seconds, 6),
                    "cpu_seconds": round(st.cpu_seconds, 6),
                    "gpu_seconds": round(st.gpu_seconds, 6),
                    "peak_vram_bytes": st.peak_vram_bytes,
                    "peak_vram_reserved_bytes": st.peak_vram_reserved_bytes,
                }
                for name, st in self.stages.items()
            },
            # Flat top-level convenience fields required by audit contract
            "model_initialization_seconds": round(self.stages.get("model_initialization", StageTiming()).wall_seconds, 6),
            "corpus_loading_seconds": round(self.stages.get("corpus_loading", StageTiming()).wall_seconds, 6),
            "bm25_seconds": round(self.stages.get("bm25", StageTiming()).wall_seconds, 6),
            "dense_retrieval_seconds": round(self.stages.get("dense_retrieval", StageTiming()).wall_seconds, 6),
            "cross_encoder_reranking_seconds": round(self.stages.get("cross_encoder_reranking", StageTiming()).wall_seconds, 6),
            "serialization_seconds": round(self.stages.get("serialization", StageTiming()).wall_seconds, 6),
        }
        return report

    @contextmanager
    def measure(self, stage_name: str):
        """Context manager to measure timing and memory for a specific stage."""
        start_gpu_event = None
        end_gpu_event = None

        if self.has_cuda and self._torch:
            self._torch.cuda.synchronize()
            start_gpu_event = self._torch.cuda.Event(enable_timing=True)
            end_gpu_event = self._torch.cuda.Event(enable_timing=True)
            start_gpu_event.record()

        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        try:
            yield
        finally:
            t1_wall = time.perf_counter()
            t1_cpu = time.process_time()

            gpu_time_s = 0.0
            stage_peak_vram = None
            stage_peak_reserved = None
            if self.has_cuda and self._torch:
                end_gpu_event.record()
                end_gpu_event.synchronize()
                gpu_time_ms = start_gpu_event.elapsed_time(end_gpu_event)
                gpu_time_s = max(0.0, gpu_time_ms / 1000.0)
                stage_peak_vram = self._torch.cuda.max_memory_allocated()
                stage_peak_reserved = self._torch.cuda.max_memory_reserved()

            timing = self.stages.get(stage_name, StageTiming())
            timing.wall_seconds += max(0.0, t1_wall - t0_wall)
            timing.cpu_seconds += max(0.0, t1_cpu - t0_cpu)
            timing.gpu_seconds += gpu_time_s
            timing.peak_vram_bytes = stage_peak_vram
            timing.peak_vram_reserved_bytes = stage_peak_reserved
            self.stages[stage_name] = timing
