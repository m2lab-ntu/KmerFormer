#!/usr/bin/env python3
"""
Lightweight resource monitor for training jobs.
Tracks GPU memory, throughput, and epoch timing.
Reconstructed from train.py call-site interface.
"""

import json
import time
import threading
from pathlib import Path

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class ResourceMonitor:
    def __init__(self, device: str = "cuda", sample_interval: float = 30.0):
        self.device = device
        self.sample_interval = sample_interval

        self._peak_gpu_mib = 0.0
        self._gpu_samples = []
        self._epoch_start_times = {}
        self._epoch_durations = {}
        self._epoch_samples = {}   # epoch → reads count
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        self._running = True
        if _HAS_TORCH and "cuda" in self.device:
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.sample_interval + 1)

    def epoch_start(self, epoch: int):
        self._epoch_start_times[epoch] = time.perf_counter()
        self._epoch_samples[epoch] = 0

    def epoch_end(self, epoch: int, train_samples: int = 0):
        t0 = self._epoch_start_times.get(epoch, time.perf_counter())
        self._epoch_durations[epoch] = time.perf_counter() - t0
        self._epoch_samples[epoch] = train_samples

    def get_epoch_throughput(self, epoch: int) -> float:
        dur = self._epoch_durations.get(epoch, 0.0)
        samples = self._epoch_samples.get(epoch, 0)
        if dur <= 0:
            return 0.0
        return samples / dur

    def snapshot(self) -> dict:
        gpu_mib = self._current_gpu_mib()
        return {
            "gpu_allocated_mib": round(gpu_mib, 1),
            "peak_gpu_mib": round(self._peak_gpu_mib, 1),
        }

    def save_report(self, path: str):
        report = {
            "peak_gpu_mib": round(self._peak_gpu_mib, 1),
            "epoch_durations_sec": {str(k): round(v, 2) for k, v in self._epoch_durations.items()},
            "epoch_throughputs_reads_per_sec": {
                str(k): round(self.get_epoch_throughput(k), 1)
                for k in self._epoch_durations
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def print_summary(self):
        print("\n=== Resource Monitor Summary ===")
        print(f"  Peak GPU memory: {self._peak_gpu_mib:.1f} MiB")
        for epoch in sorted(self._epoch_durations):
            dur = self._epoch_durations[epoch]
            thr = self.get_epoch_throughput(epoch)
            print(f"  Epoch {epoch}: {dur:.1f}s  ({thr:,.0f} reads/sec)")

    # ── internals ─────────────────────────────────────────────────────────────

    def _current_gpu_mib(self) -> float:
        if _HAS_TORCH and "cuda" in self.device:
            try:
                mib = torch.cuda.memory_allocated() / 1024 / 1024
                peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                self._peak_gpu_mib = max(self._peak_gpu_mib, peak)
                return mib
            except Exception:
                pass
        return 0.0

    def _sample_loop(self):
        while self._running:
            mib = self._current_gpu_mib()
            self._gpu_samples.append(mib)
            self._stop_event.wait(self.sample_interval)
