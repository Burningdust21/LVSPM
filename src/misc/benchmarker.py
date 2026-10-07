import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from loguru import logger


class Benchmarker:
    def __init__(self):
        self.execution_times = defaultdict(list)

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        try:
            self._synchronize_cuda()
            start_time = perf_counter()
            yield
        finally:
            self._synchronize_cuda()
            end_time = perf_counter()
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)

    @staticmethod
    def _synchronize_cuda() -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.execution_times), f)

    def dump_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(torch.cuda.memory_stats()["allocated_bytes.all.peak"], f)

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            logger.info("{}: {} calls, avg. {} seconds per call", tag, len(times), np.mean(times))

    def clear_history(self) -> None:
        self.execution_times = defaultdict(list)
