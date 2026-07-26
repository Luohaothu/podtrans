#!/usr/bin/env python3
"""Benchmark the official FunASR speaker feature extraction path."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time

import numpy as np
import torch
from funasr.models.campplus.utils import extract_feature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-threads", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--windows", type=int, default=1536)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(max(1, min(args.concurrency, 4)))
    audio = np.zeros((args.windows, 24000), dtype=np.float32)

    def prepare(rows: np.ndarray) -> tuple[int, ...]:
        tensors = [torch.from_numpy(row) for row in rows]
        features, _lengths, _times = extract_feature(tensors)
        return tuple(features.shape)

    prepare(audio[:2])
    chunks = [chunk for chunk in np.array_split(audio, args.concurrency) if len(chunk)]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        shapes = list(executor.map(prepare, chunks))
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "cpu_threads": args.cpu_threads,
                "concurrency": args.concurrency,
                "windows": args.windows,
                "seconds": round(elapsed, 6),
                "windows_per_second": round(args.windows / elapsed, 3),
                "shapes": shapes[:2],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
