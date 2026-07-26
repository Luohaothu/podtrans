#!/usr/bin/env python3
"""Validate the speaker model, clustering, batching, and GPU memory use."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diarize_funasr import (
    DEFAULT_SPEAKER_MODEL,
    ROOT,
    auto_model_kwargs,
    cached_model_path,
    utc_now,
    write_json_atomic,
)


DEFAULT_OUTPUT = ROOT / "_metadata" / "speaker_diarization_model_validation.json"


def load_wav(path: Path) -> np.ndarray:
    from funasr.utils.load_utils import load_audio_text_image_video

    samples = load_audio_text_image_video(str(path), fs=16000)
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    return np.asarray(samples, dtype=np.float32).reshape(-1)


def speaker_windows(samples: np.ndarray) -> list[np.ndarray]:
    from funasr.models.campplus.utils import sv_chunk

    duration = len(samples) / 16000.0
    return [
        np.asarray(window, dtype=np.float32)
        for _start, _end, window in sv_chunk([[0.0, duration, samples]])
    ]


def varied_windows(
    source: list[np.ndarray],
    count: int,
    *,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    for index in range(count):
        window = source[index % len(source)].copy()
        gain = 0.9 + 0.2 * ((index % 7) / 6.0)
        noise = rng.normal(0.0, 1e-5, size=window.shape).astype(np.float32)
        output.append(np.clip(window * gain + noise, -1.0, 1.0))
    return output


def infer_embeddings(model: object, arrays: list[np.ndarray], device: str) -> np.ndarray:
    import torch

    with torch.inference_mode():
        results, _meta = model.model.inference(
            data_in=arrays,
            key=[f"sample-{index}" for index in range(len(arrays))],
            device=device,
            fs=16000,
        )
    embeddings = torch.cat([item["spk_embedding"] for item in results], dim=0)
    if embeddings.shape[0] != len(arrays):
        raise RuntimeError(
            f"model returned {embeddings.shape[0]} embeddings for {len(arrays)} inputs"
        )
    return embeddings.detach().float().cpu().numpy()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(left @ right)


def best_cluster_accuracy(expected: np.ndarray, predicted: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    expected_labels = sorted(set(expected.tolist()))
    predicted_labels = sorted(set(predicted.tolist()))
    confusion = np.zeros((len(expected_labels), len(predicted_labels)), dtype=np.int64)
    for expected_value, predicted_value in zip(expected, predicted):
        confusion[
            expected_labels.index(int(expected_value)),
            predicted_labels.index(int(predicted_value)),
        ] += 1
    rows, columns = linear_sum_assignment(-confusion)
    return float(confusion[rows, columns].sum() / len(expected))


def parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive comma-separated integers")
    return sizes


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds or any(not 0 < threshold <= 1 for threshold in thresholds):
        raise argparse.ArgumentTypeError(
            "thresholds must be comma-separated numbers in (0, 1]"
        )
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-model", default=DEFAULT_SPEAKER_MODEL)
    parser.add_argument("--speaker-model-path", type=Path)
    parser.add_argument("--hub", choices=["ms", "hf"], default="ms")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=[256, 512, 1024])
    parser.add_argument(
        "--merge-thresholds",
        type=parse_thresholds,
        default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.78],
    )
    parser.add_argument("--windows-per-speaker", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import torch
    from funasr import AutoModel
    from funasr.models.campplus.cluster_backend import ClusterBackend

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for --device {args.device}")
    model_path = args.speaker_model_path or cached_model_path(args.speaker_model)
    if model_path is None:
        raise RuntimeError(
            "speaker model is not cached; download it from ModelScope before validation"
        )
    example_dir = model_path / "examples"
    example_paths = [
        example_dir / "speaker1_a_cn_16k.wav",
        example_dir / "speaker1_b_cn_16k.wav",
        example_dir / "speaker2_a_cn_16k.wav",
    ]
    missing = [str(path) for path in example_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"official validation audio is missing: {missing}")

    kwargs = auto_model_kwargs(
        args.speaker_model,
        model_path=model_path,
        hub=args.hub,
        device=args.device,
        cpu_threads=args.cpu_threads,
    )
    load_started = time.perf_counter()
    model = AutoModel(**kwargs)
    load_seconds = time.perf_counter() - load_started

    samples = [load_wav(path) for path in example_paths]
    pair_embeddings = infer_embeddings(model, samples, args.device)
    same_cosine = cosine(pair_embeddings[0], pair_embeddings[1])
    different_cosine = cosine(pair_embeddings[0], pair_embeddings[2])

    speaker_one_source = speaker_windows(samples[0]) + speaker_windows(samples[1])
    speaker_two_source = speaker_windows(samples[2])
    speaker_one = varied_windows(
        speaker_one_source,
        args.windows_per_speaker,
        seed=11,
    )
    speaker_two = varied_windows(
        speaker_two_source,
        args.windows_per_speaker,
        seed=29,
    )
    interleaved: list[np.ndarray] = []
    expected: list[int] = []
    for left, right in zip(speaker_one, speaker_two):
        interleaved.extend([left, right])
        expected.extend([0, 1])
    cluster_embeddings = infer_embeddings(model, interleaved, args.device)
    expected_array = np.asarray(expected, dtype=np.int64)
    threshold_sweep: list[dict[str, float | int]] = []
    for threshold in args.merge_thresholds:
        np.random.seed(0)
        cluster = ClusterBackend(merge_thr=threshold)
        predicted = np.asarray(cluster(cluster_embeddings), dtype=np.int64)
        threshold_sweep.append(
            {
                "merge_threshold": threshold,
                "predicted_speakers": len(set(predicted.tolist())),
                "best_label_permutation_accuracy": round(
                    best_cluster_accuracy(expected_array, predicted),
                    6,
                ),
            }
        )
    best_threshold_result = max(
        threshold_sweep,
        key=lambda item: (
            float(item["best_label_permutation_accuracy"]),
            -abs(int(item["predicted_speakers"]) - 2),
            float(item["merge_threshold"]),
        ),
    )
    cluster_accuracy = float(best_threshold_result["best_label_permutation_accuracy"])
    predicted_speakers = int(best_threshold_result["predicted_speakers"])
    recommended_merge_threshold = float(best_threshold_result["merge_threshold"])

    benchmark_source = interleaved
    benchmarks: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        arrays = [
            benchmark_source[index % len(benchmark_source)]
            for index in range(batch_size)
        ]
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            embeddings = infer_embeddings(model, arrays, args.device)
            elapsed = time.perf_counter() - started
            record: dict[str, object] = {
                "batch_size": batch_size,
                "status": "ok",
                "elapsed_seconds": round(elapsed, 6),
                "audio_seconds": batch_size * 1.5,
                "realtime_x": round(batch_size * 1.5 / elapsed, 2),
                "embedding_shape": list(embeddings.shape),
            }
            if args.device.startswith("cuda"):
                record["gpu_peak_allocated_mb"] = round(
                    torch.cuda.max_memory_allocated() / 1048576,
                    1,
                )
                record["gpu_peak_reserved_mb"] = round(
                    torch.cuda.max_memory_reserved() / 1048576,
                    1,
                )
        except RuntimeError as exc:
            elapsed = time.perf_counter() - started
            record = {
                "batch_size": batch_size,
                "status": "error",
                "elapsed_seconds": round(elapsed, 6),
                "error": repr(exc),
            }
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        benchmarks.append(record)

    successful = [item for item in benchmarks if item["status"] == "ok"]
    if not successful:
        recommended_batch_size = None
    else:
        if args.device.startswith("cuda"):
            total_mb = torch.cuda.get_device_properties(args.device).total_memory / 1048576
            within_budget = [
                item
                for item in successful
                if float(item["gpu_peak_reserved_mb"]) <= total_mb * 0.8
            ]
        else:
            within_budget = successful
        candidates = within_budget or successful
        recommended_batch_size = max(
            candidates,
            key=lambda item: float(item["realtime_x"]),
        )["batch_size"]

    checks = {
        "same_speaker_similarity_exceeds_different_by_0_2": (
            same_cosine - different_cosine >= 0.2
        ),
        "unsupervised_cluster_detects_two_speakers": predicted_speakers == 2,
        "synthetic_cluster_accuracy_at_least_0_95": cluster_accuracy >= 0.95,
        "all_requested_batches_succeeded": len(successful) == len(benchmarks),
    }
    if args.device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(args.device)
        device_info = {
            "name": properties.name,
            "total_memory_mb": round(properties.total_memory / 1048576, 1),
            "torch_cuda": torch.version.cuda,
        }
    else:
        device_info = {"name": "cpu"}
    report = {
        "generated_at": utc_now(),
        "speaker_model": args.speaker_model,
        "speaker_model_path": str(model_path),
        "precision": "fp32",
        "device": device_info,
        "model_load_seconds": round(load_seconds, 6),
        "official_pair_test": {
            "same_speaker_cosine": round(same_cosine, 6),
            "different_speaker_cosine": round(different_cosine, 6),
            "margin": round(same_cosine - different_cosine, 6),
        },
        "synthetic_diarization_test": {
            "windows": len(interleaved),
            "expected_speakers": 2,
            "predicted_speakers": predicted_speakers,
            "best_label_permutation_accuracy": round(cluster_accuracy, 6),
            "recommended_merge_threshold": recommended_merge_threshold,
            "threshold_sweep": threshold_sweep,
        },
        "batch_benchmarks": benchmarks,
        "recommended_batch_size": recommended_batch_size,
        "recommended_merge_threshold": recommended_merge_threshold,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
