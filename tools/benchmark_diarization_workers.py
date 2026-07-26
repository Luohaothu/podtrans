#!/usr/bin/env python3
"""Benchmark multiple FP32 diarization workers on one fixed episode corpus."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diarize_funasr import (
    DEFAULT_OUTPUT_PREFIX,
    ROOT,
    load_jobs,
    sort_jobs,
    utc_now,
    write_json_atomic,
)


DEFAULT_OUTPUT = ROOT / "_metadata" / "speaker_diarization_worker_benchmark.json"


def parse_worker_counts(value: str) -> list[int]:
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError(
            "worker counts must be positive comma-separated integers"
        )
    return counts


def wait_until_ready(
    processes: list[subprocess.Popen[str]],
    logs: list[Path],
    timeout_seconds: float,
) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout_seconds:
        if any(process.poll() is not None for process in processes):
            break
        ready = 0
        for path in logs:
            if path.exists() and "[diarize] batch=1 jobs=" in path.read_text(
                encoding="utf-8",
                errors="replace",
            ):
                ready += 1
        if ready == len(logs):
            return time.perf_counter() - started
        time.sleep(0.25)
    raise RuntimeError("workers did not all enter the first batch before timeout")


def parse_dmon(path: Path) -> dict[str, float | int]:
    sm_values: list[int] = []
    fb_values: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 13:
            continue
        try:
            sm_values.append(int(fields[4]))
            fb_values.append(int(fields[12]))
        except ValueError:
            continue
    if not sm_values:
        return {"samples": 0}
    return {
        "samples": len(sm_values),
        "sm_mean_pct": round(statistics.mean(sm_values), 3),
        "sm_median_pct": round(statistics.median(sm_values), 3),
        "sm_zero_pct": round(100 * sum(value == 0 for value in sm_values) / len(sm_values), 3),
        "sm_below_10_pct": round(
            100 * sum(value < 10 for value in sm_values) / len(sm_values),
            3,
        ),
        "sm_at_least_90_pct": round(
            100 * sum(value >= 90 for value in sm_values) / len(sm_values),
            3,
        ),
        "peak_fb_memory_mb": max(fb_values),
    }


def parse_worker_log(path: Path) -> dict[str, int | str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    selected_match = re.search(r"\[diarize\] selected=(\d+)", text)
    batch_matches = re.findall(
        r"processed=(\d+)/(\d+) ok=(\d+) errors=(\d+) seconds=([0-9.]+)",
        text,
    )
    result: dict[str, int | str] = {
        "path": str(path.relative_to(ROOT)),
        "selected": int(selected_match.group(1)) if selected_match else 0,
        "processed": 0,
        "ok": 0,
        "errors": 0,
    }
    if batch_matches:
        processed, _total, ok, errors, _seconds = batch_matches[-1]
        result.update(
            processed=int(processed),
            ok=int(ok),
            errors=int(errors),
        )
    return result


def output_core(episode_dir: Path, prefix: str) -> dict[str, object]:
    payload = json.loads(
        (episode_dir / f"{prefix}.json").read_text(encoding="utf-8")
    )
    return {
        "status": payload.get("status"),
        "text": payload.get("text"),
        "speakers": payload.get("speakers"),
        "segments": payload.get("segments"),
        "speaker_count": payload.get("metrics", {}).get("speaker_count"),
        "window_count": payload.get("metrics", {}).get("window_count"),
        "cluster_backend": payload.get("diarization", {}).get("cluster_backend"),
        "speaker_model_precision": payload.get("diarization", {}).get(
            "speaker_model_precision"
        ),
        "allow_tf32": payload.get("diarization", {}).get("allow_tf32"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument(
        "--worker-counts",
        type=parse_worker_counts,
        default=[1, 2, 3, 4],
    )
    parser.add_argument("--speaker-batch-size", type=int, default=256)
    parser.add_argument("--job-batch-size", type=int, default=4)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--speaker-service-socket", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    jobs, _skipped = load_jobs(
        channel_filter={args.channel},
        episode_dir_filter=None,
        output_prefix=DEFAULT_OUTPUT_PREFIX,
        include_done=True,
        retry_errors=True,
    )
    jobs = sort_jobs(jobs, "size")[: args.episodes]
    if len(jobs) != args.episodes:
        raise RuntimeError(
            f"requested {args.episodes} episodes but selected {len(jobs)}"
        )

    benchmark_dir = ROOT / "_metadata" / "speaker_worker_benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    episode_args: list[str] = []
    for job in jobs:
        episode_args.extend(["--episode-dir", str(job.episode_dir)])

    trials: list[dict[str, object]] = []
    prefixes: dict[int, str] = {}
    for workers in args.worker_counts:
        runtime = "service" if args.speaker_service_socket else "local"
        prefix = (
            f"transcript_speakers_benchmark_fp32_"
            f"{runtime}_w{workers}_b{args.speaker_batch_size}"
        )
        prefixes[workers] = prefix
        logs = [
            benchmark_dir / f"workers-{workers}-shard-{index}.log"
            for index in range(workers)
        ]
        dmon_path = benchmark_dir / f"workers-{workers}-gpu.dmon"
        processes: list[subprocess.Popen[str]] = []
        handles = []
        wall_started = time.perf_counter()
        try:
            for index, log_path in enumerate(logs):
                handle = log_path.open("w", encoding="utf-8")
                handles.append(handle)
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "diarize_funasr.py"),
                    "--device",
                    "cuda:0",
                    "--sort",
                    "size",
                    "--speaker-batch-size",
                    str(args.speaker_batch_size),
                    "--job-batch-size",
                    str(args.job_batch_size),
                    "--decode-workers",
                    str(args.decode_workers),
                    "--cpu-threads",
                    str(args.cpu_threads),
                    "--audio-chunk-seconds",
                    "300",
                    "--merge-threshold",
                    "0.60",
                    "--include-done",
                    "--no-report",
                    "--no-manifest",
                    "--output-prefix",
                    prefix,
                    "--num-shards",
                    str(workers),
                    "--shard-index",
                    str(index),
                    "--temp-dir",
                    f"/tmp/podtrans-speaker-benchmark/w{workers}-s{index}",
                    *episode_args,
                ]
                if args.speaker_service_socket is not None:
                    command.extend(
                        [
                            "--speaker-service-socket",
                            str(args.speaker_service_socket),
                            "--speaker-service-timeout",
                            "600",
                            "--vad-device",
                            "cpu",
                        ]
                    )
                environment = dict(os.environ)
                environment["PYTHONUNBUFFERED"] = "1"
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                )
            startup_seconds = wait_until_ready(processes, logs, timeout_seconds=90)
            active_started = time.perf_counter()
            with dmon_path.open("w", encoding="utf-8") as dmon_handle:
                dmon = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "dmon",
                        "-i",
                        "0",
                        "-s",
                        "pucm",
                        "-d",
                        "1",
                    ],
                    stdout=dmon_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                return_codes = [process.wait() for process in processes]
                active_seconds = time.perf_counter() - active_started
                dmon.terminate()
                try:
                    dmon.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    dmon.kill()
                    dmon.wait()
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for handle in handles:
                handle.close()

        wall_seconds = time.perf_counter() - wall_started
        worker_results = [parse_worker_log(path) for path in logs]
        selected = sum(int(item["selected"]) for item in worker_results)
        processed = sum(int(item["processed"]) for item in worker_results)
        errors = sum(int(item["errors"]) for item in worker_results)
        trial = {
            "workers": workers,
            "precision": "fp32",
            "allow_tf32": False,
            "speaker_batch_size_per_worker": args.speaker_batch_size,
            "job_batch_size_per_worker": args.job_batch_size,
            "decode_workers_per_worker": args.decode_workers,
            "startup_seconds": round(startup_seconds, 3),
            "active_seconds": round(active_seconds, 3),
            "wall_seconds": round(wall_seconds, 3),
            "episodes_selected": selected,
            "episodes_processed": processed,
            "errors": errors,
            "active_episodes_per_second": round(
                processed / max(active_seconds, 1e-9),
                6,
            ),
            "wall_episodes_per_second": round(
                processed / max(wall_seconds, 1e-9),
                6,
            ),
            "gpu": parse_dmon(dmon_path),
            "return_codes": return_codes,
            "worker_logs": worker_results,
        }
        trials.append(trial)
        print(json.dumps(trial, ensure_ascii=False), flush=True)
        time.sleep(2)

    baseline_workers = args.worker_counts[0]
    baseline_prefix = prefixes[baseline_workers]
    comparisons: list[dict[str, object]] = []
    for workers in args.worker_counts[1:]:
        candidate_prefix = prefixes[workers]
        exact = 0
        mismatches: list[str] = []
        for job in jobs:
            baseline = output_core(job.episode_dir, baseline_prefix)
            candidate = output_core(job.episode_dir, candidate_prefix)
            if baseline == candidate:
                exact += 1
            else:
                mismatches.append(str(job.episode_dir.relative_to(ROOT)))
        comparisons.append(
            {
                "baseline_workers": baseline_workers,
                "candidate_workers": workers,
                "episodes": len(jobs),
                "exact_output_core_matches": exact,
                "mismatches": mismatches,
            }
        )

    checks = {
        "all_trials_completed_without_errors": all(
            trial["errors"] == 0
            and all(code == 0 for code in trial["return_codes"])
            and trial["episodes_processed"] == len(jobs)
            for trial in trials
        ),
        "all_worker_counts_match_single_worker_outputs": all(
            comparison["exact_output_core_matches"] == len(jobs)
            for comparison in comparisons
        ),
        "all_outputs_are_strict_fp32": all(
            output_core(job.episode_dir, prefix)["speaker_model_precision"] == "fp32"
            and output_core(job.episode_dir, prefix)["allow_tf32"] is False
            for prefix in prefixes.values()
            for job in jobs
        ),
    }
    report = {
        "generated_at": utc_now(),
        "channel": args.channel,
        "episode_count": len(jobs),
        "episode_dirs": [str(job.episode_dir.relative_to(ROOT)) for job in jobs],
        "precision_policy": {
            "speaker_model_precision": "fp32",
            "allow_tf32": False,
            "speaker_runtime": (
                "single_unix_socket_service"
                if args.speaker_service_socket is not None
                else "in_process"
            ),
            "reason": (
                "The official cached ERes2NetV2 checkpoint and FunASR inference "
                "path provide FP32 only."
            ),
        },
        "trials": trials,
        "output_comparisons": comparisons,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
