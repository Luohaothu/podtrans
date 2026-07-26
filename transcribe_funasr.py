#!/usr/bin/env python3
"""Transcribe archived podcast audio with FunASR."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import archive_xiaoyuzhou_channels as archive


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = Path(
    os.environ.get("PODTRANS_DATA_DIR", PROJECT_ROOT / "data")
).expanduser().resolve()
DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
REPORT_FILE = ROOT / "_metadata" / "transcription_report.json"
MANIFEST_FILE = ROOT / "_metadata" / "transcription_manifest.jsonl"


@dataclass(frozen=True)
class Job:
    channel: str
    number: str
    title: str
    episode_dir: Path
    audio_path: Path
    transcript_json: Path
    transcript_txt: Path
    transcript_md: Path


@dataclass(frozen=True)
class ChunkTask:
    job: Job
    index: int
    wav_path: str


def log(message: str) -> None:
    print(message, flush=True)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jobs(
    *,
    channel_filter: set[str] | None,
    episode_dir_filter: set[Path] | None,
    include_done: bool,
    retry_errors: bool,
) -> tuple[list[Job], list[dict[str, Any]]]:
    channels = archive.read_channels()
    jobs: list[Job] = []
    skipped: list[dict[str, Any]] = []
    for channel in channels:
        if channel_filter and channel not in channel_filter:
            continue
        for episode in archive.load_saved_episodes(channel):
            episode_dir = archive.episode_dir(channel, episode)
            if episode_dir_filter and episode_dir.resolve() not in episode_dir_filter:
                continue
            audio_path = archive.audio_path_for(channel, episode)
            job = Job(
                channel=channel,
                number=episode.number,
                title=episode.title,
                episode_dir=episode_dir,
                audio_path=audio_path,
                transcript_json=episode_dir / "transcript.json",
                transcript_txt=episode_dir / "transcript.txt",
                transcript_md=episode_dir / "transcript.md",
            )
            if not audio_path.exists():
                skipped.append(
                    {
                        "channel": channel,
                        "number": episode.number,
                        "title": episode.title,
                        "path": rel(audio_path),
                        "status": "missing_audio",
                    }
                )
                continue
            if audio_path.stat().st_size <= 0:
                skipped.append(
                    {
                        "channel": channel,
                        "number": episode.number,
                        "title": episode.title,
                        "path": rel(audio_path),
                        "status": "empty_audio",
                    }
                )
                continue
            if not include_done and transcript_is_done(job.transcript_json, retry_errors=retry_errors):
                continue
            jobs.append(job)
    return jobs, skipped


def transcript_is_done(path: Path, *, retry_errors: bool) -> bool:
    if not path.exists():
        return False
    status = read_transcript_status(path)
    if status == "ok":
        return True
    return bool(status == "error" and not retry_errors)


def read_transcript_status(path: Path) -> str | None:
    """Read the top-level transcript status without loading large chunk payloads."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(32):
                line = fh.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped.startswith('"status"'):
                    continue
                _, raw_value = stripped.split(":", 1)
                value = json.loads(raw_value.strip().rstrip(","))
                return value if isinstance(value, str) else None
    except Exception:
        pass
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def shard_jobs(jobs: list[Job], *, shard_index: int, num_shards: int) -> list[Job]:
    if num_shards <= 1:
        return jobs
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return [job for index, job in enumerate(jobs) if index % num_shards == shard_index]


def sort_jobs(jobs: list[Job], mode: str) -> list[Job]:
    if mode == "archive":
        return jobs
    if mode == "path":
        return sorted(jobs, key=lambda job: rel(job.audio_path))
    if mode == "size":
        return sorted(jobs, key=lambda job: (job.audio_path.stat().st_size, rel(job.audio_path)))
    if mode == "channel":
        return sorted(jobs, key=lambda job: (job.channel, job.number, job.title))
    raise ValueError(f"unsupported sort mode: {mode}")


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
    except Exception as exc:
        raise RuntimeError("ffmpeg is not on PATH and imageio-ffmpeg is unavailable") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def convert_to_chunks(ffmpeg: str, job: Job, chunk_dir: Path, chunk_seconds: int) -> dict[str, Any]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = chunk_dir / "chunk-%05d.wav"
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(job.audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(output_pattern),
    ]
    started = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffmpeg exited {proc.returncode}")
    chunk_paths = sorted(chunk_dir.glob("chunk-*.wav"))
    if not chunk_paths:
        raise RuntimeError("ffmpeg produced no chunks")
    return {
        "chunk_paths": [str(path) for path in chunk_paths],
        "chunk_seconds": chunk_seconds,
        "chunk_count": len(chunk_paths),
        "decode_seconds": elapsed,
        "wav_bytes": sum(path.stat().st_size for path in chunk_paths),
    }


def convert_batch(
    ffmpeg: str,
    batch: list[Job],
    temp_dir: Path,
    workers: int,
    chunk_seconds: int,
) -> tuple[dict[Job, dict[str, Any]], dict[Job, BaseException]]:
    outputs: dict[Job, dict[str, Any]] = {}
    errors: dict[Job, BaseException] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_job = {}
        for index, job in enumerate(batch):
            chunk_dir = temp_dir / f"job-{index:04d}"
            future = executor.submit(convert_to_chunks, ffmpeg, job, chunk_dir, chunk_seconds)
            future_to_job[future] = job
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                outputs[job] = future.result()
            except Exception as exc:
                errors[job] = exc
    return outputs, errors


def create_model(args: argparse.Namespace) -> Any:
    if args.backend == "vllm":
        from funasr.auto.auto_model_vllm import AutoModelVLLM

        return AutoModelVLLM(
            model=args.model,
            device=args.device,
            dtype=args.vllm_dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            enforce_eager=args.vllm_enforce_eager,
        )
    from funasr import AutoModel

    return AutoModel(model=args.model, device=args.device, disable_update=True)


def call_generate(model: Any, wav_paths: list[str], args: argparse.Namespace) -> Any:
    kwargs: dict[str, Any] = {}
    if args.language:
        kwargs["language"] = args.language
    if args.max_length is not None:
        kwargs["max_length"] = args.max_length
    if args.backend != "vllm":
        kwargs["batch_size"] = max(1, min(len(wav_paths), args.batch_size))
    if args.batch_size_s is not None and args.backend != "vllm":
        kwargs["batch_size_s"] = args.batch_size_s
    try:
        if args.backend == "vllm":
            return model.generate(wav_paths, **kwargs)
        return model.generate(input=wav_paths if len(wav_paths) > 1 else wav_paths[0], **kwargs)
    except TypeError:
        kwargs.pop("batch_size_s", None)
        if args.backend == "vllm":
            return model.generate(wav_paths, **kwargs)
        return model.generate(input=wav_paths if len(wav_paths) > 1 else wav_paths[0], **kwargs)


def normalize_results(raw: Any, expected: int) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        results = [raw]
    elif isinstance(raw, list):
        results = raw
    else:
        results = [{"text": str(raw), "raw": raw}]
    normalized: list[dict[str, Any]] = []
    for item in results:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            normalized.append({"text": str(item), "raw": item})
    if len(normalized) == expected:
        return normalized
    if expected == 1 and normalized:
        return [normalized[0]]
    raise RuntimeError(f"model returned {len(normalized)} results for {expected} inputs")


def generate_chunk_group(model: Any, group: list[ChunkTask], args: argparse.Namespace) -> dict[ChunkTask, dict[str, Any]]:
    raw = call_generate(model, [task.wav_path for task in group], args)
    results = normalize_results(raw, len(group))
    return dict(zip(group, results))


def transcribe_chunks(
    model: Any,
    tasks: list[ChunkTask],
    args: argparse.Namespace,
) -> tuple[dict[ChunkTask, dict[str, Any]], dict[Job, BaseException]]:
    results: dict[ChunkTask, dict[str, Any]] = {}
    errors: dict[Job, BaseException] = {}
    group_index = 0
    cursor = 0
    effective_batch_size = max(1, args.batch_size)
    while cursor < len(tasks):
        group_index += 1
        group = tasks[cursor:cursor + effective_batch_size]
        try:
            results.update(generate_chunk_group(model, group, args))
            cursor += len(group)
            continue
        except Exception as exc:
            if isinstance(exc, NotImplementedError) and len(group) > 1:
                log(
                    f"[transcribe] chunk_batch={group_index} size={len(group)} "
                    f"unsupported={exc!r}; disabling chunk batching for this run"
                )
                effective_batch_size = 1
                continue
            if len(group) == 1:
                errors[group[0].job] = exc
                cursor += 1
                continue
            log(
                f"[transcribe] chunk_batch={group_index} size={len(group)} failed={exc!r}; "
                "retrying chunks individually"
            )
        for task in group:
            try:
                results.update(generate_chunk_group(model, [task], args))
            except Exception as exc:
                errors[task.job] = exc
        cursor += len(group)
    return results, errors


def transcript_text(result: dict[str, Any]) -> str:
    text = result.get("text")
    if isinstance(text, str):
        return text.strip()
    if isinstance(result.get("sentence_info"), list):
        parts = []
        for seg in result["sentence_info"]:
            if isinstance(seg, dict):
                sentence = seg.get("sentence") or seg.get("text")
                if sentence:
                    parts.append(str(sentence).strip())
        return "\n".join(part for part in parts if part)
    return json.dumps(result, ensure_ascii=False)


def aggregate_chunk_results(
    job: Job,
    chunk_results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    chunk_payloads = []
    texts = []
    for index, result in enumerate(chunk_results):
        text = transcript_text(result)
        if text:
            texts.append(text)
        chunk_payloads.append(
            {
                "index": index,
                "start_seconds": index * args.chunk_seconds,
                "end_seconds": (index + 1) * args.chunk_seconds,
                "text": text,
                "result": result,
            }
        )
    return {
        "text": "\n".join(texts),
        "chunk_seconds": args.chunk_seconds,
        "chunk_count": len(chunk_payloads),
        "chunks": chunk_payloads,
    }


def write_transcript(job: Job, result: dict[str, Any], meta: dict[str, Any], args: argparse.Namespace) -> None:
    text = transcript_text(result)
    durable_meta = {key: value for key, value in meta.items() if key not in {"wav_path", "chunk_paths"}}
    payload = {
        "status": "ok",
        "generated_at": utc_now(),
        "model": args.model,
        "backend": args.backend,
        "device": args.device,
        "channel": job.channel,
        "number": job.number,
        "title": job.title,
        "audio_path": rel(job.audio_path),
        "audio_bytes": job.audio_path.stat().st_size,
        "text": text,
        "result": result,
        "meta": durable_meta,
    }
    write_json_atomic(job.transcript_json, payload)
    job.transcript_txt.write_text(text + ("\n" if text else ""), encoding="utf-8")
    md = [
        f"# {job.number} {job.title}",
        "",
        f"- Channel: {job.channel}",
        f"- Model: {args.model}",
        f"- Backend: {args.backend}",
        f"- Device: {args.device}",
        f"- Audio: `{job.audio_path.name}`",
        f"- Generated at: {payload['generated_at']}",
        "",
        "## Transcript",
        "",
        text,
        "",
    ]
    job.transcript_md.write_text("\n".join(md), encoding="utf-8")


def write_error(job: Job, exc: BaseException, args: argparse.Namespace) -> None:
    payload = {
        "status": "error",
        "generated_at": utc_now(),
        "model": args.model,
        "backend": args.backend,
        "device": args.device,
        "channel": job.channel,
        "number": job.number,
        "title": job.title,
        "audio_path": rel(job.audio_path),
        "error": repr(exc),
    }
    write_json_atomic(job.transcript_json, payload)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_manifest(records: list[dict[str, Any]]) -> None:
    MANIFEST_FILE.parent.mkdir(exist_ok=True)
    with MANIFEST_FILE.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(*, jobs_total: int, skipped: list[dict[str, Any]], processed: int, ok: int, errors: int) -> None:
    channels = archive.read_channels()
    all_episodes = 0
    transcript_ok = 0
    transcript_error = 0
    missing_audio = 0
    pending = 0
    for channel in channels:
        for episode in archive.load_saved_episodes(channel):
            all_episodes += 1
            episode_dir = archive.episode_dir(channel, episode)
            audio_path = archive.audio_path_for(channel, episode)
            transcript_path = episode_dir / "transcript.json"
            if not audio_path.exists() or audio_path.stat().st_size <= 0:
                missing_audio += 1
                continue
            if transcript_path.exists():
                status = read_transcript_status(transcript_path)
                if status == "ok":
                    transcript_ok += 1
                elif status == "error":
                    transcript_error += 1
                else:
                    pending += 1
            else:
                pending += 1
    report = {
        "generated_at": utc_now(),
        "episodes": all_episodes,
        "transcript_ok": transcript_ok,
        "transcript_error": transcript_error,
        "missing_audio": missing_audio,
        "pending": pending,
        "this_run": {
            "jobs_selected": jobs_total,
            "processed": processed,
            "ok": ok,
            "errors": errors,
            "skipped": skipped,
        },
    }
    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_only() -> int:
    write_report(jobs_total=0, skipped=[], processed=0, ok=0, errors=0)
    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    log(
        "[transcribe-verify] "
        + " ".join(
            f"{key}={report[key]}"
            for key in ["episodes", "transcript_ok", "transcript_error", "missing_audio", "pending"]
        )
        + f" report={REPORT_FILE.relative_to(ROOT)}"
    )
    return 0 if report["pending"] == 0 and report["transcript_error"] == 0 and report["missing_audio"] == 0 else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["auto", "vllm"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--batch-size", type=int, default=4, help="audio chunks per model.generate call and AutoModel batch")
    parser.add_argument("--batch-size-s", type=int, default=None, help="FunASR batch_size_s for non-vLLM backend")
    parser.add_argument("--chunk-seconds", type=int, default=30, help="seconds per audio chunk before ASR")
    parser.add_argument("--max-length", type=int, default=256, help="FunASR max_length per chunk")
    parser.add_argument("--job-batch-size", type=int, default=2, help="episode files converted per outer batch")
    parser.add_argument("--decode-workers", type=int, default=2, help="parallel ffmpeg conversions per batch")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--vllm-dtype",
        default="bfloat16",
        help="AutoModelVLLM dtype; use bfloat16 on Ampere-or-newer GPUs",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--channel", action="append", help="only transcribe this channel; can be repeated")
    parser.add_argument("--episode-dir", action="append", help="only transcribe this episode directory; can be repeated")
    parser.add_argument("--limit", type=int, help="limit selected jobs for smoke tests")
    parser.add_argument("--sort", choices=["archive", "path", "size", "channel"], default="archive")
    parser.add_argument("--include-done", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--temp-dir", type=Path, default=ROOT / "_metadata" / "transcription_tmp")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    if args.verify_only:
        return verify_only()
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be positive")
    if args.job_batch_size <= 0:
        raise ValueError("--job-batch-size must be positive")
    if args.backend == "auto":
        args.backend = "auto_model"

    channel_filter = set(args.channel) if args.channel else None
    episode_dir_filter = {Path(path).resolve() for path in args.episode_dir} if args.episode_dir else None
    jobs, skipped = load_jobs(
        channel_filter=channel_filter,
        episode_dir_filter=episode_dir_filter,
        include_done=args.include_done,
        retry_errors=args.retry_errors,
    )
    jobs = sort_jobs(jobs, args.sort)
    jobs = shard_jobs(jobs, shard_index=args.shard_index, num_shards=args.num_shards)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    log(
        f"[transcribe] selected={len(jobs)} skipped_missing_or_empty={len(skipped)} "
        f"backend={args.backend} device={args.device} sort={args.sort}"
    )
    if not jobs:
        write_report(jobs_total=0, skipped=skipped, processed=0, ok=0, errors=0)
        return verify_only()

    ffmpeg = find_ffmpeg()
    log(f"[transcribe] ffmpeg={ffmpeg}")
    started = time.time()
    model = create_model(args)
    log(f"[transcribe] model_loaded seconds={time.time() - started:.2f}")

    processed = 0
    ok = 0
    errors = 0
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    for batch_index, batch in enumerate(chunks(jobs, args.job_batch_size), start=1):
        batch_started = time.time()
        records: list[dict[str, Any]] = []
        log(
            f"[transcribe] batch={batch_index} jobs={len(batch)} "
            f"chunk_seconds={args.chunk_seconds} chunk_batch_size={args.batch_size}"
        )
        for job in batch:
            log(f"[transcribe] batch={batch_index} job={rel(job.audio_path)} bytes={job.audio_path.stat().st_size}")
        with tempfile.TemporaryDirectory(prefix="batch-", dir=args.temp_dir) as temp_name:
            temp_dir = Path(temp_name)
            try:
                conversion, conversion_errors = convert_batch(ffmpeg, batch, temp_dir, args.decode_workers, args.chunk_seconds)
                for job, exc in conversion_errors.items():
                    log(f"[transcribe] batch={batch_index} conversion_error job={rel(job.audio_path)} error={exc!r}")
                tasks = [
                    ChunkTask(job=job, index=index, wav_path=wav_path)
                    for job in batch if job in conversion
                    for index, wav_path in enumerate(conversion[job]["chunk_paths"])
                ]
                log(f"[transcribe] batch={batch_index} chunks={len(tasks)}")
                if tasks:
                    chunk_result_map, chunk_errors = transcribe_chunks(model, tasks, args)
                else:
                    chunk_result_map, chunk_errors = {}, {}
                for job in batch:
                    job_error = conversion_errors.get(job)
                    if job not in conversion and job_error is None:
                        job_error = RuntimeError("conversion produced no output")
                    job_tasks = [task for task in tasks if task.job == job]
                    missing = [task.index for task in job_tasks if task not in chunk_result_map]
                    if job_error is None:
                        job_error = chunk_errors.get(job)
                    if missing and job_error is None:
                        job_error = RuntimeError(f"missing chunk results for indexes {missing}")
                    if job_error is not None:
                        write_error(job, job_error, args)
                        errors += 1
                        processed += 1
                        records.append(
                            {
                                "time": utc_now(),
                                "status": "error",
                                "path": rel(job.transcript_json),
                                "error": repr(job_error),
                            }
                        )
                        continue
                    chunk_results = [chunk_result_map[task] for task in sorted(job_tasks, key=lambda task: task.index)]
                    result = aggregate_chunk_results(job, chunk_results, args)
                    meta = conversion[job] | {"batch_index": batch_index}
                    write_transcript(job, result, meta, args)
                    ok += 1
                    processed += 1
                    records.append({"time": utc_now(), "status": "ok", "path": rel(job.transcript_json)})
            except Exception as exc:
                for job in batch:
                    write_error(job, exc, args)
                    errors += 1
                    processed += 1
                    records.append({"time": utc_now(), "status": "error", "path": rel(job.transcript_json), "error": repr(exc)})
            append_manifest(records)
        elapsed = time.time() - batch_started
        log(f"[transcribe] batch={batch_index} processed={processed}/{len(jobs)} ok={ok} errors={errors} seconds={elapsed:.2f}")
        write_report(jobs_total=len(jobs), skipped=skipped, processed=processed, ok=ok, errors=errors)
    write_report(jobs_total=len(jobs), skipped=skipped, processed=processed, ok=ok, errors=errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
