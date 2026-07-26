#!/usr/bin/env python3
"""Add speaker labels to existing FunASR podcast transcripts."""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import archive_xiaoyuzhou_channels as archive


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = Path(
    os.environ.get("PODTRANS_DATA_DIR", PROJECT_ROOT / "data")
).expanduser().resolve()
DEFAULT_SPEAKER_MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
DEFAULT_VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
DEFAULT_OUTPUT_PREFIX = "transcript_speakers"
DEFAULT_SPEAKER_BATCH_SIZE = 512
DEFAULT_MERGE_THRESHOLD = 0.60
MIN_VAD_CHUNK_SECONDS = 0.1
REPORT_FILE = ROOT / "_metadata" / "speaker_diarization_report.json"
MANIFEST_FILE = ROOT / "_metadata" / "speaker_diarization_manifest.jsonl"
SPEAKER_SERVICE_AUTHKEY = b"podtrans-speaker-embedding-v1"


@dataclass(frozen=True)
class Job:
    channel: str
    number: str
    title: str
    episode_dir: Path
    audio_path: Path
    transcript_json: Path
    output_json: Path
    output_txt: Path
    output_md: Path


@dataclass(frozen=True)
class AudioChunk:
    index: int
    wav_path: Path
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class DecodedEpisode:
    chunks: tuple[AudioChunk, ...]
    duration_seconds: float
    decode_seconds: float
    wav_bytes: int


@dataclass(frozen=True)
class WindowTask:
    job: Job
    start_seconds: float
    end_seconds: float
    samples: np.ndarray


@dataclass(frozen=True)
class TimedToken:
    text: str
    start_ms: int
    end_ms: int
    separator: bool = False


@dataclass(frozen=True)
class LabeledToken:
    text: str
    start_ms: int
    end_ms: int
    speaker: int
    separator: bool = False


class SpeakerEmbeddingClient:
    def __init__(self, socket_path: Path, timeout_seconds: float) -> None:
        from multiprocessing.connection import Client

        self.socket_path = socket_path
        self.timeout_seconds = timeout_seconds
        self.client_type = Client
        self.health = self._request({"op": "health"})
        self.peak_allocated_mb = 0.0
        self.peak_reserved_mb = 0.0
        self.max_service_batch_size = 0
        self.requests = 0

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        connection = self.client_type(
            str(self.socket_path),
            family="AF_UNIX",
            authkey=SPEAKER_SERVICE_AUTHKEY,
        )
        try:
            connection.send(payload)
            if not connection.poll(self.timeout_seconds):
                raise TimeoutError(
                    f"speaker service did not respond within {self.timeout_seconds}s"
                )
            response = connection.recv()
        finally:
            connection.close()
        if not isinstance(response, dict):
            raise RuntimeError(
                f"invalid speaker service response: {type(response)!r}"
            )
        if response.get("status") != "ok":
            raise RuntimeError(
                f"speaker service request failed: {response.get('error')}"
            )
        return response

    def infer(self, arrays: list[np.ndarray]) -> np.ndarray:
        shapes = {tuple(array.shape) for array in arrays}
        if len(shapes) == 1:
            audio: np.ndarray | list[np.ndarray] = np.stack(arrays).astype(
                np.float32,
                copy=False,
            )
        else:
            audio = [
                np.asarray(array, dtype=np.float32).reshape(-1)
                for array in arrays
            ]
        response = self._request({"op": "embed", "audio": audio})
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, np.ndarray):
            raise RuntimeError("speaker service returned no embedding array")
        metrics = response.get("metrics", {})
        if isinstance(metrics, dict):
            self.peak_allocated_mb = max(
                self.peak_allocated_mb,
                float(metrics.get("gpu_peak_allocated_mb", 0.0)),
            )
            self.peak_reserved_mb = max(
                self.peak_reserved_mb,
                float(metrics.get("gpu_peak_reserved_mb", 0.0)),
            )
            self.max_service_batch_size = max(
                self.max_service_batch_size,
                int(metrics.get("batch_size", 0)),
            )
        self.requests += 1
        return embeddings.astype(np.float32, copy=False)

    def consume_metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "gpu_peak_allocated_mb": round(self.peak_allocated_mb, 1),
            "gpu_peak_reserved_mb": round(self.peak_reserved_mb, 1),
            "speaker_service_max_batch_size": self.max_service_batch_size,
            "speaker_service_requests": self.requests,
        }
        self.peak_allocated_mb = 0.0
        self.peak_reserved_mb = 0.0
        self.max_service_batch_size = 0
        self.requests = 0
        return metrics


def log(message: str) -> None:
    print(message, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json_status(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(40):
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
        return None
    return None


def output_is_current(job: Job) -> bool:
    if read_json_status(job.output_json) != "ok":
        return False
    try:
        with job.output_json.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        source = payload.get("asr_source", {})
        stat = job.transcript_json.stat()
        return (
            source.get("bytes") == stat.st_size
            and source.get("mtime_ns") == stat.st_mtime_ns
        )
    except Exception:
        return False


def output_paths(episode_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    return (
        episode_dir / f"{prefix}.json",
        episode_dir / f"{prefix}.txt",
        episode_dir / f"{prefix}.md",
    )


def load_jobs(
    *,
    channel_filter: set[str] | None,
    episode_dir_filter: set[Path] | None,
    output_prefix: str,
    include_done: bool,
    retry_errors: bool,
) -> tuple[list[Job], list[dict[str, Any]]]:
    jobs: list[Job] = []
    skipped: list[dict[str, Any]] = []
    for channel in archive.read_channels():
        if channel_filter and channel not in channel_filter:
            continue
        for episode in archive.load_saved_episodes(channel):
            episode_dir = archive.episode_dir(channel, episode)
            if episode_dir_filter and episode_dir.resolve() not in episode_dir_filter:
                continue
            audio_path = archive.audio_path_for(channel, episode)
            transcript_json = episode_dir / "transcript.json"
            output_json, output_txt, output_md = output_paths(episode_dir, output_prefix)
            job = Job(
                channel=channel,
                number=episode.number,
                title=episode.title,
                episode_dir=episode_dir,
                audio_path=audio_path,
                transcript_json=transcript_json,
                output_json=output_json,
                output_txt=output_txt,
                output_md=output_md,
            )
            if not audio_path.exists() or audio_path.stat().st_size <= 0:
                skipped.append(
                    {
                        "status": "missing_audio",
                        "channel": channel,
                        "number": episode.number,
                        "path": rel(audio_path),
                    }
                )
                continue
            transcript_status = read_json_status(transcript_json)
            if transcript_status != "ok":
                skipped.append(
                    {
                        "status": "missing_or_invalid_transcript",
                        "channel": channel,
                        "number": episode.number,
                        "path": rel(transcript_json),
                        "transcript_status": transcript_status,
                    }
                )
                continue
            if not include_done and output_is_current(job):
                continue
            if (
                not include_done
                and not retry_errors
                and read_json_status(output_json) == "error"
            ):
                continue
            jobs.append(job)
    return jobs, skipped


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


def shard_jobs(jobs: list[Job], *, shard_index: int, num_shards: int) -> list[Job]:
    if num_shards <= 1:
        return jobs
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    selected: list[Job] = []
    for job in jobs:
        digest = hashlib.sha256(rel(job.audio_path).encode("utf-8")).digest()
        stable_index = int.from_bytes(digest[:8], byteorder="big") % num_shards
        if stable_index == shard_index:
            selected.append(job)
    return selected


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
    except Exception as exc:
        raise RuntimeError("ffmpeg is not on PATH and imageio-ffmpeg is unavailable") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def wav_info(path: Path) -> tuple[int, int, int, float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
    return channels, sample_width, sample_rate, frames / sample_rate


def decode_to_chunks(
    ffmpeg: str,
    job: Job,
    output_dir: Path,
    chunk_seconds: int,
) -> DecodedEpisode:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "chunk-%05d.wav"
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(job.audio_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(pattern),
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffmpeg exited {proc.returncode}")
    paths = sorted(output_dir.glob("chunk-*.wav"))
    if not paths:
        raise RuntimeError("ffmpeg produced no audio chunks")
    chunks_out: list[AudioChunk] = []
    offset = 0.0
    for index, path in enumerate(paths):
        channels, sample_width, sample_rate, duration = wav_info(path)
        if (channels, sample_width, sample_rate) != (1, 2, 16000):
            raise RuntimeError(
                f"unexpected WAV format for {path}: "
                f"channels={channels} width={sample_width} rate={sample_rate}"
            )
        chunks_out.append(
            AudioChunk(
                index=index,
                wav_path=path,
                start_seconds=offset,
                duration_seconds=duration,
            )
        )
        offset += duration
    return DecodedEpisode(
        chunks=tuple(chunks_out),
        duration_seconds=offset,
        decode_seconds=elapsed,
        wav_bytes=sum(path.stat().st_size for path in paths),
    )


def decode_batch(
    ffmpeg: str,
    jobs: list[Job],
    temp_dir: Path,
    workers: int,
    chunk_seconds: int,
) -> tuple[dict[Job, DecodedEpisode], dict[Job, BaseException]]:
    decoded: dict[Job, DecodedEpisode] = {}
    errors: dict[Job, BaseException] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                decode_to_chunks,
                ffmpeg,
                job,
                temp_dir / f"job-{index:04d}",
                chunk_seconds,
            ): job
            for index, job in enumerate(jobs)
        }
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                decoded[job] = future.result()
            except Exception as exc:
                errors[job] = exc
    return decoded, errors


def read_pcm16_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise RuntimeError(f"unsupported decoded WAV format: {path}")
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def cached_model_path(model_id: str) -> Path | None:
    if Path(model_id).exists():
        return Path(model_id).resolve()
    candidates: list[Path] = []
    modelscope_cache = os.environ.get("MODELSCOPE_CACHE")
    if modelscope_cache:
        candidates.append(Path(modelscope_cache) / model_id)
    candidates.append(Path.home() / ".cache" / "modelscope" / model_id)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def auto_model_kwargs(
    model_id: str,
    *,
    model_path: Path | None,
    hub: str,
    device: str,
    cpu_threads: int,
) -> dict[str, Any]:
    resolved_path = model_path.resolve() if model_path else cached_model_path(model_id)
    kwargs: dict[str, Any] = {
        "model": model_id,
        "hub": hub,
        "device": device,
        "ncpu": cpu_threads,
        "disable_update": True,
        "disable_pbar": True,
        "check_latest": False,
    }
    if resolved_path is not None:
        kwargs["model_path"] = str(resolved_path)
    return kwargs


def load_models(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from funasr import AutoModel
    from funasr.models.campplus.cluster_backend import ClusterBackend

    remote_speaker = args.speaker_service_socket is not None
    if not remote_speaker and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for --device {args.device}")
    if not remote_speaker and args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
    vad_device = args.vad_device or args.device
    vad_kwargs = auto_model_kwargs(
        args.vad_model,
        model_path=args.vad_model_path,
        hub=args.hub,
        device=vad_device,
        cpu_threads=args.cpu_threads,
    )
    vad_kwargs["max_single_segment_time"] = args.vad_max_segment_ms
    started = time.perf_counter()
    vad_model = AutoModel(**vad_kwargs)
    log(
        f"[diarize] vad_model_loaded device={vad_device} "
        f"seconds={time.perf_counter() - started:.2f}"
    )

    if remote_speaker:
        started = time.perf_counter()
        speaker_model = SpeakerEmbeddingClient(
            args.speaker_service_socket,
            args.speaker_service_timeout,
        )
        service = speaker_model.health.get("service", {})
        if service.get("speaker_model") != args.speaker_model:
            raise RuntimeError(
                "speaker service model mismatch: "
                f"{service.get('speaker_model')!r} != {args.speaker_model!r}"
            )
        if service.get("speaker_model_precision") != "fp32":
            raise RuntimeError("speaker service is not using strict FP32")
        if service.get("allow_tf32") is not False:
            raise RuntimeError("speaker service unexpectedly enables TF32")
        if int(service.get("max_batch_size", 0)) < args.speaker_batch_size:
            raise RuntimeError(
                "speaker service max batch is smaller than --speaker-batch-size"
            )
        log(
            f"[diarize] speaker_service_connected socket={args.speaker_service_socket} "
            f"pid={service.get('pid')} seconds={time.perf_counter() - started:.2f}"
        )
    else:
        speaker_kwargs = auto_model_kwargs(
            args.speaker_model,
            model_path=args.speaker_model_path,
            hub=args.hub,
            device=args.device,
            cpu_threads=args.cpu_threads,
        )
        started = time.perf_counter()
        speaker_model = AutoModel(**speaker_kwargs)
        log(
            f"[diarize] speaker_model_loaded seconds={time.perf_counter() - started:.2f}"
        )
    cluster_model = ClusterBackend(merge_thr=args.merge_threshold)
    return vad_model, speaker_model, cluster_model


def cluster_backend_name(window_count: int, preset_speakers: int | None) -> str:
    if window_count < 20:
        return "FunASR ClusterBackend single-speaker shortcut"
    if window_count < 2048 or preset_speakers is not None:
        return "FunASR ClusterBackend spectral clustering"
    return "FunASR ClusterBackend UMAP + HDBSCAN"


def normalize_vad_segments(
    raw_segments: Any,
    *,
    duration_seconds: float,
) -> list[tuple[float, float]]:
    if not isinstance(raw_segments, list):
        raise RuntimeError(f"unexpected VAD segments type: {type(raw_segments)!r}")
    duration_ms = int(round(duration_seconds * 1000))
    normalized: list[tuple[float, float]] = []
    for item in raw_segments:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start_ms = max(0, min(int(item[0]), duration_ms))
        end_ms = max(start_ms, min(int(item[1]), duration_ms))
        if end_ms <= start_ms:
            continue
        normalized.append((start_ms / 1000.0, end_ms / 1000.0))
    return normalized


def run_vad(vad_model: Any, chunk: AudioChunk, args: argparse.Namespace) -> list[tuple[float, float]]:
    results = vad_model.generate(
        input=str(chunk.wav_path),
        batch_size=1,
        max_single_segment_time=args.vad_max_segment_ms,
    )
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError(f"VAD returned {type(results)!r} with length {len(results)}")
    return normalize_vad_segments(
        results[0].get("value", []),
        duration_seconds=chunk.duration_seconds,
    )


def infer_speaker_embeddings(speaker_model: Any, tasks: list[WindowTask], device: str) -> np.ndarray:
    import torch

    if not tasks:
        return np.empty((0, 192), dtype=np.float32)
    arrays = [task.samples for task in tasks]
    if isinstance(speaker_model, SpeakerEmbeddingClient):
        embeddings = speaker_model.infer(arrays)
    else:
        with torch.inference_mode():
            results, _meta = speaker_model.model.inference(
                data_in=arrays,
                key=[f"window-{index}" for index in range(len(arrays))],
                device=device,
                fs=16000,
            )
        tensors = [item["spk_embedding"] for item in results]
        embeddings = torch.cat(tensors, dim=0).detach().float().cpu().numpy()
    if embeddings.shape[0] != len(tasks):
        raise RuntimeError(
            f"speaker model returned {embeddings.shape[0]} embeddings for {len(tasks)} windows"
        )
    return np.asarray(embeddings, dtype=np.float32)


def append_embedding_batch(
    speaker_model: Any,
    pending: list[WindowTask],
    embeddings_by_job: dict[Job, list[np.ndarray]],
    device: str,
) -> None:
    if not pending:
        return
    try:
        embeddings = infer_speaker_embeddings(speaker_model, pending, device)
    except Exception:
        import torch

        if len(pending) <= 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        midpoint = len(pending) // 2
        append_embedding_batch(
            speaker_model,
            pending[:midpoint],
            embeddings_by_job,
            device,
        )
        append_embedding_batch(
            speaker_model,
            pending[midpoint:],
            embeddings_by_job,
            device,
        )
        return
    for task, embedding in zip(pending, embeddings):
        embeddings_by_job.setdefault(task.job, []).append(embedding)


def diarize_decoded_batch(
    decoded: dict[Job, DecodedEpisode],
    vad_model: Any,
    speaker_model: Any,
    cluster_model: Any,
    args: argparse.Namespace,
) -> tuple[dict[Job, dict[str, Any]], dict[Job, BaseException]]:
    from funasr.models.campplus.utils import postprocess, sv_chunk

    windows_by_job: dict[Job, list[tuple[float, float]]] = {job: [] for job in decoded}
    embeddings_by_job: dict[Job, list[np.ndarray]] = {job: [] for job in decoded}
    vad_stats: dict[Job, dict[str, float | int]] = {
        job: {
            "segments": 0,
            "speech_seconds": 0.0,
            "short_chunks_skipped": 0,
        }
        for job in decoded
    }
    errors: dict[Job, BaseException] = {}
    pending: list[WindowTask] = []

    def flush_pending(force: bool = False) -> None:
        nonlocal pending
        size = len(pending) if force else args.speaker_batch_size
        while pending and (force or len(pending) >= args.speaker_batch_size):
            batch = pending[:size]
            del pending[:size]
            append_embedding_batch(
                speaker_model,
                batch,
                embeddings_by_job,
                args.device,
            )
            if not force:
                size = args.speaker_batch_size

    for job, decoded_episode in decoded.items():
        try:
            for chunk in decoded_episode.chunks:
                if chunk.duration_seconds < MIN_VAD_CHUNK_SECONDS:
                    vad_stats[job]["short_chunks_skipped"] = (
                        int(vad_stats[job]["short_chunks_skipped"]) + 1
                    )
                    continue
                local_vad = run_vad(vad_model, chunk, args)
                audio = read_pcm16_wav(chunk.wav_path)
                for local_start, local_end in local_vad:
                    start_sample = max(0, min(int(round(local_start * 16000)), len(audio)))
                    end_sample = max(start_sample, min(int(round(local_end * 16000)), len(audio)))
                    if end_sample <= start_sample:
                        continue
                    global_start = chunk.start_seconds + start_sample / 16000.0
                    global_end = chunk.start_seconds + end_sample / 16000.0
                    vad_stats[job]["segments"] = int(vad_stats[job]["segments"]) + 1
                    vad_stats[job]["speech_seconds"] = (
                        float(vad_stats[job]["speech_seconds"]) + global_end - global_start
                    )
                    speaker_windows = sv_chunk(
                        [[global_start, global_end, audio[start_sample:end_sample]]]
                    )
                    for window_start, window_end, samples in speaker_windows:
                        windows_by_job[job].append((float(window_start), float(window_end)))
                        pending.append(
                            WindowTask(
                                job=job,
                                start_seconds=float(window_start),
                                end_seconds=float(window_end),
                                samples=np.asarray(samples, dtype=np.float32),
                            )
                        )
                        flush_pending()
                del audio
        except Exception as exc:
            errors[job] = exc
            pending = [task for task in pending if task.job != job]
            windows_by_job[job].clear()
            embeddings_by_job[job].clear()

    flush_pending(force=True)
    results: dict[Job, dict[str, Any]] = {}
    for job, decoded_episode in decoded.items():
        if job in errors:
            continue
        try:
            windows = windows_by_job[job]
            embedding_rows = embeddings_by_job[job]
            if len(windows) != len(embedding_rows):
                raise RuntimeError(
                    f"window/embedding mismatch: {len(windows)} != {len(embedding_rows)}"
                )
            if not windows:
                timeline: list[list[float | int]] = []
                speaker_count = 0
            else:
                embeddings = np.stack(embedding_rows).astype(np.float32, copy=False)
                labels = cluster_model(
                    embeddings,
                    oracle_num=args.preset_speakers,
                )
                segment_rows = [[start, end, None] for start, end in windows]
                timeline = postprocess(segment_rows, None, labels, embeddings)
                speaker_count = len({int(item[2]) for item in timeline})
            results[job] = {
                "timeline": timeline,
                "cluster_backend": cluster_backend_name(
                    len(windows),
                    args.preset_speakers,
                ),
                "speaker_count": speaker_count,
                "window_count": len(windows),
                "vad_segment_count": int(vad_stats[job]["segments"]),
                "vad_short_chunks_skipped": int(
                    vad_stats[job]["short_chunks_skipped"]
                ),
                "speech_seconds": round(float(vad_stats[job]["speech_seconds"]), 3),
                "audio_duration_seconds": round(decoded_episode.duration_seconds, 3),
                "decode_seconds": round(decoded_episode.decode_seconds, 3),
                "decoded_wav_bytes": decoded_episode.wav_bytes,
            }
        except Exception as exc:
            errors[job] = exc
    return results, errors


def interpolate_chunk_tokens(
    chunk_text: str,
    timestamps: list[Any],
    *,
    chunk_start_ms: int,
    chunk_end_ms: int,
    stats: dict[str, int] | None,
) -> list[TimedToken]:
    source_tokens: list[TimedToken] = []
    for item in timestamps:
        if not isinstance(item, dict):
            continue
        token_text = str(item.get("token", ""))
        if not token_text:
            continue
        start_ms = chunk_start_ms + int(round(float(item.get("start_time", 0.0)) * 1000))
        end_ms = chunk_start_ms + int(round(float(item.get("end_time", 0.0)) * 1000))
        end_ms = max(start_ms, end_ms)
        width = max(end_ms - start_ms, 0)
        for index, char in enumerate(token_text):
            char_start = start_ms + round(width * index / len(token_text))
            char_end = start_ms + round(width * (index + 1) / len(token_text))
            source_tokens.append(TimedToken(char, char_start, max(char_start, char_end)))

    source_text = "".join(token.text for token in source_tokens)
    matcher = difflib.SequenceMatcher(a=source_text, b=chunk_text, autojunk=False)
    mapped: dict[int, TimedToken] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapped[block.b + offset] = source_tokens[block.a + offset]

    fallback_count = len(chunk_text) - len(mapped)
    if stats is not None:
        stats["synthetic_timestamp_tokens"] = (
            stats.get("synthetic_timestamp_tokens", 0) + fallback_count
        )
        if fallback_count:
            stats["chunks_with_timestamp_fallback"] = (
                stats.get("chunks_with_timestamp_fallback", 0) + 1
            )

    target_tokens: list[TimedToken] = []
    mapped_indexes = sorted(mapped)
    for index, char in enumerate(chunk_text):
        if index in mapped:
            token = mapped[index]
            target_tokens.append(TimedToken(char, token.start_ms, token.end_ms))
            continue
        previous = max((item for item in mapped_indexes if item < index), default=None)
        following = min((item for item in mapped_indexes if item > index), default=None)
        if previous is not None and following is not None:
            gap_start = mapped[previous].end_ms
            gap_end = max(gap_start, mapped[following].start_ms)
            position = index - previous
            gap_size = following - previous
        elif previous is not None:
            gap_start = mapped[previous].end_ms
            gap_end = max(gap_start, chunk_end_ms)
            position = index - previous
            gap_size = len(chunk_text) - previous
        elif following is not None:
            gap_start = chunk_start_ms
            gap_end = max(gap_start, mapped[following].start_ms)
            position = index + 1
            gap_size = following + 1
        else:
            gap_start = chunk_start_ms
            gap_end = max(gap_start, chunk_end_ms)
            position = index + 1
            gap_size = max(len(chunk_text), 1)
        width = gap_end - gap_start
        start_ms = gap_start + round(width * (position - 1) / gap_size)
        end_ms = gap_start + round(width * position / gap_size)
        target_tokens.append(TimedToken(char, start_ms, max(start_ms, end_ms)))
    return target_tokens


def extract_timed_tokens(
    payload: dict[str, Any],
    *,
    audio_duration_seconds: float | None = None,
    stats: dict[str, int] | None = None,
) -> list[TimedToken]:
    source_text = payload.get("text")
    if not isinstance(source_text, str):
        raise RuntimeError("transcript has no top-level text")
    result = payload.get("result")
    chunks = result.get("chunks") if isinstance(result, dict) else None
    if not isinstance(chunks, list):
        timestamps = result.get("timestamps") if isinstance(result, dict) else None
        result_text = result.get("text") if isinstance(result, dict) else None
        if not isinstance(result_text, str):
            raise RuntimeError(
                "transcript has neither result.chunks nor flat result timestamps"
            )
        if not isinstance(timestamps, list):
            timestamps = []
        if audio_duration_seconds is None:
            audio_duration_seconds = max(
                (
                    float(item.get("end_time", 0.0))
                    for item in timestamps
                    if isinstance(item, dict)
                ),
                default=0.0,
            )
        chunks = [
            {
                "index": 0,
                "start_seconds": 0.0,
                "end_seconds": audio_duration_seconds,
                "text": result_text,
                "result": result,
            }
        ]
        if stats is not None:
            stats["flat_result_transcript"] = 1
    timed_tokens: list[TimedToken] = []
    reconstructed_parts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_text = chunk.get("text")
        if not isinstance(chunk_text, str) or not chunk_text:
            continue
        raw_result = chunk.get("result")
        timestamps = raw_result.get("timestamps") if isinstance(raw_result, dict) else None
        if not isinstance(timestamps, list):
            timestamps = []
            if stats is not None:
                stats["chunks_without_token_timestamps"] = (
                    stats.get("chunks_without_token_timestamps", 0) + 1
                )
        token_text = "".join(
            str(item.get("token", ""))
            for item in timestamps
            if isinstance(item, dict)
        )
        if reconstructed_parts:
            boundary_ms = int(round(float(chunk.get("start_seconds", 0.0)) * 1000))
            if audio_duration_seconds is not None:
                boundary_ms = min(boundary_ms, int(round(audio_duration_seconds * 1000)))
            timed_tokens.append(
                TimedToken(
                    text="\n",
                    start_ms=boundary_ms,
                    end_ms=boundary_ms,
                    separator=True,
                )
            )
        reconstructed_parts.append(chunk_text)
        offset_ms = int(round(float(chunk.get("start_seconds", 0.0)) * 1000))
        chunk_end_ms = int(round(float(chunk.get("end_seconds", 0.0)) * 1000))
        if audio_duration_seconds is not None:
            audio_end_ms = int(round(audio_duration_seconds * 1000))
            offset_ms = min(offset_ms, audio_end_ms)
            chunk_end_ms = min(chunk_end_ms, audio_end_ms)
        if token_text == chunk_text:
            for item in timestamps:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("token", ""))
                start_ms = offset_ms + int(round(float(item.get("start_time", 0.0)) * 1000))
                end_ms = offset_ms + int(round(float(item.get("end_time", 0.0)) * 1000))
                if audio_duration_seconds is not None:
                    start_ms = min(start_ms, audio_end_ms)
                    end_ms = min(end_ms, audio_end_ms)
                timed_tokens.append(
                    TimedToken(
                        text=text,
                        start_ms=start_ms,
                        end_ms=max(start_ms, end_ms),
                    )
                )
        else:
            timed_tokens.extend(
                interpolate_chunk_tokens(
                    chunk_text,
                    timestamps,
                    chunk_start_ms=offset_ms,
                    chunk_end_ms=max(offset_ms, chunk_end_ms),
                    stats=stats,
                )
            )
    reconstructed = "\n".join(reconstructed_parts)
    if reconstructed != source_text:
        raise RuntimeError(
            f"aggregate transcript text mismatch: {len(reconstructed)} != {len(source_text)} characters"
        )
    return timed_tokens


def label_for_interval(
    start_ms: int,
    end_ms: int,
    timeline_ms: list[tuple[int, int, int]],
    cursor: int,
) -> tuple[int, int]:
    if not timeline_ms:
        return 0, cursor
    while cursor + 1 < len(timeline_ms) and timeline_ms[cursor][1] <= start_ms:
        cursor += 1
    best_speaker = timeline_ms[cursor][2]
    best_overlap = -1
    scan_start = max(0, cursor - 1)
    scan_end = scan_start
    while scan_end < len(timeline_ms) and timeline_ms[scan_end][0] <= max(end_ms, start_ms):
        seg_start, seg_end, speaker = timeline_ms[scan_end]
        overlap = max(0, min(end_ms, seg_end) - max(start_ms, seg_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
        scan_end += 1
    if best_overlap <= 0:
        midpoint = (start_ms + end_ms) / 2.0
        candidates = range(max(0, cursor - 1), min(len(timeline_ms), cursor + 2))
        best_speaker = min(
            candidates,
            key=lambda index: min(
                abs(midpoint - timeline_ms[index][0]),
                abs(midpoint - timeline_ms[index][1]),
            ),
        )
        best_speaker = timeline_ms[best_speaker][2]
    return int(best_speaker), cursor


def assign_speakers(
    tokens: list[TimedToken],
    timeline: list[list[float | int]],
) -> list[LabeledToken]:
    timeline_ms = [
        (int(round(float(start) * 1000)), int(round(float(end) * 1000)), int(speaker))
        for start, end, speaker in timeline
    ]
    labeled: list[LabeledToken] = []
    cursor = 0
    last_speaker = 0
    punctuation = set("。！？!?；;，,、：:")
    for token in tokens:
        if token.separator:
            speaker = last_speaker
        elif token.text in punctuation and labeled:
            speaker = last_speaker
        else:
            speaker, cursor = label_for_interval(
                token.start_ms,
                token.end_ms,
                timeline_ms,
                cursor,
            )
            last_speaker = speaker
        labeled.append(
            LabeledToken(
                text=token.text,
                start_ms=token.start_ms,
                end_ms=token.end_ms,
                speaker=speaker,
                separator=token.separator,
            )
        )
    return labeled


def snap_speaker_timeline(
    tokens: list[TimedToken],
    timeline: list[list[float | int]],
    *,
    max_shift_ms: int,
) -> tuple[list[list[float | int]], int]:
    if max_shift_ms <= 0 or len(timeline) < 2 or not tokens:
        return timeline, 0
    sentence_end = set("。！？!?；;")
    weak_punctuation = set("，,、：:")
    candidates: list[tuple[int, int]] = []
    for index, token in enumerate(tokens):
        if token.separator:
            candidates.append((token.start_ms, 0))
        if index + 1 >= len(tokens):
            continue
        following = tokens[index + 1]
        boundary = (token.end_ms + following.start_ms) // 2
        gap_ms = max(0, following.start_ms - token.end_ms)
        if token.text in sentence_end:
            priority = 0
        elif gap_ms >= 180:
            priority = 1
        elif token.text in weak_punctuation:
            priority = 2
        else:
            continue
        candidates.append((boundary, priority))

    collapsed: list[list[float | int]] = []
    current_start = float(timeline[0][0])
    current_end = float(timeline[0][1])
    current_speaker = int(timeline[0][2])
    snapped_count = 0
    last_boundary_ms = int(round(current_start * 1000))
    for next_start_raw, next_end_raw, next_speaker_raw in timeline[1:]:
        next_start = float(next_start_raw)
        next_end = float(next_end_raw)
        next_speaker = int(next_speaker_raw)
        if next_speaker == current_speaker:
            current_end = max(current_end, next_end)
            continue
        if next_start > current_end:
            raw_boundary_ms = int(round((current_end + next_start) * 500))
        else:
            raw_boundary_ms = int(round(current_end * 1000))
        nearby = [
            (candidate_ms, priority)
            for candidate_ms, priority in candidates
            if abs(candidate_ms - raw_boundary_ms) <= max_shift_ms
            and candidate_ms > last_boundary_ms
        ]
        if nearby:
            snapped_ms, _priority = min(
                nearby,
                key=lambda item: (abs(item[0] - raw_boundary_ms) + item[1] * 200, item[1]),
            )
        else:
            snapped_ms = raw_boundary_ms
        snapped_ms = max(last_boundary_ms, snapped_ms)
        if snapped_ms != raw_boundary_ms:
            snapped_count += 1
        collapsed.append(
            [current_start, snapped_ms / 1000.0, current_speaker]
        )
        current_start = snapped_ms / 1000.0
        current_end = max(next_end, current_start)
        current_speaker = next_speaker
        last_boundary_ms = snapped_ms
    collapsed.append([current_start, current_end, current_speaker])
    return collapsed, snapped_count


def stabilize_speaker_units(
    tokens: list[LabeledToken],
    *,
    min_gap_ms: int,
) -> tuple[list[LabeledToken], int]:
    if not tokens:
        return [], 0
    sentence_end = set("。！？!?；;")
    punctuation = sentence_end | set("，,、：:")
    units: list[list[LabeledToken]] = []
    current: list[LabeledToken] = []
    for index, token in enumerate(tokens):
        current.append(token)
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        should_flush = following is None or token.separator
        if following is not None and not following.separator:
            if token.text in sentence_end:
                should_flush = True
            elif not token.separator:
                gap_ms = max(0, following.start_ms - token.end_ms)
                if gap_ms >= min_gap_ms:
                    should_flush = True
        if should_flush:
            units.append(current)
            current = []
    if current:
        units.append(current)

    stabilized: list[LabeledToken] = []
    changed = 0
    for unit in units:
        weights: dict[int, int] = {}
        for token in unit:
            if token.separator or token.text in punctuation:
                continue
            weight = max(token.end_ms - token.start_ms, 60)
            weights[token.speaker] = weights.get(token.speaker, 0) + weight
        if weights:
            dominant = max(weights, key=lambda speaker: (weights[speaker], -speaker))
        else:
            dominant = unit[0].speaker
        for token in unit:
            if token.speaker != dominant:
                changed += 1
            stabilized.append(
                LabeledToken(
                    text=token.text,
                    start_ms=token.start_ms,
                    end_ms=token.end_ms,
                    speaker=dominant,
                    separator=token.separator,
                )
            )
    return stabilized, changed


def build_text_segments(
    tokens: list[LabeledToken],
    *,
    max_seconds: float,
    max_chars: int,
) -> list[dict[str, Any]]:
    if not tokens:
        return []
    sentence_end = set("。！？!?；;")
    segments: list[dict[str, Any]] = []
    current: list[LabeledToken] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        timed = [token for token in current if not token.separator]
        if timed:
            start_ms = timed[0].start_ms
            end_ms = timed[-1].end_ms
        else:
            start_ms = current[0].start_ms
            end_ms = current[-1].end_ms
        segments.append(
            {
                "index": len(segments),
                "speaker": f"SPEAKER_{current[0].speaker:02d}",
                "speaker_id": current[0].speaker,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": "".join(token.text for token in current),
            }
        )
        current = []

    for token in tokens:
        if current and token.speaker != current[0].speaker and not token.separator:
            flush()
        current.append(token)
        timed = [item for item in current if not item.separator]
        if not timed:
            continue
        duration = (timed[-1].end_ms - timed[0].start_ms) / 1000.0
        char_count = sum(len(item.text) for item in current)
        at_sentence_end = token.text in sentence_end
        soft_limit = duration >= max_seconds or char_count >= max_chars
        hard_limit = duration >= max_seconds * 2 or char_count >= max_chars * 2
        if (soft_limit and at_sentence_end) or hard_limit:
            flush()
    flush()
    return segments


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_time_ms(value: int, *, millis: bool = True) -> str:
    value = max(0, value)
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    if millis:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def write_success(
    job: Job,
    transcript: dict[str, Any],
    diarization: dict[str, Any],
    segments: list[dict[str, Any]],
    elapsed_seconds: float,
    args: argparse.Namespace,
) -> None:
    source_text = transcript["text"]
    reconstructed = "".join(segment["text"] for segment in segments)
    if reconstructed != source_text:
        raise RuntimeError("speaker segments do not reconstruct the source transcript exactly")
    source_stat = job.transcript_json.stat()
    speaker_ids = sorted({segment["speaker_id"] for segment in segments})
    payload = {
        "status": "ok",
        "generated_at": utc_now(),
        "channel": job.channel,
        "number": job.number,
        "title": job.title,
        "audio_path": rel(job.audio_path),
        "asr_source": {
            "path": rel(job.transcript_json),
            "bytes": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "model": transcript.get("model"),
            "backend": transcript.get("backend"),
            "text_sha256": text_sha256(source_text),
        },
        "diarization": {
            "vad_model": args.vad_model,
            "speaker_model": args.speaker_model,
            "speaker_model_precision": "fp32",
            "speaker_runtime": (
                "unix_socket_service"
                if args.speaker_service_socket is not None
                else "in_process"
            ),
            "cluster_backend": diarization["cluster_backend"],
            "merge_threshold": args.merge_threshold,
            "preset_speakers": args.preset_speakers,
            "audio_chunk_seconds": args.audio_chunk_seconds,
            "speaker_window_seconds": 1.5,
            "speaker_window_shift_seconds": 0.75,
            "speaker_boundary_snap_ms": args.speaker_boundary_snap_ms,
            "speaker_unit_gap_ms": args.speaker_unit_gap_ms,
            "device": args.device,
            "allow_tf32": args.allow_tf32,
        },
        "metrics": {
            **{
                key: value
                for key, value in diarization.items()
                if key not in {"timeline", "cluster_backend"}
            },
            "output_segment_count": len(segments),
            "processing_seconds": round(elapsed_seconds, 3),
            "rtf": round(
                elapsed_seconds / max(float(diarization["audio_duration_seconds"]), 1e-6),
                6,
            ),
        },
        "speakers": [
            {"id": f"SPEAKER_{speaker_id:02d}", "label": f"说话人 {speaker_id + 1}"}
            for speaker_id in speaker_ids
        ],
        "text": source_text,
        "segments": segments,
    }
    write_json_atomic(job.output_json, payload)

    txt_lines = [
        (
            f"[{format_time_ms(segment['start_ms'])} - "
            f"{format_time_ms(segment['end_ms'])}] "
            f"说话人 {segment['speaker_id'] + 1}: {segment['text']}"
        )
        for segment in segments
    ]
    write_text_atomic(job.output_txt, "\n\n".join(txt_lines) + ("\n" if txt_lines else ""))

    md_lines = [
        f"# {job.number} {job.title}",
        "",
        f"- Channel: {job.channel}",
        f"- ASR model: {transcript.get('model')}",
        f"- ASR backend: {transcript.get('backend')}",
        f"- Speaker model: {args.speaker_model}",
        f"- Generated at: {payload['generated_at']}",
        "",
        "## Transcript",
        "",
    ]
    for segment in segments:
        md_lines.extend(
            [
                (
                    f"**[{format_time_ms(segment['start_ms'], millis=False)}] "
                    f"说话人 {segment['speaker_id'] + 1}**"
                ),
                "",
                segment["text"],
                "",
            ]
        )
    write_text_atomic(job.output_md, "\n".join(md_lines))


def write_error(job: Job, exc: BaseException, args: argparse.Namespace) -> None:
    source_stat = job.transcript_json.stat()
    payload = {
        "status": "error",
        "generated_at": utc_now(),
        "channel": job.channel,
        "number": job.number,
        "title": job.title,
        "audio_path": rel(job.audio_path),
        "asr_source": {
            "path": rel(job.transcript_json),
            "bytes": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
        },
        "speaker_model": args.speaker_model,
        "vad_model": args.vad_model,
        "error": repr(exc),
    }
    write_json_atomic(job.output_json, payload)


def append_manifest(records: list[dict[str, Any]]) -> None:
    MANIFEST_FILE.parent.mkdir(exist_ok=True)
    with MANIFEST_FILE.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            for record in records:
                fh.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def write_report(
    *,
    output_prefix: str,
    jobs_selected: int,
    processed: int,
    ok: int,
    errors: int,
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    episodes = 0
    eligible = 0
    speaker_ok = 0
    speaker_error = 0
    pending = 0
    missing_audio = 0
    missing_or_invalid_transcript = 0
    for channel in archive.read_channels():
        for episode in archive.load_saved_episodes(channel):
            episodes += 1
            episode_dir = archive.episode_dir(channel, episode)
            audio_path = archive.audio_path_for(channel, episode)
            transcript_json = episode_dir / "transcript.json"
            output_json, _, _ = output_paths(episode_dir, output_prefix)
            if not audio_path.exists() or audio_path.stat().st_size <= 0:
                missing_audio += 1
                continue
            if read_json_status(transcript_json) != "ok":
                missing_or_invalid_transcript += 1
                continue
            eligible += 1
            status = read_json_status(output_json)
            if status == "ok":
                speaker_ok += 1
            elif status == "error":
                speaker_error += 1
            else:
                pending += 1
    report = {
        "generated_at": utc_now(),
        "episodes": episodes,
        "eligible": eligible,
        "speaker_ok": speaker_ok,
        "speaker_error": speaker_error,
        "missing_audio": missing_audio,
        "missing_or_invalid_transcript": missing_or_invalid_transcript,
        "pending": pending,
        "this_run": {
            "jobs_selected": jobs_selected,
            "processed": processed,
            "ok": ok,
            "errors": errors,
            "skipped": skipped,
        },
    }
    REPORT_FILE.parent.mkdir(exist_ok=True)
    write_json_atomic(REPORT_FILE, report)
    return report


def verify_only(output_prefix: str) -> int:
    report = write_report(
        output_prefix=output_prefix,
        jobs_selected=0,
        processed=0,
        ok=0,
        errors=0,
        skipped=[],
    )
    log(
        "[diarize-verify] "
        + " ".join(
            f"{key}={report[key]}"
            for key in [
                "episodes",
                "eligible",
                "speaker_ok",
                "speaker_error",
                "missing_audio",
                "missing_or_invalid_transcript",
                "pending",
            ]
        )
        + f" report={rel(REPORT_FILE)}"
    )
    return 0 if report["pending"] == 0 and report["speaker_error"] == 0 else 1


def process_job_result(
    job: Job,
    diarization: dict[str, Any],
    started: float,
    args: argparse.Namespace,
) -> None:
    transcript = json.loads(job.transcript_json.read_text(encoding="utf-8"))
    timestamp_stats: dict[str, int] = {}
    tokens = extract_timed_tokens(
        transcript,
        audio_duration_seconds=float(diarization["audio_duration_seconds"]),
        stats=timestamp_stats,
    )
    diarization.update(timestamp_stats)
    timeline = diarization["timeline"]
    if not timeline and tokens:
        duration = float(diarization["audio_duration_seconds"])
        timeline = [[0.0, duration, 0]]
        diarization["speaker_count"] = 1
        diarization["fallback_single_speaker"] = True
    timeline, snapped_count = snap_speaker_timeline(
        tokens,
        timeline,
        max_shift_ms=args.speaker_boundary_snap_ms,
    )
    diarization["speaker_boundaries_snapped"] = snapped_count
    labeled_tokens = assign_speakers(tokens, timeline)
    labeled_tokens, reassigned_count = stabilize_speaker_units(
        labeled_tokens,
        min_gap_ms=args.speaker_unit_gap_ms,
    )
    diarization["speaker_tokens_reassigned"] = reassigned_count
    segments = build_text_segments(
        labeled_tokens,
        max_seconds=args.max_output_segment_seconds,
        max_chars=args.max_output_segment_chars,
    )
    write_success(
        job,
        transcript,
        diarization,
        segments,
        time.perf_counter() - started,
        args,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-model", default=DEFAULT_SPEAKER_MODEL)
    parser.add_argument("--speaker-model-path", type=Path)
    parser.add_argument("--vad-model", default=DEFAULT_VAD_MODEL)
    parser.add_argument("--vad-model-path", type=Path)
    parser.add_argument("--vad-device")
    parser.add_argument("--hub", choices=["ms", "hf"], default="ms")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speaker-service-socket", type=Path)
    parser.add_argument("--speaker-service-timeout", type=float, default=300.0)
    parser.add_argument(
        "--speaker-batch-size",
        type=int,
        default=DEFAULT_SPEAKER_BATCH_SIZE,
    )
    parser.add_argument("--audio-chunk-seconds", type=int, default=300)
    parser.add_argument("--vad-max-segment-ms", type=int, default=30000)
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=DEFAULT_MERGE_THRESHOLD,
    )
    parser.add_argument("--preset-speakers", type=int)
    parser.add_argument("--speaker-boundary-snap-ms", type=int, default=1000)
    parser.add_argument("--speaker-unit-gap-ms", type=int, default=250)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--job-batch-size", type=int, default=4)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--report-every-batches", type=int, default=25)
    parser.add_argument("--max-output-segment-seconds", type=float, default=45.0)
    parser.add_argument("--max-output-segment-chars", type=int, default=400)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--channel", action="append")
    parser.add_argument("--episode-dir", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sort", choices=["archive", "path", "size", "channel"], default="size")
    parser.add_argument("--include-done", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("/tmp/podtrans-speaker-diarization"),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-manifest", action="store_true")
    args = parser.parse_args(argv)
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        parser.error("--output-prefix must be a non-empty filename prefix")
    for name in [
        "speaker_batch_size",
        "audio_chunk_seconds",
        "vad_max_segment_ms",
        "cpu_threads",
        "job_batch_size",
        "decode_workers",
        "report_every_batches",
        "max_output_segment_chars",
        "speaker_boundary_snap_ms",
        "speaker_unit_gap_ms",
    ]:
        if name in {"speaker_boundary_snap_ms", "speaker_unit_gap_ms"}:
            if getattr(args, name) < 0:
                parser.error(f"--{name.replace('_', '-')} must be non-negative")
            continue
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_output_segment_seconds <= 0:
        parser.error("--max-output-segment-seconds must be positive")
    if args.speaker_service_timeout <= 0:
        parser.error("--speaker-service-timeout must be positive")
    if args.speaker_service_socket is not None and args.allow_tf32:
        parser.error("--allow-tf32 cannot be used with --speaker-service-socket")
    if not 0 < args.merge_threshold <= 1:
        parser.error("--merge-threshold must be in (0, 1]")
    if args.preset_speakers is not None and args.preset_speakers <= 0:
        parser.error("--preset-speakers must be positive")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.verify_only:
        return verify_only(args.output_prefix)

    channel_filter = set(args.channel) if args.channel else None
    episode_dir_filter = (
        {Path(path).resolve() for path in args.episode_dir}
        if args.episode_dir
        else None
    )
    jobs, skipped = load_jobs(
        channel_filter=channel_filter,
        episode_dir_filter=episode_dir_filter,
        output_prefix=args.output_prefix,
        include_done=args.include_done,
        retry_errors=args.retry_errors,
    )
    jobs = sort_jobs(jobs, args.sort)
    jobs = shard_jobs(jobs, shard_index=args.shard_index, num_shards=args.num_shards)
    if args.limit is not None:
        jobs = jobs[:args.limit]
    log(
        f"[diarize] selected={len(jobs)} skipped={len(skipped)} "
        f"speaker_model={args.speaker_model} device={args.device} "
        f"speaker_batch_size={args.speaker_batch_size}"
    )
    if not jobs:
        if not args.no_report:
            write_report(
                output_prefix=args.output_prefix,
                jobs_selected=0,
                processed=0,
                ok=0,
                errors=0,
                skipped=skipped,
            )
        return 0

    ffmpeg = find_ffmpeg()
    log(f"[diarize] ffmpeg={ffmpeg}")
    vad_model, speaker_model, cluster_model = load_models(args)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    ok = 0
    errors = 0
    for batch_index, batch in enumerate(batched(jobs, args.job_batch_size), start=1):
        batch_started = time.perf_counter()
        records: list[dict[str, Any]] = []
        log(f"[diarize] batch={batch_index} jobs={len(batch)}")
        with tempfile.TemporaryDirectory(prefix="batch-", dir=args.temp_dir) as temp_name:
            decoded, batch_errors = decode_batch(
                ffmpeg,
                batch,
                Path(temp_name),
                args.decode_workers,
                args.audio_chunk_seconds,
            )
            for job, exc in batch_errors.items():
                log(f"[diarize] decode_error job={rel(job.audio_path)} error={exc!r}")
            try:
                import torch

                measure_cuda = (
                    args.speaker_service_socket is None
                    and args.device.startswith("cuda")
                    and torch.cuda.is_available()
                )
                if measure_cuda:
                    torch.cuda.reset_peak_memory_stats()
                diarization, model_errors = diarize_decoded_batch(
                    decoded,
                    vad_model,
                    speaker_model,
                    cluster_model,
                    args,
                )
                if measure_cuda:
                    peak_allocated_mb = round(
                        torch.cuda.max_memory_allocated() / 1048576,
                        1,
                    )
                    peak_reserved_mb = round(
                        torch.cuda.max_memory_reserved() / 1048576,
                        1,
                    )
                    for result in diarization.values():
                        result["gpu_peak_allocated_mb"] = peak_allocated_mb
                        result["gpu_peak_reserved_mb"] = peak_reserved_mb
                elif isinstance(speaker_model, SpeakerEmbeddingClient):
                    service_metrics = speaker_model.consume_metrics()
                    for result in diarization.values():
                        result.update(service_metrics)
                batch_errors.update(model_errors)
            except Exception as exc:
                diarization = {}
                for job in decoded:
                    batch_errors[job] = exc
            for job in batch:
                exc = batch_errors.get(job)
                if exc is None and job not in diarization:
                    exc = RuntimeError("diarization produced no result")
                if exc is not None:
                    write_error(job, exc, args)
                    errors += 1
                    processed += 1
                    records.append(
                        {
                            "time": utc_now(),
                            "status": "error",
                            "path": rel(job.output_json),
                            "error": repr(exc),
                        }
                    )
                    continue
                try:
                    process_job_result(
                        job,
                        diarization[job],
                        batch_started,
                        args,
                    )
                except Exception as output_exc:
                    write_error(job, output_exc, args)
                    errors += 1
                    processed += 1
                    records.append(
                        {
                            "time": utc_now(),
                            "status": "error",
                            "path": rel(job.output_json),
                            "error": repr(output_exc),
                        }
                    )
                    continue
                result = diarization[job]
                log(
                    f"[diarize] ok job={rel(job.audio_path)} "
                    f"speakers={result['speaker_count']} windows={result['window_count']}"
                )
                ok += 1
                processed += 1
                records.append(
                    {
                        "time": utc_now(),
                        "status": "ok",
                        "path": rel(job.output_json),
                        "speakers": result["speaker_count"],
                        "windows": result["window_count"],
                    }
                )
        if not args.no_manifest:
            append_manifest(records)
        log(
            f"[diarize] batch={batch_index} processed={processed}/{len(jobs)} "
            f"ok={ok} errors={errors} seconds={time.perf_counter() - batch_started:.2f}"
        )
        if not args.no_report and batch_index % args.report_every_batches == 0:
            write_report(
                output_prefix=args.output_prefix,
                jobs_selected=len(jobs),
                processed=processed,
                ok=ok,
                errors=errors,
                skipped=skipped,
            )
    if not args.no_report:
        write_report(
            output_prefix=args.output_prefix,
            jobs_selected=len(jobs),
            processed=processed,
            ok=ok,
            errors=errors,
            skipped=skipped,
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
