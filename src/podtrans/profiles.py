"""Hardware-aware starting configurations for podtrans inference."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    vram_gb: int | None
    examples: tuple[str, ...]
    asr_backend: str
    asr_batch_size: int
    asr_job_batch_size: int
    asr_decode_workers: int
    vllm_memory_utilization: float | None
    speaker_batch_size: int
    diarize_job_batch_size: int
    diarize_decode_workers: int
    max_worker_shards: int
    handler_workers: int
    max_queued_requests: int
    notes: str
    validated_on_host: bool = False


PROFILES: dict[str, HardwareProfile] = {
    "cpu": HardwareProfile(
        name="cpu",
        vram_gb=None,
        examples=("x86-64 CPU", "Apple Silicon CPU"),
        asr_backend="auto",
        asr_batch_size=1,
        asr_job_batch_size=1,
        asr_decode_workers=2,
        vllm_memory_utilization=None,
        speaker_batch_size=32,
        diarize_job_batch_size=1,
        diarize_decode_workers=2,
        max_worker_shards=1,
        handler_workers=0,
        max_queued_requests=0,
        notes="CPU-only fallback; runs diarization in-process without the GPU service.",
    ),
    "gpu-4gb": HardwareProfile(
        name="gpu-4gb",
        vram_gb=4,
        examples=("GeForce GTX 1650 4GB", "GeForce RTX 3050 4GB"),
        asr_backend="auto",
        asr_batch_size=1,
        asr_job_batch_size=1,
        asr_decode_workers=1,
        vllm_memory_utilization=None,
        speaker_batch_size=32,
        diarize_job_batch_size=1,
        diarize_decode_workers=1,
        max_worker_shards=1,
        handler_workers=1,
        max_queued_requests=2,
        notes="Tight VRAM tier; use AutoModel and fall back to CPU if model loading fails.",
    ),
    "gpu-6gb": HardwareProfile(
        name="gpu-6gb",
        vram_gb=6,
        examples=("GeForce RTX 2060 6GB", "GeForce RTX 3050 6GB"),
        asr_backend="auto",
        asr_batch_size=1,
        asr_job_batch_size=1,
        asr_decode_workers=2,
        vllm_memory_utilization=None,
        speaker_batch_size=64,
        diarize_job_batch_size=1,
        diarize_decode_workers=2,
        max_worker_shards=2,
        handler_workers=1,
        max_queued_requests=2,
        notes="Uses the lower-overhead AutoModel backend; no forced low precision.",
    ),
    "gpu-8gb": HardwareProfile(
        name="gpu-8gb",
        vram_gb=8,
        examples=(
            "GeForce RTX 3060 Ti 8GB",
            "GeForce RTX 4060 8GB",
            "GeForce RTX 5060 8GB",
        ),
        asr_backend="auto",
        asr_batch_size=2,
        asr_job_batch_size=2,
        asr_decode_workers=2,
        vllm_memory_utilization=None,
        speaker_batch_size=128,
        diarize_job_batch_size=2,
        diarize_decode_workers=2,
        max_worker_shards=2,
        handler_workers=2,
        max_queued_requests=4,
        notes="Conservative CUDA profile for common 8GB cards.",
    ),
    "gpu-12gb": HardwareProfile(
        name="gpu-12gb",
        vram_gb=12,
        examples=(
            "GeForce RTX 3060 12GB",
            "GeForce RTX 4070 12GB",
            "GeForce RTX 5070 12GB",
        ),
        asr_backend="vllm",
        asr_batch_size=4,
        asr_job_batch_size=2,
        asr_decode_workers=3,
        vllm_memory_utilization=0.84,
        speaker_batch_size=256,
        diarize_job_batch_size=2,
        diarize_decode_workers=3,
        max_worker_shards=3,
        handler_workers=2,
        max_queued_requests=4,
        notes="BF16 vLLM starting point for Ampere-or-newer GPUs.",
    ),
    "gpu-16gb": HardwareProfile(
        name="gpu-16gb",
        vram_gb=16,
        examples=(
            "GeForce RTX 4060 Ti 16GB",
            "GeForce RTX 4070 Ti SUPER 16GB",
            "GeForce RTX 4080 / 5080 16GB",
            "GeForce RTX 5060 Ti / 5070 Ti 16GB",
        ),
        asr_backend="vllm",
        asr_batch_size=6,
        asr_job_batch_size=3,
        asr_decode_workers=4,
        vllm_memory_utilization=0.87,
        speaker_batch_size=384,
        diarize_job_batch_size=3,
        diarize_decode_workers=4,
        max_worker_shards=4,
        handler_workers=2,
        max_queued_requests=4,
        notes="Mainstream 16GB BF16 vLLM profile.",
    ),
    "gpu-24gb": HardwareProfile(
        name="gpu-24gb",
        vram_gb=24,
        examples=(
            "GeForce RTX 3090 / 3090 Ti 24GB",
            "GeForce RTX 4090 24GB",
        ),
        asr_backend="vllm",
        asr_batch_size=8,
        asr_job_batch_size=4,
        asr_decode_workers=4,
        vllm_memory_utilization=0.90,
        speaker_batch_size=512,
        diarize_job_batch_size=4,
        diarize_decode_workers=4,
        max_worker_shards=6,
        handler_workers=3,
        max_queued_requests=6,
        notes="High-throughput single-GPU profile; increase only after a short validation run.",
    ),
    "gpu-32gb": HardwareProfile(
        name="gpu-32gb",
        vram_gb=32,
        examples=("GeForce RTX 5090 32GB",),
        asr_backend="vllm",
        asr_batch_size=12,
        asr_job_batch_size=4,
        asr_decode_workers=6,
        vllm_memory_utilization=0.92,
        speaker_batch_size=1024,
        diarize_job_batch_size=4,
        diarize_decode_workers=4,
        max_worker_shards=8,
        handler_workers=3,
        max_queued_requests=6,
        notes="32GB consumer flagship starting point.",
    ),
    "gpu-48gb": HardwareProfile(
        name="gpu-48gb",
        vram_gb=48,
        examples=("48GB modified or workstation-class NVIDIA GPU",),
        asr_backend="vllm",
        asr_batch_size=16,
        asr_job_batch_size=6,
        asr_decode_workers=8,
        vllm_memory_utilization=0.92,
        speaker_batch_size=1536,
        diarize_job_batch_size=4,
        diarize_decode_workers=4,
        max_worker_shards=16,
        handler_workers=3,
        max_queued_requests=6,
        notes="Validated project host profile; one shared FP32 speaker service.",
        validated_on_host=True,
    ),
}


def get_profile(name: str) -> HardwareProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(PROFILES)
        raise ValueError(f"unknown profile {name!r}; choose one of: {choices}") from exc


def select_profile(vram_gb: float | None) -> HardwareProfile:
    if vram_gb is None or vram_gb < 4:
        return PROFILES["cpu"]
    gpu_profiles = [
        profile for profile in PROFILES.values() if profile.vram_gb is not None
    ]
    eligible = [profile for profile in gpu_profiles if profile.vram_gb <= vram_gb]
    if not eligible:
        return PROFILES["cpu"]
    return max(eligible, key=lambda profile: profile.vram_gb or 0)


def resolved_profile(
    profile: HardwareProfile,
    *,
    cpu_cores: int | None = None,
) -> dict[str, Any]:
    cores = max(1, cpu_cores or os.cpu_count() or 1)
    if profile.name == "cpu":
        worker_shards = 1
        worker_cpu_threads = min(16, cores)
        decode_workers = min(profile.diarize_decode_workers, max(1, cores // 4))
    else:
        worker_shards = min(profile.max_worker_shards, max(1, cores // 4))
        worker_cpu_threads = max(1, min(4, cores // worker_shards))
        decode_workers = min(
            profile.diarize_decode_workers,
            max(1, cores // max(1, worker_shards * 2)),
        )
    asr_decode_workers = min(
        profile.asr_decode_workers,
        max(1, cores // 4),
    )
    service_cpu_threads = min(8, max(1, cores // 2))
    handler_workers = min(
        profile.handler_workers,
        max(1, cores // 8),
    )
    payload = asdict(profile)
    payload.update(
        {
            "cpu_cores": cores,
            "worker_shards": worker_shards,
            "worker_cpu_threads": worker_cpu_threads,
            "asr_decode_workers": asr_decode_workers,
            "diarize_decode_workers": decode_workers,
            "service_cpu_threads": service_cpu_threads,
            "handler_workers": handler_workers,
        }
    )
    return payload


def command_arguments(
    profile: HardwareProfile,
    command: str,
    *,
    cpu_cores: int | None = None,
) -> list[str]:
    config = resolved_profile(profile, cpu_cores=cpu_cores)
    if command == "transcribe":
        args = [
            "--backend",
            profile.asr_backend,
            "--device",
            "cpu" if profile.name == "cpu" else "cuda",
            "--batch-size",
            str(profile.asr_batch_size),
            "--job-batch-size",
            str(profile.asr_job_batch_size),
            "--decode-workers",
            str(config["asr_decode_workers"]),
        ]
        if profile.vllm_memory_utilization is not None:
            args.extend(
                [
                    "--vllm-dtype",
                    "bfloat16",
                    "--vllm-gpu-memory-utilization",
                    str(profile.vllm_memory_utilization),
                ]
            )
        return args
    if command == "diarize":
        return [
            "--device",
            "cpu" if profile.name == "cpu" else "cuda:0",
            "--vad-device",
            "cpu",
            "--speaker-batch-size",
            str(profile.speaker_batch_size),
            "--job-batch-size",
            str(profile.diarize_job_batch_size),
            "--decode-workers",
            str(config["diarize_decode_workers"]),
            "--cpu-threads",
            str(config["worker_cpu_threads"]),
        ]
    if command == "speaker-service":
        if profile.name == "cpu":
            raise ValueError(
                "the CPU profile runs diarization in-process and does not use "
                "speaker-service"
            )
        return [
            "--device",
            "cuda:0",
            "--max-batch-size",
            str(profile.speaker_batch_size),
            "--warmup-batch-size",
            str(profile.speaker_batch_size),
            "--cpu-threads",
            str(config["service_cpu_threads"]),
            "--handler-workers",
            str(config["handler_workers"]),
            "--max-queued-requests",
            str(profile.max_queued_requests),
        ]
    raise ValueError(f"profiles are not supported by command {command!r}")


def detect_nvidia_gpu() -> tuple[str, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    devices: list[tuple[str, float]] = []
    for line in result.stdout.splitlines():
        try:
            name, memory_mib = line.rsplit(",", 1)
            nominal_gb = round(float(memory_mib.strip()) / 1024, 1)
            devices.append((name.strip(), nominal_gb))
        except ValueError:
            continue
    return max(devices, key=lambda item: item[1]) if devices else None


def profile_json(profile: HardwareProfile, *, cpu_cores: int | None = None) -> str:
    return json.dumps(
        resolved_profile(profile, cpu_cores=cpu_cores),
        ensure_ascii=False,
        indent=2,
    )
