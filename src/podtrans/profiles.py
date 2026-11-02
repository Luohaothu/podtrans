"""Hardware-aware starting configurations for podtrans inference."""

from __future__ import annotations

import json
import os
import platform as py_platform
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal


ASR_NANO = "FunAudioLLM/Fun-ASR-Nano-2512"
SPEAKER_ERES2NET_V2 = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
VAD_FSMN = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
VLLM_DTYPE = "bfloat16"

# Listed catalog covers generations 1..N. get_profile / query_apple_profile
# still materialize later M-series generations from the same recipe table.
LISTED_APPLE_GENERATION_MAX = 5

_APPLE_NAME = re.compile(r"^apple-m(\d+)-(\d+)gb$")
_APPLE_BRAND = re.compile(r"^Apple M(\d+)\b")
_PROBE_TIMEOUT_SECONDS = 5

ProbeRunner = Callable[[Sequence[str]], Any]


class Family(Enum):
    CPU = "cpu"
    NVIDIA = "nvidia"
    APPLE = "apple"


@dataclass(frozen=True)
class Vram:
    """Discrete NVIDIA memory, in GB. Never unified memory."""

    gb: float

    def __post_init__(self) -> None:
        if self.gb <= 0:
            raise ValueError("VRAM must be positive")


@dataclass(frozen=True)
class AppleChip:
    """Detected Apple Silicon. Selection uses generation and unified_gb only."""

    generation: int
    unified_gb: float
    brand: str

    def __post_init__(self) -> None:
        if self.generation < 1:
            raise ValueError("Apple generation must be >= 1")
        if self.unified_gb <= 0:
            raise ValueError("unified memory must be positive")


@dataclass(frozen=True)
class Machine:
    """One detected or previewed computer. Discrete VRAM and an Apple chip never coexist."""

    family: Family
    label: str
    vram: Vram | None = None
    apple: AppleChip | None = None

    def __post_init__(self) -> None:
        if self.family is Family.NVIDIA:
            if self.vram is None or self.apple is not None:
                raise ValueError("an NVIDIA machine carries discrete VRAM and no Apple chip")
        elif self.family is Family.APPLE:
            if self.apple is None or self.vram is not None:
                raise ValueError("an Apple machine carries an Apple chip and no discrete VRAM")
        elif self.family is Family.CPU:
            if self.vram is not None or self.apple is not None:
                raise ValueError("a CPU machine carries neither VRAM nor an Apple chip")
        else:
            raise ValueError(f"unknown machine family {self.family!r}")


@dataclass(frozen=True)
class FunasrAuto:
    model: str
    device: Literal["cpu", "cuda", "mps"]

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("auto backend device must be cpu, cuda, or mps")


@dataclass(frozen=True)
class FunasrVllm:
    model: str
    memory_utilization: float
    dtype: str = VLLM_DTYPE
    device: Literal["cuda"] = "cuda"

    def __post_init__(self) -> None:
        if self.device != "cuda":
            raise ValueError("vLLM only runs on CUDA")
        if not 0 < self.memory_utilization < 1:
            raise ValueError("vLLM memory utilization must be in (0, 1)")


MLX_NANO = "fun-asr-nano-mlx"
MLX_NANO_VARIANT = "w4a16"  # only variant the catalog emits


@dataclass(frozen=True)
class FunasrMlx:
    model: str = MLX_NANO
    variant: str = MLX_NANO_VARIANT

    def __post_init__(self) -> None:
        if self.model != MLX_NANO:
            raise ValueError("funasr-mlx catalog model must be fun-asr-nano-mlx")
        if self.variant != MLX_NANO_VARIANT:
            raise ValueError("funasr-mlx catalog variant must be w4a16")


AsrPair = FunasrAuto | FunasrVllm | FunasrMlx


@dataclass(frozen=True)
class InProcessSpeaker:
    model: str
    device: Literal["cpu", "mps"]
    vad_model: str
    vad_device: Literal["cpu", "mps"]
    batch_size: int

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "mps"}:
            raise ValueError("in-process speaker device must be cpu or mps")
        if self.vad_device not in {"cpu", "mps"}:
            raise ValueError("VAD device must be cpu or mps")
        if self.batch_size <= 0:
            raise ValueError("speaker batch size must be positive")


@dataclass(frozen=True)
class SpeakerService:
    """Shared CUDA speaker embedding server."""

    model: str
    batch_size: int
    device: Literal["cuda"] = "cuda"

    def __post_init__(self) -> None:
        if self.device != "cuda":
            raise ValueError("speaker-service only runs on CUDA")
        if self.batch_size <= 0:
            raise ValueError("speaker batch size must be positive")


SpeakerPlacement = InProcessSpeaker | SpeakerService


@dataclass(frozen=True)
class CpuKey:
    pass


@dataclass(frozen=True)
class VramKey:
    gb: int


@dataclass(frozen=True)
class UnifiedKey:
    """Apple profile match. memory_gb is the catalog floor, not raw unified size."""

    generation: int
    memory_gb: int


MatchKey = CpuKey | VramKey | UnifiedKey


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    match: MatchKey
    examples: tuple[str, ...]
    asr: AsrPair
    speaker: SpeakerPlacement
    asr_batch_size: int
    asr_job_batch_size: int
    asr_decode_workers: int
    diarize_job_batch_size: int
    diarize_decode_workers: int
    max_worker_shards: int
    handler_workers: int
    max_queued_requests: int
    notes: str
    validated_on_host: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.match, VramKey):
            if not isinstance(self.speaker, SpeakerService):
                raise ValueError("NVIDIA profiles use speaker-service")
            if not isinstance(self.asr, FunasrAuto | FunasrVllm) or self.asr.device != "cuda":
                raise ValueError("NVIDIA ASR must run on CUDA")
        elif isinstance(self.match, UnifiedKey):
            cpu_auto = isinstance(self.asr, FunasrAuto) and self.asr.device == "cpu"
            if not isinstance(self.asr, FunasrMlx) and not cpu_auto:
                raise ValueError("Apple ASR is funasr-mlx, or FunASR auto on CPU")
            if not isinstance(self.speaker, InProcessSpeaker):
                raise ValueError("Apple profiles run diarization in-process")
            if self.speaker.device != "cpu" or self.speaker.vad_device != "cpu":
                raise ValueError("Apple speaker and VAD stay on CPU")
        elif isinstance(self.match, CpuKey):
            if not isinstance(self.asr, FunasrAuto) or self.asr.device != "cpu":
                raise ValueError("the CPU profile runs ASR on CPU")
            if not isinstance(self.speaker, InProcessSpeaker) or self.speaker.device != "cpu":
                raise ValueError("the CPU profile runs diarization in-process on CPU")
        else:
            raise ValueError("unknown profile match")

    @property
    def uses_speaker_service(self) -> bool:
        return isinstance(self.speaker, SpeakerService)

    @property
    def family(self) -> Family:
        if isinstance(self.match, VramKey):
            return Family.NVIDIA
        if isinstance(self.match, UnifiedKey):
            return Family.APPLE
        return Family.CPU

    def memory_label(self) -> str:
        if isinstance(self.match, VramKey):
            return f"{self.match.gb}GB vram"
        if isinstance(self.match, UnifiedKey):
            return f"{self.match.memory_gb}GB unified"
        return "CPU"

    def asr_pair_label(self) -> str:
        asr = self.asr
        if isinstance(asr, FunasrMlx):
            return f"mlx/{asr.model}@{asr.variant}"
        backend = "vllm" if isinstance(asr, FunasrVllm) else "auto"
        model = asr.model.rsplit("/", 1)[-1]
        return f"{backend}/{model}@{asr.device}"

    def mode_label(self) -> str:
        if self.uses_speaker_service:
            return "service"
        return "in-process"


def _nano(device: Literal["cpu", "cuda", "mps"]) -> FunasrAuto:
    return FunasrAuto(ASR_NANO, device)


def _apple_floor_asr(memory_gb: int) -> FunasrAuto | FunasrMlx:
    if memory_gb == 8:
        return _nano("cpu")
    if memory_gb > 8:
        return FunasrMlx()
    raise ValueError(f"Apple memory floor must be at least 8GB, got {memory_gb}")


@dataclass(frozen=True)
class AppleRecipe:
    """One pre-compiled Apple catalog row: generation band × unified-memory floor."""

    generation_lo: int
    generation_hi: int | None
    memory_gb: int
    asr: FunasrAuto | FunasrMlx
    asr_batch_size: int
    asr_job_batch_size: int
    asr_decode_workers: int
    speaker_batch_size: int
    diarize_job_batch_size: int
    diarize_decode_workers: int
    notes: str

    def covers(self, generation: int) -> bool:
        if generation < self.generation_lo:
            return False
        if self.generation_hi is None:
            return True
        return generation <= self.generation_hi


# Pre-compiled recipe table. Selection is a table query, not ad-hoc branching.
APPLE_RECIPE_TABLE: tuple[AppleRecipe, ...] = (
    AppleRecipe(
        1,
        2,
        8,
        _apple_floor_asr(8),
        1,
        1,
        1,
        16,
        1,
        1,
        (
            "Fun-ASR-Nano on CPU with small batches. "
            "The 8GB floor stays on CPU FunASR. "
            "Speaker and VAD stay on CPU. Unvalidated starting point."
        ),
    ),
    AppleRecipe(
        1,
        2,
        16,
        _apple_floor_asr(16),
        1,
        1,
        2,
        32,
        1,
        2,
        (
            "funasr-mlx fun-asr-nano-mlx w4a16. "
            "Speaker and VAD stay on CPU. Unvalidated starting point."
        ),
    ),
    AppleRecipe(
        1,
        2,
        32,
        _apple_floor_asr(32),
        1,
        1,
        2,
        32,
        1,
        2,
        (
            "funasr-mlx fun-asr-nano-mlx w4a16. "
            "Speaker stays on CPU. Unvalidated starting point."
        ),
    ),
    AppleRecipe(
        1,
        2,
        64,
        _apple_floor_asr(64),
        2,
        1,
        2,
        64,
        1,
        2,
        (
            "funasr-mlx fun-asr-nano-mlx w4a16, with a larger ASR batch. "
            "Unvalidated starting point."
        ),
    ),
    AppleRecipe(
        3,
        None,
        8,
        _apple_floor_asr(8),
        1,
        1,
        2,
        16,
        1,
        2,
        (
            "Fun-ASR-Nano on CPU. The 8GB floor stays on CPU FunASR. "
            "Speaker and VAD stay on CPU. Unvalidated starting point."
        ),
    ),
    AppleRecipe(
        3,
        None,
        16,
        _apple_floor_asr(16),
        1,
        1,
        2,
        32,
        1,
        2,
        (
            "funasr-mlx fun-asr-nano-mlx w4a16, batch 1. "
            "Speaker and VAD stay on CPU. Unvalidated starting point."
        ),
    ),
    AppleRecipe(
        3,
        None,
        32,
        _apple_floor_asr(32),
        2,
        2,
        4,
        64,
        2,
        2,
        "funasr-mlx fun-asr-nano-mlx w4a16, batch 2. Unvalidated starting point.",
    ),
    AppleRecipe(
        3,
        None,
        64,
        _apple_floor_asr(64),
        4,
        2,
        4,
        128,
        2,
        4,
        (
            "funasr-mlx fun-asr-nano-mlx w4a16, batch 4. "
            "Still one in-process diarize worker. "
            "Unvalidated starting point."
        ),
    ),
)


def _recipes_for(generation: int) -> tuple[AppleRecipe, ...]:
    rows = tuple(recipe for recipe in APPLE_RECIPE_TABLE if recipe.covers(generation))
    if not rows:
        raise ValueError(f"no Apple recipes cover generation {generation}")
    return rows


def _materialize_apple(generation: int, recipe: AppleRecipe) -> HardwareProfile:
    return HardwareProfile(
        name=f"apple-m{generation}-{recipe.memory_gb}gb",
        match=UnifiedKey(generation, recipe.memory_gb),
        examples=(f"Apple M{generation} with at least {recipe.memory_gb}GB unified",),
        asr=recipe.asr,
        speaker=InProcessSpeaker(
            model=SPEAKER_ERES2NET_V2,
            device="cpu",
            vad_model=VAD_FSMN,
            vad_device="cpu",
            batch_size=recipe.speaker_batch_size,
        ),
        asr_batch_size=recipe.asr_batch_size,
        asr_job_batch_size=recipe.asr_job_batch_size,
        asr_decode_workers=recipe.asr_decode_workers,
        diarize_job_batch_size=recipe.diarize_job_batch_size,
        diarize_decode_workers=recipe.diarize_decode_workers,
        max_worker_shards=1,
        handler_workers=0,
        max_queued_requests=0,
        notes=recipe.notes,
    )


@dataclass(frozen=True)
class _AppleColumn:
    """Pre-compiled floor column for one generation: sorted memory → profile."""

    floors: tuple[int, ...]
    profiles: tuple[HardwareProfile, ...]

    def query(self, unified_gb: float) -> HardwareProfile:
        chosen = 0
        for index, floor in enumerate(self.floors):
            if floor <= unified_gb:
                chosen = index
            else:
                break
        return self.profiles[chosen]


def _compile_apple_column(generation: int) -> _AppleColumn:
    recipes = _recipes_for(generation)
    floors = tuple(recipe.memory_gb for recipe in recipes)
    if floors != tuple(sorted(floors)) or len(floors) != len(set(floors)):
        raise RuntimeError(f"Apple memory floors for M{generation} must be strictly increasing")
    profiles = tuple(_materialize_apple(generation, recipe) for recipe in recipes)
    return _AppleColumn(floors, profiles)


def query_apple_profile(generation: int, unified_gb: float) -> HardwareProfile:
    """Table-based Apple selection: floor unified memory within the generation column."""
    if generation < 1:
        raise ValueError("Apple generation must be >= 1")
    column = _APPLE_QUERY.get(generation)
    if column is None:
        column = _compile_apple_column(generation)
    return column.query(unified_gb)


def _cpu_profile() -> HardwareProfile:
    return HardwareProfile(
        name="cpu",
        match=CpuKey(),
        examples=("x86-64 CPU",),
        asr=_nano("cpu"),
        speaker=InProcessSpeaker(
            SPEAKER_ERES2NET_V2,
            "cpu",
            VAD_FSMN,
            "cpu",
            batch_size=32,
        ),
        asr_batch_size=1,
        asr_job_batch_size=1,
        asr_decode_workers=2,
        diarize_job_batch_size=1,
        diarize_decode_workers=2,
        max_worker_shards=1,
        handler_workers=0,
        max_queued_requests=0,
        notes=(
            "CPU-only fallback; runs diarization in-process without the GPU service. "
            "Apple Silicon machines resolve to apple-m* profiles, not this one."
        ),
    )


def _gpu(
    gb: int,
    examples: tuple[str, ...],
    asr: AsrPair,
    asr_batch_size: int,
    asr_job_batch_size: int,
    asr_decode_workers: int,
    speaker_batch_size: int,
    diarize_job_batch_size: int,
    diarize_decode_workers: int,
    max_worker_shards: int,
    handler_workers: int,
    max_queued_requests: int,
    notes: str,
    *,
    validated_on_host: bool = False,
) -> HardwareProfile:
    return HardwareProfile(
        name=f"gpu-{gb}gb",
        match=VramKey(gb),
        examples=examples,
        asr=asr,
        speaker=SpeakerService(SPEAKER_ERES2NET_V2, speaker_batch_size),
        asr_batch_size=asr_batch_size,
        asr_job_batch_size=asr_job_batch_size,
        asr_decode_workers=asr_decode_workers,
        diarize_job_batch_size=diarize_job_batch_size,
        diarize_decode_workers=diarize_decode_workers,
        max_worker_shards=max_worker_shards,
        handler_workers=handler_workers,
        max_queued_requests=max_queued_requests,
        notes=notes,
        validated_on_host=validated_on_host,
    )


def _nvidia_profiles() -> tuple[HardwareProfile, ...]:
    auto = _nano("cuda")
    return (
        _gpu(
            4,
            ("GeForce GTX 1650 4GB", "GeForce RTX 3050 4GB"),
            auto,
            1,
            1,
            1,
            32,
            1,
            1,
            1,
            1,
            2,
            "Tight VRAM tier; use AutoModel and fall back to CPU if model loading fails.",
        ),
        _gpu(
            6,
            ("GeForce RTX 2060 6GB", "GeForce RTX 3050 6GB"),
            auto,
            1,
            1,
            2,
            64,
            1,
            2,
            2,
            1,
            2,
            "Uses the lower-overhead AutoModel backend; no forced low precision.",
        ),
        _gpu(
            8,
            (
                "GeForce RTX 3060 Ti 8GB",
                "GeForce RTX 4060 8GB",
                "GeForce RTX 5060 8GB",
            ),
            auto,
            2,
            2,
            2,
            128,
            2,
            2,
            2,
            2,
            4,
            "Conservative CUDA profile for common 8GB cards.",
        ),
        _gpu(
            12,
            (
                "GeForce RTX 3060 12GB",
                "GeForce RTX 4070 12GB",
                "GeForce RTX 5070 12GB",
            ),
            FunasrVllm(ASR_NANO, 0.84),
            4,
            2,
            3,
            256,
            2,
            3,
            3,
            2,
            4,
            "BF16 vLLM starting point for Ampere-or-newer GPUs.",
        ),
        _gpu(
            16,
            (
                "GeForce RTX 4060 Ti 16GB",
                "GeForce RTX 4070 Ti SUPER 16GB",
                "GeForce RTX 4080 / 5080 16GB",
                "GeForce RTX 5060 Ti / 5070 Ti 16GB",
            ),
            FunasrVllm(ASR_NANO, 0.87),
            6,
            3,
            4,
            384,
            3,
            4,
            4,
            2,
            4,
            "Mainstream 16GB BF16 vLLM profile.",
        ),
        _gpu(
            24,
            (
                "GeForce RTX 3090 / 3090 Ti 24GB",
                "GeForce RTX 4090 24GB",
            ),
            FunasrVllm(ASR_NANO, 0.90),
            8,
            4,
            4,
            512,
            4,
            4,
            6,
            3,
            6,
            "High-throughput single-GPU profile; increase only after a short validation run.",
        ),
        _gpu(
            32,
            ("GeForce RTX 5090 32GB",),
            FunasrVllm(ASR_NANO, 0.92),
            12,
            4,
            6,
            1024,
            4,
            4,
            8,
            3,
            6,
            "32GB consumer flagship starting point.",
        ),
        _gpu(
            48,
            ("48GB modified or workstation-class NVIDIA GPU",),
            FunasrVllm(ASR_NANO, 0.92),
            16,
            6,
            8,
            1536,
            4,
            4,
            16,
            3,
            6,
            "Validated project host profile; one shared FP32 speaker service.",
            validated_on_host=True,
        ),
    )


def _apple_profiles() -> tuple[HardwareProfile, ...]:
    return tuple(
        profile
        for generation in range(1, LISTED_APPLE_GENERATION_MAX + 1)
        for profile in _compile_apple_column(generation).profiles
    )


def _vram_key_gb(profile: HardwareProfile) -> int:
    match = profile.match
    if not isinstance(match, VramKey):
        raise RuntimeError(f"{profile.name} is not a VRAM profile")
    return match.gb


def _validate_catalog(profiles: tuple[HardwareProfile, ...]) -> None:
    names = [profile.name for profile in profiles]
    if len(names) != len(set(names)):
        raise RuntimeError("profile names must be unique")
    for generation in (1, 3):
        floors = [recipe.memory_gb for recipe in _recipes_for(generation)]
        if floors != sorted(floors) or len(floors) != len(set(floors)):
            raise RuntimeError("Apple memory floors must be strictly increasing")
        for recipe in _recipes_for(generation):
            if recipe.asr != _apple_floor_asr(recipe.memory_gb):
                raise RuntimeError(
                    f"Apple floor {recipe.memory_gb}GB does not match the ASR floor rule"
                )
    nvidia = [profile for profile in profiles if isinstance(profile.match, VramKey)]
    expected = {
        "gpu-4gb": 4,
        "gpu-6gb": 6,
        "gpu-8gb": 8,
        "gpu-12gb": 12,
        "gpu-16gb": 16,
        "gpu-24gb": 24,
        "gpu-32gb": 32,
        "gpu-48gb": 48,
    }
    if {profile.name: _vram_key_gb(profile) for profile in nvidia} != expected:
        raise RuntimeError("NVIDIA tiers drifted from the historical VRAM floors")
    for profile in nvidia:
        gb = _vram_key_gb(profile)
        if gb <= 8 and not isinstance(profile.asr, FunasrAuto):
            raise RuntimeError(f"{profile.name} stays on the auto backend")
        if gb >= 12 and not isinstance(profile.asr, FunasrVllm):
            raise RuntimeError(f"{profile.name} stays on vLLM")
    listed_apple = {
        f"apple-m{generation}-{recipe.memory_gb}gb"
        for generation in range(1, LISTED_APPLE_GENERATION_MAX + 1)
        for recipe in _recipes_for(generation)
    }
    found_apple = {profile.name for profile in profiles if profile.family is Family.APPLE}
    if found_apple != listed_apple:
        raise RuntimeError("listed Apple profiles must be the generation by floor catalog")
    for profile in profiles:
        if profile.family is Family.APPLE and profile.validated_on_host:
            raise RuntimeError(f"{profile.name} is not validated on the project host")
        if isinstance(profile.asr, FunasrVllm) and not isinstance(profile.match, VramKey):
            raise RuntimeError("vLLM profiles require a VRAM key")


_CPU_PROFILE = _cpu_profile()
_NVIDIA_PROFILES = _nvidia_profiles()
_APPLE_QUERY: dict[int, _AppleColumn] = {
    generation: _compile_apple_column(generation)
    for generation in range(1, LISTED_APPLE_GENERATION_MAX + 1)
}
_LISTED = (
    _CPU_PROFILE,
    *_NVIDIA_PROFILES,
    *(
        profile
        for generation in range(1, LISTED_APPLE_GENERATION_MAX + 1)
        for profile in _APPLE_QUERY[generation].profiles
    ),
)
_validate_catalog(_LISTED)
PROFILES: dict[str, HardwareProfile] = {profile.name: profile for profile in _LISTED}


def listed_profiles() -> tuple[HardwareProfile, ...]:
    return _LISTED


def _unknown_profile(name: str) -> ValueError:
    choices = ", ".join(profile.name for profile in _LISTED)
    return ValueError(f"unknown profile {name!r}; choose one of: {choices}")


def get_profile(name: str) -> HardwareProfile:
    found = PROFILES.get(name)
    if found is not None:
        return found
    match = _APPLE_NAME.fullmatch(name)
    if match is None:
        raise _unknown_profile(name)
    generation = int(match.group(1))
    memory = int(match.group(2))
    if generation < 1 or f"apple-m{generation}-{memory}gb" != name:
        raise _unknown_profile(name)
    for recipe in _recipes_for(generation):
        if recipe.memory_gb == memory:
            return _materialize_apple(generation, recipe)
    if memory <= 0:
        raise _unknown_profile(name)
    floored = query_apple_profile(generation, float(memory))
    raise ValueError(
        f"unknown profile {name!r}; {memory}GB is not an Apple memory floor, "
        f"{floored.name} is the floor for that chip"
    )


def _select_nvidia(vram: Vram | None) -> HardwareProfile:
    if vram is None or vram.gb < 4:
        return _CPU_PROFILE
    eligible = [
        profile for profile in _NVIDIA_PROFILES if _vram_key_gb(profile) <= vram.gb
    ]
    if not eligible:
        return _CPU_PROFILE
    return max(eligible, key=_vram_key_gb)


def select_machine(machine: Machine) -> HardwareProfile:
    if machine.family is Family.NVIDIA:
        return _select_nvidia(machine.vram)
    if machine.family is Family.APPLE:
        chip = machine.apple
        if chip is None:
            raise ValueError("an Apple machine requires an Apple chip")
        return query_apple_profile(chip.generation, chip.unified_gb)
    return _CPU_PROFILE


def select_profile(vram_gb: float | None) -> HardwareProfile:
    """Map discrete NVIDIA VRAM GB to a profile. Prefer select_machine for new code."""
    if vram_gb is None or vram_gb < 4:
        return select_machine(Machine(Family.CPU, "CPU"))
    return select_machine(Machine(Family.NVIDIA, "GPU", vram=Vram(vram_gb)))


def resolved_profile(
    profile: HardwareProfile,
    *,
    cpu_cores: int | None = None,
) -> dict[str, Any]:
    cores = max(1, cpu_cores or os.cpu_count() or 1)
    if isinstance(profile.speaker, InProcessSpeaker):
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
    asr = profile.asr
    speaker = profile.speaker
    match = profile.match
    if isinstance(speaker, InProcessSpeaker):
        vad_model = speaker.vad_model
        vad_device = speaker.vad_device
    else:
        vad_model = VAD_FSMN
        vad_device = "cpu"
    if isinstance(asr, FunasrMlx):
        asr_backend = "mlx"
        asr_device = None
        asr_mlx_variant: str | None = asr.variant
    elif isinstance(asr, FunasrVllm):
        asr_backend = "vllm"
        asr_device = asr.device
        asr_mlx_variant = None
    else:
        asr_backend = "auto"
        asr_device = asr.device
        asr_mlx_variant = None
    return {
        "name": profile.name,
        "vram_gb": match.gb if isinstance(match, VramKey) else None,
        "examples": list(profile.examples),
        "asr_backend": asr_backend,
        "asr_batch_size": profile.asr_batch_size,
        "asr_job_batch_size": profile.asr_job_batch_size,
        "asr_decode_workers": asr_decode_workers,
        "vllm_memory_utilization": (
            asr.memory_utilization if isinstance(asr, FunasrVllm) else None
        ),
        "speaker_batch_size": speaker.batch_size,
        "diarize_job_batch_size": profile.diarize_job_batch_size,
        "diarize_decode_workers": decode_workers,
        "max_worker_shards": profile.max_worker_shards,
        "handler_workers": handler_workers,
        "max_queued_requests": profile.max_queued_requests,
        "notes": profile.notes,
        "validated_on_host": profile.validated_on_host,
        "cpu_cores": cores,
        "worker_shards": worker_shards,
        "worker_cpu_threads": worker_cpu_threads,
        "service_cpu_threads": service_cpu_threads,
        "family": profile.family.value,
        "asr_model": asr.model,
        "asr_device": asr_device,
        "asr_mlx_variant": asr_mlx_variant,
        "speaker_model": speaker.model,
        "speaker_device": speaker.device,
        "vad_model": vad_model,
        "vad_device": vad_device,
        "speaker_mode": "speaker_service" if profile.uses_speaker_service else "in_process",
        "apple_generation": match.generation if isinstance(match, UnifiedKey) else None,
        "unified_memory_gb": match.memory_gb if isinstance(match, UnifiedKey) else None,
    }


def _flag_device(device: str, *, cuda_index: bool) -> str:
    if cuda_index and device == "cuda":
        return "cuda:0"
    return device


def command_arguments(
    profile: HardwareProfile,
    command: str,
    *,
    cpu_cores: int | None = None,
) -> list[str]:
    config = resolved_profile(profile, cpu_cores=cpu_cores)
    asr = profile.asr
    if command == "transcribe":
        if isinstance(asr, FunasrMlx):
            return [
                "--model",
                asr.model,
                "--backend",
                "mlx",
                "--mlx-variant",
                asr.variant,
                "--batch-size",
                str(profile.asr_batch_size),
                "--job-batch-size",
                str(profile.asr_job_batch_size),
                "--decode-workers",
                str(config["asr_decode_workers"]),
            ]
        args = [
            "--model",
            asr.model,
            "--backend",
            "vllm" if isinstance(asr, FunasrVllm) else "auto",
            "--device",
            asr.device,
            "--batch-size",
            str(profile.asr_batch_size),
            "--job-batch-size",
            str(profile.asr_job_batch_size),
            "--decode-workers",
            str(config["asr_decode_workers"]),
        ]
        if isinstance(asr, FunasrVllm):
            args.extend(
                [
                    "--vllm-dtype",
                    asr.dtype,
                    "--vllm-gpu-memory-utilization",
                    str(asr.memory_utilization),
                ]
            )
        return args
    if command == "diarize":
        speaker = profile.speaker
        if isinstance(speaker, InProcessSpeaker):
            vad_model = speaker.vad_model
            vad_device = speaker.vad_device
        else:
            vad_model = VAD_FSMN
            vad_device = "cpu"
        return [
            "--speaker-model",
            speaker.model,
            "--vad-model",
            vad_model,
            "--device",
            _flag_device(speaker.device, cuda_index=True),
            "--vad-device",
            vad_device,
            "--speaker-batch-size",
            str(speaker.batch_size),
            "--job-batch-size",
            str(profile.diarize_job_batch_size),
            "--decode-workers",
            str(config["diarize_decode_workers"]),
            "--cpu-threads",
            str(config["worker_cpu_threads"]),
        ]
    if command == "speaker-service":
        if not isinstance(profile.speaker, SpeakerService):
            raise ValueError(
                f"{profile.name} runs diarization in-process and does not use "
                "speaker-service"
            )
        return [
            "--device",
            "cuda:0",
            "--max-batch-size",
            str(profile.speaker.batch_size),
            "--warmup-batch-size",
            str(profile.speaker.batch_size),
            "--cpu-threads",
            str(config["service_cpu_threads"]),
            "--handler-workers",
            str(config["handler_workers"]),
            "--max-queued-requests",
            str(profile.max_queued_requests),
        ]
    raise ValueError(f"profiles are not supported by command {command!r}")


def _run_probe(command: Sequence[str], runner: ProbeRunner | None) -> Any:
    if runner is not None:
        return runner(command)
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )


def detect_nvidia_gpu(runner: ProbeRunner | None = None) -> tuple[str, float] | None:
    try:
        result = _run_probe(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            runner,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    devices: list[tuple[str, float]] = []
    for line in str(result.stdout).splitlines():
        try:
            name, memory_mib = line.rsplit(",", 1)
            nominal_gb = round(float(memory_mib.strip()) / 1024, 1)
        except ValueError:
            continue
        if nominal_gb <= 0:
            continue
        devices.append((name.strip(), nominal_gb))
    return max(devices, key=lambda item: item[1]) if devices else None


def _detect_apple(
    *,
    platform: str | None = None,
    machine: str | None = None,
    runner: ProbeRunner | None = None,
) -> AppleChip | None:
    if (sys.platform if platform is None else platform) != "darwin":
        return None
    arch = machine if machine is not None else py_platform.machine()
    if arch not in {"arm64", "aarch64"}:
        return None
    try:
        result = _run_probe(
            ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
            runner,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    lines = str(result.stdout).splitlines()
    if len(lines) < 2:
        return None
    brand = lines[0].strip()
    brand_match = _APPLE_BRAND.match(brand)
    if brand_match is None:
        return None
    try:
        mem_bytes = int(lines[1].strip())
    except ValueError:
        return None
    if mem_bytes <= 0:
        return None
    unified_gb = round(mem_bytes / 1024**3, 1)
    if unified_gb <= 0:
        return None
    return AppleChip(int(brand_match.group(1)), unified_gb, brand)


def detect_machine(
    *,
    platform: str | None = None,
    machine: str | None = None,
    nvidia_runner: ProbeRunner | None = None,
    apple_runner: ProbeRunner | None = None,
) -> Machine:
    detected = detect_nvidia_gpu(runner=nvidia_runner)
    if detected is not None:
        name, gb = detected
        return Machine(Family.NVIDIA, name, vram=Vram(gb))
    chip = _detect_apple(platform=platform, machine=machine, runner=apple_runner)
    if chip is not None:
        return Machine(Family.APPLE, chip.brand, apple=chip)
    return Machine(Family.CPU, "CPU")


def machine_from_probe(
    *,
    cpu_only: bool = False,
    vram_gb: float | None = None,
    apple_generation: int | None = None,
    unified_gb: float | None = None,
) -> Machine:
    has_generation = apple_generation is not None
    has_unified = unified_gb is not None
    if has_generation != has_unified:
        raise ValueError("--apple-generation and --unified-gb must be set together")
    selected = sum((bool(cpu_only), vram_gb is not None, has_generation))
    if selected > 1:
        raise ValueError(
            "choose only one of --cpu-only, --vram-gb, or Apple generation and unified memory"
        )
    if vram_gb is not None:
        if vram_gb <= 0:
            raise ValueError("--vram-gb must be positive")
        return Machine(Family.NVIDIA, "GPU", vram=Vram(vram_gb))
    if has_generation:
        if apple_generation is None or apple_generation < 1:
            raise ValueError("--apple-generation must be >= 1")
        if unified_gb is None or unified_gb <= 0:
            raise ValueError("--unified-gb must be positive")
        brand = f"Apple M{apple_generation}"
        return Machine(
            Family.APPLE,
            brand,
            apple=AppleChip(apple_generation, unified_gb, brand),
        )
    if cpu_only:
        return Machine(Family.CPU, "CPU")
    return detect_machine()


def floor_note(machine: Machine, profile: HardwareProfile) -> str:
    notes = profile.notes
    chip = machine.apple
    match = profile.match
    if chip is not None and isinstance(match, UnifiedKey) and chip.unified_gb > match.memory_gb:
        notes = (
            f"{notes} {chip.unified_gb:.1f}GB unified floors to the {match.memory_gb}GB rung."
        )
    return notes


_IMPLICIT_COMMANDS = frozenset({"transcribe", "diarize", "speaker-service"})
FRIENDLY_PROFILE_LEVELS = frozenset({"auto", "quality", "speed"})


def _implicit_profile(command: str, *, platform: str | None = None) -> bool:
    current = sys.platform if platform is None else platform
    return command in _IMPLICIT_COMMANDS and current == "darwin"


def _skips_profile(command: str, argv: Sequence[str]) -> bool:
    if "-h" in argv or "--help" in argv:
        return True
    return command == "speaker-service" and "--check" in argv


@dataclass(frozen=True)
class SelectionRequest:
    """Hardware / friendly-level selection stripped from a functional command argv."""

    profile: str | None = None
    apple_generation: int | None = None
    unified_gb: float | None = None
    vram_gb: float | None = None
    cpu_only: bool = False


def extract_profile(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    """Backward-compatible helper: return only --profile and the remaining argv."""
    request, remaining = extract_selection(argv)
    return request.profile, remaining


def _take_flag(
    values: list[str],
    index: int,
    *,
    flag: str,
    cast,
) -> tuple[object, int]:
    value = values[index]
    if value == flag:
        if index + 1 >= len(values):
            raise ValueError(f"{flag} requires a value")
        return cast(values[index + 1]), index + 2
    prefix = f"{flag}="
    if value.startswith(prefix):
        return cast(value[len(prefix) :]), index + 1
    raise ValueError(f"internal flag parse error for {flag}")


def extract_selection(argv: Sequence[str]) -> tuple[SelectionRequest, list[str]]:
    profile: str | None = None
    apple_generation: int | None = None
    unified_gb: float | None = None
    vram_gb: float | None = None
    cpu_only = False
    remaining: list[str] = []
    index = 0
    values = list(argv)
    while index < len(values):
        value = values[index]
        if value == "--profile" or value.startswith("--profile="):
            parsed, index = _take_flag(values, index, flag="--profile", cast=str)
            profile = str(parsed)
            continue
        if value == "--apple-generation" or value.startswith("--apple-generation="):
            parsed, index = _take_flag(values, index, flag="--apple-generation", cast=int)
            apple_generation = int(parsed)
            continue
        if value == "--unified-gb" or value.startswith("--unified-gb="):
            parsed, index = _take_flag(values, index, flag="--unified-gb", cast=float)
            unified_gb = float(parsed)
            continue
        if value == "--vram-gb" or value.startswith("--vram-gb="):
            parsed, index = _take_flag(values, index, flag="--vram-gb", cast=float)
            vram_gb = float(parsed)
            continue
        if value == "--cpu-only":
            cpu_only = True
            index += 1
            continue
        remaining.append(value)
        index += 1
    return (
        SelectionRequest(
            profile=profile,
            apple_generation=apple_generation,
            unified_gb=unified_gb,
            vram_gb=vram_gb,
            cpu_only=cpu_only,
        ),
        remaining,
    )


def _apply_friendly_level(profile: HardwareProfile, level: str) -> HardwareProfile:
    if level == "auto":
        return profile
    if level == "quality":
        if profile.family is Family.APPLE and isinstance(profile.asr, FunasrMlx):
            return replace(profile, asr=FunasrAuto(ASR_NANO, "cpu"))
        return profile
    if level == "speed":
        return profile
    raise ValueError(f"unknown friendly profile level {level!r}")


def resolve_profile_selection(
    request: SelectionRequest,
    *,
    machine: Machine | None = None,
    platform: str | None = None,
    detect: bool = True,
) -> tuple[HardwareProfile, str | None]:
    """Resolve a catalog or friendly-level profile.

    Returns (profile, announcement_name). announcement_name is set when the
    selection should print `podtrans: using profile ...` to stderr.
    """
    name = request.profile
    has_hardware = (
        request.cpu_only
        or request.vram_gb is not None
        or request.apple_generation is not None
        or request.unified_gb is not None
    )
    concrete = name is not None and name not in FRIENDLY_PROFILE_LEVELS
    if concrete and has_hardware:
        raise ValueError(
            "concrete --profile cannot be combined with "
            "--apple-generation, --unified-gb, --vram-gb, or --cpu-only"
        )
    if concrete:
        assert name is not None
        return get_profile(name), None

    level = name or "auto"
    if level not in FRIENDLY_PROFILE_LEVELS:
        raise ValueError(
            f"unknown profile {level!r}; choose auto, quality, speed, or a catalog id"
        )
    if has_hardware:
        chosen = machine_from_probe(
            cpu_only=request.cpu_only,
            vram_gb=request.vram_gb,
            apple_generation=request.apple_generation,
            unified_gb=request.unified_gb,
        )
    elif machine is not None:
        chosen = machine
    elif detect:
        chosen = detect_machine(platform=platform)
    else:
        raise ValueError("profile selection requires hardware detection or hardware flags")
    profile = _apply_friendly_level(select_machine(chosen), level)
    label = profile.name if level == "auto" else f"{profile.name} ({level})"
    return profile, label


def _last_option(argv: Sequence[str], flag: str) -> tuple[bool, str | None]:
    prefix = f"{flag}="
    found = False
    value: str | None = None
    index = 0
    values = list(argv)
    while index < len(values):
        token = values[index]
        if token == flag:
            found = True
            if index + 1 < len(values) and not values[index + 1].startswith("--"):
                value = values[index + 1]
                index += 2
                continue
            value = None
            index += 1
            continue
        if token.startswith(prefix):
            found = True
            value = token[len(prefix) :]
        index += 1
    return found, value


def _replace_option(args: list[str], flag: str, value: str) -> list[str]:
    prefix = f"{flag}="
    updated: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == flag and index + 1 < len(args):
            updated.extend((flag, value))
            index += 2
            continue
        if token.startswith(prefix):
            updated.append(f"{flag}={value}")
            index += 1
            continue
        updated.append(token)
        index += 1
    return updated


def _drop_option(args: list[str], flag: str) -> list[str]:
    prefix = f"{flag}="
    updated: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == flag:
            index += 2 if index + 1 < len(args) else 1
            continue
        if token.startswith(prefix):
            index += 1
            continue
        updated.append(token)
        index += 1
    return updated


def _apply_mlx_benchmark_override(
    profile: HardwareProfile,
    profile_args: list[str],
    user_args: Sequence[str],
) -> list[str]:
    if not isinstance(profile.asr, FunasrMlx):
        return profile_args
    device_passed, _device = _last_option(user_args, "--device")
    backend_passed, backend = _last_option(user_args, "--backend")
    if not device_passed and not (backend_passed and backend != "mlx"):
        return profile_args
    rewritten = _replace_option(profile_args, "--model", ASR_NANO)
    rewritten = _drop_option(rewritten, "--mlx-variant")
    if backend_passed:
        return _drop_option(rewritten, "--backend")
    return _replace_option(rewritten, "--backend", "auto")


def resolve_command_argv(
    command: str,
    argv: Sequence[str],
    *,
    machine: Machine | None = None,
    platform: str | None = None,
) -> list[str]:
    args = list(argv)
    if _skips_profile(command, args):
        return args
    request, rest = extract_selection(args)
    has_hardware = (
        request.cpu_only
        or request.vram_gb is not None
        or request.apple_generation is not None
        or request.unified_gb is not None
    )
    wants_selection = (
        request.profile is not None
        or has_hardware
        or _implicit_profile(command, platform=platform)
    )
    if not wants_selection:
        return args
    profile, announcement = resolve_profile_selection(
        request,
        machine=machine,
        platform=platform,
        detect=True,
    )
    if announcement is not None:
        print(f"podtrans: using profile {announcement}", file=sys.stderr)
    profile_args = command_arguments(profile, command)
    if command == "transcribe":
        profile_args = _apply_mlx_benchmark_override(profile, profile_args, rest)
    return profile_args + rest


def profile_json(profile: HardwareProfile, *, cpu_cores: int | None = None) -> str:
    return json.dumps(
        resolved_profile(profile, cpu_cores=cpu_cores),
        ensure_ascii=False,
        indent=2,
    )
