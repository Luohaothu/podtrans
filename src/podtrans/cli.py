"""Unified command-line entry point for podtrans."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Callable

from . import __version__
from .profiles import (
    PROFILES,
    command_arguments,
    detect_nvidia_gpu,
    get_profile,
    profile_json,
    resolved_profile,
    select_profile,
)


Command = tuple[str, str]
COMMANDS: dict[str, Command] = {
    "archive": ("podtrans.archive", "main"),
    "transcribe": ("podtrans.transcribe", "main"),
    "diarize": ("podtrans.diarize", "main"),
    "diarize-batch": ("podtrans.batch_diarize", "main"),
    "speaker-service": ("podtrans.speaker_service", "main"),
    "validate-archive": ("podtrans.tools.validate_archive", "main"),
    "validate-diarization": (
        "podtrans.tools.validate_diarization",
        "main",
    ),
    "benchmark-workers": ("podtrans.tools.benchmark_workers", "main"),
    "benchmark-fbank": ("podtrans.tools.benchmark_fbank", "main"),
}
PROFILE_AWARE_COMMANDS = {"transcribe", "diarize", "speaker-service"}
INFERENCE_MODULES = {
    "numpy",
    "torch",
    "torchaudio",
    "funasr",
    "modelscope",
    "imageio_ffmpeg",
    "vllm",
}


def print_help() -> None:
    print(
        """usage: podtrans <command> [options]

Podcast archiving, transcription, and speaker diarization.

Core commands:
  archive                 Archive RSS metadata and podcast audio
  transcribe              Transcribe archived audio with FunASR
  diarize                 Add speaker labels to transcripts
  diarize-batch           Run concurrent diarization with one GPU model service
  speaker-service         Run or check the shared GPU embedding service

Configuration:
  profiles                List built-in hardware profiles
  recommend               Detect hardware and recommend a profile

Validation and benchmarks:
  validate-archive
  validate-diarization
  benchmark-workers
  benchmark-fbank

Use `podtrans <command> --help` for command-specific options.
Profile-aware commands accept `--profile NAME`; later options override it.
"""
    )


def extract_profile(argv: list[str]) -> tuple[str | None, list[str]]:
    profile: str | None = None
    remaining: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--profile":
            if index + 1 >= len(argv):
                raise ValueError("--profile requires a value")
            profile = argv[index + 1]
            index += 2
            continue
        if value.startswith("--profile="):
            profile = value.split("=", 1)[1]
            index += 1
            continue
        remaining.append(value)
        index += 1
    return profile, remaining


def profiles_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="podtrans profiles")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--cpu-cores", type=int)
    args = parser.parse_args(argv)
    if args.cpu_cores is not None and args.cpu_cores <= 0:
        parser.error("--cpu-cores must be positive")
    if args.json:
        import json

        payload = [
            resolved_profile(profile, cpu_cores=args.cpu_cores)
            for profile in PROFILES.values()
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print("PROFILE    VRAM   ASR       SPK_BATCH  MAX_WORKERS  EXAMPLES")
    for profile in PROFILES.values():
        config = resolved_profile(profile, cpu_cores=args.cpu_cores)
        vram = "CPU" if profile.vram_gb is None else f"{profile.vram_gb}GB"
        examples = "; ".join(profile.examples)
        print(
            f"{profile.name:<10} {vram:<6} {profile.asr_backend:<9} "
            f"{profile.speaker_batch_size:<10} "
            f"{config['worker_shards']:<12} {examples}"
        )
    return 0


def recommend_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="podtrans recommend")
    parser.add_argument("--vram-gb", type=float)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--cpu-cores", type=int)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true")
    output.add_argument("--name-only", action="store_true")
    args = parser.parse_args(argv)
    if args.vram_gb is not None and args.vram_gb <= 0:
        parser.error("--vram-gb must be positive")
    if args.cpu_cores is not None and args.cpu_cores <= 0:
        parser.error("--cpu-cores must be positive")

    detected_name: str | None = None
    detected_vram = args.vram_gb
    if args.cpu_only:
        detected_vram = None
    elif detected_vram is None:
        detected = detect_nvidia_gpu()
        if detected is not None:
            detected_name, detected_vram = detected

    profile = select_profile(detected_vram)
    if args.name_only:
        print(profile.name)
        return 0
    if args.json:
        print(profile_json(profile, cpu_cores=args.cpu_cores))
        return 0

    hardware = (
        f"{detected_name or 'GPU'} ({detected_vram:.1f}GB)"
        if detected_vram is not None
        else f"CPU ({args.cpu_cores or os.cpu_count() or 1} cores)"
    )
    config = resolved_profile(profile, cpu_cores=args.cpu_cores)
    print(f"hardware: {hardware}")
    print(f"profile: {profile.name}")
    print(f"notes: {profile.notes}")
    print()
    print(f"podtrans transcribe --profile {profile.name}")
    if profile.name == "cpu":
        print(f"podtrans diarize --profile {profile.name}")
    else:
        print(f"podtrans diarize-batch --profile {profile.name}")
        print(f"recommended worker shards: {config['worker_shards']}")
    return 0


def load_command(command: str) -> Callable[[list[str]], int]:
    module_name, function_name = COMMANDS[command]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in INFERENCE_MODULES:
            raise RuntimeError(
                f"missing inference dependency {exc.name!r}; install "
                "`podtrans[inference]` or `podtrans[vllm]`"
            ) from exc
        raise
    return getattr(module, function_name)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print_help()
        return 0
    if args[0] in {"-V", "--version"}:
        print(f"podtrans {__version__}")
        return 0
    if args[0] == "help":
        if len(args) == 1:
            print_help()
            return 0
        args = [args[1], "--help", *args[2:]]

    command, command_args = args[0], args[1:]
    try:
        if command == "profiles":
            return profiles_main(command_args)
        if command == "recommend":
            return recommend_main(command_args)
        if command not in COMMANDS:
            raise ValueError(f"unknown command {command!r}")
        profile_name = None
        if command in PROFILE_AWARE_COMMANDS:
            profile_name, command_args = extract_profile(command_args)
        if profile_name is not None:
            profile = get_profile(profile_name)
            command_args = command_arguments(profile, command) + command_args
        entry = load_command(command)
        try:
            return int(entry(command_args))
        except ModuleNotFoundError as exc:
            if exc.name in INFERENCE_MODULES:
                raise RuntimeError(
                    f"missing inference dependency {exc.name!r}; install "
                    "`podtrans[inference]` or `podtrans[vllm]`"
                ) from exc
            raise
    except (RuntimeError, ValueError) as exc:
        print(f"podtrans: error: {exc}", file=sys.stderr)
        return 2
