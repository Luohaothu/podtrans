"""Unified command-line entry point for podtrans."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable

from . import __version__
from .profiles import extract_selection, resolve_command_argv


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
  transcribe              Transcribe archived audio with FunASR or an OpenAI-compatible endpoint
  diarize                 Add speaker labels to transcripts
  diarize-batch           Run concurrent diarization with one GPU model service
  speaker-service         Run or check the shared GPU embedding service

Validation and benchmarks:
  validate-archive
  validate-diarization
  benchmark-workers
  benchmark-fbank

Use `podtrans <command> --help` for command-specific options.
Functional commands accept hardware selection flags and `--profile`:
  --profile auto|quality|speed|<catalog-id>
  --apple-generation N --unified-gb GB
  --vram-gb GB | --cpu-only
Later options override profile defaults.
On Apple Silicon, transcribe, diarize, and speaker-service select a profile when
`--profile` and hardware flags are omitted. Linux keeps command defaults unless
selection flags are provided. diarize-batch always selects a profile.
Apple and CPU profiles diarize in-process.
"""
    )


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    prefix = flag + "="
    return any(arg == flag or arg.startswith(prefix) for arg in argv)


def _dependency_error(exc: ModuleNotFoundError) -> RuntimeError | None:
    if exc.name in {"funasr_mlx", "mlx"}:
        return RuntimeError(
            f"missing dependency {exc.name!r}; install `podtrans[mlx]`, "
            "the Apple Silicon extra"
        )
    if exc.name in INFERENCE_MODULES:
        return RuntimeError(
            f"missing inference dependency {exc.name!r}; install "
            "`podtrans[inference]` or `podtrans[vllm]`"
        )
    return None


def load_command(command: str) -> Callable[[list[str]], int]:
    module_name, function_name = COMMANDS[command]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        dependency_error = _dependency_error(exc)
        if dependency_error is not None:
            raise dependency_error from exc
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
        if command not in COMMANDS:
            raise ValueError(f"unknown command {command!r}")
        if command in PROFILE_AWARE_COMMANDS:
            request, rest = extract_selection(command_args)
            if (
                command == "transcribe"
                and request.profile is not None
                and _argv_has_flag(rest, "--asr-url")
            ):
                raise ValueError(
                    "--profile configures in-process FunASR; "
                    "--asr-url does not take a hardware profile"
                )
            command_args = resolve_command_argv(command, command_args)
        entry = load_command(command)
        try:
            return int(entry(command_args))
        except ModuleNotFoundError as exc:
            dependency_error = _dependency_error(exc)
            if dependency_error is not None:
                raise dependency_error from exc
            raise
    except (RuntimeError, ValueError) as exc:
        print(f"podtrans: error: {exc}", file=sys.stderr)
        return 2
