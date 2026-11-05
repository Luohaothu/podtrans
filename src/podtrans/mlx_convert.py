"""Convert local Nano weights to q8 without redistributing checkpoints.

Tokenizer sidecars come from the published MLX w4a16 variant because its extra
special tokens differ from the PyTorch ModelScope tokenizer.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from .profiles import ASR_NANO, MLX_NANO, MLX_NANO_VARIANT

Q8_BITS = 8
Q8_GROUP_SIZE = 64
Q8_MODE = "affine"
Q8_TARGETS = (
    "llm.model.layers.",
    "audio_encoder.",
    "audio_adaptor.",
    "llm.model.embed_tokens",
)
FUNASR_MLX_CONVERT_FAMILY = "fun-asr-nano"
SOURCE_RULE = "fun-asr-nano"
CONVERT_DTYPE = "bfloat16"
_SIDECAR_VARIANT = "w4a16"
_STAMP_NAME = ".podtrans-q8-stamp.json"
_TOKENIZER_FILES = ("tokenizer_config.json", "vocab.json", "merges.txt")
_SIDECAR_ALLOW_PATTERNS = tuple(
    f"{_SIDECAR_VARIANT}/{name}"
    for name in (
        "config.json",
        *_TOKENIZER_FILES,
        "generation_config.json",
        "chat_template.json",
        "preprocessor_config.json",
    )
)
_SOURCE_SHARD_BYTES = 400_000_000


def ensure_q8_checkpoint(cache: Path, *, model_pt: Path | None = None) -> Path:
    """Return a complete local q8 checkpoint, converting only on a cache miss."""
    cache = Path(cache)
    checkpoint = cache / MLX_NANO / MLX_NANO_VARIANT
    version = metadata.version("funasr-mlx")
    source = Path(model_pt) if model_pt is not None else _find_model_pt()
    if source is not None and not source.is_file():
        source = None
    quantization = {
        "bits": Q8_BITS,
        "group_size": Q8_GROUP_SIZE,
        "mode": Q8_MODE,
        "targets": list(Q8_TARGETS),
    }
    stamp = {
        "podtrans_convert_version": 1,
        "funasr_mlx_version": version,
        **quantization,
        "family": FUNASR_MLX_CONVERT_FAMILY,
        "source_rule": SOURCE_RULE,
        "dtype": CONVERT_DTYPE,
        "sidecar_variant": _SIDECAR_VARIANT,
    }
    if source is not None:
        stat = source.stat()
        stamp.update(model_pt_size=stat.st_size, model_pt_mtime_ns=stat.st_mtime_ns)
    if _cache_matches(checkpoint, stamp):
        return checkpoint
    if source is None:
        raise FileNotFoundError(
            "No usable local q8 checkpoint or Fun-ASR-Nano-2512 model.pt. "
            "Download the PyTorch weights first (~2.1 GB via ModelScope), "
            "for example by running the CPU or MPS path once."
        )

    converter = _converter(version)
    sidecars = cache / MLX_NANO / _SIDECAR_VARIANT
    if not _has_sidecars(sidecars):
        if any(
            os.environ.get(name, "").lower() not in {"", "0", "false", "no", "off"}
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        ):
            raise FileNotFoundError(
                f"Offline local q8 conversion needs published MLX w4a16 sidecars at {sidecars}"
            )
        print("[transcribe] downloading published MLX w4a16 tokenizer/config sidecars", flush=True)
        sidecars = _download_sidecars(cache)
    if not _has_sidecars(sidecars):
        raise RuntimeError(f"Published MLX tokenizer/config files are incomplete at {sidecars}")
    config = json.loads((sidecars / "config.json").read_text())
    config.pop("quantization", None)
    config["quantization"] = quantization

    build_root = checkpoint.parent / ".q8-build"
    build_root.mkdir(parents=True, exist_ok=True)
    print("[transcribe] converting local Fun-ASR-Nano model.pt to MLX q8", flush=True)
    with tempfile.TemporaryDirectory(dir=build_root) as temporary:
        work = Path(temporary)
        source_shards = work / "source"
        converted = work / MLX_NANO_VARIANT
        _write_source_shards(source, source_shards)
        converter(
            source_shards,
            converted,
            model=FUNASR_MLX_CONVERT_FAMILY,
            dtype=CONVERT_DTYPE,
            source_rule=SOURCE_RULE,
            config=config,
            quantization=quantization,
        )
        for file in sidecars.iterdir():
            if file.is_file() and file.suffix in {".json", ".txt"}:
                if not file.name.startswith("model") and file.name != "config.json":
                    shutil.copyfile(file, converted / file.name)
        if not _complete(converted):
            raise RuntimeError("Local q8 converter produced an incomplete checkpoint")
        (converted / _STAMP_NAME).write_text(json.dumps(stamp, indent=2) + "\n")
        _publish(converted, checkpoint)
    return checkpoint


def _download_sidecars(cache: Path) -> Path:
    repo = f"baicai1145/{MLX_NANO}"
    destination = cache / MLX_NANO
    options = {
        "local_dir": str(destination),
        "allow_patterns": list(_SIDECAR_ALLOW_PATTERNS),
    }
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        snapshot_download(model_id=repo, revision="master", **options)
    except Exception as exc:
        from huggingface_hub import snapshot_download

        print(
            f"[transcribe] ModelScope sidecar fetch failed ({type(exc).__name__}); "
            "trying Hugging Face",
            flush=True,
        )
        snapshot_download(repo_id=repo, revision="main", **options)
    return destination / _SIDECAR_VARIANT


def _find_model_pt() -> Path | None:
    roots = [Path.home() / ".cache" / "modelscope"]
    if cache := os.environ.get("MODELSCOPE_CACHE"):
        roots.insert(0, Path(cache).expanduser())
    for root in roots:
        for base in (root, root / "models"):
            for name in (ASR_NANO, ASR_NANO.replace("/", "--")):
                repo = base / name
                candidates = [repo / "model.pt", repo / "snapshots" / "master" / "model.pt"]
                candidates.extend(sorted(repo.glob("snapshots/*/model.pt")))
                for candidate in candidates:
                    if candidate.is_file():
                        return candidate
    return None


def _has_sidecars(directory: Path) -> bool:
    return all(
        (directory / name).is_file() and (directory / name).stat().st_size > 0
        for name in ("config.json", *_TOKENIZER_FILES)
    )


def _complete(directory: Path) -> bool:
    return _has_sidecars(directory) and any(
        shard.is_file() and shard.stat().st_size > 0 for shard in directory.glob("*.safetensors")
    )


def _cache_matches(checkpoint: Path, expected: dict[str, Any]) -> bool:
    if not _complete(checkpoint):
        return False
    try:
        stamp = json.loads((checkpoint / _STAMP_NAME).read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(stamp, dict)
        and all(stamp.get(key) == value for key, value in expected.items())
        and all(isinstance(stamp.get(key), int) for key in ("model_pt_size", "model_pt_mtime_ns"))
    )


def _converter(version: str) -> Callable[..., Any]:
    try:
        from funasr_mlx.checkpoints import convert_checkpoint

        inspect.signature(convert_checkpoint).bind(
            Path("source"),
            Path("destination"),
            model=FUNASR_MLX_CONVERT_FAMILY,
            dtype=CONVERT_DTYPE,
            source_rule=SOURCE_RULE,
            config={},
            quantization={},
        )
    except (ImportError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"funasr-mlx {version} is incompatible with podtrans local q8 conversion; "
            "need >=0.2.3,<0.3 with convert_checkpoint(source, destination, model, dtype, "
            "source_rule, config, quantization)"
        ) from exc
    return convert_checkpoint


def _write_source_shards(model_pt: Path, destination: Path) -> None:
    import torch
    from safetensors.torch import save_file

    state = torch.load(model_pt, map_location="cpu", weights_only=True, mmap=True)
    destination.mkdir()
    shard: dict[str, Any] = {}
    size = 0
    number = 0
    for name in sorted(state):
        if name.startswith(("ctc_decoder.", "ctc.")):
            continue
        tensor = state[name].contiguous()
        tensor_bytes = tensor.numel() * tensor.element_size()
        if shard and size + tensor_bytes > _SOURCE_SHARD_BYTES:
            save_file(shard, str(destination / f"part-{number:03d}.safetensors"))
            shard, size, number = {}, 0, number + 1
        shard[name] = tensor
        size += tensor_bytes
    if not shard:
        raise RuntimeError("Fun-ASR-Nano model.pt contains no convertible weights")
    save_file(shard, str(destination / f"part-{number:03d}.safetensors"))


def _publish(converted: Path, checkpoint: Path) -> None:
    if not checkpoint.exists():
        converted.rename(checkpoint)
        return
    import ctypes

    # Darwin RENAME_SWAP preserves the previous directory if publication fails.
    # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/stdio.h
    swap = ctypes.CDLL(None, use_errno=True).renamex_np
    swap.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    swap.restype = ctypes.c_int
    if swap(os.fsencode(converted), os.fsencode(checkpoint), 0x00000002) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(checkpoint))
