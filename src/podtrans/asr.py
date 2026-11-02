from __future__ import annotations

import ipaddress
import json
import math
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import archive

DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
_DEFAULT_DEVICE = "cuda"
_DEFAULT_MAX_LENGTH = 256
_DEFAULT_BATCH_SIZE = 4
_DEFAULT_VLLM_DTYPE = "bfloat16"
_DEFAULT_TENSOR_PARALLEL_SIZE = 1
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.8
_DEFAULT_MAX_MODEL_LEN = 4096
_DEFAULT_REMOTE_TIMEOUT_SECONDS = 600.0
_TRANSPORT_ATTEMPTS = 4
_RETRYABLE_HTTP_STATUS = frozenset({429, 503})
_ERROR_BODY_LIMIT = 500

SHARED_ASR_FLAGS = frozenset({
    "--model",
    "--language",
    "--batch-size",
    "--chunk-seconds",
    "--job-batch-size",
    "--decode-workers",
    "--channel",
    "--episode-dir",
    "--limit",
    "--sort",
    "--include-done",
    "--retry-errors",
    "--num-shards",
    "--shard-index",
    "--temp-dir",
    "--verify-only",
})
IN_PROCESS_ASR_FLAGS = frozenset({
    "--backend",
    "--device",
    "--mlx-variant",
    "--batch-size-s",
    "--max-length",
    "--tensor-parallel-size",
    "--vllm-dtype",
    "--vllm-gpu-memory-utilization",
    "--vllm-max-model-len",
    "--vllm-enforce-eager",
})
REMOTE_ASR_FLAGS = frozenset({"--asr-url"})

_IN_PROCESS_FIELDS = (
    ("backend", "--backend"),
    ("device", "--device"),
    ("mlx_variant", "--mlx-variant"),
    ("batch_size_s", "--batch-size-s"),
    ("max_length", "--max-length"),
    ("tensor_parallel_size", "--tensor-parallel-size"),
    ("vllm_dtype", "--vllm-dtype"),
    ("vllm_gpu_memory_utilization", "--vllm-gpu-memory-utilization"),
    ("vllm_max_model_len", "--vllm-max-model-len"),
    ("vllm_enforce_eager", "--vllm-enforce-eager"),
)


@dataclass(frozen=True)
class WordStamp:
    token: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class AsrSource:
    model: str
    backend: str
    device: str
    endpoint: str | None = None


@dataclass(frozen=True)
class AutoModelAsr:
    model: str
    device: str
    language: str | None
    batch_size: int
    batch_size_s: int | None
    max_length: int | None


@dataclass(frozen=True)
class VllmOptions:
    dtype: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    enforce_eager: bool


@dataclass(frozen=True)
class VllmAsr:
    model: str
    device: str
    language: str | None
    batch_size: int
    max_length: int | None
    options: VllmOptions


@dataclass(frozen=True)
class RemoteAsr:
    base_url: str
    model: str
    api_key: str | None = field(repr=False, default=None)
    language: str | None = None
    batch_size: int = _DEFAULT_BATCH_SIZE
    timeout_seconds: float = _DEFAULT_REMOTE_TIMEOUT_SECONDS

    @property
    def transcription_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/audio/transcriptions"


@dataclass(frozen=True)
class MlxAsr:
    model: str
    variant: str
    batch_size: int


AsrSpec = AutoModelAsr | VllmAsr | RemoteAsr | MlxAsr


class AsrSession(Protocol):
    source: AsrSource
    batch_size: int

    def prepare(self) -> None:
        """Load local models before any job work. Remote sessions are a no-op."""
        raise NotImplementedError

    def transcribe(self, wav_paths: Sequence[str]) -> list[dict[str, Any]]:
        """Return one dict per path, in the same order.

        Remote dicts include timestamps. Local dicts are FunASR payloads.
        Read every wav before returning. Raise instead of returning a short list.
        """
        raise NotImplementedError


class _FormatRejected(Exception):
    pass


def parse_asr_spec(args: Any) -> AsrSpec:
    batch_size = _require_positive_int(getattr(args, "batch_size", None), "--batch-size")
    if args.asr_url:
        for attr, flag in _IN_PROCESS_FIELDS:
            if getattr(args, attr) is not None:
                raise ValueError(f"{flag} cannot be combined with --asr-url")
        model = getattr(args, "model", None)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("--model is required with --asr-url")
        base_url = _asr_endpoint(args.asr_url)
        language = getattr(args, "language", None)
        if language in (None, "", "auto"):
            language = None
        api_key = os.environ.get("PODTRANS_ASR_API_KEY")
        if api_key == "":
            api_key = None
        if api_key is not None and any(char in api_key for char in "\r\n"):
            raise ValueError("PODTRANS_ASR_API_KEY must not contain a newline")
        if api_key is not None and not _key_transport_is_safe(base_url):
            raise ValueError(
                "PODTRANS_ASR_API_KEY requires an https --asr-url unless the host is loopback"
            )
        return RemoteAsr(
            base_url=base_url,
            model=model.strip(),
            api_key=api_key,
            language=language,
            batch_size=batch_size,
        )

    backend = getattr(args, "backend", None) or "auto"
    if backend == "mlx":
        model = getattr(args, "model", None)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("--model is required with --backend mlx")
        variant = getattr(args, "mlx_variant", None)
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError("--mlx-variant is required with --backend mlx")
        for attr, flag in _IN_PROCESS_FIELDS:
            if attr in {"backend", "mlx_variant"}:
                continue
            if getattr(args, attr, None) is not None:
                raise ValueError(f"{flag} cannot be combined with --backend mlx")
        language = getattr(args, "language", None)
        if language not in (None, "", "auto"):
            raise ValueError("--language must be omitted or auto with --backend mlx")
        return MlxAsr(model=model.strip(), variant=variant.strip(), batch_size=batch_size)

    model = getattr(args, "model", None) or DEFAULT_MODEL
    device = getattr(args, "device", None) or _DEFAULT_DEVICE
    language = getattr(args, "language", None) or None
    max_length = getattr(args, "max_length", None)
    if max_length is None:
        max_length = _DEFAULT_MAX_LENGTH
    if backend == "auto":
        return AutoModelAsr(
            model=model,
            device=device,
            language=language,
            batch_size=batch_size,
            batch_size_s=getattr(args, "batch_size_s", None),
            max_length=max_length,
        )
    if backend == "vllm":
        eager = getattr(args, "vllm_enforce_eager", None)
        return VllmAsr(
            model=model,
            device=device,
            language=language,
            batch_size=batch_size,
            max_length=max_length,
            options=VllmOptions(
                dtype=getattr(args, "vllm_dtype", None) or _DEFAULT_VLLM_DTYPE,
                tensor_parallel_size=(
                    _DEFAULT_TENSOR_PARALLEL_SIZE
                    if getattr(args, "tensor_parallel_size", None) is None
                    else args.tensor_parallel_size
                ),
                gpu_memory_utilization=(
                    _DEFAULT_GPU_MEMORY_UTILIZATION
                    if getattr(args, "vllm_gpu_memory_utilization", None) is None
                    else args.vllm_gpu_memory_utilization
                ),
                max_model_len=(
                    _DEFAULT_MAX_MODEL_LEN
                    if getattr(args, "vllm_max_model_len", None) is None
                    else args.vllm_max_model_len
                ),
                enforce_eager=False if eager is None else bool(eager),
            ),
        )
    raise ValueError(f"unsupported --backend {backend!r}")


def open_asr(spec: AsrSpec) -> AsrSession:
    if isinstance(spec, RemoteAsr):
        return _RemoteAsrSession(spec)
    if isinstance(spec, MlxAsr):
        return _MlxAsrSession(spec)
    return _LocalAsrSession(spec)


def _parse_transcription_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("transcription response must be a JSON object")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("transcription response is missing text")
    stamps = _stamps_for_payload(payload, text)
    return {
        "text": text,
        "timestamps": [
            {
                "token": stamp.token,
                "start_time": stamp.start_time,
                "end_time": stamp.end_time,
            }
            for stamp in stamps
        ],
    }


def _stamps_for_payload(payload: dict[str, Any], text: str) -> list[WordStamp]:
    words = _dict_list(payload.get("words"), "words")
    allow_text = False
    if words is None:
        segments = payload.get("segments")
        if segments is None:
            return []
        segment_list = _dict_list(segments, "segments")
        if segment_list is None:
            return []
        flattened: list[dict[str, Any]] = []
        for segment in segment_list:
            nested = segment.get("words")
            if nested is None:
                continue
            flattened.extend(_dict_list(nested, "words") or [])
        if flattened:
            words = flattened
        else:
            words = segment_list
            allow_text = True
    raw_tokens: list[str] = []
    times: list[tuple[float, float]] = []
    for item in words:
        raw_tokens.append(_item_token(item, allow_text=allow_text))
        times.append(_item_times(item))
    stripped = [token.lstrip(" ") for token in raw_tokens]
    chosen = raw_tokens if "".join(raw_tokens) == text else stripped
    return [
        WordStamp(token=token, start_time=start, end_time=end)
        for token, (start, end) in zip(chosen, times, strict=True)
    ]


def _dict_list(value: Any, label: str) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"transcription {label} must be a list")
    if not value:
        return None
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"transcription {label} must be objects")
    return list(value)


def _item_token(item: dict[str, Any], *, allow_text: bool) -> str:
    for key in ("word", "token"):
        if item.get(key) is not None:
            return str(item[key])
    if allow_text and item.get("text") is not None:
        return str(item["text"])
    raise ValueError("transcription token is missing")


def _item_times(item: dict[str, Any]) -> tuple[float, float]:
    return (_pick_time(item, "start", "start_time"), _pick_time(item, "end", "end_time"))


def _pick_time(item: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in item and item[key] is not None:
            return _finite_number(item[key])
    raise ValueError("transcription timestamp is missing")


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("transcription timestamp is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("transcription timestamp is not finite")
    return number


def _require_positive_int(value: Any, flag: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{flag} must be a positive integer")
    return value


def _asr_endpoint(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError("--asr-url must be an http or https URL")
    parsed = urllib.parse.urlparse(raw.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--asr-url must be an http or https URL")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError("--asr-url must not include userinfo")
    if parsed.query:
        raise ValueError("--asr-url must not include a query string")
    if parsed.fragment:
        raise ValueError("--asr-url must not include a fragment")
    path = parsed.path.rstrip("/")
    transcription_suffix = "/audio/transcriptions"
    if path.endswith(transcription_suffix):
        parent = path[: -len(transcription_suffix)].rstrip("/")
        return _url_with_path(parsed, parent)
    if path.endswith("/v1"):
        return _url_with_path(parsed, path)
    raise ValueError("--asr-url must end in /v1 or /audio/transcriptions")


def _url_with_path(parsed: urllib.parse.ParseResult, path: str) -> str:
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _key_transport_is_safe(base_url: str) -> bool:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http" or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


_OPENER = urllib.request.build_opener(_NoRedirectHandler)


def _open(request: urllib.request.Request, *, timeout: float):
    return _OPENER.open(request, timeout=timeout)


class _RemoteAsrSession:
    def __init__(self, spec: RemoteAsr) -> None:
        self.source = AsrSource(
            model=spec.model,
            backend="openai",
            device="remote",
            endpoint=spec.base_url,
        )
        self.batch_size = spec.batch_size
        self._spec = spec

    def prepare(self) -> None:
        return None

    def transcribe(self, wav_paths: Sequence[str]) -> list[dict[str, Any]]:
        paths = [str(path) for path in wav_paths]
        if not paths:
            return []
        results: list[dict[str, Any] | None] = [None] * len(paths)
        errors: list[BaseException | None] = [None] * len(paths)

        def worker(index: int, path: str) -> None:
            try:
                wav_bytes = Path(path).read_bytes()
                results[index] = self._post_transcription(wav_bytes, Path(path).name)
            except Exception as exc:
                errors[index] = exc

        with ThreadPoolExecutor(max_workers=len(paths)) as executor:
            futures = [
                executor.submit(worker, index, path) for index, path in enumerate(paths)
            ]
            for future in futures:
                future.result()
        for error in errors:
            if error is not None:
                raise error
        parsed = [item for item in results if item is not None]
        if len(parsed) != len(paths):
            raise RuntimeError(
                f"transcription returned {len(parsed)} results for {len(paths)} inputs"
            )
        return parsed

    def _post_transcription(self, wav_bytes: bytes, filename: str) -> dict[str, Any]:
        fields = [("model", self._spec.model)]
        if self._spec.language is not None:
            fields.append(("language", self._spec.language))
        try:
            return self._post_fields(
                wav_bytes,
                filename,
                [
                    *fields,
                    ("response_format", "verbose_json"),
                    ("timestamp_granularities[]", "word"),
                ],
                fallback_on_400=True,
            )
        except _FormatRejected:
            return self._post_fields(
                wav_bytes,
                filename,
                [*fields, ("response_format", "json")],
                fallback_on_400=False,
            )

    def _post_fields(
        self,
        wav_bytes: bytes,
        filename: str,
        fields: list[tuple[str, str]],
        *,
        fallback_on_400: bool,
    ) -> dict[str, Any]:
        last_transport_error: BaseException | None = None
        for attempt in range(_TRANSPORT_ATTEMPTS):
            try:
                return self._post_once(wav_bytes, filename, fields, fallback_on_400=fallback_on_400)
            except _FormatRejected:
                raise
            except urllib.error.HTTPError as exc:
                detail = _http_error_text(exc)
                if exc.code in _RETRYABLE_HTTP_STATUS and attempt + 1 < _TRANSPORT_ATTEMPTS:
                    last_transport_error = RuntimeError(detail)
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise RuntimeError(detail) from None
            except urllib.error.URLError as exc:
                last_transport_error = exc
                if attempt + 1 < _TRANSPORT_ATTEMPTS:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise RuntimeError(f"ASR transcription failed: {exc.reason}") from None
        raise RuntimeError(f"ASR transcription failed: {last_transport_error}")

    def _post_once(
        self,
        wav_bytes: bytes,
        filename: str,
        fields: list[tuple[str, str]],
        *,
        fallback_on_400: bool,
    ) -> dict[str, Any]:
        body, boundary = _multipart_body(fields, filename, wav_bytes)
        request = urllib.request.Request(
            self._spec.transcription_url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        if self._spec.api_key is not None:
            request.add_header("Authorization", f"Bearer {self._spec.api_key}")
        try:
            with _open(request, timeout=self._spec.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 400 and fallback_on_400:
                raise _FormatRejected(_http_error_text(exc)) from None
            raise
        return _parse_transcription_payload(_decode_json_object(raw))


class _MlxAsrSession:
    def __init__(self, spec: MlxAsr) -> None:
        self.source = AsrSource(
            model=spec.model,
            backend="mlx",
            device="mlx",
            endpoint=None,
        )
        self.batch_size = spec.batch_size
        self._spec = spec
        self._worker: Any = None

    def prepare(self) -> None:
        if self._worker is not None:
            return
        from funasr_mlx.api import TranscribeWorker, download_model

        cache = archive.ROOT / "models" / "funasr-mlx"
        downloaded = download_model(
            self._spec.model,
            variant=self._spec.variant,
            dest=cache,
        )
        checkpoint = Path(downloaded)
        if checkpoint.name != self._spec.variant:
            raise ValueError(
                f"funasr-mlx returned {checkpoint.name!r}, "
                f"expected variant {self._spec.variant!r}"
            )
        # The variant directory is named w4a16, so family cannot be inferred from it.
        self._worker = TranscribeWorker(checkpoint, family="fun-asr-nano")

    def transcribe(self, wav_paths: Sequence[str]) -> list[dict[str, Any]]:
        paths = [str(path) for path in wav_paths]
        if not paths:
            return []
        self.prepare()
        results: list[dict[str, Any]] = []
        for path in paths:
            payload = self._worker.transcribe(path)
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str):
                raise RuntimeError("funasr-mlx transcription is missing text")
            results.append({"text": text, "timestamps": []})
        if len(results) != len(paths):
            raise RuntimeError(
                f"transcription returned {len(results)} results for {len(paths)} inputs"
            )
        return results


class _LocalAsrSession:
    def __init__(self, spec: AutoModelAsr | VllmAsr) -> None:
        self._spec = spec
        self._force_single = False
        self.batch_size = spec.batch_size
        self.source = AsrSource(
            model=spec.model,
            backend="vllm" if isinstance(spec, VllmAsr) else "auto_model",
            device=spec.device,
            endpoint=None,
        )
        self._model: Any = None

    def prepare(self) -> None:
        self._ensure_model()

    def transcribe(self, wav_paths: Sequence[str]) -> list[dict[str, Any]]:
        paths = [str(path) for path in wav_paths]
        if not paths:
            return []
        if self._force_single and len(paths) > 1:
            results: list[dict[str, Any]] = []
            for path in paths:
                results.extend(self.transcribe([path]))
            return results
        self._ensure_model()
        try:
            raw = self._call_generate(paths)
        except NotImplementedError:
            if len(paths) == 1:
                raise
            self._force_single = True
            print(
                "[transcribe] multi-chunk generate is unsupported; sending one wav at a time",
                flush=True,
            )
            return self.transcribe(paths)
        normalized = _normalize_results(raw, len(paths))
        if len(normalized) != len(paths):
            raise RuntimeError(
                f"model returned {len(normalized)} results for {len(paths)} inputs"
            )
        return normalized

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        spec = self._spec
        if isinstance(spec, VllmAsr):
            from funasr.auto.auto_model_vllm import AutoModelVLLM

            self._model = AutoModelVLLM(
                model=spec.model,
                device=spec.device,
                dtype=spec.options.dtype,
                tensor_parallel_size=spec.options.tensor_parallel_size,
                gpu_memory_utilization=spec.options.gpu_memory_utilization,
                max_model_len=spec.options.max_model_len,
                enforce_eager=spec.options.enforce_eager,
            )
            return
        from funasr import AutoModel

        self._model = AutoModel(model=spec.model, device=spec.device, disable_update=True)

    def _call_generate(self, wav_paths: list[str]) -> Any:
        spec = self._spec
        kwargs: dict[str, Any] = {}
        if spec.language:
            kwargs["language"] = spec.language
        if spec.max_length is not None:
            kwargs["max_length"] = spec.max_length
        vllm = isinstance(spec, VllmAsr)
        if not vllm:
            kwargs["batch_size"] = max(1, min(len(wav_paths), spec.batch_size))
            if isinstance(spec, AutoModelAsr) and spec.batch_size_s is not None:
                kwargs["batch_size_s"] = spec.batch_size_s
        try:
            return self._generate(wav_paths, kwargs, vllm=vllm)
        except TypeError:
            kwargs.pop("batch_size_s", None)
            return self._generate(wav_paths, kwargs, vllm=vllm)

    def _generate(self, wav_paths: list[str], kwargs: dict[str, Any], *, vllm: bool) -> Any:
        if vllm:
            return self._model.generate(wav_paths, **kwargs)
        target: str | list[str] = wav_paths if len(wav_paths) > 1 else wav_paths[0]
        return self._model.generate(input=target, **kwargs)


def _normalize_results(raw: Any, expected: int) -> list[dict[str, Any]]:
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


def _multipart_body(
    fields: list[tuple[str, str]],
    filename: str,
    wav_bytes: bytes,
) -> tuple[bytes, str]:
    boundary = "podtrans" + secrets.token_hex(16)
    for _ in range(3):
        if boundary.encode("ascii") not in wav_bytes:
            break
        boundary = "podtrans" + secrets.token_hex(16)
    safe_name = filename.replace("\\", "/").rsplit("/", 1)[-1].replace('"', "") or "audio.wav"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        )
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + wav_bytes
        + b"\r\n"
    )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def _decode_json_object(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("ASR transcription failed: invalid JSON response") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("ASR transcription failed: response is not a JSON object")
    return payload


def _http_error_text(exc: urllib.error.HTTPError) -> str:
    raw = exc.read() if exc.fp is not None else b""
    body = raw.decode("utf-8", errors="replace").strip()
    if len(body) > _ERROR_BODY_LIMIT:
        body = body[:_ERROR_BODY_LIMIT]
    return f"ASR transcription failed: HTTP {exc.code} {body}".rstrip()
