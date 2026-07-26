#!/usr/bin/env python3
"""Serve one FP32 ERes2NetV2 instance over a local Unix socket."""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Any

import numpy as np

from .diarize import (
    DEFAULT_SPEAKER_MODEL,
    ROOT,
    SPEAKER_SERVICE_AUTHKEY,
    auto_model_kwargs,
    cached_model_path,
    utc_now,
    write_json_atomic,
)


DEFAULT_SOCKET = Path("/tmp/podtrans-speaker-embedding.sock")
DEFAULT_STATUS = ROOT / "_metadata" / "speaker_embedding_service_status.json"
AUTHKEY = SPEAKER_SERVICE_AUTHKEY


@dataclass
class PreparedRequest:
    features: Any
    feature_seconds: float
    enqueued_at: float
    event: threading.Event
    embeddings: np.ndarray | None = None
    metrics: dict[str, Any] | None = None
    error: BaseException | None = None


def request(
    socket_path: Path,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    connection = Client(str(socket_path), family="AF_UNIX", authkey=AUTHKEY)
    try:
        connection.send(payload)
        if not connection.poll(timeout_seconds):
            raise TimeoutError(
                f"speaker service did not respond within {timeout_seconds}s"
            )
        response = connection.recv()
    finally:
        connection.close()
    if not isinstance(response, dict):
        raise RuntimeError(f"invalid speaker service response: {type(response)!r}")
    return response


class SpeakerEmbeddingServer:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from funasr import AutoModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the speaker embedding service")
        if args.allow_tf32:
            raise RuntimeError("TF32 is disabled by the strict FP32 precision policy")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(args.handler_workers, 4)))

        model_path = args.speaker_model_path or cached_model_path(args.speaker_model)
        kwargs = auto_model_kwargs(
            args.speaker_model,
            model_path=model_path,
            hub=args.hub,
            device=args.device,
            cpu_threads=args.cpu_threads,
        )
        load_started = time.perf_counter()
        auto_model = AutoModel(**kwargs)
        self.model = auto_model.model
        self.model.eval()
        parameter_dtypes = {str(parameter.dtype) for parameter in self.model.parameters()}
        if parameter_dtypes != {"torch.float32"}:
            raise RuntimeError(
                f"official speaker model is not strict FP32: {sorted(parameter_dtypes)}"
            )

        self.args = args
        self.torch = torch
        self.model_path = model_path
        self.load_seconds = time.perf_counter() - load_started
        self.status_lock = threading.Lock()
        self.request_queue: queue.Queue[PreparedRequest] = queue.Queue(
            maxsize=args.max_queued_requests
        )
        self.gpu_thread = threading.Thread(
            target=self.gpu_loop,
            name="speaker-gpu",
            daemon=True,
        )
        self.executor = ThreadPoolExecutor(
            max_workers=args.handler_workers,
            thread_name_prefix="speaker-request",
        )
        self.listener: Listener | None = None
        self.stop_event = threading.Event()
        self.started_at = utc_now()
        self.requests = 0
        self.windows = 0
        self.gpu_batches = 0
        self.max_observed_batch_size = 0
        self.max_peak_allocated_mb = 0.0
        self.max_peak_reserved_mb = 0.0
        self.last_request: dict[str, Any] | None = None
        self.warmup: dict[str, Any] | None = None
        self.write_status("loading")

    def status_payload(self, state: str) -> dict[str, Any]:
        properties = self.torch.cuda.get_device_properties(self.args.device)
        return {
            "state": state,
            "generated_at": utc_now(),
            "started_at": self.started_at,
            "pid": os.getpid(),
            "socket": str(self.args.socket),
            "speaker_model": self.args.speaker_model,
            "speaker_model_path": (
                str(self.model_path) if self.model_path is not None else None
            ),
            "speaker_model_precision": "fp32",
            "allow_tf32": False,
            "device": self.args.device,
            "device_name": properties.name,
            "total_memory_mb": round(properties.total_memory / 1048576, 1),
            "max_batch_size": self.args.max_batch_size,
            "model_load_seconds": round(self.load_seconds, 6),
            "requests": self.requests,
            "windows": self.windows,
            "gpu_batches": self.gpu_batches,
            "max_observed_batch_size": self.max_observed_batch_size,
            "max_peak_allocated_mb": round(self.max_peak_allocated_mb, 1),
            "max_peak_reserved_mb": round(self.max_peak_reserved_mb, 1),
            "last_request": self.last_request,
            "warmup": self.warmup,
        }

    def write_status(self, state: str = "ready") -> None:
        self.args.status_file.parent.mkdir(parents=True, exist_ok=True)
        with self.status_lock:
            write_json_atomic(self.args.status_file, self.status_payload(state))

    def prepare_features(self, audio: np.ndarray | list[np.ndarray]) -> Any:
        from funasr.models.campplus.utils import extract_feature

        tensors = [
            self.torch.from_numpy(np.asarray(row, dtype=np.float32).reshape(-1))
            for row in audio
        ]
        features, _lengths, _times = extract_feature(tensors)
        return features.pin_memory()

    def validate_audio(self, audio: np.ndarray | list[np.ndarray]) -> int:
        if isinstance(audio, np.ndarray):
            if audio.dtype != np.float32 or audio.ndim != 2:
                raise ValueError(
                    f"audio array must be float32 [N, T], got {audio.dtype} "
                    f"{tuple(audio.shape)}"
                )
            batch_size = int(audio.shape[0])
        elif isinstance(audio, list):
            if not all(
                isinstance(row, np.ndarray)
                and row.dtype == np.float32
                and row.ndim == 1
                and row.size > 0
                for row in audio
            ):
                raise ValueError("audio list must contain non-empty float32 vectors")
            batch_size = len(audio)
        else:
            raise ValueError("audio must be a NumPy matrix or a list of vectors")
        if batch_size <= 0 or batch_size > self.args.max_batch_size:
            raise ValueError(
                f"batch size must be in [1, {self.args.max_batch_size}], got {batch_size}"
            )

        return batch_size

    def forward_features(self, features: Any) -> tuple[np.ndarray, dict[str, Any]]:
        batch_size = int(features.shape[0])
        self.torch.cuda.reset_peak_memory_stats()
        gpu_started = time.perf_counter()
        try:
            speech = features.to(
                device=self.args.device,
                dtype=self.torch.float32,
                non_blocking=True,
            )
            with self.torch.inference_mode():
                embeddings = self.model.forward(speech)
            output = embeddings.detach().float().cpu().numpy()
            self.torch.cuda.synchronize()
        except Exception:
            self.torch.cuda.empty_cache()
            raise
        gpu_seconds = time.perf_counter() - gpu_started
        metrics = {
            "combined_batch_size": batch_size,
            "gpu_seconds": round(gpu_seconds, 6),
            "gpu_peak_allocated_mb": round(
                self.torch.cuda.max_memory_allocated() / 1048576,
                1,
            ),
            "gpu_peak_reserved_mb": round(
                self.torch.cuda.max_memory_reserved() / 1048576,
                1,
            ),
            "gpu_current_reserved_mb": round(
                self.torch.cuda.memory_reserved() / 1048576,
                1,
            ),
        }
        del speech, embeddings
        return output, metrics

    def gpu_loop(self) -> None:
        deferred: PreparedRequest | None = None
        while not self.stop_event.is_set() or deferred is not None or not self.request_queue.empty():
            if deferred is not None:
                first = deferred
                deferred = None
            else:
                try:
                    first = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
            requests = [first]
            combined_size = int(first.features.shape[0])
            feature_shape = tuple(first.features.shape[1:])
            deadline = time.perf_counter() + self.args.batch_wait_ms / 1000.0
            while combined_size < self.args.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    candidate = self.request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                candidate_size = int(candidate.features.shape[0])
                if (
                    tuple(candidate.features.shape[1:]) != feature_shape
                    or combined_size + candidate_size > self.args.max_batch_size
                ):
                    deferred = candidate
                    break
                requests.append(candidate)
                combined_size += candidate_size

            gpu_started = time.perf_counter()
            try:
                combined = self.torch.cat(
                    [item.features for item in requests],
                    dim=0,
                )
                output, gpu_metrics = self.forward_features(combined)
                cursor = 0
                for item in requests:
                    size = int(item.features.shape[0])
                    item.embeddings = output[cursor:cursor + size]
                    item.metrics = {
                        **gpu_metrics,
                        "batch_size": size,
                        "gpu_queue_wait_seconds": round(
                            gpu_started - item.enqueued_at,
                            6,
                        ),
                    }
                    cursor += size
            except BaseException as exc:
                for item in requests:
                    item.error = exc
            finally:
                for item in requests:
                    item.event.set()
                    self.request_queue.task_done()

            with self.status_lock:
                self.gpu_batches += 1
                self.max_observed_batch_size = max(
                    self.max_observed_batch_size,
                    combined_size,
                )
                if requests[0].metrics is not None:
                    batch_metrics = requests[0].metrics
                    self.max_peak_allocated_mb = max(
                        self.max_peak_allocated_mb,
                        float(batch_metrics["gpu_peak_allocated_mb"]),
                    )
                    self.max_peak_reserved_mb = max(
                        self.max_peak_reserved_mb,
                        float(batch_metrics["gpu_peak_reserved_mb"]),
                    )

    def infer(
        self,
        audio: np.ndarray | list[np.ndarray],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        batch_size = self.validate_audio(audio)
        request_started = time.perf_counter()
        features = self.prepare_features(audio)
        feature_seconds = time.perf_counter() - request_started
        prepared = PreparedRequest(
            features=features,
            feature_seconds=feature_seconds,
            enqueued_at=time.perf_counter(),
            event=threading.Event(),
        )
        self.request_queue.put(
            prepared,
            timeout=self.args.request_timeout_seconds,
        )
        if not prepared.event.wait(self.args.request_timeout_seconds):
            raise TimeoutError("speaker GPU queue request timed out")
        if prepared.error is not None:
            raise prepared.error
        if prepared.embeddings is None or prepared.metrics is None:
            raise RuntimeError("speaker GPU queue returned no result")
        metrics = {
            **prepared.metrics,
            "feature_seconds": round(feature_seconds, 6),
            "total_seconds": round(time.perf_counter() - request_started, 6),
        }
        with self.status_lock:
            self.requests += 1
            self.windows += batch_size
            self.last_request = metrics
        self.write_status()
        return prepared.embeddings, metrics

    def warm_up(self) -> None:
        if self.args.warmup_batch_size <= 0:
            return
        started = time.perf_counter()
        audio = np.zeros(
            (self.args.warmup_batch_size, 24000),
            dtype=np.float32,
        )
        features = self.prepare_features(audio)
        _embeddings, metrics = self.forward_features(features)
        self.warmup = {
            **metrics,
            "total_seconds": round(time.perf_counter() - started, 6),
        }
        self.max_observed_batch_size = max(
            self.max_observed_batch_size,
            self.args.warmup_batch_size,
        )
        self.max_peak_allocated_mb = max(
            self.max_peak_allocated_mb,
            float(metrics["gpu_peak_allocated_mb"]),
        )
        self.max_peak_reserved_mb = max(
            self.max_peak_reserved_mb,
            float(metrics["gpu_peak_reserved_mb"]),
        )

    def handle_connection(self, connection: Connection) -> None:
        try:
            payload = connection.recv()
            if not isinstance(payload, dict):
                raise ValueError("request payload must be an object")
            operation = payload.get("op")
            if operation == "health":
                connection.send(
                    {
                        "status": "ok",
                        "service": self.status_payload("ready"),
                    }
                )
                return
            if operation != "embed":
                raise ValueError(f"unsupported operation: {operation!r}")
            audio = payload.get("audio")
            if not isinstance(audio, (np.ndarray, list)):
                raise ValueError("embed request has no audio batch")
            embeddings, metrics = self.infer(audio)
            connection.send(
                {
                    "status": "ok",
                    "embeddings": embeddings,
                    "metrics": metrics,
                }
            )
        except BaseException as exc:
            try:
                connection.send(
                    {
                        "status": "error",
                        "error": repr(exc),
                    }
                )
            except Exception:
                pass
        finally:
            connection.close()

    def stop(self, _signum: int | None = None, _frame: Any = None) -> None:
        self.stop_event.set()
        if self.listener is not None:
            self.listener.close()

    def serve(self) -> None:
        self.warm_up()
        self.gpu_thread.start()
        self.args.socket.parent.mkdir(parents=True, exist_ok=True)
        self.args.socket.unlink(missing_ok=True)
        self.listener = Listener(
            str(self.args.socket),
            family="AF_UNIX",
            authkey=AUTHKEY,
        )
        self.listener._listener._socket.settimeout(1.0)
        os.chmod(self.args.socket, 0o600)
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        self.write_status("ready")
        print(
            json.dumps(self.status_payload("ready"), ensure_ascii=False),
            flush=True,
        )
        try:
            while not self.stop_event.is_set():
                try:
                    connection = self.listener.accept()
                except socket.timeout:
                    continue
                except (OSError, socket.error):
                    if self.stop_event.is_set():
                        break
                    raise
                self.executor.submit(self.handle_connection, connection)
        finally:
            if self.listener is not None:
                self.listener.close()
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.gpu_thread.join(timeout=5)
            self.args.socket.unlink(missing_ok=True)
            self.write_status("stopped")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="podtrans speaker-service",
        description=__doc__,
    )
    parser.add_argument("--speaker-model", default=DEFAULT_SPEAKER_MODEL)
    parser.add_argument("--speaker-model-path", type=Path)
    parser.add_argument("--hub", choices=["ms", "hf"], default="ms")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--handler-workers", type=int, default=4)
    parser.add_argument("--max-queued-requests", type=int, default=8)
    parser.add_argument("--batch-wait-ms", type=float, default=50.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--warmup-batch-size", type=int, default=512)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.max_batch_size <= 0:
        parser.error("--max-batch-size must be positive")
    if (
        args.cpu_threads <= 0
        or args.handler_workers <= 0
        or args.max_queued_requests <= 0
    ):
        parser.error(
            "--cpu-threads, --handler-workers, and --max-queued-requests "
            "must be positive"
        )
    if args.batch_wait_ms < 0 or args.request_timeout_seconds <= 0:
        parser.error("--batch-wait-ms must be non-negative and request timeout positive")
    if not 0 <= args.warmup_batch_size <= args.max_batch_size:
        parser.error("--warmup-batch-size must be in [0, --max-batch-size]")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        response = request(
            args.socket,
            {"op": "health"},
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(response, ensure_ascii=False))
        return 0 if response.get("status") == "ok" else 1
    server = SpeakerEmbeddingServer(args)
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
