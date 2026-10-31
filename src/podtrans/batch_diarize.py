"""Run concurrent diarization with one shared speaker model service.

Pass --attach to use a speaker-service that is already running.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from . import archive
from .profiles import (
    PROFILES,
    HardwareProfile,
    command_arguments,
    detect_nvidia_gpu,
    get_profile,
    resolved_profile,
    select_profile,
)


DEFAULT_SOCKET = Path("/tmp/podtrans-speaker-embedding.sock")


class SpeakerLease(Protocol):
    socket: Path

    def release(self) -> None:
        """Idempotent. A spawned service stops its child. An attached service returns."""


@dataclass
class SpawnedSpeakerService:
    socket: Path
    process: subprocess.Popen[str]
    _log: TextIO | None = None
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            stop_process(self.process)
        finally:
            self.socket.unlink(missing_ok=True)
            log = self._log
            self._log = None
            if log is not None and not log.closed:
                log.close()


@dataclass(frozen=True)
class AttachedSpeakerService:
    socket: Path

    def release(self) -> None:
        return


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def strip_worker_separator(diarize_args: list[str]) -> list[str]:
    if diarize_args[:1] == ["--"]:
        return list(diarize_args[1:])
    return list(diarize_args)


def reject_worker_socket_flag(forwarded: list[str]) -> None:
    for arg in forwarded:
        if arg == "--speaker-service-socket" or arg.startswith("--speaker-service-socket="):
            raise ValueError(
                "do not forward --speaker-service-socket; "
                "diarize-batch passes the batch socket itself"
            )


def _socket_is_healthy(socket_path: Path, timeout_seconds: float) -> bool:
    from multiprocessing.context import AuthenticationError

    from .speaker_service import request

    try:
        response = request(
            socket_path,
            {"op": "health"},
            timeout_seconds=min(5, timeout_seconds),
        )
    except (
        AuthenticationError,
        ConnectionError,
        EOFError,
        OSError,
        TimeoutError,
        RuntimeError,
    ):
        return False
    return response.get("status") == "ok"


def _wait_until_ready(
    socket_path: Path,
    timeout_seconds: float,
    process: subprocess.Popen[str] | None,
) -> None:
    from multiprocessing.context import AuthenticationError

    from .speaker_service import request

    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process is not None:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"speaker service exited before becoming ready: {exit_code}"
                )
        try:
            response = request(
                socket_path,
                {"op": "health"},
                timeout_seconds=min(5, timeout_seconds),
            )
            if response.get("status") == "ok":
                return
            last_error = RuntimeError(
                f"speaker service health status is {response.get('status')!r}"
            )
        except (
            AuthenticationError,
            ConnectionError,
            EOFError,
            OSError,
            TimeoutError,
            RuntimeError,
        ) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(
        f"speaker service did not become ready within {timeout_seconds}s: "
        f"{last_error!r}"
    )


def spawn_speaker_service(
    socket: Path,
    profile: HardwareProfile,
    timeout_seconds: float,
) -> SpawnedSpeakerService:
    """Start speaker-service unless the socket is already healthy."""
    if socket.exists() and _socket_is_healthy(socket, timeout_seconds):
        raise RuntimeError(
            f"speaker socket {socket} is already healthy; "
            "pass --attach or stop the service"
        )
    socket.parent.mkdir(parents=True, exist_ok=True)
    socket.unlink(missing_ok=True)
    metadata_dir = archive.ROOT / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (metadata_dir / "speaker_embedding_service.log").open("w", encoding="utf-8")
    service_status_path = metadata_dir / "speaker_embedding_service_status.json"
    command = [
        sys.executable,
        "-m",
        "podtrans",
        "speaker-service",
        *command_arguments(profile, "speaker-service"),
        "--socket",
        str(socket),
        "--status-file",
        str(service_status_path),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=archive.ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        log_handle.close()
        raise
    try:
        _wait_until_ready(socket, timeout_seconds, process)
    except Exception:
        stop_process(process)
        socket.unlink(missing_ok=True)
        log_handle.close()
        raise
    return SpawnedSpeakerService(socket=socket, process=process, _log=log_handle)


def attach_speaker_service(socket: Path, timeout_seconds: float) -> AttachedSpeakerService:
    """Wait until health status is ok. Do not start, stop, or unlink the service."""
    _wait_until_ready(socket, timeout_seconds, None)
    return AttachedSpeakerService(socket)


def run_workers(
    *,
    socket: Path,
    profile: HardwareProfile,
    workers: int,
    forwarded: list[str],
    args: argparse.Namespace,
    worker_processes: list[subprocess.Popen[str]],
    service_pid: int | None,
) -> int:
    """Start shard workers against socket. Does not start or stop a service."""
    metadata_dir = archive.ROOT / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    worker_pid_path = metadata_dir / "speaker_diarization_worker_pids"
    executable = [sys.executable, "-m", "podtrans"]
    worker_logs = []
    try:
        pid_records: list[dict[str, int]] = []
        for shard_index in range(workers):
            log_path = metadata_dir / f"speaker_diarization_worker_{shard_index}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            worker_logs.append(log_handle)
            command = [
                *executable,
                "diarize",
                *command_arguments(profile, "diarize"),
                *forwarded,
                "--speaker-service-socket",
                str(socket),
                "--speaker-service-timeout",
                str(args.service_timeout),
                "--retry-errors",
                "--num-shards",
                str(workers),
                "--shard-index",
                str(shard_index),
                "--no-report",
            ]
            process = subprocess.Popen(
                command,
                cwd=archive.ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            worker_processes.append(process)
            pid_records.append({"shard": shard_index, "pid": process.pid})
            if shard_index + 1 < workers:
                time.sleep(args.worker_start_stagger)
        worker_pid_path.write_text(
            "\n".join(
                f"{record['shard']} {record['pid']}" for record in pid_records
            )
            + "\n",
            encoding="utf-8",
        )
        exit_codes = [process.wait() for process in worker_processes]
        failed = [code for code in exit_codes if code != 0]
        if failed:
            print(
                f"[batch-diarize] worker failures={failed}",
                file=sys.stderr,
            )
            return failed[0]
        if not args.no_final_verify:
            from .diarize import main as diarize_main

            verify_exit = diarize_main(["--verify-only"])
            if verify_exit:
                return verify_exit
        payload: dict[str, object] = {
            "status": "complete",
            "profile": profile.name,
            "workers": workers,
        }
        if service_pid is not None:
            payload["service_pid"] = service_pid
        print(json.dumps(payload))
        return 0
    finally:
        for handle in worker_logs:
            handle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="podtrans diarize-batch",
        description=__doc__,
    )
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--service-timeout", type=float, default=300)
    parser.add_argument("--worker-start-stagger", type=float, default=2)
    parser.add_argument("--no-final-verify", action="store_true")
    parser.add_argument(
        "--attach",
        action="store_true",
        help="use an already running speaker-service and do not stop it",
    )
    parser.add_argument(
        "diarize_args",
        nargs=argparse.REMAINDER,
        help="arguments after -- are forwarded to the diarize command",
    )
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers <= 0:
        parser.error("--workers must be positive")
    if args.service_timeout <= 0 or args.worker_start_stagger < 0:
        parser.error("timeouts must be positive and stagger must be non-negative")

    if args.profile:
        profile = get_profile(args.profile)
    else:
        detected = detect_nvidia_gpu()
        profile = select_profile(detected[1] if detected else None)
    config = resolved_profile(profile)
    forwarded = strip_worker_separator(args.diarize_args)
    reject_worker_socket_flag(forwarded)

    if args.attach:
        lease: SpeakerLease = attach_speaker_service(args.socket, args.service_timeout)
    elif profile.name == "cpu":
        if args.workers not in {None, 1}:
            parser.error("the CPU profile supports exactly one in-process worker")
        from .diarize import main as diarize_main

        return diarize_main(command_arguments(profile, "diarize") + forwarded)
    else:
        lease = spawn_speaker_service(args.socket, profile, args.service_timeout)

    workers = args.workers or int(config["worker_shards"])
    worker_processes: list[subprocess.Popen[str]] = []
    previous_handlers = {}
    service_pid = (
        lease.process.pid if isinstance(lease, SpawnedSpeakerService) else None
    )

    def stop_workers(_signum: int | None = None, _frame: object | None = None) -> None:
        for process in worker_processes:
            stop_process(process)
        lease.release()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, stop_workers)
        if service_pid is not None:
            print(
                f"[batch-diarize] profile={profile.name} service_pid={service_pid} "
                f"workers={workers} socket={lease.socket}",
                flush=True,
            )
        else:
            print(
                f"[batch-diarize] profile={profile.name} workers={workers} "
                f"socket={lease.socket}",
                flush=True,
            )
        return run_workers(
            socket=lease.socket,
            profile=profile,
            workers=workers,
            forwarded=forwarded,
            args=args,
            worker_processes=worker_processes,
            service_pid=service_pid,
        )
    finally:
        for process in worker_processes:
            stop_process(process)
        lease.release()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
