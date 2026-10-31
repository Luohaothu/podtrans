"""Run concurrent diarization with one shared speaker model service."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import archive
from .profiles import (
    detect_machine,
    get_profile,
    resolve_command_argv,
    resolved_profile,
    select_machine,
)


DEFAULT_SOCKET = Path("/tmp/podtrans-speaker-embedding.sock")


def wait_for_service(
    process: subprocess.Popen[str],
    socket_path: Path,
    timeout_seconds: float,
) -> None:
    from .speaker_service import request

    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
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
        except (ConnectionError, OSError, TimeoutError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(
        f"speaker service did not become ready within {timeout_seconds}s: "
        f"{last_error!r}"
    )


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="podtrans diarize-batch",
        description=__doc__,
    )
    parser.add_argument("--profile")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--service-timeout", type=float, default=300)
    parser.add_argument("--worker-start-stagger", type=float, default=2)
    parser.add_argument("--no-final-verify", action="store_true")
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
        try:
            profile = get_profile(args.profile)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        profile = select_machine(detect_machine())
        print(f"podtrans: using profile {profile.name}", file=sys.stderr)
    config = resolved_profile(profile)
    forwarded = list(args.diarize_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    if not profile.uses_speaker_service:
        if args.workers not in {None, 1}:
            parser.error(f"{profile.name} supports exactly one in-process worker")
        from .diarize import main as diarize_main

        return diarize_main(
            resolve_command_argv(
                "diarize",
                ["--profile", profile.name, *forwarded],
            )
        )

    workers = args.workers or int(config["worker_shards"])
    metadata_dir = archive.ROOT / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    service_log_path = metadata_dir / "speaker_embedding_service.log"
    worker_pid_path = metadata_dir / "speaker_diarization_worker_pids"
    service_status_path = metadata_dir / "speaker_embedding_service_status.json"
    executable = [sys.executable, "-m", "podtrans"]
    service_command = [
        *executable,
        "speaker-service",
        "--profile",
        profile.name,
        "--socket",
        str(args.socket),
        "--status-file",
        str(service_status_path),
    ]

    service: subprocess.Popen[str] | None = None
    worker_processes: list[subprocess.Popen[str]] = []
    worker_logs = []
    previous_handlers: dict[int, object] = {}

    def stop_all(_signum: int | None = None, _frame: object | None = None) -> None:
        for process in worker_processes:
            stop_process(process)
        stop_process(service)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, stop_all)
        service_log = service_log_path.open("w", encoding="utf-8")
        try:
            service = subprocess.Popen(
                service_command,
                cwd=archive.ROOT,
                stdout=service_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            wait_for_service(service, args.socket, args.service_timeout)
            print(
                f"[batch-diarize] profile={profile.name} service_pid={service.pid} "
                f"workers={workers} socket={args.socket}",
                flush=True,
            )

            pid_records: list[dict[str, int]] = []
            for shard_index in range(workers):
                log_path = (
                    metadata_dir
                    / f"speaker_diarization_worker_{shard_index}.log"
                )
                log_handle = log_path.open("w", encoding="utf-8")
                worker_logs.append(log_handle)
                command = [
                    *executable,
                    "diarize",
                    "--profile",
                    profile.name,
                    *forwarded,
                    "--speaker-service-socket",
                    str(args.socket),
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
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "profile": profile.name,
                        "workers": workers,
                        "service_pid": service.pid,
                    }
                )
            )
            return 0
        finally:
            service_log.close()
    finally:
        stop_all()
        args.socket.unlink(missing_ok=True)
        for handle in worker_logs:
            handle.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
