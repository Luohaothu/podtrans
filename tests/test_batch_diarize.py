import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from podtrans import archive, batch_diarize
from podtrans.profiles import get_profile


SERVICE_LOG = archive.ROOT / "_metadata" / "speaker_embedding_service.log"


class _PreservedServiceLog:
    def __enter__(self) -> None:
        self._existed = SERVICE_LOG.exists()
        self._previous = SERVICE_LOG.read_bytes() if self._existed else None
        SERVICE_LOG.parent.mkdir(parents=True, exist_ok=True)
        SERVICE_LOG.write_text("keep-this-log\n", encoding="utf-8")

    def __exit__(self, *exc: object) -> None:
        if self._previous is None:
            SERVICE_LOG.unlink(missing_ok=True)
        else:
            SERVICE_LOG.write_bytes(self._previous)


class BatchDiarizeTests(unittest.TestCase):
    def test_cpu_profile_runs_in_process_without_service(self) -> None:
        with patch("podtrans.diarize.main", return_value=0) as diarize_main:
            exit_code = batch_diarize.main(
                ["--profile", "cpu", "--", "--limit", "1"]
            )
        self.assertEqual(exit_code, 0)
        args = diarize_main.call_args.args[0]
        self.assertIn("cpu", args)
        self.assertEqual(args[-2:], ["--limit", "1"])

    def test_forwarded_speaker_socket_flag_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "--speaker-service-socket"):
            batch_diarize.main(
                ["--profile", "cpu", "--", "--speaker-service-socket", "/tmp/x.sock"]
            )
        with self.assertRaisesRegex(ValueError, "--speaker-service-socket"):
            batch_diarize.main(
                [
                    "--profile",
                    "gpu-8gb",
                    "--attach",
                    "--",
                    "--speaker-service-socket=/tmp/x.sock",
                ]
            )

    def test_attach_leaves_a_healthy_socket_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = Path(temp_dir) / "speaker.sock"
            socket_path.write_bytes(b"live")
            with (
                _PreservedServiceLog(),
                patch(
                    "podtrans.speaker_service.request",
                    return_value={"status": "ok"},
                ),
                patch("podtrans.batch_diarize.subprocess.Popen") as popen,
                patch.object(batch_diarize, "run_workers", return_value=0) as run_workers,
                patch("podtrans.diarize.main") as diarize_main,
            ):
                exit_code = batch_diarize.main(
                    [
                        "--profile",
                        "gpu-8gb",
                        "--attach",
                        "--socket",
                        str(socket_path),
                        "--",
                        "--limit",
                        "1",
                    ]
                )
                self.assertEqual(SERVICE_LOG.read_text(encoding="utf-8"), "keep-this-log\n")
                popen.assert_not_called()
                diarize_main.assert_not_called()
            self.assertEqual(exit_code, 0)
            self.assertEqual(socket_path.read_bytes(), b"live")
            self.assertEqual(run_workers.call_args.kwargs["socket"], socket_path)
            self.assertIsNone(run_workers.call_args.kwargs["service_pid"])
            self.assertEqual(run_workers.call_args.kwargs["forwarded"], ["--limit", "1"])

    def test_attach_release_is_a_second_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = Path(temp_dir) / "speaker.sock"
            socket_path.write_bytes(b"live")
            with patch(
                "podtrans.speaker_service.request",
                return_value={"status": "ok"},
            ):
                lease = batch_diarize.attach_speaker_service(socket_path, 1)
            lease.release()
            lease.release()
            self.assertTrue(socket_path.exists())
            self.assertIsInstance(lease, batch_diarize.AttachedSpeakerService)

    def test_cpu_attach_does_not_run_in_process(self) -> None:
        lease = batch_diarize.AttachedSpeakerService(Path("/tmp/podtrans-speaker.sock"))
        with (
            patch.object(batch_diarize, "attach_speaker_service", return_value=lease),
            patch.object(batch_diarize, "run_workers", return_value=0) as run_workers,
            patch("podtrans.diarize.main") as diarize_main,
        ):
            exit_code = batch_diarize.main(
                ["--profile", "cpu", "--attach", "--workers", "2", "--", "--limit", "1"]
            )
        self.assertEqual(exit_code, 0)
        diarize_main.assert_not_called()
        self.assertEqual(run_workers.call_args.kwargs["workers"], 2)
        self.assertIsNone(run_workers.call_args.kwargs["service_pid"])

    def test_spawn_refuses_to_unlink_a_healthy_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = Path(temp_dir) / "speaker.sock"
            socket_path.write_bytes(b"live")
            with (
                _PreservedServiceLog(),
                patch(
                    "podtrans.speaker_service.request",
                    return_value={"status": "ok"},
                ),
                patch("podtrans.batch_diarize.subprocess.Popen") as popen,
            ):
                with self.assertRaisesRegex(RuntimeError, "--attach"):
                    batch_diarize.spawn_speaker_service(
                        socket_path,
                        get_profile("gpu-8gb"),
                        5,
                    )
                self.assertEqual(SERVICE_LOG.read_text(encoding="utf-8"), "keep-this-log\n")
                popen.assert_not_called()
            self.assertEqual(socket_path.read_bytes(), b"live")

    def test_spawn_cleans_up_on_keyboard_interrupt_while_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = Path(temp_dir) / "speaker.sock"
            process = Mock()
            process.poll.return_value = None
            process.wait.return_value = 0
            with (
                _PreservedServiceLog(),
                patch(
                    "podtrans.speaker_service.request",
                    side_effect=ConnectionRefusedError("down"),
                ),
                patch(
                    "podtrans.batch_diarize.subprocess.Popen",
                    return_value=process,
                ),
                patch(
                    "podtrans.batch_diarize._wait_until_ready",
                    side_effect=KeyboardInterrupt(),
                ),
                patch.object(batch_diarize, "stop_process") as stop_process,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    batch_diarize.spawn_speaker_service(
                        socket_path,
                        get_profile("gpu-8gb"),
                        5,
                    )
            stop_process.assert_called_once_with(process)
            self.assertFalse(socket_path.exists())

    def test_spawn_replaces_a_stale_socket_and_closes_the_child_if_startup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = Path(temp_dir) / "speaker.sock"
            socket_path.write_bytes(b"stale")
            with (
                _PreservedServiceLog(),
                patch(
                    "podtrans.speaker_service.request",
                    side_effect=ConnectionRefusedError("down"),
                ),
                patch(
                    "podtrans.batch_diarize.subprocess.Popen",
                    side_effect=OSError("boom"),
                ) as popen,
            ):
                with self.assertRaises(OSError):
                    batch_diarize.spawn_speaker_service(
                        socket_path,
                        get_profile("gpu-8gb"),
                        5,
                    )
            self.assertFalse(socket_path.exists())
            popen.assert_called_once()

    def test_spawned_release_stops_the_child_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = Path(temp_dir) / "speaker.sock"
            socket_path.write_bytes(b"owned")
            process = Mock()
            process.poll.return_value = None
            process.wait.return_value = 0
            lease = batch_diarize.SpawnedSpeakerService(socket=socket_path, process=process)
            lease.release()
            lease.release()
            process.terminate.assert_called_once()
            self.assertFalse(socket_path.exists())

    def test_main_releases_a_spawned_service_when_workers_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = Path(temp_dir) / "speaker.sock"
            socket_path.write_bytes(b"owned")
            process = Mock()
            process.pid = 4321
            process.poll.return_value = None
            process.wait.return_value = 0
            lease = batch_diarize.SpawnedSpeakerService(socket=socket_path, process=process)
            with (
                patch.object(batch_diarize, "spawn_speaker_service", return_value=lease),
                patch.object(
                    batch_diarize,
                    "run_workers",
                    side_effect=RuntimeError("worker failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    batch_diarize.main(
                        ["--profile", "gpu-8gb", "--workers", "1", "--socket", str(socket_path)]
                    )
            process.terminate.assert_called_once()
            self.assertFalse(socket_path.exists())


if __name__ == "__main__":
    unittest.main()
