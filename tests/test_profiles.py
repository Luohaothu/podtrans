import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from podtrans import cli
from podtrans.profiles import (
    PROFILES,
    command_arguments,
    detect_nvidia_gpu,
    resolved_profile,
    select_profile,
)


class HardwareProfileTests(unittest.TestCase):
    def test_selects_largest_safe_vram_tier(self) -> None:
        self.assertEqual(select_profile(None).name, "cpu")
        self.assertEqual(select_profile(3.9).name, "cpu")
        self.assertEqual(select_profile(10).name, "gpu-8gb")
        self.assertEqual(select_profile(20).name, "gpu-16gb")
        self.assertEqual(select_profile(32).name, "gpu-32gb")
        self.assertEqual(select_profile(80).name, "gpu-48gb")

    def test_cpu_count_bounds_worker_concurrency(self) -> None:
        config = resolved_profile(PROFILES["gpu-48gb"], cpu_cores=12)
        self.assertEqual(config["worker_shards"], 3)
        self.assertLessEqual(
            config["diarize_decode_workers"] * config["worker_shards"],
            12,
        )

    def test_vllm_profiles_use_bfloat16(self) -> None:
        args = command_arguments(PROFILES["gpu-24gb"], "transcribe")
        self.assertIn("vllm", args)
        self.assertIn("bfloat16", args)
        small_args = command_arguments(PROFILES["gpu-8gb"], "transcribe")
        self.assertIn("auto", small_args)
        self.assertNotIn("--vllm-dtype", small_args)

    def test_cpu_profile_rejects_gpu_service(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not use speaker-service"):
            command_arguments(PROFILES["cpu"], "speaker-service")

    def test_detected_memory_rounds_to_nominal_vram_tier(self) -> None:
        result = SimpleNamespace(stdout="NVIDIA GeForce RTX 4090, 49140\n")
        with patch("podtrans.profiles.subprocess.run", return_value=result):
            detected = detect_nvidia_gpu()
        self.assertEqual(detected, ("NVIDIA GeForce RTX 4090", 48.0))
        self.assertEqual(select_profile(detected[1]).name, "gpu-48gb")


class CliProfileTests(unittest.TestCase):
    def test_user_arguments_follow_and_override_profile_arguments(self) -> None:
        captured: list[str] = []

        def command(args: list[str]) -> int:
            captured.extend(args)
            return 0

        with patch("podtrans.cli.load_command", return_value=command):
            exit_code = cli.main(
                [
                    "transcribe",
                    "--profile",
                    "gpu-24gb",
                    "--batch-size",
                    "3",
                ]
            )
        self.assertEqual(exit_code, 0)
        occurrences = [
            index for index, value in enumerate(captured) if value == "--batch-size"
        ]
        self.assertEqual(len(occurrences), 2)
        self.assertEqual(captured[occurrences[-1] + 1], "3")

    def test_recommend_name_only(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = cli.main(
                ["recommend", "--vram-gb", "24", "--name-only"]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().strip(), "gpu-24gb")

    def test_base_cli_help_does_not_import_inference_stack(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = cli.main(["--help"])
        self.assertEqual(exit_code, 0)
        self.assertIn("diarize-batch", output.getvalue())

    def test_runtime_missing_dependency_has_install_hint(self) -> None:
        error = ModuleNotFoundError("No module named 'funasr'", name="funasr")

        def command(_args: list[str]) -> int:
            raise error

        output = io.StringIO()
        with (
            patch("podtrans.cli.load_command", return_value=command),
            redirect_stderr(output),
        ):
            exit_code = cli.main(["transcribe"])
        self.assertEqual(exit_code, 2)
        self.assertIn("podtrans[inference]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
