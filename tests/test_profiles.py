import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from podtrans import cli
from podtrans.profiles import (
    PROFILES,
    ASR_NANO,
    AppleChip,
    Family,
    FunasrVllm,
    Machine,
    SpeakerService,
    SPEAKER_ERES2NET_V2,
    VAD_FSMN,
    Vram,
    command_arguments,
    detect_machine,
    detect_nvidia_gpu,
    get_profile,
    listed_profiles,
    resolved_profile,
    resolve_command_argv,
    select_machine,
    select_profile,
)


def flag(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def apple(generation: int, unified_gb: float, brand: str | None = None) -> Machine:
    label = brand or f"Apple M{generation}"
    return Machine(
        Family.APPLE,
        label,
        apple=AppleChip(generation, unified_gb, label),
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

    def test_cpu_profile_emits_nano_on_cpu(self) -> None:
        args = command_arguments(PROFILES["cpu"], "transcribe", cpu_cores=8)
        self.assertEqual(flag(args, "--model"), ASR_NANO)
        self.assertEqual(flag(args, "--backend"), "auto")
        self.assertEqual(flag(args, "--device"), "cpu")

    def test_nvidia_diarize_keeps_cuda_service_devices(self) -> None:
        args = command_arguments(PROFILES["gpu-24gb"], "diarize", cpu_cores=16)
        self.assertEqual(flag(args, "--device"), "cuda:0")
        self.assertEqual(flag(args, "--vad-device"), "cpu")
        self.assertEqual(flag(args, "--speaker-model"), SPEAKER_ERES2NET_V2)
        self.assertEqual(flag(args, "--vad-model"), VAD_FSMN)
        service = command_arguments(PROFILES["gpu-24gb"], "speaker-service", cpu_cores=16)
        self.assertEqual(flag(service, "--device"), "cuda:0")
        self.assertEqual(flag(service, "--max-batch-size"), "512")

    def test_detected_memory_rounds_to_nominal_vram_tier(self) -> None:
        result = SimpleNamespace(stdout="NVIDIA GeForce RTX 4090, 49140\n")
        with patch("podtrans.profiles.subprocess.run", return_value=result):
            detected = detect_nvidia_gpu()
        self.assertEqual(detected, ("NVIDIA GeForce RTX 4090", 48.0))
        self.assertEqual(select_profile(detected[1]).name, "gpu-48gb")


class AppleSelectionTests(unittest.TestCase):
    def test_generation_and_memory_choose_the_rung(self) -> None:
        cases = (
            (1, 8, "apple-m1-8gb", "cpu", "1", "1"),
            (1, 16, "apple-m1-16gb", "cpu", "1", "1"),
            (2, 18, "apple-m2-16gb", "cpu", "1", "1"),
            (2, 32, "apple-m2-32gb", "mps", "1", "1"),
            (2, 64, "apple-m2-64gb", "mps", "2", "1"),
            (2, 7.5, "apple-m2-8gb", "cpu", "1", "1"),
            (3, 8, "apple-m3-8gb", "cpu", "1", "1"),
            (3, 16, "apple-m3-16gb", "mps", "1", "1"),
            (4, 24, "apple-m4-16gb", "mps", "1", "1"),
            (4, 36, "apple-m4-32gb", "mps", "2", "2"),
            (4, 48, "apple-m4-32gb", "mps", "2", "2"),
            (4, 96, "apple-m4-64gb", "mps", "4", "2"),
            (5, 64, "apple-m5-64gb", "mps", "4", "2"),
            (6, 24, "apple-m6-16gb", "mps", "1", "1"),
        )
        for generation, memory, name, device, batch, job_batch in cases:
            profile = select_machine(apple(generation, memory))
            args = command_arguments(profile, "transcribe", cpu_cores=16)
            self.assertEqual(profile.name, name)
            self.assertEqual(flag(args, "--model"), ASR_NANO)
            self.assertEqual(flag(args, "--backend"), "auto")
            self.assertEqual(flag(args, "--device"), device)
            self.assertEqual(flag(args, "--batch-size"), batch)
            self.assertEqual(flag(args, "--job-batch-size"), job_batch)
            self.assertNotIn("--vllm-dtype", args)

    def test_variant_does_not_change_the_profile(self) -> None:
        pro = select_machine(apple(4, 24, "Apple M4 Pro"))
        plain = select_machine(apple(4, 24, "Apple M4"))
        self.assertEqual(pro.name, plain.name)

    def test_unified_memory_does_not_enter_the_vram_floor(self) -> None:
        apple_profile = select_machine(apple(4, 24))
        nvidia_profile = select_machine(Machine(Family.NVIDIA, "GPU", vram=Vram(24)))
        self.assertEqual(apple_profile.name, "apple-m4-16gb")
        self.assertEqual(nvidia_profile.name, "gpu-24gb")
        self.assertEqual(select_machine(Machine(Family.CPU, "CPU")).name, "cpu")

    def test_nvidia_floor_matches_the_historical_tiers(self) -> None:
        expectations = (
            (None, "cpu"),
            (3.9, "cpu"),
            (10, "gpu-8gb"),
            (20, "gpu-16gb"),
            (32, "gpu-32gb"),
            (80, "gpu-48gb"),
        )
        for vram, name in expectations:
            self.assertEqual(select_profile(vram).name, name)
            if vram is not None and vram >= 4:
                chosen = select_machine(Machine(Family.NVIDIA, "GPU", vram=Vram(vram)))
                self.assertEqual(chosen.name, name)

    def test_listed_apple_rows_stay_on_nano_and_in_process(self) -> None:
        apple_rows = [
            profile for profile in listed_profiles() if profile.family is Family.APPLE
        ]
        names = [profile.name for profile in apple_rows]
        self.assertIn("apple-m1-8gb", names)
        self.assertIn("apple-m5-64gb", names)
        self.assertNotIn("apple-m6-16gb", names)
        self.assertEqual(len(apple_rows), 20)
        for profile in apple_rows:
            transcribe = command_arguments(profile, "transcribe", cpu_cores=16)
            diarize = command_arguments(profile, "diarize", cpu_cores=16)
            self.assertEqual(flag(transcribe, "--model"), ASR_NANO)
            self.assertNotIn("paraformer-zh", transcribe)
            self.assertEqual(flag(diarize, "--device"), "cpu")
            self.assertEqual(flag(diarize, "--vad-device"), "cpu")
            self.assertEqual(flag(diarize, "--speaker-model"), SPEAKER_ERES2NET_V2)
            self.assertEqual(flag(diarize, "--vad-model"), VAD_FSMN)
            self.assertFalse(profile.uses_speaker_service)
            self.assertFalse(profile.validated_on_host)
            with self.assertRaisesRegex(ValueError, "does not use speaker-service"):
                command_arguments(profile, "speaker-service")
        self.assertTrue(PROFILES["gpu-48gb"].validated_on_host)

    def test_speaker_batches_scale_with_the_rung(self) -> None:
        expectations = {
            "apple-m1-8gb": "16",
            "apple-m2-64gb": "64",
            "apple-m4-16gb": "32",
            "apple-m4-32gb": "64",
            "apple-m4-64gb": "128",
        }
        for name, batch in expectations.items():
            args = command_arguments(get_profile(name), "diarize", cpu_cores=16)
            self.assertEqual(flag(args, "--speaker-batch-size"), batch)

    def test_pinned_name_does_not_floor(self) -> None:
        self.assertEqual(get_profile("apple-m4-16gb").name, "apple-m4-16gb")
        self.assertEqual(get_profile("apple-m6-32gb").name, "apple-m6-32gb")
        with self.assertRaisesRegex(ValueError, "apple-m4-16gb"):
            get_profile("apple-m4-24gb")

    def test_apple_m4_transcribe_emits_model_and_mps(self) -> None:
        args = command_arguments(get_profile("apple-m4-16gb"), "transcribe", cpu_cores=8)
        self.assertEqual(
            args[:6],
            ["--model", ASR_NANO, "--backend", "auto", "--device", "mps"],
        )

    def test_machine_cannot_carry_both_memory_domains(self) -> None:
        chip = AppleChip(4, 16, "Apple M4")
        with self.assertRaisesRegex(ValueError, "no Apple chip"):
            Machine(Family.NVIDIA, "GPU", vram=Vram(8), apple=chip)
        with self.assertRaisesRegex(ValueError, "no discrete VRAM"):
            Machine(Family.APPLE, "Apple M4", vram=Vram(8), apple=chip)
        with self.assertRaises(ValueError):
            Machine(Family.CPU, "CPU", vram=Vram(8))
        with self.assertRaises(ValueError):
            Vram(0)
        with self.assertRaises(ValueError):
            AppleChip(0, 16, "Apple M0")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            FunasrVllm(ASR_NANO, 0.9, **{"device": "cpu"})
        with self.assertRaisesRegex(ValueError, "CUDA"):
            SpeakerService(SPEAKER_ERES2NET_V2, 32, **{"device": "mps"})

    def test_in_process_profiles_use_one_worker_shard(self) -> None:
        for name in ("cpu", "apple-m4-64gb"):
            config = resolved_profile(get_profile(name), cpu_cores=32)
            self.assertEqual(config["worker_shards"], 1)
            self.assertEqual(config["speaker_mode"], "in_process")
            self.assertIsNone(config["vram_gb"])
        nvidia = resolved_profile(PROFILES["gpu-48gb"], cpu_cores=12)
        self.assertEqual(nvidia["worker_shards"], 3)
        self.assertEqual(nvidia["speaker_mode"], "speaker_service")
        self.assertEqual(nvidia["vllm_memory_utilization"], 0.92)


class MachineProbeTests(unittest.TestCase):
    def test_linux_probe_skips_sysctl_and_stays_on_cpu(self) -> None:
        def fail(_command: list[str]) -> SimpleNamespace:
            raise AssertionError("sysctl should not run")

        machine = detect_machine(
            platform="linux",
            nvidia_runner=self._missing,
            apple_runner=fail,
        )
        self.assertEqual(machine.family, Family.CPU)
        self.assertEqual(select_machine(machine).name, "cpu")

    def test_apple_sysctl_selects_the_floored_rung(self) -> None:
        def sysctl(command: list[str]) -> SimpleNamespace:
            self.assertEqual(
                list(command),
                ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
            )
            return SimpleNamespace(stdout="Apple M4 Pro\n25769803776\n")

        machine = detect_machine(
            platform="darwin",
            nvidia_runner=self._missing,
            apple_runner=sysctl,
        )
        self.assertEqual(machine.family, Family.APPLE)
        self.assertIsNone(machine.vram)
        self.assertIsNotNone(machine.apple)
        assert machine.apple is not None
        self.assertEqual(machine.apple.generation, 4)
        self.assertEqual(machine.apple.unified_gb, 24.0)
        self.assertEqual(machine.label, "Apple M4 Pro")
        self.assertEqual(select_machine(machine).name, "apple-m4-16gb")

    def test_non_apple_brand_stays_on_cpu(self) -> None:
        def sysctl(_command: list[str]) -> SimpleNamespace:
            return SimpleNamespace(stdout="Intel(R) Core(TM) i7\n17179869184\n")

        machine = detect_machine(
            platform="darwin",
            nvidia_runner=self._missing,
            apple_runner=sysctl,
        )
        self.assertEqual(machine.family, Family.CPU)

    def test_nvidia_probe_wins_and_keeps_the_vram_floor(self) -> None:
        def nvidia(_command: list[str]) -> SimpleNamespace:
            return SimpleNamespace(stdout="NVIDIA GeForce RTX 4090, 24576\n")

        def apple(_command: list[str]) -> SimpleNamespace:
            raise AssertionError("sysctl should not run when NVIDIA is present")

        machine = detect_machine(
            platform="darwin",
            nvidia_runner=nvidia,
            apple_runner=apple,
        )
        self.assertEqual(machine.family, Family.NVIDIA)
        self.assertIsNone(machine.apple)
        self.assertEqual(select_machine(machine).name, "gpu-24gb")

    @staticmethod
    def _missing(_command: list[str]) -> SimpleNamespace:
        raise FileNotFoundError("nvidia-smi")


class ResolveArgvTests(unittest.TestCase):
    def test_darwin_prepends_and_user_flag_wins(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            args = resolve_command_argv(
                "transcribe",
                ["--batch-size", "3"],
                machine=apple(4, 24),
                platform="darwin",
            )
        self.assertEqual(flag(args, "--device"), "mps")
        self.assertEqual(flag(args, "--model"), ASR_NANO)
        occurrences = [index for index, value in enumerate(args) if value == "--batch-size"]
        self.assertEqual(args[occurrences[-1] + 1], "3")
        self.assertIn("podtrans: using profile apple-m4-16gb", stderr.getvalue())

    def test_linux_bare_command_does_not_prepend(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            args = resolve_command_argv(
                "transcribe",
                ["--batch-size", "3"],
                machine=apple(4, 24),
                platform="linux",
            )
        self.assertEqual(args, ["--batch-size", "3"])
        self.assertEqual(stderr.getvalue(), "")

    def test_explicit_profile_skips_detection_message(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            args = resolve_command_argv(
                "transcribe",
                ["--profile", "gpu-24gb", "--batch-size", "3"],
                platform="darwin",
            )
        self.assertEqual(flag(args, "--backend"), "vllm")
        self.assertEqual(args[args.index("--batch-size", args.index("--batch-size") + 1) + 1], "3")
        self.assertEqual(stderr.getvalue(), "")

    def test_last_explicit_profile_wins(self) -> None:
        args = resolve_command_argv(
            "transcribe",
            ["--profile", "cpu", "--profile=gpu-8gb"],
            platform="darwin",
        )
        self.assertEqual(flag(args, "--device"), "cuda")
        self.assertEqual(flag(args, "--backend"), "auto")
        self.assertNotIn("--profile", args)
        self.assertNotIn("--profile=gpu-8gb", args)

    def test_help_does_not_select_a_profile(self) -> None:
        args = resolve_command_argv(
            "transcribe",
            ["--help"],
            machine=apple(4, 24),
            platform="darwin",
        )
        self.assertEqual(args, ["--help"])


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
        text = output.getvalue()
        self.assertIn("diarize-batch", text)
        self.assertIn("Apple Silicon", text)
        self.assertIn("Linux commands without `--profile`", text)

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

    def test_profiles_lists_apple_rows(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = cli.main(["profiles"])
        text = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("apple-m1-8gb", text)
        self.assertIn("apple-m4-16gb", text)
        self.assertIn("apple-m5-64gb", text)
        self.assertIn("auto/Fun-ASR-Nano-2512@mps", text)
        self.assertIn("vllm/Fun-ASR-Nano-2512@cuda", text)
        self.assertIn("gpu-48gb", text)
        self.assertNotIn("paraformer", text)
        self.assertNotIn("apple-m6", text)

    def test_recommend_apple_preview_floors_and_stays_in_process(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = cli.main(
                [
                    "recommend",
                    "--apple-generation",
                    "4",
                    "--unified-gb",
                    "24",
                    "--cpu-cores",
                    "8",
                ]
            )
        text = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("hardware: Apple M4 (24.0GB unified)", text)
        self.assertIn("profile: apple-m4-16gb", text)
        self.assertIn("24.0GB unified floors to the 16GB rung.", text)
        self.assertIn("podtrans diarize --profile apple-m4-16gb", text)
        self.assertNotIn("diarize-batch", text)

    def test_recommend_preview_names(self) -> None:
        cases = (
            (["--apple-generation", "1", "--unified-gb", "16"], "apple-m1-16gb"),
            (["--apple-generation", "3", "--unified-gb", "7"], "apple-m3-8gb"),
            (["--apple-generation", "2", "--unified-gb", "32"], "apple-m2-32gb"),
            (["--cpu-only"], "cpu"),
        )
        for args, name in cases:
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = cli.main(["recommend", *args, "--name-only"])
            self.assertEqual(exit_code, 0)
            self.assertEqual(output.getvalue().strip(), name)

    def test_recommend_rejects_mixed_hardware_flags(self) -> None:
        stderr = io.StringIO()
        with (
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as caught,
        ):
            cli.main(
                [
                    "recommend",
                    "--vram-gb",
                    "16",
                    "--apple-generation",
                    "4",
                    "--unified-gb",
                    "16",
                ]
            )
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("choose only one", stderr.getvalue())

    def test_recommend_requires_apple_generation_and_memory_together(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            cli.main(["recommend", "--unified-gb", "16"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("must be set together", stderr.getvalue())

    def test_recommend_apple_json_stays_unvalidated(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = cli.main(
                [
                    "recommend",
                    "--apple-generation",
                    "4",
                    "--unified-gb",
                    "16",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["name"], "apple-m4-16gb")
        self.assertEqual(payload["asr_model"], ASR_NANO)
        self.assertEqual(payload["asr_device"], "mps")
        self.assertEqual(payload["speaker_device"], "cpu")
        self.assertEqual(payload["vad_device"], "cpu")
        self.assertEqual(payload["speaker_mode"], "in_process")
        self.assertFalse(payload["validated_on_host"])
        self.assertIsNone(payload["vram_gb"])
        self.assertEqual(payload["unified_memory_gb"], 16)
        self.assertEqual(payload["family"], "apple")

    def test_recommend_nvidia_still_uses_the_service_command(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = cli.main(
                ["recommend", "--vram-gb", "24", "--cpu-cores", "12"]
            )
        text = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("hardware: GPU (24.0GB)", text)
        self.assertIn("profile: gpu-24gb", text)
        self.assertIn("podtrans diarize-batch --profile gpu-24gb", text)
        self.assertIn("recommended worker shards: 3", text)

    def test_recommend_prints_detected_apple_brand(self) -> None:
        output = io.StringIO()
        with (
            patch("podtrans.profiles.detect_machine", return_value=apple(4, 24, "Apple M4 Pro")),
            redirect_stdout(output),
        ):
            exit_code = cli.main(["recommend"])
        text = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("hardware: Apple M4 Pro (24.0GB unified)", text)
        self.assertIn("profile: apple-m4-16gb", text)
        self.assertIn("24.0GB unified floors to the 16GB rung.", text)

    def test_darwin_cli_prepends_and_user_flag_wins(self) -> None:
        captured: list[str] = []

        def command(args: list[str]) -> int:
            captured.extend(args)
            return 0

        stderr = io.StringIO()
        with (
            patch("podtrans.cli.load_command", return_value=command),
            patch("podtrans.profiles.sys.platform", "darwin"),
            patch("podtrans.profiles.detect_machine", return_value=apple(4, 24)),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(["transcribe", "--batch-size", "3"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(flag(captured, "--model"), ASR_NANO)
        self.assertEqual(flag(captured, "--device"), "mps")
        sizes = [index for index, value in enumerate(captured) if value == "--batch-size"]
        self.assertEqual(captured[sizes[-1] + 1], "3")
        self.assertIn("podtrans: using profile apple-m4-16gb", stderr.getvalue())

    def test_linux_cli_does_not_prepend_or_probe(self) -> None:
        captured: list[str] = []

        def command(args: list[str]) -> int:
            captured.extend(args)
            return 0

        stderr = io.StringIO()
        with (
            patch("podtrans.cli.load_command", return_value=command),
            patch("podtrans.profiles.sys.platform", "linux"),
            patch("podtrans.profiles.detect_machine") as detect,
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(["transcribe", "--batch-size", "3"])
        self.assertEqual(exit_code, 0)
        detect.assert_not_called()
        self.assertEqual(captured, ["--batch-size", "3"])
        self.assertNotIn("using profile", stderr.getvalue())

    def test_darwin_speaker_service_rejects_the_apple_profile(self) -> None:
        stderr = io.StringIO()
        with (
            patch("podtrans.profiles.sys.platform", "darwin"),
            patch("podtrans.profiles.detect_machine", return_value=apple(4, 16)),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(["speaker-service"])
        self.assertEqual(exit_code, 2)
        text = stderr.getvalue()
        self.assertIn("podtrans: using profile apple-m4-16gb", text)
        self.assertIn("does not use speaker-service", text)

    def test_darwin_help_does_not_probe(self) -> None:
        with (
            patch("podtrans.profiles.sys.platform", "darwin"),
            patch("podtrans.profiles.detect_machine") as detect,
            redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            cli.main(["transcribe", "--help"])
        self.assertEqual(caught.exception.code, 0)
        detect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
