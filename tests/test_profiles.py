import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from podtrans import cli
from podtrans.profiles import (
    APPLE_RECIPE_TABLE,
    PROFILES,
    ASR_NANO,
    AppleChip,
    Family,
    FunasrAuto,
    FunasrMlx,
    FunasrVllm,
    Machine,
    SelectionRequest,
    SpeakerService,
    SPEAKER_ERES2NET_V2,
    VAD_FSMN,
    Vram,
    command_arguments,
    detect_machine,
    detect_nvidia_gpu,
    get_profile,
    listed_profiles,
    query_apple_profile,
    resolved_profile,
    resolve_command_argv,
    resolve_profile_selection,
    select_machine,
    select_profile,
)

NANO_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
MLX_PREFIX = [
    "--model",
    "fun-asr-nano-mlx",
    "--backend",
    "mlx",
    "--mlx-variant",
    "q8",
]


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
            (1, 8, "apple-m1-8gb", False, "1", "1"),
            (1, 16, "apple-m1-16gb", True, "1", "1"),
            (2, 18, "apple-m2-16gb", True, "1", "1"),
            (2, 32, "apple-m2-32gb", True, "1", "1"),
            (2, 64, "apple-m2-64gb", True, "2", "1"),
            (2, 7.5, "apple-m2-8gb", False, "1", "1"),
            (3, 8, "apple-m3-8gb", False, "1", "1"),
            (3, 16, "apple-m3-16gb", True, "1", "1"),
            (4, 24, "apple-m4-16gb", True, "1", "1"),
            (4, 36, "apple-m4-32gb", True, "2", "2"),
            (4, 48, "apple-m4-32gb", True, "2", "2"),
            (4, 96, "apple-m4-64gb", True, "4", "2"),
            (5, 64, "apple-m5-64gb", True, "4", "2"),
            (6, 24, "apple-m6-16gb", True, "1", "1"),
        )
        for generation, memory, name, mlx, batch, job_batch in cases:
            profile = select_machine(apple(generation, memory))
            args = command_arguments(profile, "transcribe", cpu_cores=16)
            self.assertEqual(profile.name, name)
            self.assertEqual(flag(args, "--batch-size"), batch)
            self.assertEqual(flag(args, "--job-batch-size"), job_batch)
            self.assertNotIn("--vllm-dtype", args)
            if mlx:
                self.assertEqual(args[:6], MLX_PREFIX)
                self.assertNotIn("--device", args)
                self.assertNotIn("mps", args)
            else:
                self.assertEqual(flag(args, "--model"), NANO_MODEL)
                self.assertEqual(flag(args, "--backend"), "auto")
                self.assertEqual(flag(args, "--device"), "cpu")
        m4 = command_arguments(get_profile("apple-m4-16gb"), "transcribe", cpu_cores=16)
        for name in ("apple-m1-16gb", "apple-m2-16gb"):
            args = command_arguments(get_profile(name), "transcribe", cpu_cores=16)
            diarize = command_arguments(get_profile(name), "diarize", cpu_cores=16)
            self.assertEqual(args[:6], m4[:6])
            self.assertEqual(flag(diarize, "--device"), "cpu")

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

    def test_listed_apple_rows_keep_the_asr_floor_and_cpu_diarize(self) -> None:
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
            if profile.name.endswith("-8gb"):
                self.assertEqual(flag(transcribe, "--model"), NANO_MODEL)
                self.assertEqual(flag(transcribe, "--backend"), "auto")
                self.assertEqual(flag(transcribe, "--device"), "cpu")
            else:
                self.assertEqual(transcribe[:6], MLX_PREFIX)
                self.assertNotIn("--device", transcribe)
                self.assertNotIn("mps", transcribe)
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
        self.assertNotIn("funasr_mlx", sys.modules)

    def test_apple_recipe_table_stores_no_mps_auto_device(self) -> None:
        saw_cpu_floor = False
        for recipe in APPLE_RECIPE_TABLE:
            if isinstance(recipe.asr, FunasrAuto):
                self.assertNotEqual(recipe.asr.device, "mps")
                self.assertEqual(recipe.asr.device, "cpu")
                self.assertEqual(recipe.memory_gb, 8)
                saw_cpu_floor = True
            else:
                self.assertIsInstance(recipe.asr, FunasrMlx)
                self.assertGreater(recipe.memory_gb, 8)
        self.assertTrue(saw_cpu_floor)

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

    def test_apple_m4_transcribe_emits_mlx(self) -> None:
        args = command_arguments(get_profile("apple-m4-16gb"), "transcribe", cpu_cores=8)
        self.assertEqual(args[:6], MLX_PREFIX)
        self.assertNotIn("--device", args)
        self.assertNotIn("mps", args)
        config = resolved_profile(get_profile("apple-m4-16gb"), cpu_cores=8)
        self.assertEqual(config["asr_backend"], "mlx")
        self.assertEqual(config["asr_model"], "fun-asr-nano-mlx")
        self.assertIsNone(config["asr_device"])
        self.assertEqual(config["asr_mlx_variant"], "q8")
        self.assertEqual(
            get_profile("apple-m4-16gb").asr_pair_label(),
            "mlx/fun-asr-nano-mlx@q8",
        )
        self.assertFalse(config["validated_on_host"])
        cpu = resolved_profile(get_profile("apple-m4-8gb"), cpu_cores=8)
        self.assertEqual(cpu["asr_backend"], "auto")
        self.assertEqual(cpu["asr_model"], NANO_MODEL)
        self.assertEqual(cpu["asr_device"], "cpu")
        self.assertIsNone(cpu["asr_mlx_variant"])

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
        with self.assertRaisesRegex(ValueError, "fun-asr-nano-mlx"):
            FunasrMlx(model="other")
        for variant in ("bf16", "w4a16"):
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, "q8"):
                    FunasrMlx(variant=variant)
        with self.assertRaisesRegex(ValueError, "Apple ASR"):
            replace(
                get_profile("apple-m4-16gb"),
                asr=FunasrAuto(NANO_MODEL, "mps"),
            )

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
            machine="arm64",
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

    def test_rosetta_process_stays_on_cpu(self) -> None:
        def sysctl(_command: list[str]) -> SimpleNamespace:
            raise AssertionError("sysctl should not run under Rosetta")

        machine = detect_machine(
            platform="darwin",
            machine="x86_64",
            nvidia_runner=self._missing,
            apple_runner=sysctl,
        )
        self.assertEqual(machine.family, Family.CPU)
        self.assertEqual(select_machine(machine).name, "cpu")

    def test_non_apple_brand_stays_on_cpu(self) -> None:
        def sysctl(_command: list[str]) -> SimpleNamespace:
            return SimpleNamespace(stdout="Intel(R) Core(TM) i7\n17179869184\n")

        machine = detect_machine(
            platform="darwin",
            machine="arm64",
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
            machine="arm64",
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
        self.assertEqual(args[:6], MLX_PREFIX)
        self.assertNotIn("--device", args)
        self.assertNotIn("mps", args)
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

    def test_hardware_flags_select_without_darwin(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            args = resolve_command_argv(
                "transcribe",
                ["--apple-generation", "5", "--unified-gb", "64", "--batch-size", "3"],
                platform="linux",
            )
        self.assertEqual(args[:6], MLX_PREFIX)
        self.assertNotIn("--device", args)
        self.assertNotIn("mps", args)
        self.assertEqual(args[args.index("--batch-size", args.index("--batch-size") + 1) + 1], "3")
        self.assertIn("podtrans: using profile apple-m5-64gb", stderr.getvalue())
        self.assertNotIn("--apple-generation", args)

    def test_profile_auto_on_linux_detects(self) -> None:
        stderr = io.StringIO()
        with (
            patch("podtrans.profiles.detect_machine", return_value=apple(4, 24)),
            redirect_stderr(stderr),
        ):
            args = resolve_command_argv(
                "transcribe",
                ["--profile", "auto"],
                platform="linux",
            )
        self.assertEqual(args[:6], MLX_PREFIX)
        self.assertNotIn("--device", args)
        self.assertNotIn("mps", args)
        self.assertIn("podtrans: using profile apple-m4-16gb", stderr.getvalue())


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

    def test_base_cli_help_does_not_import_inference_stack(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = cli.main(["--help"])
        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("diarize-batch", text)
        self.assertIn("Apple Silicon", text)
        self.assertIn("--apple-generation", text)
        self.assertIn("auto|quality|speed", text)
        self.assertNotIn("  profiles", text)
        self.assertNotIn("  recommend", text)

    def test_profile_with_asr_url_is_rejected_before_transcribe(self) -> None:
        output = io.StringIO()
        with (
            patch("podtrans.cli.load_command") as load_command,
            redirect_stderr(output),
        ):
            exit_code = cli.main(
                [
                    "transcribe",
                    "--profile",
                    "gpu-24gb",
                    "--asr-url",
                    "http://127.0.0.1:8000/v1",
                    "--model",
                    "whisper",
                ]
            )
        self.assertEqual(exit_code, 2)
        load_command.assert_not_called()
        self.assertIn(
            "--profile configures in-process FunASR; "
            "--asr-url does not take a hardware profile",
            output.getvalue(),
        )

        output.truncate(0)
        output.seek(0)
        with (
            patch("podtrans.cli.load_command") as load_command,
            redirect_stderr(output),
        ):
            exit_code = cli.main(
                [
                    "transcribe",
                    "--asr-url=http://127.0.0.1:8000/v1",
                    "--profile=cpu",
                    "--model",
                    "whisper",
                ]
            )
        self.assertEqual(exit_code, 2)
        load_command.assert_not_called()

    def test_darwin_asr_url_skips_the_implicit_apple_profile(self) -> None:
        captured: list[str] = []

        def command(args: list[str]) -> int:
            captured.extend(args)
            return 0

        def detect_should_not_run(*_args: object, **_kwargs: object) -> Machine:
            raise AssertionError("detect_machine should not run for --asr-url")

        stderr = io.StringIO()
        argv = [
            "transcribe",
            "--asr-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "whisper",
        ]
        with (
            patch("podtrans.cli.load_command", return_value=command),
            patch("podtrans.profiles.sys.platform", "darwin"),
            patch("podtrans.profiles.detect_machine", detect_should_not_run),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(argv)
        self.assertEqual(exit_code, 0)
        self.assertEqual(captured, argv[1:])
        self.assertNotIn("--backend", captured)
        self.assertNotIn("--device", captured)
        self.assertNotIn("--mlx-variant", captured)
        self.assertNotIn("using profile", stderr.getvalue())

        direct = resolve_command_argv(
            "transcribe",
            argv[1:],
            machine=apple(5, 64),
            platform="darwin",
        )
        self.assertEqual(direct, argv[1:])

    def test_darwin_hardware_flags_with_asr_url_are_rejected(self) -> None:
        stderr = io.StringIO()
        with (
            patch("podtrans.cli.load_command") as load_command,
            patch("podtrans.profiles.sys.platform", "darwin"),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(
                [
                    "transcribe",
                    "--apple-generation",
                    "5",
                    "--unified-gb",
                    "64",
                    "--asr-url",
                    "http://127.0.0.1:8000/v1",
                    "--model",
                    "whisper",
                ]
            )
        self.assertEqual(exit_code, 2)
        load_command.assert_not_called()
        self.assertIn("--asr-url does not take hardware flags", stderr.getvalue())

        with self.assertRaisesRegex(ValueError, "does not take hardware flags"):
            resolve_command_argv(
                "transcribe",
                [
                    "--cpu-only",
                    "--asr-url",
                    "http://127.0.0.1:8000/v1",
                    "--model",
                    "whisper",
                ],
                platform="darwin",
            )

    def test_darwin_explicit_profile_with_asr_url_is_rejected(self) -> None:
        stderr = io.StringIO()
        with (
            patch("podtrans.cli.load_command") as load_command,
            patch("podtrans.profiles.sys.platform", "darwin"),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(
                [
                    "transcribe",
                    "--profile",
                    "apple-m5-64gb",
                    "--asr-url",
                    "http://127.0.0.1:8000/v1",
                    "--model",
                    "whisper",
                ]
            )
        self.assertEqual(exit_code, 2)
        load_command.assert_not_called()
        self.assertIn(
            "--profile configures in-process FunASR; "
            "--asr-url does not take a hardware profile",
            stderr.getvalue(),
        )

    def test_asr_url_without_profile_reaches_transcribe(self) -> None:
        captured: list[str] = []

        def command(args: list[str]) -> int:
            captured.extend(args)
            return 0

        with patch("podtrans.cli.load_command", return_value=command):
            exit_code = cli.main(
                [
                    "transcribe",
                    "--asr-url",
                    "http://127.0.0.1:8000/v1",
                    "--model",
                    "whisper",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            captured,
            ["--asr-url", "http://127.0.0.1:8000/v1", "--model", "whisper"],
        )

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

    def test_runtime_missing_mlx_extra_names_the_apple_silicon_extra(self) -> None:
        for name in ("funasr_mlx", "mlx"):
            error = ModuleNotFoundError(f"No module named {name}", name=name)

            def command(_args: list[str], error: ModuleNotFoundError = error) -> int:
                raise error

            output = io.StringIO()
            with (
                patch("podtrans.cli.load_command", return_value=command),
                redirect_stderr(output),
            ):
                exit_code = cli.main(["transcribe"])
            self.assertEqual(exit_code, 2)
            text = output.getvalue()
            self.assertIn("`podtrans[mlx]`", text)
            self.assertIn("Apple Silicon", text)
            self.assertNotIn("podtrans[inference]", text)

    def test_removed_configuration_subcommands(self) -> None:
        for command in ("profiles", "recommend"):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = cli.main([command])
            self.assertEqual(exit_code, 2)
            self.assertIn(f"unknown command {command!r}", stderr.getvalue())

    def test_apple_hardware_flags_on_transcribe(self) -> None:
        captured: list[str] = []

        def command(args: list[str]) -> int:
            captured.extend(args)
            return 0

        stderr = io.StringIO()
        with (
            patch("podtrans.cli.load_command", return_value=command),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(
                [
                    "transcribe",
                    "--apple-generation",
                    "4",
                    "--unified-gb",
                    "24",
                    "--batch-size",
                    "3",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(captured[:6], MLX_PREFIX)
        self.assertNotIn("--device", captured)
        self.assertNotIn("mps", captured)
        sizes = [index for index, value in enumerate(captured) if value == "--batch-size"]
        self.assertEqual(captured[sizes[-1] + 1], "3")
        self.assertIn("podtrans: using profile apple-m4-16gb", stderr.getvalue())
        self.assertNotIn("--apple-generation", captured)
        self.assertNotIn("--unified-gb", captured)

    def test_profile_quality_keeps_apple_asr_on_cpu(self) -> None:
        captured: list[str] = []

        def command(args: list[str]) -> int:
            captured.extend(args)
            return 0

        stderr = io.StringIO()
        with (
            patch("podtrans.cli.load_command", return_value=command),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(
                [
                    "transcribe",
                    "--profile",
                    "quality",
                    "--apple-generation",
                    "4",
                    "--unified-gb",
                    "24",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(flag(captured, "--model"), NANO_MODEL)
        self.assertEqual(flag(captured, "--backend"), "auto")
        self.assertEqual(flag(captured, "--device"), "cpu")
        self.assertNotIn("--mlx-variant", captured)
        self.assertNotIn("fun-asr-nano-mlx", captured)
        self.assertIn("apple-m4-16gb (quality)", stderr.getvalue())

    def test_profile_speed_keeps_mlx_on_early_16gb(self) -> None:
        captured: list[str] = []

        def command(args: list[str]) -> int:
            captured.extend(args)
            return 0

        stderr = io.StringIO()
        with (
            patch("podtrans.cli.load_command", return_value=command),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(
                [
                    "transcribe",
                    "--profile",
                    "speed",
                    "--apple-generation",
                    "1",
                    "--unified-gb",
                    "16",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(captured[:6], MLX_PREFIX)
        self.assertNotIn("--device", captured)
        self.assertNotIn("mps", captured)
        self.assertIn("apple-m1-16gb (speed)", stderr.getvalue())

    def test_hardware_flags_reject_mixed_domains(self) -> None:
        stderr = io.StringIO()
        with (
            patch("podtrans.cli.load_command", return_value=lambda _args: 0),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(
                [
                    "transcribe",
                    "--vram-gb",
                    "16",
                    "--apple-generation",
                    "4",
                    "--unified-gb",
                    "16",
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("choose only one", stderr.getvalue())

    def test_concrete_profile_rejects_hardware_flags(self) -> None:
        stderr = io.StringIO()
        with (
            patch("podtrans.cli.load_command", return_value=lambda _args: 0),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(
                [
                    "transcribe",
                    "--profile",
                    "cpu",
                    "--apple-generation",
                    "4",
                    "--unified-gb",
                    "16",
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("cannot be combined", stderr.getvalue())

    def test_query_apple_table_matches_select_machine(self) -> None:
        self.assertEqual(query_apple_profile(5, 64).name, "apple-m5-64gb")
        self.assertEqual(query_apple_profile(4, 24).name, "apple-m4-16gb")
        profile, label = resolve_profile_selection(
            SelectionRequest(profile="auto", apple_generation=5, unified_gb=64)
        )
        self.assertEqual(profile.name, "apple-m5-64gb")
        self.assertEqual(label, "apple-m5-64gb")

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
        self.assertEqual(captured[:6], MLX_PREFIX)
        self.assertNotIn("--device", captured)
        self.assertNotIn("mps", captured)
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


class MlxBenchmarkOverrideTests(unittest.TestCase):
    def test_device_mps_selects_pytorch_nano(self) -> None:
        args = resolve_command_argv(
            "transcribe",
            ["--profile", "apple-m4-16gb", "--device", "mps"],
        )
        self.assertEqual(flag(args, "--model"), NANO_MODEL)
        self.assertEqual(flag(args, "--backend"), "auto")
        self.assertEqual(flag(args, "--device"), "mps")
        self.assertNotIn("--mlx-variant", args)
        self.assertNotIn("fun-asr-nano-mlx", args)
        self.assertEqual(flag(args, "--batch-size"), "1")
        self.assertEqual(flag(args, "--job-batch-size"), "1")
        self.assertGreater(args.index("--device"), args.index("--decode-workers"))

    def test_device_cpu_selects_pytorch_nano(self) -> None:
        args = resolve_command_argv(
            "transcribe",
            ["--profile", "apple-m4-16gb", "--device", "cpu"],
        )
        self.assertEqual(flag(args, "--model"), NANO_MODEL)
        self.assertEqual(flag(args, "--backend"), "auto")
        self.assertEqual(flag(args, "--device"), "cpu")
        self.assertNotIn("--mlx-variant", args)
        self.assertNotIn("fun-asr-nano-mlx", args)

    def test_device_equals_form_selects_pytorch_nano(self) -> None:
        args = resolve_command_argv(
            "transcribe",
            ["--profile", "apple-m4-16gb", "--device=cpu"],
        )
        self.assertIn("--device=cpu", args)
        self.assertEqual(flag(args, "--model"), NANO_MODEL)
        self.assertEqual(flag(args, "--backend"), "auto")
        self.assertNotIn("--mlx-variant", args)
        self.assertNotIn("fun-asr-nano-mlx", args)

    def test_backend_mlx_keeps_the_mlx_flags(self) -> None:
        spaced = resolve_command_argv(
            "transcribe",
            ["--profile", "apple-m4-16gb", "--backend", "mlx"],
        )
        self.assertEqual(spaced[:6], MLX_PREFIX)
        self.assertNotIn("--device", spaced)
        equals = resolve_command_argv(
            "transcribe",
            ["--profile", "apple-m4-16gb", "--backend=mlx"],
        )
        self.assertEqual(equals[:6], MLX_PREFIX)
        self.assertIn("--backend=mlx", equals)

    def test_other_backend_drops_the_profile_backend(self) -> None:
        args = resolve_command_argv(
            "transcribe",
            ["--profile", "apple-m4-16gb", "--backend", "auto"],
        )
        self.assertEqual(flag(args, "--model"), NANO_MODEL)
        self.assertEqual(
            [args[index + 1] for index, value in enumerate(args) if value == "--backend"],
            ["auto"],
        )
        self.assertNotIn("--mlx-variant", args)
        self.assertNotIn("fun-asr-nano-mlx", args)

    def test_quality_on_the_8gb_floor_stays_cpu_funasr(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            args = resolve_command_argv(
                "transcribe",
                [
                    "--profile",
                    "quality",
                    "--apple-generation",
                    "4",
                    "--unified-gb",
                    "8",
                ],
            )
        self.assertEqual(flag(args, "--model"), NANO_MODEL)
        self.assertEqual(flag(args, "--backend"), "auto")
        self.assertEqual(flag(args, "--device"), "cpu")
        self.assertIn("apple-m4-8gb (quality)", stderr.getvalue())

    def test_speed_on_the_8gb_floor_does_not_select_mps(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            args = resolve_command_argv(
                "transcribe",
                [
                    "--profile",
                    "speed",
                    "--apple-generation",
                    "1",
                    "--unified-gb",
                    "8",
                ],
            )
        self.assertEqual(flag(args, "--model"), NANO_MODEL)
        self.assertEqual(flag(args, "--backend"), "auto")
        self.assertEqual(flag(args, "--device"), "cpu")
        self.assertNotIn("mps", args)
        self.assertIn("apple-m1-8gb (speed)", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
