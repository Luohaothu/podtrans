import ctypes
import errno
import fnmatch
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from podtrans import mlx_convert


class FakeTensor:
    def __init__(self, size: int = 8) -> None:
        self.size = size

    def contiguous(self):
        return self

    def numel(self) -> int:
        return self.size

    def element_size(self) -> int:
        return 2


class Q8ConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.cache = self.root / "data" / "models" / "funasr-mlx"
        self.model_pt = self.root / "model.pt"
        self.model_pt.write_bytes(b"local PyTorch weights")
        self.sidecar_fixture = self.root / "published"
        self.sidecar_fixture.mkdir()
        for name in (
            "tokenizer_config.json", "vocab.json", "generation_config.json",
            "chat_template.json", "preprocessor_config.json",
        ):
            (self.sidecar_fixture / name).write_text('{"published": true}')
        (self.sidecar_fixture / "merges.txt").write_text("#version: 0.2\na b\n")
        (self.sidecar_fixture / "config.json").write_text(
            json.dumps({"model_type": "fun-asr-nano", "quantization": {"bits": 4}})
        )
        (self.sidecar_fixture / "model.safetensors.index.json").write_text("{}")
        for name in ("model.safetensors", "model.npz", "pytorch_model.bin", "model.pt", "weights.pth"):
            (self.sidecar_fixture / name).write_bytes(b"w4a16 weights")
        (self.sidecar_fixture / "pytorch_model.bin.index.json").write_text("{}")
        (self.sidecar_fixture / "README.md").write_text("Do not copy")
        self.saved_shards = []
        self.conversions = []
        self.fetched_files = []
        self.download = Mock(side_effect=self._download)
        self.hf_download = Mock(side_effect=self._download)
        self.unfiltered_download = Mock(
            side_effect=AssertionError("Unfiltered catalog download must not run")
        )
        self.load = Mock(return_value={
            "llm.model.layers.0.weight": FakeTensor(),
            "audio_encoder.weight": FakeTensor(),
            "ctc_decoder.weight": FakeTensor(),
            "ctc.weight": FakeTensor(),
        })
        api = ModuleType("funasr_mlx.api")
        api.download_model = self.unfiltered_download
        modelscope_snapshot = ModuleType("modelscope.hub.snapshot_download")
        modelscope_snapshot.snapshot_download = self.download
        huggingface = ModuleType("huggingface_hub")
        huggingface.snapshot_download = self.hf_download
        checkpoints = ModuleType("funasr_mlx.checkpoints")
        checkpoints.convert_checkpoint = self._convert
        self.checkpoints = checkpoints
        torch = ModuleType("torch")
        torch.load = self.load
        safetensors_torch = ModuleType("safetensors.torch")
        safetensors_torch.save_file = self._save_file
        self.stack.enter_context(patch.dict(sys.modules, {
            "funasr_mlx": ModuleType("funasr_mlx"),
            "funasr_mlx.api": api,
            "funasr_mlx.checkpoints": checkpoints,
            "modelscope": ModuleType("modelscope"),
            "modelscope.hub": ModuleType("modelscope.hub"),
            "modelscope.hub.snapshot_download": modelscope_snapshot,
            "huggingface_hub": huggingface,
            "torch": torch,
            "safetensors": ModuleType("safetensors"),
            "safetensors.torch": safetensors_torch,
        }))
        self.stack.enter_context(patch.object(mlx_convert.metadata, "version", return_value="0.2.3"))
        self.stack.enter_context(patch.object(Path, "home", return_value=self.root / "home"))
        self.stack.enter_context(patch.dict(os.environ, {
            "MODELSCOPE_CACHE": str(self.root / "modelscope"),
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
        }))

    def _download(self, model_id=None, *, repo_id=None, revision, local_dir, allow_patterns):
        self.assertEqual(model_id or repo_id, "baicai1145/fun-asr-nano-mlx")
        self.assertEqual(revision, "master" if model_id else "main")
        self.assertEqual(Path(local_dir), self.cache / "fun-asr-nano-mlx")
        for variant in ("w4a16", "fp16"):
            for source in self.sidecar_fixture.iterdir():
                remote_path = f"{variant}/{source.name}"
                if any(fnmatch.fnmatchcase(remote_path, pattern) for pattern in allow_patterns):
                    target = Path(local_dir) / remote_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    self.fetched_files.append(remote_path)
        return str(local_dir)

    def _save_file(self, tensors, filename):
        self.saved_shards.append(dict(tensors))
        Path(filename).write_bytes(b"source shard")

    def _convert(self, source, destination, *, model, dtype, source_rule, config, quantization):
        self.assertFalse(destination.exists())
        self.assertTrue(list(source.glob("*.safetensors")))
        self.conversions.append({
            "model": model,
            "dtype": dtype,
            "source_rule": source_rule,
            "config": config,
            "quantization": quantization,
        })
        destination.mkdir()
        (destination / "config.json").write_text(json.dumps(config))
        (destination / "model-00001.safetensors").write_bytes(b"q8 weights")

    def _ensure(self):
        return mlx_convert.ensure_q8_checkpoint(self.cache, model_pt=self.model_pt)

    def _assert_temp_empty(self):
        temporary = self.cache / "fun-asr-nano-mlx" / ".q8-build"
        self.assertEqual(list(temporary.iterdir()) if temporary.exists() else [], [])

    def _change_source(self):
        self.model_pt.write_bytes(b"updated PyTorch model.pt weights")

    def _swap(self, source, destination, flags):
        self.assertEqual(flags, 0x00000002)
        source, destination = Path(os.fsdecode(source)), Path(os.fsdecode(destination))
        backup = source.parent / "old"
        destination.rename(backup)
        source.rename(destination)
        backup.rename(source)
        return 0

    def test_converts_once_and_reuses_the_complete_cache(self):
        first = self._ensure()
        self.assertEqual(first, self.cache / "fun-asr-nano-mlx" / "q8")
        self.assertEqual(self._ensure(), first)
        self.assertEqual(len(self.conversions), 1)
        self.download.assert_called_once()
        self.hf_download.assert_not_called()
        self.load.assert_called_once_with(
            self.model_pt, map_location="cpu", weights_only=True, mmap=True
        )
        self.assertEqual(self.model_pt.read_bytes(), b"local PyTorch weights")
        self._assert_temp_empty()

    def test_conversion_recipe_and_tokenizer_match_the_published_mlx_source(self):
        checkpoint = self._ensure()
        call = self.conversions[0]
        expected_quantization = {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
            "targets": [
                "llm.model.layers.", "audio_encoder.", "audio_adaptor.", "llm.model.embed_tokens"
            ],
        }
        self.assertEqual(call["quantization"], expected_quantization)
        self.assertEqual(call["model"], "fun-asr-nano")
        self.assertEqual(call["source_rule"], "fun-asr-nano")
        self.assertEqual(call["dtype"], "bfloat16")
        self.assertEqual(json.loads((checkpoint / "config.json").read_text()), {
            "model_type": "fun-asr-nano", "quantization": expected_quantization
        })
        self.assertEqual((checkpoint / "vocab.json").read_text(), '{"published": true}')
        self.assertTrue((checkpoint / "merges.txt").is_file())
        self.assertTrue((checkpoint / "generation_config.json").is_file())
        self.assertFalse((checkpoint / "model.safetensors.index.json").exists())
        self.assertFalse((checkpoint / "README.md").exists())
        saved_names = {name for shard in self.saved_shards for name in shard}
        self.assertEqual(saved_names, {"llm.model.layers.0.weight", "audio_encoder.weight"})

    def test_sidecar_snapshot_filters_weights_and_preserves_every_sidecar(self):
        checkpoint = self._ensure()
        expected = [
            "w4a16/config.json",
            "w4a16/tokenizer_config.json",
            "w4a16/vocab.json",
            "w4a16/merges.txt",
            "w4a16/generation_config.json",
            "w4a16/chat_template.json",
            "w4a16/preprocessor_config.json",
        ]
        patterns = self.download.call_args.kwargs["allow_patterns"]
        self.assertEqual(patterns, expected)
        self.assertEqual(set(self.fetched_files), set(expected))
        source = self.cache / "fun-asr-nano-mlx" / "w4a16"
        self.assertEqual(
            {path.name for path in source.iterdir()},
            {Path(name).name for name in expected},
        )
        for path in expected:
            name = Path(path).name
            if name != "config.json":
                self.assertEqual(
                    (checkpoint / name).read_bytes(),
                    (self.sidecar_fixture / name).read_bytes(),
                )
        self.assertEqual(
            json.loads((checkpoint / "config.json").read_text())["quantization"]["bits"], 8
        )
        for variant in ("w4a16", "fp16"):
            for file in self.sidecar_fixture.iterdir():
                remote_path = f"{variant}/{file.name}"
                if remote_path not in expected:
                    self.assertFalse(
                        any(fnmatch.fnmatchcase(remote_path, pattern) for pattern in patterns)
                    )
        self.unfiltered_download.assert_not_called()
        self.hf_download.assert_not_called()

    def test_sidecar_fetch_falls_back_to_hugging_face_with_the_same_filter(self):
        self.download.side_effect = OSError("ModelScope is unavailable")
        checkpoint = self._ensure()
        self.download.assert_called_once()
        self.hf_download.assert_called_once()
        primary = self.download.call_args.kwargs
        fallback = self.hf_download.call_args.kwargs
        self.assertEqual(primary["model_id"], fallback["repo_id"])
        self.assertEqual(primary["revision"], "master")
        self.assertEqual(fallback["revision"], "main")
        self.assertEqual(primary["local_dir"], fallback["local_dir"])
        self.assertEqual(primary["allow_patterns"], fallback["allow_patterns"])
        self.assertTrue((checkpoint / "tokenizer_config.json").is_file())
        self.assertTrue((checkpoint / "vocab.json").is_file())
        self.assertTrue((checkpoint / "merges.txt").is_file())
        self.unfiltered_download.assert_not_called()

    def test_source_writer_shards_without_splitting_tensors(self):
        self.load.return_value = {
            "audio_encoder.a": FakeTensor(150_000_000),
            "audio_encoder.b": FakeTensor(150_000_000),
        }
        self._ensure()
        self.assertEqual([list(shard) for shard in self.saved_shards], [
            ["audio_encoder.a"], ["audio_encoder.b"]
        ])

    def test_cache_hit_works_without_model_pt_and_without_network(self):
        checkpoint = self._ensure()
        self.model_pt.unlink()
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            self.assertEqual(self._ensure(), checkpoint)
        self.assertEqual(len(self.conversions), 1)
        self.download.assert_called_once()
        self.hf_download.assert_not_called()

    def test_missing_model_pt_names_required_weights_and_download_size(self):
        self.model_pt.unlink()
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with self.assertRaisesRegex(FileNotFoundError, r"q8.*Fun-ASR-Nano-2512.*2\.1 GB"):
                self._ensure()
        self.download.assert_not_called()
        self.hf_download.assert_not_called()
        self.load.assert_not_called()
        self.assertFalse((self.cache / "fun-asr-nano-mlx" / "q8").exists())

    def test_source_lookup_supports_modelscope_layouts(self):
        roots = [self.root / "modelscope", self.root / "home" / ".cache" / "modelscope"]
        for root in roots:
            for prefix in (Path(), Path("models")):
                for repo in ("FunAudioLLM/Fun-ASR-Nano-2512", "FunAudioLLM--Fun-ASR-Nano-2512"):
                    for suffix in ("model.pt", "snapshots/master/model.pt", "snapshots/revision/model.pt"):
                        candidate = root / prefix / repo / suffix
                        with self.subTest(path=candidate):
                            candidate.parent.mkdir(parents=True, exist_ok=True)
                            candidate.write_bytes(b"weights")
                            self.assertEqual(mlx_convert._find_model_pt(), candidate)
                            candidate.unlink()

    def test_source_lookup_finds_default_modelscope_hub_without_cache_env(self):
        hub = self.root / "home" / ".cache" / "modelscope" / "hub"
        candidate = hub / "models" / "FunAudioLLM/Fun-ASR-Nano-2512" / "model.pt"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"weights")
        with patch.dict(os.environ):
            os.environ.pop("MODELSCOPE_CACHE", None)
            self.assertEqual(mlx_convert._find_model_pt(), candidate)

    def test_source_lookup_prefers_explicit_cache_over_default_hub(self):
        default = self.root / "home" / ".cache" / "modelscope" / "hub"
        explicit = self.root / "modelscope"
        repo = Path("models/FunAudioLLM/Fun-ASR-Nano-2512/model.pt")
        for root in (default, explicit):
            candidate = root / repo
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(b"weights")
        self.assertEqual(mlx_convert._find_model_pt(), explicit / repo)

    def test_cached_sidecars_allow_conversion_offline(self):
        target = self.cache / "fun-asr-nano-mlx" / "w4a16"
        shutil.copytree(self.sidecar_fixture, target)
        with patch.dict(os.environ, {"TRANSFORMERS_OFFLINE": "1"}):
            self._ensure()
        self.download.assert_not_called()
        self.hf_download.assert_not_called()
        self.assertEqual(len(self.conversions), 1)

    def test_missing_sidecars_fail_explicitly_offline(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with self.assertRaisesRegex(FileNotFoundError, "Offline.*w4a16 sidecars"):
                self._ensure()
        self.download.assert_not_called()
        self.hf_download.assert_not_called()
        self.load.assert_not_called()

    def test_converter_failure_leaves_no_partial_q8_or_temporary_files(self):
        def fail(source, destination, **kwargs):
            destination.mkdir()
            (destination / "partial.safetensors").write_bytes(b"partial")
            raise RuntimeError("conversion failed")

        self.checkpoints.convert_checkpoint = fail
        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            self._ensure()
        self.assertFalse((self.cache / "fun-asr-nano-mlx" / "q8").exists())
        self._assert_temp_empty()

    def test_failed_rebuild_preserves_previous_good_q8(self):
        checkpoint = self._ensure()
        old_stamp = (checkpoint / ".podtrans-q8-stamp.json").read_bytes()
        self._change_source()

        def fail(source, destination, **kwargs):
            destination.mkdir()
            raise RuntimeError("conversion failed")

        self.checkpoints.convert_checkpoint = fail
        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            self._ensure()
        self.assertEqual((checkpoint / ".podtrans-q8-stamp.json").read_bytes(), old_stamp)
        self.assertEqual((checkpoint / "model-00001.safetensors").read_bytes(), b"q8 weights")
        self._assert_temp_empty()

    def test_source_changes_replace_the_cache_with_an_atomic_swap(self):
        checkpoint = self._ensure()
        self._change_source()
        swap = Mock(side_effect=self._swap)
        with patch.object(ctypes, "CDLL", return_value=SimpleNamespace(renamex_np=swap)):
            self.assertEqual(self._ensure(), checkpoint)
        self.assertEqual(len(self.conversions), 2)
        swap.assert_called_once()
        self.assertEqual(swap.argtypes, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint])
        self.assertIs(swap.restype, ctypes.c_int)
        stamp = json.loads((checkpoint / ".podtrans-q8-stamp.json").read_text())
        self.assertEqual(stamp["model_pt_size"], self.model_pt.stat().st_size)
        self._assert_temp_empty()

    def test_failed_atomic_swap_preserves_previous_good_q8(self):
        checkpoint = self._ensure()
        old_stamp = (checkpoint / ".podtrans-q8-stamp.json").read_bytes()
        self._change_source()
        swap = Mock(return_value=-1)
        with patch.object(ctypes, "CDLL", return_value=SimpleNamespace(renamex_np=swap)):
            with patch.object(ctypes, "get_errno", return_value=errno.ENOTSUP):
                with self.assertRaises(OSError) as raised:
                    self._ensure()
        self.assertEqual(raised.exception.errno, errno.ENOTSUP)
        self.assertEqual((checkpoint / ".podtrans-q8-stamp.json").read_bytes(), old_stamp)
        self._assert_temp_empty()

    def test_missing_tokenizer_file_invalidates_cache(self):
        checkpoint = self._ensure()
        (checkpoint / "merges.txt").unlink()
        swap = Mock(side_effect=self._swap)
        with patch.object(ctypes, "CDLL", return_value=SimpleNamespace(renamex_np=swap)):
            self._ensure()
        self.assertEqual(len(self.conversions), 2)
        self.assertTrue((checkpoint / "merges.txt").is_file())

    def test_signature_drift_fails_before_loading_source(self):
        def incompatible(source, destination, *, model, dtype):
            raise AssertionError("Incompatible converter must not run")

        self.checkpoints.convert_checkpoint = incompatible
        with self.assertRaisesRegex(RuntimeError, r"funasr-mlx 0\.2\.3.*need >=0\.2\.3,<0\.3"):
            self._ensure()
        self.load.assert_not_called()
        self.download.assert_not_called()
        self.hf_download.assert_not_called()

    def test_positional_only_signature_drift_is_incompatible(self):
        def incompatible(source, destination, model, dtype, source_rule, config, quantization, /):
            raise AssertionError("Incompatible converter must not run")

        self.checkpoints.convert_checkpoint = incompatible
        with self.assertRaisesRegex(RuntimeError, "incompatible.*convert_checkpoint"):
            self._ensure()
        self.load.assert_not_called()

    def test_missing_converter_import_has_a_compatible_version_message(self):
        del self.checkpoints.convert_checkpoint
        with self.assertRaisesRegex(RuntimeError, "funasr-mlx 0.2.3.*incompatible"):
            self._ensure()
        self.load.assert_not_called()
