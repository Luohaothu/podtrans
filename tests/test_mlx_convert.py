import ctypes
import errno
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
        for name in ("tokenizer_config.json", "vocab.json", "generation_config.json"):
            (self.sidecar_fixture / name).write_text('{"published": true}')
        (self.sidecar_fixture / "merges.txt").write_text("#version: 0.2\na b\n")
        (self.sidecar_fixture / "config.json").write_text(
            json.dumps({"model_type": "fun-asr-nano", "quantization": {"bits": 4}})
        )
        (self.sidecar_fixture / "model.safetensors.index.json").write_text("{}")
        (self.sidecar_fixture / "model.safetensors").write_bytes(b"w4a16 weights")
        (self.sidecar_fixture / "README.md").write_text("Do not copy")
        self.saved_shards = []
        self.conversions = []
        self.download = Mock(side_effect=self._download)
        self.load = Mock(return_value={
            "llm.model.layers.0.weight": FakeTensor(),
            "audio_encoder.weight": FakeTensor(),
            "ctc_decoder.weight": FakeTensor(),
            "ctc.weight": FakeTensor(),
        })
        api = ModuleType("funasr_mlx.api")
        api.download_model = self.download
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

    def _download(self, repo, *, variant, dest):
        self.assertEqual(repo, "fun-asr-nano-mlx")
        self.assertEqual(variant, "w4a16")
        self.assertEqual(dest, self.cache)
        target = Path(dest) / repo / variant
        shutil.copytree(self.sidecar_fixture, target, dirs_exist_ok=True)
        return target

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

    def test_missing_model_pt_names_required_weights_and_download_size(self):
        self.model_pt.unlink()
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with self.assertRaisesRegex(FileNotFoundError, r"q8.*Fun-ASR-Nano-2512.*2\.1 GB"):
                self._ensure()
        self.download.assert_not_called()
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

    def test_cached_sidecars_allow_conversion_offline(self):
        self._download("fun-asr-nano-mlx", variant="w4a16", dest=self.cache)
        with patch.dict(os.environ, {"TRANSFORMERS_OFFLINE": "1"}):
            self._ensure()
        self.download.assert_not_called()
        self.assertEqual(len(self.conversions), 1)

    def test_missing_sidecars_fail_explicitly_offline(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with self.assertRaisesRegex(FileNotFoundError, "Offline.*w4a16 sidecars"):
                self._ensure()
        self.download.assert_not_called()
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
