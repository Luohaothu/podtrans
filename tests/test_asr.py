import io
import json
import os
import sys
import tempfile
import unittest
from email.message import EmailMessage
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from podtrans.asr import (
    DEFAULT_MODEL,
    AutoModelAsr,
    RemoteAsr,
    VllmAsr,
    _parse_transcription_payload,
    open_asr,
    parse_asr_spec,
)


URL = "http://127.0.0.1:8000/v1"
TRANSCRIPTION_URL = "http://127.0.0.1:8000/v1/audio/transcriptions"


def spec_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "asr_url": None,
        "model": None,
        "language": "auto",
        "batch_size": 4,
        "backend": None,
        "device": None,
        "batch_size_s": None,
        "max_length": None,
        "tensor_parallel_size": None,
        "vllm_dtype": None,
        "vllm_gpu_memory_utilization": None,
        "vllm_max_model_len": None,
        "vllm_enforce_eager": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def http_error(code: int, body: bytes, headers: EmailMessage | None = None) -> HTTPError:
    return HTTPError(
        TRANSCRIPTION_URL,
        code,
        "error",
        headers or EmailMessage(),
        io.BytesIO(body),
    )


class TranscriptionPayloadTests(unittest.TestCase):
    def test_top_level_words_become_token_timestamps(self) -> None:
        result = _parse_transcription_payload(
            {
                "text": "你好世界",
                "words": [
                    {"word": "你好", "start": 0, "end": 0.5},
                    {"token": "世界", "start_time": 0.5, "end_time": 1},
                ],
                "segments": [
                    {
                        "text": "ignored",
                        "start": 0,
                        "end": 1,
                        "words": [{"word": "ignored", "start": 0, "end": 1}],
                    }
                ],
            }
        )
        self.assertEqual(
            result,
            {
                "text": "你好世界",
                "timestamps": [
                    {"token": "你好", "start_time": 0.0, "end_time": 0.5},
                    {"token": "世界", "start_time": 0.5, "end_time": 1.0},
                ],
            },
        )

    def test_segment_words_are_used_when_top_level_words_are_absent(self) -> None:
        result = _parse_transcription_payload(
            {
                "text": "hello world",
                "words": [],
                "segments": [
                    {
                        "text": "hello",
                        "start": 0,
                        "end": 0.4,
                        "words": [{"word": "hello", "start": 0.0, "end": 0.4}],
                    },
                    {
                        "text": " world",
                        "start": 0.4,
                        "end": 0.9,
                        "words": [{"word": " world", "start": 0.4, "end": 0.9}],
                    },
                ],
            }
        )
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(
            [item["token"] for item in result["timestamps"]],
            ["hello", " world"],
        )

    def test_leading_spaces_are_stripped_when_that_spelling_matches_text(self) -> None:
        result = _parse_transcription_payload(
            {
                "text": "helloworld",
                "words": [
                    {"word": " hello", "start": 0.0, "end": 0.4},
                    {"word": " world", "start": 0.4, "end": 0.8},
                ],
            }
        )
        self.assertEqual(
            [item["token"] for item in result["timestamps"]],
            ["hello", "world"],
        )

    def test_stripped_tokens_remain_when_neither_spelling_matches_text(self) -> None:
        result = _parse_transcription_payload(
            {
                "text": "hello world",
                "words": [
                    {"word": " hi", "start": 0.0, "end": 0.2},
                    {"word": " there", "start": 0.2, "end": 0.4},
                ],
            }
        )
        self.assertEqual(
            [item["token"] for item in result["timestamps"]],
            ["hi", "there"],
        )

    def test_segments_without_words_become_one_stamp_each(self) -> None:
        result = _parse_transcription_payload(
            {
                "text": "hello world",
                "segments": [
                    {"text": "hello ", "start": 0.0, "end": 1.0},
                    {"text": "world", "start": 1.0, "end": 2.0},
                ],
            }
        )
        self.assertEqual(
            result["timestamps"],
            [
                {"token": "hello ", "start_time": 0.0, "end_time": 1.0},
                {"token": "world", "start_time": 1.0, "end_time": 2.0},
            ],
        )

    def test_text_only_response_has_empty_timestamps(self) -> None:
        self.assertEqual(
            _parse_transcription_payload({"text": "hello"}),
            {"text": "hello", "timestamps": []},
        )

    def test_missing_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _parse_transcription_payload({"words": [{"word": "hi", "start": 0, "end": 1}]})


class ParseAsrSpecTests(unittest.TestCase):
    def test_unset_flags_build_auto_model_with_historical_defaults(self) -> None:
        spec = parse_asr_spec(spec_args())
        self.assertIsInstance(spec, AutoModelAsr)
        assert isinstance(spec, AutoModelAsr)
        self.assertEqual(spec.model, DEFAULT_MODEL)
        self.assertEqual(spec.device, "cuda")
        self.assertEqual(spec.language, "auto")
        self.assertEqual(spec.batch_size, 4)
        self.assertIsNone(spec.batch_size_s)
        self.assertEqual(spec.max_length, 256)
        session = open_asr(spec)
        self.assertEqual(session.source.backend, "auto_model")
        self.assertEqual(session.source.device, "cuda")
        self.assertIsNone(session.source.endpoint)
        self.assertEqual(session.batch_size, 4)

    def test_explicit_auto_backend_stays_auto_model(self) -> None:
        spec = parse_asr_spec(spec_args(backend="auto", batch_size_s=30, device="cpu"))
        self.assertIsInstance(spec, AutoModelAsr)
        assert isinstance(spec, AutoModelAsr)
        self.assertEqual(spec.device, "cpu")
        self.assertEqual(spec.batch_size_s, 30)

    def test_vllm_backend_fills_option_defaults(self) -> None:
        spec = parse_asr_spec(spec_args(backend="vllm", model="custom", device="cuda:1"))
        self.assertIsInstance(spec, VllmAsr)
        assert isinstance(spec, VllmAsr)
        self.assertEqual(spec.model, "custom")
        self.assertEqual(spec.device, "cuda:1")
        self.assertEqual(spec.max_length, 256)
        self.assertEqual(spec.options.dtype, "bfloat16")
        self.assertEqual(spec.options.tensor_parallel_size, 1)
        self.assertEqual(spec.options.gpu_memory_utilization, 0.8)
        self.assertEqual(spec.options.max_model_len, 4096)
        self.assertIs(spec.options.enforce_eager, False)
        self.assertEqual(open_asr(spec).source.backend, "vllm")

    def test_vllm_uses_explicit_options(self) -> None:
        spec = parse_asr_spec(
            spec_args(
                backend="vllm",
                vllm_dtype="float16",
                tensor_parallel_size=2,
                vllm_gpu_memory_utilization=0.5,
                vllm_max_model_len=1024,
                vllm_enforce_eager=True,
                max_length=128,
            )
        )
        assert isinstance(spec, VllmAsr)
        self.assertEqual(spec.max_length, 128)
        self.assertEqual(spec.options.dtype, "float16")
        self.assertEqual(spec.options.tensor_parallel_size, 2)
        self.assertEqual(spec.options.gpu_memory_utilization, 0.5)
        self.assertEqual(spec.options.max_model_len, 1024)
        self.assertIs(spec.options.enforce_eager, True)

    def test_remote_url_shapes(self) -> None:
        root = parse_asr_spec(spec_args(asr_url=URL + "/", model="whisper"))
        copied = parse_asr_spec(
            spec_args(
                asr_url="https://api.openai.com/v1/audio/transcriptions",
                model="whisper-1",
            )
        )
        assert isinstance(root, RemoteAsr)
        assert isinstance(copied, RemoteAsr)
        self.assertEqual(root.base_url, URL)
        self.assertEqual(root.transcription_url, TRANSCRIPTION_URL)
        self.assertEqual(copied.base_url, "https://api.openai.com/v1")
        self.assertEqual(
            copied.transcription_url,
            "https://api.openai.com/v1/audio/transcriptions",
        )
        session = open_asr(root)
        self.assertEqual(session.source.backend, "openai")
        self.assertEqual(session.source.device, "remote")
        self.assertEqual(session.source.endpoint, URL)
        self.assertEqual(session.source.model, "whisper")

    def test_remote_language_auto_is_omitted_and_other_codes_are_kept(self) -> None:
        omitted = parse_asr_spec(spec_args(asr_url=URL, model="whisper"))
        chinese = parse_asr_spec(
            spec_args(asr_url=URL, model="whisper", language="zh")
        )
        assert isinstance(omitted, RemoteAsr)
        assert isinstance(chinese, RemoteAsr)
        self.assertIsNone(omitted.language)
        self.assertEqual(chinese.language, "zh")

    def test_remote_requires_model_and_rejects_in_process_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "--model"):
            parse_asr_spec(spec_args(asr_url=URL))
        rejected = {
            "backend": "auto",
            "device": "cuda",
            "batch_size_s": 10,
            "max_length": 128,
            "tensor_parallel_size": 2,
            "vllm_dtype": "float16",
            "vllm_gpu_memory_utilization": 0.5,
            "vllm_max_model_len": 2048,
            "vllm_enforce_eager": True,
        }
        for field_name, value in rejected.items():
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(ValueError, "--asr-url"):
                    parse_asr_spec(
                        spec_args(asr_url=URL, model="whisper", **{field_name: value})
                    )

    def test_remote_urls_with_userinfo_query_fragment_or_bare_origin_are_rejected(self) -> None:
        rejected = [
            "http://user:secret@127.0.0.1:8000/v1",
            "http://token@127.0.0.1:8000/v1",
            "http://127.0.0.1:8000/v1?key=secret",
            "http://127.0.0.1:8000/v1#fragment",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8000/v1/audio",
            "ftp://127.0.0.1:8000/v1",
        ]
        for url in rejected:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    parse_asr_spec(spec_args(asr_url=url, model="whisper"))

    def test_api_key_comes_from_the_environment_and_stays_out_of_repr(self) -> None:
        with patch.dict("os.environ", {"PODTRANS_ASR_API_KEY": "sk-test-value"}):
            spec = parse_asr_spec(spec_args(asr_url=URL, model="whisper"))
        assert isinstance(spec, RemoteAsr)
        self.assertEqual(spec.api_key, "sk-test-value")
        self.assertNotIn("sk-test-value", repr(spec))
        source = open_asr(spec).source
        self.assertEqual(source.endpoint, URL)
        self.assertNotIn("sk-test-value", repr(source))

        with patch.dict("os.environ", {"PODTRANS_ASR_API_KEY": ""}):
            empty = parse_asr_spec(spec_args(asr_url=URL, model="whisper"))
        assert isinstance(empty, RemoteAsr)
        self.assertIsNone(empty.api_key)

        with patch.dict("os.environ"):
            os.environ.pop("PODTRANS_ASR_API_KEY", None)
            missing = parse_asr_spec(spec_args(asr_url=URL, model="whisper"))
        assert isinstance(missing, RemoteAsr)
        self.assertIsNone(missing.api_key)

    def test_non_positive_batch_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "--batch-size"):
            parse_asr_spec(spec_args(batch_size=0))


class RemoteSessionTests(unittest.TestCase):
    def test_verbose_json_round_trip_does_not_import_funasr(self) -> None:
        sys.modules.pop("funasr", None)
        spec = parse_asr_spec(spec_args(asr_url=URL, model="whisper", batch_size=2))
        session = open_asr(spec)
        payload = {
            "text": "hi",
            "words": [{"word": "hi", "start": 0.0, "end": 0.2}],
        }
        requests = []
        timeouts = []

        def urlopen(request, timeout=None):
            requests.append(request)
            timeouts.append(timeout)
            return io.BytesIO(json.dumps(payload).encode())

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "chunk.wav"
            wav_path.write_bytes(b"RIFFwav")
            with patch("urllib.request.urlopen", urlopen):
                result = session.transcribe([str(wav_path)])
        self.assertNotIn("funasr", sys.modules)
        self.assertEqual(result[0]["timestamps"][0]["token"], "hi")
        self.assertEqual(requests[0].full_url, TRANSCRIPTION_URL)
        self.assertIsNone(requests[0].get_header("Authorization"))
        self.assertIn(b"verbose_json", requests[0].data)
        self.assertIn(b"timestamp_granularities[]", requests[0].data)
        self.assertIn(b'filename="chunk.wav"', requests[0].data)
        self.assertIn(b"RIFFwav", requests[0].data)
        self.assertEqual(timeouts, [600.0])

    def test_authorization_bearer_is_sent_only_when_a_key_is_set(self) -> None:
        with patch.dict("os.environ", {"PODTRANS_ASR_API_KEY": "sk-local"}):
            spec = parse_asr_spec(spec_args(asr_url=URL, model="whisper"))
        requests = []

        def urlopen(request, timeout=None):
            requests.append(request)
            return io.BytesIO(json.dumps({"text": "hi"}).encode())

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "a.wav"
            wav_path.write_bytes(b"wav")
            with patch("urllib.request.urlopen", urlopen):
                result = open_asr(spec).transcribe([str(wav_path)])
        self.assertEqual(result, [{"text": "hi", "timestamps": []}])
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer sk-local")
        self.assertNotIn("sk-local", json.dumps(result))

    def test_results_stay_aligned_when_responses_finish_out_of_order(self) -> None:
        spec = parse_asr_spec(spec_args(asr_url=URL, model="whisper"))

        def urlopen(request, timeout=None):
            name = "b.wav" if b'filename="b.wav"' in request.data else "a.wav"
            if name == "a.wav":
                __import__("time").sleep(0.05)
            body = {"text": name}
            return io.BytesIO(json.dumps(body).encode())

        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "a.wav"
            second = Path(temp_dir) / "b.wav"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            with patch("urllib.request.urlopen", urlopen):
                result = open_asr(spec).transcribe([str(first), str(second)])
        self.assertEqual([item["text"] for item in result], ["a.wav", "b.wav"])

    def test_http_400_falls_back_to_plain_json_once(self) -> None:
        spec = parse_asr_spec(spec_args(asr_url=URL, model="whisper", language="zh"))
        requests = []

        def urlopen(request, timeout=None):
            requests.append(request)
            if len(requests) == 1:
                raise http_error(400, b"word timestamps unsupported")
            return io.BytesIO(json.dumps({"text": "only text"}).encode())

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "a.wav"
            wav_path.write_bytes(b"wav")
            with patch("urllib.request.urlopen", urlopen):
                result = open_asr(spec).transcribe([str(wav_path)])
        self.assertEqual(result, [{"text": "only text", "timestamps": []}])
        self.assertIn(b"verbose_json", requests[0].data)
        self.assertIn(b"zh", requests[0].data)
        self.assertNotIn(b"verbose_json", requests[1].data)
        self.assertNotIn(b"timestamp_granularities", requests[1].data)
        self.assertIn(b"json", requests[1].data)

    def test_other_client_errors_are_not_retried_and_omit_headers(self) -> None:
        spec = parse_asr_spec(spec_args(asr_url=URL, model="whisper"))
        headers = EmailMessage()
        headers["X-Secret"] = "super-secret-header"
        calls = 0

        def urlopen(request, timeout=None):
            nonlocal calls
            calls += 1
            raise http_error(401, b"missing-token-body", headers)

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "a.wav"
            wav_path.write_bytes(b"wav")
            with patch("urllib.request.urlopen", urlopen):
                with self.assertRaises(RuntimeError) as raised:
                    open_asr(spec).transcribe([str(wav_path)])
        self.assertEqual(calls, 1)
        message = str(raised.exception)
        self.assertIn("401", message)
        self.assertIn("missing-token-body", message)
        self.assertNotIn("super-secret-header", message)

    def test_transport_errors_retry_then_fail_without_a_partial_list(self) -> None:
        spec = parse_asr_spec(spec_args(asr_url=URL, model="whisper"))
        calls: list[str] = []

        def urlopen(request, timeout=None):
            name = "bad.wav" if b'filename="bad.wav"' in request.data else "ok.wav"
            calls.append(name)
            if name == "bad.wav":
                raise URLError("connection refused")
            return io.BytesIO(json.dumps({"text": "ok"}).encode())

        with tempfile.TemporaryDirectory() as temp_dir:
            ok_path = Path(temp_dir) / "ok.wav"
            bad_path = Path(temp_dir) / "bad.wav"
            ok_path.write_bytes(b"ok")
            bad_path.write_bytes(b"bad")
            with (
                patch("urllib.request.urlopen", urlopen),
                patch("podtrans.asr.time.sleep"),
            ):
                with self.assertRaises(RuntimeError):
                    open_asr(spec).transcribe([str(ok_path), str(bad_path)])
        self.assertEqual(calls.count("ok.wav"), 1)
        self.assertEqual(calls.count("bad.wav"), 4)

    def test_retryable_status_then_success(self) -> None:
        spec = parse_asr_spec(spec_args(asr_url=URL, model="whisper"))
        calls = 0

        def urlopen(request, timeout=None):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise http_error(503, b"busy")
            return io.BytesIO(json.dumps({"text": "later"}).encode())

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "a.wav"
            wav_path.write_bytes(b"wav")
            with (
                patch("urllib.request.urlopen", urlopen),
                patch("podtrans.asr.time.sleep"),
            ):
                result = open_asr(spec).transcribe([str(wav_path)])
        self.assertEqual(calls, 3)
        self.assertEqual(result[0]["text"], "later")


class LocalSessionTests(unittest.TestCase):
    def test_auto_model_falls_back_to_one_wav_and_remembers_that(self) -> None:
        calls: list[tuple[object, dict[str, object]]] = []
        constructed: list[dict[str, object]] = []

        class FakeAutoModel:
            def __init__(self, **kwargs: object) -> None:
                constructed.append(kwargs)

            def generate(self, input: object, **kwargs: object) -> dict[str, str]:
                calls.append((input, kwargs))
                if isinstance(input, list):
                    raise NotImplementedError("batch")
                return {"text": str(input)}

        fake = ModuleType("funasr")
        fake.AutoModel = FakeAutoModel
        spec = parse_asr_spec(
            spec_args(model="m", device="cpu", language="zh", max_length=32, batch_size_s=8)
        )
        session = open_asr(spec)
        with patch.dict(sys.modules, {"funasr": fake}):
            first = session.transcribe(["a.wav", "b.wav"])
            second = session.transcribe(["c.wav", "d.wav"])
        self.assertEqual([item["text"] for item in first], ["a.wav", "b.wav"])
        self.assertEqual([item["text"] for item in second], ["c.wav", "d.wav"])
        self.assertEqual(constructed, [{"model": "m", "device": "cpu", "disable_update": True}])
        self.assertEqual(calls[0][0], ["a.wav", "b.wav"])
        self.assertEqual(calls[0][1]["batch_size"], 2)
        self.assertEqual(calls[0][1]["batch_size_s"], 8)
        self.assertEqual(calls[0][1]["language"], "zh")
        self.assertEqual(calls[0][1]["max_length"], 32)
        self.assertEqual([call[0] for call in calls[1:]], ["a.wav", "b.wav", "c.wav", "d.wav"])

    def test_type_error_retries_without_batch_size_s(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeAutoModel:
            def __init__(self, **kwargs: object) -> None:
                return None

            def generate(self, input: object, **kwargs: object) -> dict[str, str]:
                calls.append(kwargs)
                if "batch_size_s" in kwargs:
                    raise TypeError("unexpected keyword")
                return {"text": "ok"}

        fake = ModuleType("funasr")
        fake.AutoModel = FakeAutoModel
        spec = parse_asr_spec(spec_args(batch_size_s=12, language=None))
        with patch.dict(sys.modules, {"funasr": fake}):
            result = open_asr(spec).transcribe(["a.wav"])
        self.assertEqual(result, [{"text": "ok"}])
        self.assertIn("batch_size_s", calls[0])
        self.assertNotIn("batch_size_s", calls[1])
        self.assertNotIn("language", calls[1])

    def test_single_wav_not_implemented_is_reraised(self) -> None:
        class FakeAutoModel:
            def __init__(self, **kwargs: object) -> None:
                return None

            def generate(self, input: object, **kwargs: object) -> dict[str, str]:
                raise NotImplementedError("nope")

        fake = ModuleType("funasr")
        fake.AutoModel = FakeAutoModel
        with patch.dict(sys.modules, {"funasr": fake}):
            with self.assertRaises(NotImplementedError):
                open_asr(parse_asr_spec(spec_args())).transcribe(["a.wav"])

    def test_vllm_generate_receives_the_path_list(self) -> None:
        calls: list[tuple[object, dict[str, object]]] = []
        constructed: list[dict[str, object]] = []

        class FakeVLLM:
            def __init__(self, **kwargs: object) -> None:
                constructed.append(kwargs)

            def generate(self, wavs: object, **kwargs: object) -> list[dict[str, str]]:
                calls.append((wavs, kwargs))
                return [{"text": "a"}, {"text": "b"}]

        funasr = ModuleType("funasr")
        auto = ModuleType("funasr.auto")
        vllm_mod = ModuleType("funasr.auto.auto_model_vllm")
        vllm_mod.AutoModelVLLM = FakeVLLM
        spec = parse_asr_spec(spec_args(backend="vllm", language="zh"))
        with patch.dict(
            sys.modules,
            {
                "funasr": funasr,
                "funasr.auto": auto,
                "funasr.auto.auto_model_vllm": vllm_mod,
            },
        ):
            result = open_asr(spec).transcribe(["a.wav", "b.wav"])
        self.assertEqual(result, [{"text": "a"}, {"text": "b"}])
        self.assertEqual(calls[0][0], ["a.wav", "b.wav"])
        self.assertNotIn("batch_size", calls[0][1])
        self.assertEqual(calls[0][1]["language"], "zh")
        self.assertEqual(constructed[0]["dtype"], "bfloat16")
        self.assertEqual(constructed[0]["tensor_parallel_size"], 1)
        self.assertIs(constructed[0]["enforce_eager"], False)


if __name__ == "__main__":
    unittest.main()
