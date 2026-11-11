# podtrans

English | [简体中文](https://github.com/Luohaothu/podtrans/blob/main/README.zh-CN.md)

`podtrans` is an installable CLI for archiving, transcribing, and speaker-labeling podcasts. The project uses uv to manage dependencies and the lock file. After installation, everything runs through `podtrans <command>`; you do not need to run any Python or shell scripts from the repository.

## Features

- Fetch episode metadata, audio, and show notes from RSS, and generate an integrity report.
- Batch-transcribe archived audio with `FunAudioLLM/Fun-ASR-Nano-2512`.
- Point transcription at a service compatible with OpenAI `/v1/audio/transcriptions`, or use `diarize-batch --attach` to connect to a speaker service that is already running.
- Generate speaker labels with FunASR VAD and `iic/speech_eres2netv2_sv_zh-cn_16k-common`.
- Load the speaker model in one dedicated GPU service; multiple workers share its weights over a Unix socket.
- Support CPU-only runs, task sharding, error retries, final verification, and hardware presets.
- Provide VRAM tiers from 4GB to 48GB, covering mainstream GeForce consumer GPUs.

## Installation

Python 3.11 and 3.12 are supported.
Only Linux and macOS are supported.

### uv

Archiving and configuration queries only:

```bash
uv sync
uv run podtrans --help
```

Install FunASR, PyTorch, and the speaker-labeling dependencies:

```bash
uv sync --extra inference
uv run podtrans --version
```

Use the vLLM backend:

```bash
uv sync --extra vllm
```

Nodes in mainland China can use the TUNA mirror to speed up Python package resolution and downloads:

```bash
uv sync \
  --extra vllm \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

Models are downloaded from ModelScope by default. Set `MODELSCOPE_CACHE=/path/to/cache` to keep models on a fast local disk.

### pip

The repository can be installed directly as a standard Python package:

```bash
python -m pip install .
podtrans --help
```

Install the inference features:

```bash
python -m pip install ".[inference]"
```

Install the vLLM inference features:

```bash
python -m pip install ".[vllm]"
```

The `mlx` extra is for Apple Silicon macOS. The Apple MLX path is unvalidated.

```bash
python -m pip install ".[mlx]"
```

You can also build a wheel first and install it in another environment:

```bash
uv build
python -m pip install "dist/podtrans-0.2.0-py3-none-any.whl[inference]"
```

For CUDA or ROCm environments, install the `torch` and `torchaudio` builds that match your driver by following the official PyTorch instructions. The project does not force-replace a hardware-specific PyTorch you have already installed.

## CLI

```text
podtrans archive
podtrans transcribe
podtrans diarize
podtrans diarize-batch
podtrans speaker-service
podtrans validate-archive
podtrans validate-diarization
podtrans benchmark-workers
podtrans benchmark-fbank
```

Show the options for a subcommand:

```bash
podtrans transcribe --help
podtrans help diarize
```

Real episodes, audio, transcripts, run logs, and model caches never go into Git. By default the program uses `data/` in the current directory. You can also set it explicitly:

```bash
export PODTRANS_DATA_DIR=/path/to/podcast-data
```

## Hardware profiles

### Apple Silicon ASR

`profiles` and `recommend` are gone. Hardware selection is flags on `transcribe`, `diarize`, `diarize-batch`, and `speaker-service`.

Apple Silicon catalog floors above 8GB default to FunASR auto with `FunAudioLLM/Fun-ASR-Nano-2512` on PyTorch MPS. The default chunk duration is 30 seconds. Start with either command on Apple Silicon macOS:

```bash
podtrans transcribe
podtrans transcribe --profile auto
```

The 8GB floor stays on CPU FunASR. `quality` selects CPU FunASR for every Apple row. `speed` uses the same ASR as `auto`. To select CPU explicitly, run `podtrans transcribe --device cpu`. Speaker and VAD stay on CPU. Apple rows are unvalidated. Install the extra with `pip install 'podtrans[mlx]'` on Apple Silicon macOS. That extra includes FunASR, PyTorch, and funasr-mlx.

MLX remains a manual opt-in. The first command uses the detected Apple profile. The second pins a catalog row and supplies the model and local `q8` variant explicitly:

```bash
podtrans transcribe --backend mlx
podtrans transcribe --profile apple-m4-16gb --backend mlx --model fun-asr-nano-mlx --mlx-variant q8
```

MLX uses a local 8-bit `q8` checkpoint and a 512-token output cap. Only the MLX path caps chunks below its frontend limit, typically to 29 seconds. A chunk that reaches the output cap without EOS emits a warning and still returns its text. MLX can emit `<>` in transcripts. See the known limitation in [issue #6](https://github.com/Luohaothu/podtrans/issues/6).

Before the first MLX prepare, obtain the Fun-ASR-Nano-2512 PyTorch `model.pt` weights through ModelScope, about 2.1 GB. Running the CPU or MPS path once downloads them. MLX prepare searches `$MODELSCOPE_CACHE` and `~/.cache/modelscope`, including the flattened repository snapshot layout. If `model.pt` is missing, prepare reports how to obtain it instead of starting that download.

The first prepare converts `model.pt` locally and caches the result at `$PODTRANS_DATA_DIR/models/funasr-mlx/fun-asr-nano-mlx/q8/`, or `./data/models/funasr-mlx/fun-asr-nano-mlx/q8/` by default. It downloads only seven small tokenizer and config files from the published `w4a16` package, about 4.5 MB in the inspected file set. The hub requests exclude weight shards and weight indexes. Inference loads the locally converted q8 weights. This keeps the MLX special-token ids aligned with its decoder. A complete cached q8 checkpoint needs no source files or network access. To convert offline, cache both `model.pt` and the MLX sidecars first. Otherwise prepare reports the missing source files. Conversion intermediates are temporary. Existing ModelScope and Hugging Face caches remain intact.

Local conversion relies on the funasr-mlx `convert_checkpoint` signature and requires `funasr-mlx>=0.2.3,<0.3`. Prepare checks that signature before conversion. Truncation detection uses the Qwen3 EOS ids `151643`, `151645`, and `151671` from funasr-mlx 0.2.3 `llm_asr.py`. Recheck both dependencies before widening the version range.

Select hardware directly on the working commands. There are no separate `profiles` / `recommend` subcommands anymore.

```bash
podtrans transcribe --apple-generation 4 --unified-gb 24
podtrans transcribe --profile auto
podtrans transcribe --profile quality
podtrans transcribe --profile speed
podtrans diarize --vram-gb 16
podtrans diarize-batch --cpu-only
```

`--profile` accepts a friendly profile name or a specific catalog id:

| Value | Meaning |
| --- | --- |
| `auto` | Query the catalog from the detected machine or the hardware flags. |
| `quality` | Same Apple catalog row, with ASR on CPU FunASR. |
| `speed` | Same catalog row and ASR as `auto`, including MPS above the 8GB floor. |
| `cpu` / `gpu-*` / `apple-m*-*gb` | One catalog id. |

`--apple-generation` and `--unified-gb` must be given together, and cannot be combined with `--vram-gb`, `--cpu-only`, or a specific catalog id passed to `--profile`.

On macOS, when `--profile` and the hardware flags are omitted, `transcribe`, `diarize`, and `speaker-service` automatically use the detected profile and write its name to standard error, for example `podtrans: using profile apple-m4-16gb`. Options given later override options of the same name in the profile. On Linux, when the selection flags are omitted, these three commands keep their own defaults; a profile is selected only when you pass `--profile auto` or a hardware flag. `diarize-batch` selects a profile on every system. A Rosetta (x86_64) process is not detected as an Apple profile and uses `cpu` instead.

Apple profile names follow `apple-m{generation}-{tier}gb`. The built-in table covers M1 through M5. Newer chips are still looked up in the same precompiled table until they are added to the list, for example `apple-m6-16gb`. Generations 1 and 2 share one set of batch sizes; generation 3 and later share another. Unified memory is rounded down to 8, 16, 32, or 64GB. An M4 with 24GB gets `apple-m4-16gb`. Apple chips with less than 8GB use that generation's 8GB profile. An M5 Pro with 64GB gets `apple-m5-64gb`.

Apple rows above 8GB use FunASR auto with Fun-ASR-Nano on MPS and 30-second chunks. The 8GB floor uses CPU FunASR. MLX requires an explicit `--backend mlx` and uses local `q8`, converted from Fun-ASR-Nano `model.pt`. Speaker and VAD stay on CPU. Diarize runs in the current process. Apple rows are unvalidated.

Pin a profile, or change just one option:

```bash
podtrans transcribe --profile apple-m4-16gb
podtrans transcribe --profile apple-m4-16gb --device cpu
podtrans transcribe --profile apple-m4-16gb --device mps
podtrans diarize --profile apple-m4-16gb --speaker-batch-size 16
```

`speaker-service --profile apple-m4-16gb` fails, because this profile does not use the speaker service.

The recommended values are conservative starting points. The worker count is also scaled down to the number of CPU cores on the machine, so that each process's feature-extraction threads do not oversubscribe the CPU.

CPU-only profiles scale with the core count:

| CPU cores | ASR decode | Diarize workers | Model CPU threads |
| ---: | ---: | ---: | ---: |
| 4 | 1 | 1 | 4 |
| 8 | 2 | 1 | 8 |
| 16 or more | 2 | 1 | Up to 16 |

GPU profiles use `min(profile max workers, CPU cores // 4)`, and still limit decode and model threads per worker. When VRAM is plentiful but CPU cores are few, the program does not blindly start a large number of processes.

| Profile | Typical GPUs | ASR backend | ASR batch | Speaker batch | Max workers |
| --- | --- | --- | ---: | ---: | ---: |
| `cpu` | x86-64 CPU | AutoModel/CPU | 1 | 32 | 1 |
| `gpu-4gb` | GTX 1650, RTX 3050 4GB | AutoModel | 1 | 32 | 1 |
| `gpu-6gb` | RTX 2060, RTX 3050 6GB | AutoModel | 1 | 64 | 2 |
| `gpu-8gb` | RTX 3060 Ti, RTX 4060, RTX 5060 | AutoModel | 2 | 128 | 2 |
| `gpu-12gb` | RTX 3060, RTX 4070, RTX 5070 | vLLM BF16 | 4 | 256 | 3 |
| `gpu-16gb` | RTX 4060 Ti, 4070 Ti SUPER, 4080, 5060 Ti, 5070 Ti, 5080 | vLLM BF16 | 6 | 384 | 4 |
| `gpu-24gb` | RTX 3090, RTX 4090 | vLLM BF16 | 8 | 512 | 6 |
| `gpu-32gb` | RTX 5090 | vLLM BF16 | 12 | 1024 | 8 |
| `gpu-48gb` | 48GB modded cards or workstation cards | vLLM BF16 | 16 | 1536 | 16 |

VRAM sizes that are not in the table, such as 10GB, 11GB, or 20GB, select the nearest tier that does not exceed the actual VRAM. For example, 10GB uses `gpu-8gb` and 20GB uses `gpu-16gb`.

NVIDIA CUDA is the currently validated path. AMD RX 7600 8GB, 7700 XT 12GB, 7800 XT/7900 GRE 16GB, 7900 XT 20GB, and 7900 XTX 24GB can use the tier with the same VRAM as a starting point for options, but need a compatible ROCm PyTorch/FunASR/vLLM environment. If runtime compatibility cannot be met, use `cpu`.

`gpu-48gb` comes from validation on the project's own node. All other NVIDIA profiles and all Apple profiles are scaled conservatively by VRAM, unified memory, and CPU core count. Validate them on a few episodes first, then raise the options based on peak VRAM, GPU utilization, and CPU feature-extraction speed. In the catalog, only `gpu-48gb` has `validated_on_host` set to true.

## Configuration

```bash
mkdir -p data
cp examples/channels.example.txt data/channels.txt
```

Edit `data/channels.txt` and put one podcast name per line. Resolve the RSS feeds and write `data/feeds.json`:

```bash
podtrans archive --resolve --metadata-only
```

You can also prepare the feed mapping by hand, using `examples/feeds.example.json` as a reference.

## Archiving

```bash
podtrans archive --download --workers 4
podtrans archive --verify
podtrans validate-archive --workers 8
```

Tasks can be re-run safely; files that already exist and pass verification are not downloaded again. `--retry-problems` retries only the problem entries from the last verification report.

## Transcription

CPU:

```bash
podtrans transcribe --profile cpu
```

Select by hardware flags or a friendly profile name:

```bash
podtrans transcribe --apple-generation 4 --unified-gb 24
podtrans transcribe --profile auto
podtrans transcribe --profile speed
```

On Apple Silicon you can also just run `podtrans transcribe`. The command applies the detected profile and writes `podtrans: using profile <name>` to standard error. Options given later still take precedence. On Linux, when the selection flags are omitted, this command keeps its own defaults.

Pick the 24GB profile and override one preset option:

```bash
podtrans transcribe \
  --profile gpu-24gb \
  --batch-size 6
```

User options come after the preset options, so they take precedence. The 4GB to 8GB tiers use FunASR AutoModel by default to avoid the extra VRAM overhead of vLLM; NVIDIA GPUs with 12GB or more on Ampere or newer use BF16 vLLM by default. The ASR model stays the same.

## External transcription services

`podtrans transcribe --asr-url` sends chunked audio to a service compatible with OpenAI `/v1/audio/transcriptions`. HTTP servers for Faster-Whisper and Whisper.cpp, as well as cloud APIs that require a key, all fall into this category. This path neither installs nor imports FunASR. The machine still needs `ffmpeg` on `PATH`.

The key is read only from the `PODTRANS_ASR_API_KEY` environment variable. Neither the command line nor `transcript.json` contains the key. A successful transcription records only the validated service root URL in `meta.asr_url`.

```bash
export PODTRANS_ASR_API_KEY=sk-local

podtrans transcribe \
  --asr-url http://127.0.0.1:8000/v1 \
  --model Systran/faster-whisper-large-v3 \
  --language zh \
  --chunk-seconds 300 \
  --batch-size 4
```

If the service does not check keys, you can leave `PODTRANS_ASR_API_KEY` unset.

`--asr-url` is the API root that already includes `/v1`. A full URL ending in `/audio/transcriptions` also works. URLs with user info or a query string fail. Remote mode requires an explicit `--model` and does not fall back to the FunASR default model name. With `--language auto`, the request omits the language and the service detects it. `--batch-size` is the number of chunk requests sent concurrently in one batch. For Whisper-style services, set `--chunk-seconds` to `300`. If you omit it, the default remains `30` so local FunASR chunking does not change.

`--profile`, `--backend`, `--device`, `--batch-size-s`, `--max-length`, `--tensor-parallel-size`, and all `--vllm-*` options describe only the in-process FunASR. The command fails if any of them is combined with `--asr-url`.

When the service returns only text without word-level timestamps, `timestamps` in the chunk is an empty array `[]`. Speaker labeling later spreads the text across that chunk's time window using the existing rules.

## Speaker labeling

CPU mode loads the model in the current process:

```bash
podtrans diarize --profile cpu
```

Apple Silicon profiles also load the speaker model in the current process:

```bash
podtrans diarize --profile apple-m4-16gb
```

With an Apple or CPU profile, `diarize-batch` does not start `speaker-service` and allows only one worker.

GPU high-concurrency mode:

```bash
podtrans diarize-batch --profile gpu-24gb
```

Without `--attach`, `diarize-batch` runs in this order:

1. Start a dedicated speaker-embedding service.
2. Wait for the Unix socket health check to pass.
3. Start sharded workers based on the profile and CPU core count.
4. Workers handle only decoding, VAD, clustering, and writing results; they do not load the GPU speaker model again.
5. Wait for all workers to finish, run a final `--verify-only`, then stop the service.

Limit the workers and pass the remaining options to each diarize worker:

```bash
podtrans diarize-batch \
  --profile gpu-16gb \
  --workers 2 \
  -- \
  --channel "Example Podcast" \
  --limit 10
```

Replace `Example Podcast` with a podcast name exactly as it appears in `data/channels.txt`.

You can also manage the shared service by hand. A single worker takes the socket directly. A batch of workers must be given `--attach` explicitly. The batch command does not switch to attach mode on its own when the socket already exists.

```bash
podtrans speaker-service --profile gpu-24gb

podtrans diarize \
  --profile gpu-24gb \
  --speaker-service-socket /tmp/podtrans-speaker-embedding.sock

podtrans diarize-batch \
  --profile gpu-24gb \
  --attach \
  -- \
  --channel "Example Podcast" \
  --limit 10
```

`--attach` only checks the socket's health and then starts the sharded workers. It does not delete the socket, does not clear the service log, and does not stop the service on exit or on SIGINT. The batch command passes `--speaker-service-socket` to the workers itself. Forwarding that option again after `--` fails.

Without `--attach`, if the socket is already healthy, `diarize-batch` refuses to delete it and suggests using `--attach` or stopping the service first. Without `--attach`, a CPU profile still runs single-stream labeling in the current process. With `--attach`, the speaker model goes through the socket, even though VAD stays on CPU.

The speaker model keeps the official FP32 weights as loaded by FunASR, and does not call `half()` or any other forced low-precision conversion. TF32 is not enabled implicitly either.

## Output

Each episode directory may contain:

- `transcript.json`, `transcript.txt`, `transcript.md`: raw ASR results.
- `transcript_speakers.json`, `.txt`, `.md`: results with `Speaker N` labels.
- `episode.json`, `shownotes.md`: archive metadata and show notes.

Global reports, progress manifests, and service logs are written to `data/_metadata/`.

## Development and verification

```bash
uv lock --check
uv sync
uv run python -m unittest discover -s tests -v
uv build
```

CI uses the committed `uv.lock` to create the environment, run the tests, verify the CLI, and build the wheel and source distribution.

## Acceptable use

Archive only public content that you have the right to download and process, and follow the terms of the RSS services, content platforms, and copyright holders. This project does not bypass logins, paywalls, DRM, or access controls.

## Models and licenses

The code in this repository is licensed under [MIT](LICENSE). The repository does **not** include model weights; users download them at runtime.

The models used by default and their licenses are listed below (the model cards are authoritative):

- `FunAudioLLM/Fun-ASR-Nano-2512` (default ASR): [Apache-2.0](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) (Hugging Face card; the License field for this model on ModelScope is empty, so the Hugging Face card is used here). These weights are also subject to the [FunASR Model License v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE): when using, copying, modifying, or sharing them, you must give attribution and author information and keep the relevant model names.
- `baicai1145/fun-asr-nano-mlx` (tokenizer / config sidecars for the MLX path): [Apache-2.0](https://huggingface.co/baicai1145/fun-asr-nano-mlx) (apache-2.0 on both Hugging Face and ModelScope; the repository includes a NOTICE).
- `iic/speech_eres2netv2_sv_zh-cn_16k-common` (speaker): [Apache License 2.0](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common) (ModelScope).
- `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` (VAD): [Apache License 2.0](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch) (ModelScope).

The FunASR inference library itself is MIT. When using the weights above, follow their licenses, especially the attribution requirements of the FunASR Model License.

## License

[MIT](LICENSE)
