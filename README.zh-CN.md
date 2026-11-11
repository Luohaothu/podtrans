# podtrans

[English](README.md) | 简体中文

`podtrans` 是一套可安装的播客归档、转录和说话人标注 CLI。项目使用 uv 管理依赖与锁文件；安装后统一通过 `podtrans <command>` 使用，不需要执行仓库中的 Python 或 Shell 脚本。

## 功能

- 从 RSS 拉取节目元数据、音频和 shownotes，并生成完整性报告。
- 使用 `FunAudioLLM/Fun-ASR-Nano-2512` 批量转录归档音频。
- 可以把转录指到兼容 OpenAI `/v1/audio/transcriptions` 的服务，也可以用 `diarize-batch --attach` 接入已经启动的说话人服务。
- 使用 FunASR VAD 与 `iic/speech_eres2netv2_sv_zh-cn_16k-common` 生成说话人标签。
- 用一个独立 GPU 服务加载说话人模型，多个 worker 通过 Unix socket 共享权重。
- 支持 CPU-only 运行、任务分片、错误重试、最终校验和硬件预设。
- 提供 4GB 至 48GB 显存档位，覆盖主流 GeForce 消费级显卡。

## 安装

支持 Python 3.11 和 3.12。
只支持 Linux 和 macOS。

### uv

只使用归档和配置查询功能：

```bash
uv sync
uv run podtrans --help
```

安装 FunASR、PyTorch 和说话人标注依赖：

```bash
uv sync --extra inference
uv run podtrans --version
```

使用 vLLM 后端：

```bash
uv sync --extra vllm
```

国内节点可以使用 TUNA 加速 Python 包解析和下载：

```bash
uv sync \
  --extra vllm \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

模型默认通过 ModelScope 下载。可用 `MODELSCOPE_CACHE=/path/to/cache` 将模型放在本地高速盘。

### pip

仓库可直接作为标准 Python 包安装：

```bash
python -m pip install .
podtrans --help
```

安装推理功能：

```bash
python -m pip install ".[inference]"
```

安装 vLLM 推理功能：

```bash
python -m pip install ".[vllm]"
```

`mlx` 附加依赖用于 Apple Silicon macOS。Apple MLX 路径尚未验证。

```bash
python -m pip install ".[mlx]"
```

也可以先构建 wheel，再在其他环境安装：

```bash
uv build
python -m pip install "dist/podtrans-0.2.0-py3-none-any.whl[inference]"
```

CUDA 或 ROCm 环境应按 PyTorch 官方说明安装与驱动匹配的 `torch` 和 `torchaudio` 构建。项目不会强制替换用户已经安装的硬件专用 PyTorch。

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

查看子命令参数：

```bash
podtrans transcribe --help
podtrans help diarize
```

真实节目、音频、转录结果、运行日志和模型缓存均不进入 Git。程序默认使用当前目录的 `data/`，也可显式设置：

```bash
export PODTRANS_DATA_DIR=/path/to/podcast-data
```

## 硬件档位

### Apple Silicon ASR

`profiles` 和 `recommend` 已移除。硬件选择改为 `transcribe`、`diarize`、`diarize-batch` 和 `speaker-service` 上的旗标。

Apple Silicon 目录中 8GB 以上的档位默认使用 FunASR auto，在 PyTorch MPS 上运行 `FunAudioLLM/Fun-ASR-Nano-2512`。默认分片时长为 30 秒。在 Apple Silicon macOS 上可以用以下任一命令开始：

```bash
podtrans transcribe
podtrans transcribe --profile auto
```

8GB 最低档仍使用 CPU FunASR。`quality` 对所有 Apple 档位都选择 CPU FunASR。`speed` 使用与 `auto` 相同的 ASR。要显式选择 CPU，运行 `podtrans transcribe --device cpu`。说话人模型和 VAD 始终在 CPU 上运行。Apple 档位尚未验证。在 Apple Silicon macOS 上用 `pip install 'podtrans[mlx]'` 安装该附加依赖，其中包含 FunASR、PyTorch 和 funasr-mlx。

MLX 仍需手动启用。第一条命令使用检测到的 Apple 档位；第二条固定一个目录档位，并显式指定模型和本地 `q8` 变体：

```bash
podtrans transcribe --backend mlx
podtrans transcribe --profile apple-m4-16gb --backend mlx --model fun-asr-nano-mlx --mlx-variant q8
```

MLX 使用本地 8-bit `q8` checkpoint，输出上限为 512 个 token。只有 MLX 路径会把分片限制在其前端上限以下，通常为 29 秒。某个分片达到输出上限仍未出现 EOS 时，会发出警告并照常返回文本。MLX 可能在转录中输出 `<>`，见 [issue #6](https://github.com/Luohaothu/podtrans/issues/6) 中的已知限制。

首次执行 MLX prepare 之前，需要先通过 ModelScope 获取 Fun-ASR-Nano-2512 的 PyTorch `model.pt` 权重，约 2.1 GB。运行一次 CPU 或 MPS 路径就会下载它。MLX prepare 会搜索 `$MODELSCOPE_CACHE` 和 `~/.cache/modelscope`，包括扁平化的仓库快照布局。如果缺少 `model.pt`，prepare 会提示如何获取，而不会自行开始下载。

首次 prepare 会在本地转换 `model.pt`，并把结果缓存到 `$PODTRANS_DATA_DIR/models/funasr-mlx/fun-asr-nano-mlx/q8/`，默认为 `./data/models/funasr-mlx/fun-asr-nano-mlx/q8/`。它只从已发布的 `w4a16` 包下载七个小的 tokenizer 和 config 文件，按已核查的文件集约 4.5 MB。对模型仓库的请求不包含权重分片和权重索引。推理加载的是本地转换得到的 q8 权重，这样 MLX 的特殊 token id 与其解码器保持一致。已完整缓存的 q8 checkpoint 不需要源文件，也不需要联网。要离线转换，请先缓存 `model.pt` 和 MLX sidecar 文件，否则 prepare 会报告缺少的源文件。转换过程的中间文件是临时的。已有的 ModelScope 和 Hugging Face 缓存保持不变。

本地转换依赖 funasr-mlx 的 `convert_checkpoint` 函数签名，要求 `funasr-mlx>=0.2.3,<0.3`。prepare 在转换前会检查该签名。截断检测使用 funasr-mlx 0.2.3 `llm_asr.py` 中的 Qwen3 EOS id `151643`、`151645` 和 `151671`。放宽版本范围前，请重新核对这两处依赖。

在功能命令上直接选择硬件，不再提供独立的 `profiles` / `recommend` 子命令。

```bash
podtrans transcribe --apple-generation 4 --unified-gb 24
podtrans transcribe --profile auto
podtrans transcribe --profile quality
podtrans transcribe --profile speed
podtrans diarize --vram-gb 16
podtrans diarize-batch --cpu-only
```

`--profile` 接受友好档位或具体目录 id：

| 取值 | 含义 |
| --- | --- |
| `auto` | 根据检测到的机器或硬件旗标查询目录。 |
| `quality` | 同一 Apple 目录档位，ASR 改用 CPU FunASR。 |
| `speed` | 与 `auto` 相同的目录档位和 ASR，包括 8GB 最低档以上使用 MPS。 |
| `cpu` / `gpu-*` / `apple-m*-*gb` | 单个目录 id。 |

`--apple-generation` 与 `--unified-gb` 必须一起给出，且不能与 `--vram-gb`、`--cpu-only` 或具体目录 `--profile` 混用。

在 macOS 上，`transcribe`、`diarize` 和 `speaker-service` 在省略 `--profile` 与硬件旗标时会自动选用检测到的档位，并把名称写到标准错误，例如 `podtrans: using profile apple-m4-16gb`。写在后面的参数覆盖档位里的同名参数。Linux 上省略选择旗标时，这三个命令保持各自的默认参数；传入 `--profile auto` 或硬件旗标时才会选择。`diarize-batch` 在每个系统上都会选择一个档位。Rosetta（x86_64）进程不会被检测为 Apple 档位，而是使用 `cpu`。

Apple 档位的名称是 `apple-m{代}-{档}gb`。内置表覆盖 M1 到 M5；更新的芯片在被写进列表之前，仍然按同一张预编译表查询，例如 `apple-m6-16gb`。代际 1 和 2 共用一套批次；代际 3 及以后共用另一套。统一内存按 8、16、32、64GB 向下取整。24GB 的 M4 得到 `apple-m4-16gb`。低于 8GB 的 Apple 芯片使用该代的 8GB 档。M5 Pro 64GB 得到 `apple-m5-64gb`。

8GB 以上的 Apple 档位使用 FunASR auto，在 MPS 上运行 Fun-ASR-Nano，分片 30 秒。8GB 最低档使用 CPU FunASR。MLX 需要显式传入 `--backend mlx`，使用由 Fun-ASR-Nano `model.pt` 转换得到的本地 `q8`。说话人模型和 VAD 始终在 CPU 上运行。说话人标注在当前进程中运行。Apple 档位尚未验证。

固定档位，或只改一个参数：

```bash
podtrans transcribe --profile apple-m4-16gb
podtrans transcribe --profile apple-m4-16gb --device cpu
podtrans transcribe --profile apple-m4-16gb --device mps
podtrans diarize --profile apple-m4-16gb --speaker-batch-size 16
```

`speaker-service --profile apple-m4-16gb` 会失败，因为这个档位不使用说话人服务。

推荐值是保守起点，worker 数还会按照本机 CPU 核数收缩，避免每个进程的特征提取线程造成 CPU 过量争用。

CPU-only 档位按核心数调整：

| CPU 核心 | ASR 解码 | 说话人标注 worker | 模型 CPU 线程 |
| ---: | ---: | ---: | ---: |
| 4 | 1 | 1 | 4 |
| 8 | 2 | 1 | 8 |
| 16 及以上 | 2 | 1 | 最高 16 |

GPU 档位使用 `min(profile 最大 worker, CPU 核心数 // 4)`，并继续限制每个 worker 的 decode 和模型线程数。显存足够但 CPU 核数较少时不会盲目启动大量进程。

| 档位 | 典型显卡 | ASR 后端 | ASR batch | 说话人 batch | 最大 worker |
| --- | --- | --- | ---: | ---: | ---: |
| `cpu` | x86-64 CPU | AutoModel/CPU | 1 | 32 | 1 |
| `gpu-4gb` | GTX 1650、RTX 3050 4GB | AutoModel | 1 | 32 | 1 |
| `gpu-6gb` | RTX 2060、RTX 3050 6GB | AutoModel | 1 | 64 | 2 |
| `gpu-8gb` | RTX 3060 Ti、RTX 4060、RTX 5060 | AutoModel | 2 | 128 | 2 |
| `gpu-12gb` | RTX 3060、RTX 4070、RTX 5070 | vLLM BF16 | 4 | 256 | 3 |
| `gpu-16gb` | RTX 4060 Ti、4070 Ti SUPER、4080、5060 Ti、5070 Ti、5080 | vLLM BF16 | 6 | 384 | 4 |
| `gpu-24gb` | RTX 3090、RTX 4090 | vLLM BF16 | 8 | 512 | 6 |
| `gpu-32gb` | RTX 5090 | vLLM BF16 | 12 | 1024 | 8 |
| `gpu-48gb` | 48GB 改装卡或工作站卡 | vLLM BF16 | 16 | 1536 | 16 |

10GB、11GB、20GB 等非表格档位会选择不超过实际显存的最近档位。例如 10GB 使用 `gpu-8gb`，20GB 使用 `gpu-16gb`。

NVIDIA CUDA 是当前验证路径。AMD RX 7600 8GB、7700 XT 12GB、7800 XT/7900 GRE 16GB、7900 XT 20GB、7900 XTX 24GB 可按同显存档位作为参数起点，但需要兼容的 ROCm PyTorch/FunASR/vLLM 环境；无法满足运行时兼容性时应使用 `cpu`。

`gpu-48gb` 来自项目实际节点验证。其他 NVIDIA 档位和全部 Apple 档位按显存、统一内存和 CPU 核数保守缩放，应先用少量节目验证，再根据峰值显存、GPU 利用率和 CPU 特征提取速度提高参数。目录里只有 `gpu-48gb` 的 `validated_on_host` 为 true。

## 准备配置

```bash
mkdir -p data
cp examples/channels.example.txt data/channels.txt
```

编辑 `data/channels.txt`，每行填写一个节目名称。解析 RSS 并写入 `data/feeds.json`：

```bash
podtrans archive --resolve --metadata-only
```

也可以参考 `examples/feeds.example.json` 手工准备 feed 映射。

## 归档

```bash
podtrans archive --download --workers 4
podtrans archive --verify
podtrans validate-archive --workers 8
```

任务可重复执行；已有且校验通过的文件不会重复下载。`--retry-problems` 只重试上次验证报告中的异常条目。

## 转录

CPU：

```bash
podtrans transcribe --profile cpu
```

按硬件旗标或友好档位选择：

```bash
podtrans transcribe --apple-generation 4 --unified-gb 24
podtrans transcribe --profile auto
podtrans transcribe --profile speed
```

在 Apple Silicon 上也可以直接运行 `podtrans transcribe`。命令会套用检测到的档位，并把 `podtrans: using profile <name>` 写到标准错误。后面的参数仍然优先。Linux 上省略选择旗标时，本命令保持自己的默认值。

指定 24GB 档位，并覆盖一个预设参数：

```bash
podtrans transcribe \
  --profile gpu-24gb \
  --batch-size 6
```

用户参数位于预设参数之后，因此具有更高优先级。4GB 至 8GB 档位默认使用 FunASR AutoModel，避免 vLLM 额外显存开销；12GB 及以上 Ampere 或更新的 NVIDIA GPU 默认使用 BF16 vLLM。ASR 模型不变。

## 外部转录服务

`podtrans transcribe --asr-url` 把分片音频发给兼容 OpenAI `/v1/audio/transcriptions` 的服务。Faster-Whisper、Whisper.cpp 的 HTTP 服务，以及带密钥的云端接口都属于这一类。这条路径不安装，也不导入 FunASR。机器上仍需要 `ffmpeg` 在 `PATH` 里。

密钥只从环境变量 `PODTRANS_ASR_API_KEY` 读取。命令行和 `transcript.json` 都不写密钥。成功的转录在 `meta.asr_url` 里只记录校验后的服务根地址。

```bash
export PODTRANS_ASR_API_KEY=sk-local

podtrans transcribe \
  --asr-url http://127.0.0.1:8000/v1 \
  --model Systran/faster-whisper-large-v3 \
  --language zh \
  --chunk-seconds 300 \
  --batch-size 4
```

服务不校验密钥时，可以不设置 `PODTRANS_ASR_API_KEY`。

`--asr-url` 是已经包含 `/v1` 的 API 根。以 `/audio/transcriptions` 结尾的完整地址也可以。带用户信息或查询字符串的 URL 会失败。远程模式必须写出 `--model`，不会套用 FunASR 的默认模型名。`--language auto` 时请求不带 language，由服务自己检测。`--batch-size` 是同一批里同时发出的分片请求数。Whisper 一类服务建议把 `--chunk-seconds` 设为 `300`。不写这个参数时默认仍是 `30`，以免改变本机 FunASR 的切分。

`--profile`、`--backend`、`--device`、`--batch-size-s`、`--max-length`，以及 `--tensor-parallel-size` 和全部 `--vllm-*`，只描述本进程里的 FunASR。它们和 `--asr-url` 一起出现时命令失败。

服务只返回文本、没有词级时间时，分片里的 `timestamps` 是空数组 `[]`。后面的说话人标注会按原有规则把文本摊到这个分片的时间窗上。

## 说话人标注

CPU 模式在当前进程中加载模型：

```bash
podtrans diarize --profile cpu
```

Apple Silicon 档位同样在当前进程加载说话人模型：

```bash
podtrans diarize --profile apple-m4-16gb
```

`diarize-batch` 遇到 Apple 档位或 CPU 档位时不启动 `speaker-service`，并且只允许一个 worker。

GPU 高并发模式：

```bash
podtrans diarize-batch --profile gpu-24gb
```

不带 `--attach` 时，`diarize-batch` 的执行顺序：

1. 启动一个独立说话人嵌入服务。
2. 等待 Unix socket 健康检查通过。
3. 按 profile 和 CPU 核数启动分片 worker。
4. worker 只负责解码、VAD、聚类和写结果，不重复加载 GPU 说话人模型。
5. 等待全部 worker 完成，执行最终 `--verify-only`，再停止服务。

限制 worker，并把其余参数传给每个 diarize worker：

```bash
podtrans diarize-batch \
  --profile gpu-16gb \
  --workers 2 \
  -- \
  --channel "示例播客" \
  --limit 10
```

请把 `示例播客` 换成 `data/channels.txt` 中的节目名称，写法需完全一致。

也可以手工管理共享服务。单个 worker 直接带上 socket。一批 worker 必须显式传入 `--attach`。socket 已经存在时，批命令不会自己改成附着模式。

```bash
podtrans speaker-service --profile gpu-24gb

podtrans diarize \
  --profile gpu-24gb \
  --speaker-service-socket /tmp/podtrans-speaker-embedding.sock

podtrans diarize-batch \
  --profile gpu-24gb \
  --attach \
  -- \
  --channel "示例播客" \
  --limit 10
```

`--attach` 只检查 socket 健康，然后启动分片 worker。它不删除 socket，不清空服务日志，也不会在退出或收到 SIGINT 时停止这个服务。worker 使用的 `--speaker-service-socket` 由批命令自己传入。在 `--` 后面再转发这个参数会失败。

不带 `--attach` 时，如果 socket 已经健康，`diarize-batch` 拒绝删掉它，并提示改用 `--attach` 或先停止服务。CPU profile 在没有 `--attach` 时仍在当前进程里做单路标注。加上 `--attach` 后，说话人模型走 socket，即使 VAD 留在 CPU。

说话人模型保持 FunASR 加载后的官方 FP32 权重，不调用 `half()` 或其他强制低精度转换。TF32 也不会隐式启用。

## 输出

每个节目目录可能包含：

- `transcript.json`、`transcript.txt`、`transcript.md`：ASR 原始结果。
- `transcript_speakers.json`、`.txt`、`.md`：带 `Speaker N` 标签的结果。
- `episode.json`、`shownotes.md`：归档元数据和 shownotes。

全局报告、进度清单和服务日志写入 `data/_metadata/`。

## 开发与验证

```bash
uv lock --check
uv sync
uv run python -m unittest discover -s tests -v
uv build
```

CI 使用提交到仓库的 `uv.lock` 创建环境、运行测试、验证 CLI 并构建 wheel 和源码包。

## 使用边界

仅归档你有权下载和处理的公开内容，并遵守 RSS 服务、内容平台和版权方的条款。该项目不绕过登录、付费墙、DRM 或访问控制。

## 模型与许可证

本仓库的代码以 [MIT](LICENSE) 许可。仓库**不包含**模型权重；运行时由用户自行下载。

默认使用的模型及许可证如下（以模型卡片为准）：

- `FunAudioLLM/Fun-ASR-Nano-2512`（默认 ASR）：[Apache-2.0](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)（Hugging Face 卡片；ModelScope 上该模型的 License 字段为空，此处以 Hugging Face 为准）。这些权重同时还受 [FunASR Model License v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE) 约束：使用、复制、修改和分享时必须注明出处和作者信息，并保留相关模型名称。
- `baicai1145/fun-asr-nano-mlx`（MLX 路径用的 tokenizer / config sidecar）：[Apache-2.0](https://huggingface.co/baicai1145/fun-asr-nano-mlx)（Hugging Face 与 ModelScope 均为 apache-2.0；该仓库带 NOTICE）。
- `iic/speech_eres2netv2_sv_zh-cn_16k-common`（说话人）：[Apache License 2.0](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common)（ModelScope）。
- `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`（VAD）：[Apache License 2.0](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)（ModelScope）。

FunASR 推理库本身是 MIT。使用上述权重时请遵守对应许可证，尤其是 FunASR Model License 的署名要求。

## 许可证

[MIT](LICENSE)
