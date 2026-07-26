# podtrans

`podtrans` 是一套可安装的播客归档、FunASR 转录和说话人标注 CLI。项目使用 uv 管理依赖与锁文件；安装后统一通过 `podtrans <command>` 使用，不需要执行仓库中的 Python 或 Shell 脚本。

## 功能

- 从 RSS 拉取节目元数据、音频和 shownotes，并生成完整性报告。
- 使用 `FunAudioLLM/Fun-ASR-Nano-2512` 批量转录归档音频。
- 使用 FunASR VAD 与 `iic/speech_eres2netv2_sv_zh-cn_16k-common` 生成说话人标签。
- 用一个独立 GPU 服务加载说话人模型，多个 worker 通过 Unix socket 共享权重。
- 支持 CPU-only 运行、任务分片、错误重试、最终校验和硬件预设。
- 提供 4GB 至 48GB 显存档位，覆盖主流 GeForce 消费级显卡。

## 安装

支持 Python 3.11 和 3.12。

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
podtrans profiles
podtrans recommend
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

## 硬件推荐

列出所有内置档位：

```bash
podtrans profiles
podtrans profiles --json
```

自动读取 NVIDIA GPU 名称和显存并给出配置：

```bash
podtrans recommend
```

没有 NVIDIA GPU 时自动选择 CPU。也可手工指定：

```bash
podtrans recommend --vram-gb 16 --cpu-cores 12
podtrans recommend --cpu-only --cpu-cores 16
```

推荐值是保守起点，worker 数还会按照本机 CPU 核数收缩，避免每个进程的特征提取线程造成 CPU 过量争用。

CPU-only 档位按核心数调整：

| CPU 核心 | ASR decode | Diarize worker | 模型 CPU threads |
| ---: | ---: | ---: | ---: |
| 4 | 1 | 1 | 4 |
| 8 | 2 | 1 | 8 |
| 16 及以上 | 2 | 1 | 最高 16 |

GPU 档位使用 `min(profile 最大 worker, CPU 核心数 // 4)`，并继续限制每个 worker 的 decode 和模型线程数。显存足够但 CPU 核数较少时不会盲目启动大量进程。

| Profile | 典型显卡 | ASR 后端 | ASR batch | 说话人 batch | 最大 worker |
| --- | --- | --- | ---: | ---: | ---: |
| `cpu` | 无可用 GPU | AutoModel/CPU | 1 | 32 | 1 |
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

`gpu-48gb` 来自项目实际节点验证。其他档位按显存和 CPU 供给保守缩放，应先用少量节目验证，再根据峰值显存、GPU 利用率和 CPU 特征提取速度提高参数。

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

自动推荐配置：

```bash
profile="$(podtrans recommend --name-only)"
podtrans transcribe --profile "${profile}"
```

指定 24GB 档位，并覆盖一个预设参数：

```bash
podtrans transcribe \
  --profile gpu-24gb \
  --batch-size 6
```

用户参数位于预设参数之后，因此具有更高优先级。4GB 至 8GB 档位默认使用 FunASR AutoModel，避免 vLLM 额外显存开销；12GB 及以上 Ampere 或更新的 NVIDIA GPU 默认使用 BF16 vLLM。ASR 模型不变。

## 说话人标注

CPU 模式在当前进程中加载模型：

```bash
podtrans diarize --profile cpu
```

GPU 高并发模式：

```bash
podtrans diarize-batch --profile gpu-24gb
```

`diarize-batch` 的执行顺序：

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

也可以手工管理共享服务：

```bash
podtrans speaker-service --profile gpu-24gb

podtrans diarize \
  --profile gpu-24gb \
  --speaker-service-socket /tmp/podtrans-speaker-embedding.sock
```

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
uv run podtrans profiles
uv build
```

CI 使用提交到仓库的 `uv.lock` 创建环境、运行测试、验证 CLI 并构建 wheel 和源码包。

## 使用边界

仅归档你有权下载和处理的公开内容，并遵守 RSS 服务、内容平台和版权方的条款。该项目不绕过登录、付费墙、DRM 或访问控制。

## License

[MIT](LICENSE)
