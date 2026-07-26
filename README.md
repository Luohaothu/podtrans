# podtrans

`podtrans` 是一套可恢复运行的播客归档、FunASR 转录和说话人标注工具。它将节目数据与源码分离，并通过单独的说话人嵌入服务让多个并发 worker 共享一份 GPU 模型权重。

## 功能

- 从 RSS 拉取节目元数据、音频和 shownotes，并生成完整性报告。
- 使用 `FunAudioLLM/Fun-ASR-Nano-2512` 批量转录归档音频。
- 使用 FunASR VAD 与 `iic/speech_eres2netv2_sv_zh-cn_16k-common` 生成说话人标签。
- 由一个 GPU 嵌入服务集中加载说话人模型，多个 worker 通过 Unix socket 并发提交批次。
- 支持断点续跑、错误重试、任务分片、结果清单和最终校验。

## 目录

```text
podtrans/
├── archive_xiaoyuzhou_channels.py  # RSS 解析、归档和校验
├── transcribe_funasr.py            # ASR 转录
├── diarize_funasr.py               # 说话人标注
├── speaker_embedding_service.py    # 单实例 GPU 嵌入服务
├── scripts/                        # 并发服务和 systemd 启动脚本
├── tools/                          # 归档校验与性能基准
├── tests/                          # 说话人时间轴单元测试
└── examples/                       # 可提交的配置示例
```

真实节目、音频、转录结果、运行日志和模型缓存均不进入 Git。程序默认使用仓库下的 `data/`，也可通过 `PODTRANS_DATA_DIR` 指向独立磁盘或 NAS。

## 安装

要求 Linux、Python 3.11 或 3.12，以及与显卡驱动匹配的 CUDA 版 PyTorch。`ffmpeg` 可由系统提供；没有系统 `ffmpeg` 时会使用 `imageio-ffmpeg`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements.txt
```

需要 vLLM 后端时安装：

```bash
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-vllm.txt
```

FunASR 模型默认从 ModelScope 获取，`--hub ms` 也是说话人标注的默认值。可用 `MODELSCOPE_CACHE=/path/to/cache` 将模型缓存放到高速本地盘。

## 准备配置

```bash
mkdir -p data
cp examples/channels.example.txt data/channels.txt
```

编辑 `data/channels.txt`，每行填写一个节目名称。归档器可通过 iTunes Search 解析 RSS 地址并写入 `data/feeds.json`：

```bash
PODTRANS_DATA_DIR="$PWD/data" \
  .venv/bin/python archive_xiaoyuzhou_channels.py \
  --resolve --metadata-only
```

也可以参考 `examples/feeds.example.json` 手工准备 feed 映射。

## 归档

```bash
export PODTRANS_DATA_DIR="$PWD/data"

# 下载全部音频，4 路并发
.venv/bin/python archive_xiaoyuzhou_channels.py --download --workers 4

# 校验本地归档
.venv/bin/python archive_xiaoyuzhou_channels.py --verify
.venv/bin/python tools/validate_archive.py --workers 8
```

任务可重复执行；已有且校验通过的文件不会重复下载。`--retry-problems` 只重试上次验证报告中的异常条目。

## 转录

精度基线使用 FunASR `AutoModel`：

```bash
export PODTRANS_DATA_DIR="$PWD/data"
.venv/bin/python transcribe_funasr.py \
  --backend auto \
  --device cuda \
  --batch-size 4 \
  --job-batch-size 2 \
  --decode-workers 2
```

在支持 BF16 的 Ampere 或更新 GPU 上，可使用 vLLM 提高吞吐：

```bash
.venv/bin/python transcribe_funasr.py \
  --backend vllm \
  --vllm-dtype bfloat16 \
  --vllm-gpu-memory-utilization 0.90 \
  --batch-size 8 \
  --job-batch-size 4 \
  --decode-workers 4
```

普通模式和 vLLM 模式使用同一个 ASR 模型。实际批大小应通过短任务逐步提高，避免解码端供给不足或显存溢出。

## 说话人标注

推荐用用户级 systemd 启动完整流程：

```bash
export PODTRANS_DATA_DIR="$PWD/data"
bash scripts/start_speaker_diarization.sh
```

启动器先创建一个 `speaker_embedding_service.py` GPU 服务，等待 socket 健康检查通过后，再启动多个分片 worker。worker 负责音频解码、VAD、聚类和结果写入，但不会重复加载说话人嵌入模型。

默认参数偏保守：

| 环境变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `SPEAKER_WORKERS` | `4` | 并发任务分片数 |
| `SPEAKER_BATCH_SIZE` | `512` | 单次嵌入窗口上限 |
| `SPEAKER_JOB_BATCH_SIZE` | `4` | 每个 worker 的节目批次 |
| `SPEAKER_DECODE_WORKERS` | `4` | 每个 worker 的解码并发 |
| `SPEAKER_SERVICE_SOCKET` | `/tmp/podtrans-speaker-embedding.sock` | 服务 socket |

48GB 显存节点可从下列配置开始验证：

```bash
SPEAKER_WORKERS=16 \
SPEAKER_BATCH_SIZE=1536 \
PODTRANS_DATA_DIR="$PWD/data" \
bash scripts/start_speaker_diarization.sh
```

说话人模型保持 FunASR 加载后的原始精度，不调用 `half()` 或其他强制低精度转换。`--allow-tf32` 只有显式传入时才启用。

## 输出

每个节目目录可能包含：

- `transcript.json`、`transcript.txt`、`transcript.md`：ASR 原始结果。
- `transcript_speakers.json`、`.txt`、`.md`：带 `Speaker N` 标签的结果。
- `episode.json`、`shownotes.md`：归档元数据和 shownotes。

全局报告、进度清单和服务日志写入 `data/_metadata/`。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q .
bash -n scripts/*.sh
```

模型与吞吐验证工具：

```bash
.venv/bin/python tools/validate_diarize_funasr.py --help
.venv/bin/python tools/benchmark_diarization_workers.py --help
.venv/bin/python tools/benchmark_fbank.py --help
```

## 使用边界

仅归档你有权下载和处理的公开内容，并遵守 RSS 服务、内容平台和版权方的条款。该项目不绕过登录、付费墙、DRM 或访问控制。

## License

[MIT](LICENSE)
