# llama.cpp for MTT S80 (b12600 / MoE + MTP 版)

在摩尔线程 MTT S80 上用**官方桌面驱动 + MUSA 4.0.1**运行 llama.cpp，针对
Qwen3.8-35B-A3B MoE 模型做了实测调优，配合 MTP 推测解码约 **15 t/s**。

- `master` 分支：原 9B dense 快照（约 24 t/s，稳定版）
- 本分支 `moe-b12600`：较新的 llama.cpp b12600 基座 + S80 patch，跑 MoE + MTP

两个分支都可在 GitHub 页面直接 Download ZIP。

## 环境要求

| 项目 | 版本 |
|---|---|
| GPU | Moore Threads MTT S80 (16 GB, mp_21) |
| 驱动 | DESKTOP 桌面驱动 3.0.0（不是 compute-only） |
| MUSA SDK | 4.0.1，装在 `/usr/local/musa` |
| 系统 | Ubuntu 24.04，x86-64，建议 32 GB 内存（16 线程 CPU） |
| 内核 | 6.8.0-136-generic（锁定，勿升级） |

驱动下载：https://www.mthreads.com/pes/drivers/driver-info/DESKTOP_MTT_S80/release-note/v3.0.0?productType=DESKTOP&osVersion=MTT_S80_Ubuntu

Moore Threads 3.0.0 驱动通过 DKMS 编译，只支持到内核 6.8.0-136。不要升级内核，
否则 mtgpu 模块无法编译，S80 不可用。

图形桌面与 LLM 共用 S80，桌面本身占约 2 GB 显存。MoE 模型 14 GB + draft 1.9 GB，
显存余量很小，按下面的参数用。

## 版本信息

- 上游 llama.cpp：`29b52d398b294509746ec0a3f28948a3107c0213`（release b12600）
- 内嵌 WebUI：`b10726`
- S80 内核优化 patch 来源：pcalau12i，https://www.foleosoft.com/software/S80-Optimization
- 源码快照已包含 patch 及 MUSA 4.0.1 编译适配，构建时无需再 apply patch

## 模型下载

用 hf-mirror 下载 35B-A3B MoE 主模型和 MTP draft 头。仓库里的文件名与本地文件名不同，
下载时直接重命名：

```sh
mkdir -p ~/models
BASE=https://hf-mirror.com/IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-MiniPlus-V2.1-Abliterated-GGUF/resolve/main
wget -c "$BASE/Qwen3.8-35B-A3B-Distill.APEX-I-MiniPlus-V2.1-Abliterated.gguf" \
  -O ~/models/Qwen3.8-35B-A3B-Distill-Abliterated-APEX.gguf
wget -c "$BASE/mtp-Qwen3.8-35B-A3B-Distill-Q8_0.gguf" \
  -O ~/models/mtp-Qwen3.8-35B-A3B-Q8_0.gguf
```

## 推荐配置（实测）

```sh
./build/bin/llama-server \
  -m ~/models/Qwen3.8-35B-A3B-Distill-Abliterated-APEX.gguf \
  -ngl 999 -c 8192 -np 1 --n-cpu-moe 16 -t 16 -tb 16 \
  --model-draft ~/models/mtp-Qwen3.8-35B-A3B-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 127.0.0.1 --port 8080 --no-warmup
```

`./run.sh` 已固化这套参数。

关键点：

- **不要加 `--load-mode none`**。在 CPU/GPU 混合放专家权重时，这个开关会让输出退化成
  重复 token（如 "3333..." / "-registration-..."）。用默认 mmap 正常。
- `--n-cpu-moe 16` 表示前 16 层专家放 CPU、后 24 层放 GPU，实测最快（约 15.4 t/s）。
- MTP draft 头必须显式用 `--model-draft` 和 `--spec-type draft-mtp` 指定，否则不启用。
- 带 draft 时最多 **32 层专家放 GPU**（`--n-cpu-moe 8`，约 15.0 t/s）；36 层以上 draft
  加载会因显存不足失败。不带 draft 可放满 40 层，但显存 16.1 GB / 16.4 GB 太极限且更慢。
- 上下文只实测了 8192。64K 会增加 KV cache 占用，未在本版本验证，别直接套用。

| GPU 专家层数 | `--n-cpu-moe` | 带 MTP | 显存 |
|---|---|---|---|
| 0 | 40 | 15.0 t/s | ~5.5 GB |
| 8 | 32 | 15.3 t/s | ~8.1 GB |
| 16 | 24 | 15.2 t/s | - |
| **24** | **16** | **15.4 t/s** | **12.9 GB** |
| 32 | 8 | 15.0 t/s | 15.3 GB |
| 36+ | 4 或更小 | draft OOM | - |

## 使用

```sh
./setup.sh   # 下载匹配版本 WebUI dist，cmake 编译
./run.sh     # 默认绑 127.0.0.1；内网访问用 LAN_ACCESS=1 ./run.sh
```

`run.sh` 支持的环境变量：`MODEL`、`DRAFT_MODEL`、`CTX`（默认 8192）、
`N_CPU_MOE`（默认 16）、`PORT`（8080）、`DRAFTS`（3）、`LAN_ACCESS`。

启动后用 OpenAI 兼容 API：`http://127.0.0.1:8080/v1`。
