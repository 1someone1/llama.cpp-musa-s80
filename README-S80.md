# llama.cpp for MTT S80 (MUSA)

在摩尔线程 MTT S80 上用**官方桌面驱动**运行 llama.cpp。
本仓库提供编译好的源码快照和一键脚本：clone 后 `./setup.sh` 编译，`./run.sh` 启动。

## 环境要求

| 项目 | 版本 |
|---|---|
| GPU | Moore Threads MTT S80 (16 GB, mp_21) |
| 驱动 | DESKTOP 桌面驱动 3.0.0（不是 compute-only） |
| MUSA SDK | 4.0.1，装在 `/usr/local/musa` |
| 系统 | Ubuntu 24.04，x86-64 |
| 内核 | 6.8.0-136-generic（锁定，勿升级） |

驱动下载：https://www.mthreads.com/pes/drivers/driver-info/DESKTOP_MTT_S80/release-note/v3.0.0?productType=DESKTOP&osVersion=MTT_S80_Ubuntu

Moore Threads 3.0.0 驱动通过 DKMS 编译，只支持到内核 6.8.0-136。不要升级内核，
否则 mtgpu 模块无法编译，S80 不可用。

## 推荐模型

本版本的实测甜点是 **9B dense + MTP**，显存占用低、速度快（decode 约 24 t/s），
支持图片输入。图形桌面与 LLM 共用 S80，14B 以上或 128K 上下文会挤爆显示驱动。

### 首选：Qwen3.8-9B-Distill-uncensored-heretic Q4_K_M

- 9B dense，无限制（heretic），带 MTP 头（开启后 decode 翻倍）
- 5.9 GB，配 64K 上下文约 10 GB 显存，安全
- 视觉：需配 mmproj（0.6 GB），加载后 vision/video 可用

下载（hf-mirror）：

```sh
BASE=https://hf-mirror.com/mradermacher/Qwen3.8-9B-Distill-uncensored-heretic-GGUF/resolve/main
wget -c "$BASE/Qwen3.8-9B-Distill-uncensored-heretic.Q4_K_M.gguf" -O ~/models/Qwen3.8-9B-Q4_K_M.gguf
wget -c "$BASE/Qwen3.8-9B-Distill-uncensored-heretic.mmproj-Q8_0.gguf" -O ~/models/Qwen3.8-9B-mmproj-Q8_0.gguf
```

启动：

```sh
MODEL=~/models/Qwen3.8-9B-Q4_K_M.gguf \
MMPROJ=~/models/Qwen3.8-9B-mmproj-Q8_0.gguf \
./run.sh
```

### 备选：Qwen3.5-9B-Ultra-Uncensored-Heretic Q4_K_M

同为 9B dense 无限制模型，无 MTP 头。
https://hf-mirror.com/llmfan46/Qwen3.5-9B-Ultra-Uncensored-Heretic-GGUF

### 不建议在本版本使用 MoE 模型

35B-A3B 类 MoE（如 Qwen3.8-35B-A3B-Distill）需要专家权重卸载到系统 RAM，
在图形桌面共用 GPU 的环境下会把系统拖到卡死（Wayland/Xorg 均如此）。
如需运行 MoE，需无图形会话 + compute_only 驱动模式，属另一种配置。

## 使用

```sh
./setup.sh   # 下匹配版本 WebUI dist，cmake 编译
./run.sh     # 默认绑 127.0.0.1；内网访问用 LAN_ACCESS=1 ./run.sh
```

`run.sh` 支持的环境变量：`MODEL`（必填）、`MMPROJ`、`CTX`（默认 65536）、
`PORT`（8080）、`LAN_ACCESS`、`LAN_IP_OVERRIDE`、`DRAFT_MODEL`、`DRAFTS`。

启动后用 OpenAI 兼容 API：`http://127.0.0.1:8080/v1`。

## 版本信息

- 上游 llama.cpp：`85c55223caf0a2ad0d1d88e5a73ab3fe36107867`（2026-08-31）
- 内嵌 WebUI：`b10726`
- S80 内核优化 patch 来源：pcalau12i，https://www.foleosoft.com/software/S80-Optimization

源码快照已包含该 patch 及在 MUSA 4.0.1 下编译所需的适配。
