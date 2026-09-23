# AI 中文双人播客生成系统

输入一个中文商业故事主题与两位主播的性别，输出一期完整的中文双人对话播客。

- 文稿：三幕式（开场铺垫 → 主体讲述 → 分析总结），由 Qwen3.8-27B 生成
- 音频：MOSS-TTSD 双人对话合成，5–15 分钟 mp3
- 封面：Z-Image-Turbo 生成的 1024×1024 PNG

全部模型均为 Apache-2.0，全程离线运行，**无需任何付费 API**。

## 快速开始

> ⚠️ **第 5、6 步是强制门槛**：默认（llamacpp）后端的三个 GGUF 与音色参考
> 音频没配齐，后端会直接报错或产出性别错误的音频。见下文《部署前必办》。

```bash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 跑单元测试（无需 GPU）
python -m pytest tests/ -v

# 3. 拉取模型（磁盘要预留约 150GB，见下文《资源需求》）
bash scripts/download_models.sh /models

# 4. 编译 llama.cpp（针对 A100 的 sm_80）
bash scripts/build_llamacpp.sh /opt/llama.cpp

# 5. 【强制】配置 TTS 的三个 GGUF 与二进制
#    默认配置已指向 /opt/llama.cpp/openmoss/build-cuda/bin/llama-moss-tts
#    （与 build_llamacpp.sh 的产物一致），但仍需：
#      a) ls /opt/llama.cpp/openmoss/build-cuda/bin/  确认二进制存在
#      b) 按 fork 文档转换出 backbone/encoder/decoder 三个 GGUF，
#         把路径填进 tts.model_path / tts.audio_encoder_model /
#         tts.audio_decoder_model（见下文《部署前必办》第 1、3 节）
#      c) 用 llama-moss-tts --help 核对标志名

# 6. 【强制】回填音色参考音频（默认 llamacpp 后端必需）
#    见下文《部署前必办》第 2 节

# 7. 启动 LLM 服务
/opt/llama.cpp/upstream/build-cuda/bin/llama-server \
    -m /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q5_K_XL.gguf \
    --port 8080 -ngl 99 --host 127.0.0.1

# 8. 环境自检（务必先跑这一步）
python scripts/smoke_test.py --config configs/default.yaml

# 9. 标定语速，把结果填回 configs/default.yaml
python scripts/calibrate_length.py --config configs/default.yaml

# 10. 生成播客
python scripts/generate.py \
    --input case.json --output-dir out/
```

`case.json`：

```json
{"topic": "星巴克国内运营转移", "speaker_gender1": "男", "speaker_gender2": "女"}
```

输出到 `out/`：`podcast.mp3`、`cover.png`、`script.json`。
（只有三步全部成功才会发布这三个文件；任何一步失败都不会留下半成品。）

`configs/default.yaml` 里 `tts.model_path` 与 `cover.model_path` 写的是
`/models/...` 绝对路径。模型挂载点不同时，三种改法按优先级排列：
`--models-root /挂载点` > 环境变量 `MODELS_ROOT=/挂载点` > 配置文件里的
`models_root`。（相对路径按挂载点解析，绝对路径原样使用。）

## ⚠️ 部署前必办（模型已下载 → 第一次成功运行之间的强制门槛）

默认的 `llamacpp` 后端要打通三件事，缺一件就会启动失败、或产出与请求无关的
音频。**这三件事不做，冒烟测试必然不通过。**

### 1. 二进制路径与参数

编译产物是 `/opt/llama.cpp/openmoss/build-cuda/bin/llama-moss-tts`
（OpenMOSS fork 的 `llama-moss-tts` 目标），`configs/default.yaml` 已指向它
（该路径由 `scripts/build_llamacpp.sh` 的 `${PREFIX}/openmoss/build-cuda/bin`
决定，`PREFIX` 默认 `/opt/llama.cpp`）。若你的安装前缀不同，改 `tts.binary`。

适配器发出的参数已按 fork 的**首方文档与源码**（`docs/moss-tts-firstclass-e2e_zh.md`、
`tools/tts/run-moss-tts-delay.cpp`）对齐：

```
-m <backbone.gguf> [--audio-encoder-model <enc.gguf>] --audio-decoder-model <dec.gguf>
  --text "[S1]…[S2]…" --reference-audio <合并后的参考音频.wav> --wav-out <输出.wav>
  --max-new-tokens <N>
```

不发送二进制没有的标志（`--temperature`/`--top-p`/`--top-k`/`--repetition-penalty`/
`--reference-text`/`--text-normalize` 等属官方 `inference.py`）；`-ngl` 也不必发，
该二进制默认 `-1`（全部层上 GPU）。**上机仍要跑一次
`/opt/llama.cpp/openmoss/build-cuda/bin/llama-moss-tts --help` 核对**：标志名若与
本系统不一致，全部可在 `configs/default.yaml` 里改（`model_flag`、`output_flag`、
`audio_encoder_flag`、`audio_decoder_flag`、`reference_audio_flag`、
`reference_text_flag`），不必动代码。参数对不上时进程会以非零码退出，
`smoke_test.py` 第 4 步会把它连同 stderr 一起报出来。

### 2. 音色参考音频（不改则必然失败）

脚本里的 `[S1]`/`[S2]` 只标明轮次归属、**不含性别**，所以默认的 `llamacpp`
后端只能靠**参考音频**控制音色性别——缺了它，合成出的两个音色与请求的
`speaker_gender1/2` 无关，会直接踩中「性别必须符合要求」这条 0 分门限。
因此 `src/aiboke/tts.py` 的默认后端在缺 `reference_audio` 时**主动报错**，
而不是静默合成一段性别未知的音频。

`llama-moss-tts` 只接受**一个** `--reference-audio`，所以适配器会把两位主播的
参考音频按 `[S1]→[S2]` 顺序**自动拼接**成一个临时文件（用 ffmpeg，随用随删），
与提示文本 `[S1]…[S2]…` 的分段顺序一致。因此两个参考 wav 应当是**同一采样率**
（该二进制要求 24kHz），否则拼接会失败并报 `TtsError`。

`configs/voice_presets.json` 里发布出去的四个预设**都没有填**
`reference_audio`，必须二选一：

1. **生成音色后回填（推荐）** —— 用 MOSS-VoiceGenerator 按预设的
   `description` 文本生成四个音色 wav，把路径写进每个预设的
   `reference_audio` 字段：

   ```bash
   # 权重：ModelScope 上的 openmoss/MOSS-VoiceGenerator
   modelscope download --model openmoss/MOSS-VoiceGenerator --local_dir /models/MOSS-VoiceGenerator
   ```

   ```json
   {
     "id": "m_calm",
     "gender": "男",
     "description": "一位成熟稳重的男声，音色低沉有磁性，语速从容偏慢，语调沉稳，适合讲述深度商业分析",
     "reference_audio": "/models/voices/m_calm.wav"
   }
   ```

2. **换后端** —— 把 `configs/default.yaml` 的 `tts.backend` 改成
   `transformers`。该后端把音色 `description` 文本直接传给模型，不需要参考
   音频；代价是权重约 19GB，与 LLM 无法同时常驻（显存不够）。它需要官方
   仓库的 `inference.py`：把 MOSS-TTSD 官方仓库拉到本地，并把
   `tts.inference_script` 指到该文件的绝对路径（默认值 `inference.py` 按
   cwd 解析，容器里通常不成立）。

生成后务必**人工试听**：确认音色区分度、性别与预设标注一致（见设计文档 7.2）。
`python scripts/smoke_test.py` 的第 4 步也会因此报错或通过。

### 3. 模型文件的形态（native 路径需要三个 GGUF）

fork 的原生路径要**三个** GGUF 文件：first-class backbone（必须由完整权重
转换，**不是** `MOSS-TTS-GGUF` 仓库里的通用量化文件）、audio encoder、
audio decoder——后两个由 fork 自带的
`convert_moss_audio_tokenizer_split_to_gguf.py` 从完整权重转换。转换命令与
参数以 fork 仓库的 `docs/moss-tts-firstclass-e2e_zh.md` 为准
（`scripts/build_llamacpp.sh` 结束时也会提示）。

对应到配置：

```yaml
tts:
  model_path: /models/moss-tts-gguf/moss_delay_firstclass_f16.gguf   # -m，backbone **文件**
  audio_encoder_model: /models/moss-tts-gguf/moss_tts_audio_encoder_f16.gguf
  audio_decoder_model: /models/moss-tts-gguf/moss_tts_audio_decoder_f16.gguf
```

注意 `model_path` 对两个后端的含义不同：`llamacpp` 要 backbone GGUF **文件**
（`-m` 的值），`transformers` 要完整权重**目录**。

`scripts/download_models.sh` 拉的是通用 GGUF 与 ONNX 编解码器：够
transformers 后备与 hybrid 路径用，但**不足以直接跑 native 路径**。若时间
不允许打通 native 路径，就按上面第 2 条的「换后端」走 transformers。

## 两种调用方式

```bash
# CLI
python scripts/generate.py --topic "星巴克国内运营转移" --g1 男 --g2 女 --output-dir out/

# HTTP 常驻服务
#   注意：create_app 需要一个配置路径参数，所以用一行 python 启动；
#   `uvicorn aiboke.server:create_app --factory` 会因缺参数而报 TypeError。
python -c "import uvicorn; from pathlib import Path; from aiboke.server import create_app; uvicorn.run(create_app(Path('configs/default.yaml')), host='0.0.0.0', port=8000)"
#   GET  /health
#   POST /generate  {"topic": "...", "speaker_gender1": "男", "speaker_gender2": "女"}
```

## 四个 0 分门限

赛题规定命中任一条即整案 0 分，因此这四项是系统的最高优先级：

| 门限 | 防御 |
|---|---|
| 必须中文 | Prompt 硬约束 + CJK 占比检测（阈值 0.85） |
| 5–15 分钟 | 三重保险：字数反算 + `max_new_tokens` 硬限 + ffprobe 实测校验 |
| 双人对话 | `[S1]`/`[S2]` 结构强制 + 轮次均衡校验（各占 ≥25%） |
| 性别匹配 | 按性别选音色（构造保证）+ 配置层校验 |

时长目标设为 8.5 分钟，距上下限各有充分余量。

## ⚠️ llama.cpp 静默乱码陷阱

llama.cpp 低于 **b10450** 时，Qwen3.8 的 DeltaNet 层 CUDA 路径存在 bug，
症状极具欺骗性：

> 模型正常加载、显存占用正常、推理速度正常、**无任何报错**，但输出全是乱码。

由于文稿占 70 分且可能触发「非中文则 0 分」，这个 bug 必须主动拦截。
`scripts/smoke_test.py` 会生成一句中文并**校验其内容**（而非只看调用是否成功），
不通过则拒绝进入批处理。

另一个坑：只替换 `llama-server` 二进制不够——旧的 `libggml-cuda.so` 仍会被
`LD_LIBRARY_PATH` 加载。必须同时更新共享库，并用 `ldd` 验证。

## 资源需求（实测与设计估算的差异）

**结论先说**：设计文档「四个模型全部常驻、合计约 37–39GB」的预算**装不进
40GB**——按实测数字合计约 43GB。这一点**必须在目标机上用 `nvidia-smi` 实测
确认**，本仓库没有 GPU，无法证明它装得下。磁盘也远大于设计估算。

### 磁盘（2026-09 实测的仓库体积）

约 **75GB**（未设 `HF_TOKEN`）到 **135GB**（设了 `HF_TOKEN`，MOSS-TTS-GGUF
整仓），挂载盘建议预留 **150GB**。设计文档写的 42–43GB 是按「量化后权重」
算的，与实际仓库体积相差约 3 倍。

| 组件 | 磁盘 | 备注 |
|---|---|---|
| Qwen3.8-27B | ≈ 21 GB | 只取 `Q5_K_XL` 一个量化文件；整仓有 25 个量化版本（>400GB），故下载脚本带 `--include` 过滤 |
| MOSS-TTSD | ≈ 17 GB / 68 GB | 未设 `HF_TOKEN`：完整权重（transformers 后备）；设了：gated GGUF 整仓 |
| MOSS-Audio-Tokenizer | ≈ 14 GB | ONNX **fp32** 权重（设计文档写的 2–3GB 是显存估算，不是磁盘） |
| Z-Image-Turbo | ≈ 33 GB | 仓库**只提供 bf16/fp32 权重，没有预量化 FP8** |

### 显存（A100-40GB）

| 组件 | 精度 | 显存 | 来源 |
|---|---|---|---|
| Qwen3.8-27B | GGUF Q5_K_XL | ~19 GB | 设计估算 |
| MOSS-TTSD 8B | GGUF Q6_K | ~8–9 GB | 设计估算 |
| MOSS-Audio-Tokenizer | ONNX | ~2–3 GB | 设计估算 |
| Z-Image-Turbo | **bf16 加载** | **~12 GB** | 实测权重分片（24.6GB fp32 ≈ 6.15B 参数）÷ 2，`cover_runner.py` 以 `torch_dtype=bfloat16` 加载；设计文档的 8GB 是 FP8 假设，而仓库没有 FP8 权重 |
| **合计** | | **~41–43 GB** | 超出 40GB，**须实测确认** |

若实测装不下，降级顺序（按代价从低到高）：

1. **LLM 降到 Q4_K_M**（约 17GB，省 2GB）——该量化文件需另行下载
   （`modelscope download --model unsloth/Qwen3.8-27B-GGUF --include '*Q4_K_M*'`），
   并把 `llama-server -m` 指向它；
2. **封面阶段串行化**：LLM 与 Z-Image 不同时常驻，封面在文稿与语音完成后再加载
   （省约 12GB，代价是每集多一次模型加载）；
3. `python scripts/generate.py --fallback-cover`（纯色兜底图，放弃封面那 10 分）。

**上机必做**：跑完一次完整生成，用 `nvidia-smi` 记录四个模型同时驻留时的实际
占用；上面的合计是推算值，不是测量值。

## 容器运行

模型不烘进镜像，运行时挂载：

```bash
docker build -t aiboke .
docker run --gpus all -it --rm \
    -v /host/models:/models \
    -v /host/out:/out \
    -e MODELS_ROOT=/models \
    aiboke bash

# 容器内（llama-server 已在 PATH 上，由 Dockerfile 的 ENV 提供）：
llama-server -m /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q5_K_XL.gguf \
    --port 8080 -ngl 99 --host 127.0.0.1 &
python scripts/smoke_test.py --config configs/default.yaml
python scripts/generate.py --input /case.json --output-dir /out
```

镜像里的两份 llama.cpp 分别在 `/opt/llama.cpp/upstream`（文稿）与
`/opt/llama.cpp/openmoss`（语音，`moss-tts-firstclass` 分支），两个
`build-cuda/bin` 都已加入 `PATH` 与 `LD_LIBRARY_PATH`。

## 目录结构

```
src/aiboke/     核心库（schema / length / gates / prompts 为纯逻辑，可无 GPU 单测）
configs/        配置与音色预设
scripts/        下载、编译、标定、自检、CLI
tests/          单元测试
docs/           设计文档与实现计划
DESIGN.md       设计文档副本（便于随镜像源码一并阅读）
```

## 许可与致谢

本项目代码：见仓库 LICENSE。

依赖的模型与数据全部为 Apache-2.0：

- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — 文稿生成
- [MOSS-TTSD](https://github.com/OpenMOSS/MOSS-TTSD) — 双人对话语音合成
- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) — 封面生成
- [AISHELL-3](https://www.openslr.org/93/) — 备选参考音色（Apache-2.0，允许商用）

`configs/podcast_styles.json` 中的长度标定表移植自
[uiuing/ai-podcast-workflow](https://github.com/uiuing/ai-podcast-workflow)（MIT License）。
