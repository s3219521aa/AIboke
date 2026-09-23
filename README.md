# AI 中文双人播客生成系统

输入一个中文商业故事主题与两位主播的性别，输出一期完整的中文双人对话播客。

- 文稿：三幕式（开场铺垫 → 主体讲述 → 分析总结），由 Qwen3.8-27B 生成
- 音频：MOSS-TTSD 双人对话合成，5–15 分钟 mp3
- 封面：Z-Image-Turbo 生成的 1024×1024 PNG

全部模型均为 Apache-2.0，全程离线运行，**无需任何付费 API**。

## 快速开始

> ⚠️ **第 5、6 步是强制门槛**：默认（llamacpp）后端的三个 GGUF 与音色参考
> 音频没配齐，后端会直接报错或产出性别错误的音频。参考音频还必须是 24kHz
> 且两人性别可区分（这两项只能人工核对）。见下文《部署前必办》。

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
#    第 4 步（TTS）需要上面第 5、6 步已完成；音色/GGUF 还没就绪时可以只跑
#    其余步骤：python scripts/smoke_test.py --skip-tts
#    （跳过的步骤**不算通过**，脚本结尾会明确说明）

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

自检与标定脚本用的音色就是 `configs/voice_presets.json` 里发布的那四条（按
请求的性别解析），不是脚本自带的假预设——只有显式加 `--no-reference-audio`
才会退回内置假预设，那只是为了在音色尚未生成的过渡期验证其余环境。

输出到 `out/`：`podcast.mp3`、`cover.png`、`script.json`。
（只有三步全部成功才会发布这三个文件；任何一步失败都不会留下半成品。）

`configs/default.yaml` 里 `tts.model_path` 与 `cover.model_path` 写的是
`/models/...` 绝对路径。模型挂载点不同时，三种改法按优先级排列：
`--models-root /挂载点` > 环境变量 `MODELS_ROOT=/挂载点` > 配置文件里的
`models_root`。（相对路径按挂载点解析，绝对路径原样使用。）

## ⚠️ 部署前必办（模型已下载 → 第一次成功运行之间的强制门槛）

默认的 `llamacpp` 后端要打通下面第 1–3 条，缺一条就会启动失败、或产出与请求
无关的音频；第 4 条只在改用备选 `transformers` 后端时才需要，但那条**没有
代码级兜底**，必须人工核对。**前三条不做，冒烟测试必然不通过。**

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
与提示文本 `[S1]…[S2]…` 的分段顺序一致。因此两个参考 wav **都必须是
24kHz**（该二进制的硬要求）。注意采样率不一致**不会**报 `TtsError`：拼接走
ffmpeg 的滤镜图，它会自动插入重采样把两段对齐，另一段的音色被**静默改变**。
`scripts/smoke_test.py` 的第 4 步会用 ffprobe 核对两个参考音频的采样率并
因此失败（给出转换命令），但**请把它当作必查项自行复核**：

```bash
ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate \
    -of default=noprint_wrappers=1:nokey=1 <参考音频.wav>   # 期望 24000
```

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

对应到配置（`configs/default.yaml` 已经按这份清单写好）：

```yaml
tts:
  model_path: /models/moss-tts-gguf/moss_delay_firstclass_f16.gguf   # -m，backbone **文件**
  audio_encoder_model: /models/moss-tts-gguf/moss_tts_audio_encoder_f16.gguf
  audio_decoder_model: /models/moss-tts-gguf/moss_tts_audio_decoder_f16.gguf  # 必需
```

注意 `model_path` 对两个后端的含义不同：`llamacpp` 要 backbone GGUF **文件**
（`-m` 的值），`transformers` 要完整权重**目录**。

**`audio_decoder_model` 是 native 路径的必需参数**：没有它二进制无法把生成的
token 还原成 wav，所以适配器在**起进程之前**就会以 `TtsError` 拦下缺失的配置
（消息里给出两条出路），而不是让你等一整轮合成跑完再收到一个含糊的 CLI 报错。
若你的 fork 确实自带 decoder 或改用别的参数，把这道校验显式关掉：
`tts.require_audio_decoder: false`。

`scripts/download_models.sh` 拉的是通用权重（Qwen GGUF / MOSS-TTSD / Z-Image），
**不足以直接跑 native 路径**（native 要的是上面那三个转换出来的 GGUF）。该脚本
也不再拉 ~14GB 的 `MOSS-Audio-Tokenizer-ONNX`——本仓库没有任何代码路径消费它
（native 用转换出的 GGUF，transformers 后备用官方仓库自己的编解码器）。若时间
不允许打通 native 路径，就按上面第 2 条的「换后端」走 transformers。

### 4. 备选（transformers）后端：官方 `inference.py` 的输入契约（部署前必查）

这条必须**先核对再上线**。适配器与官方脚本之间是靠一份 JSONL 与一组命令行
标志约定的，而这两处约定**没有代码级兜底**：键名对不上时脚本会安静地忽略掉
音色条件，产出两个与请求性别无关的音色——正是「性别必须符合要求」这条 0 分
门限最怕的静默退化。

官方 MOSS-TTSD（v1.0）README 的「JSONL Input Format」一节列出的键是：

| 键 | 含义 |
| --- | --- |
| `text` | `[S1]…[S2]…` 的对话稿 |
| `prompt_audio_speakerN` | 第 N 位说话人的参考音频路径 |
| `prompt_text_speakerN` | 第 N 位说话人的参考文本 |
| `base_path`（可选） | 参考音频的公共前缀目录 |

N 支持 1–5。**注意官方列出的键里没有 `voice_description_speakerN`**——
`format_input_jsonl` 会额外写上这两个键，但按上述契约它们很可能被忽略。
也就是说：**transformers 后端下，音色性别的唯一载体是
`prompt_audio_speakerN` 指向的参考音频**；`reference_audio` 为空时该字段是空
串，模型就只会自己编一个音色（连"按描述生成"都没有代码保证）。

命令行侧，适配器发送的是 `--model_path` / `--input_jsonl` / `--save_dir` /
`--mode voice_clone_and_continuation` / `--max_new_tokens` / `--temperature` /
`--top_p` / `--top_k` / `--repetition_penalty` / `--text_normalize` /
`--sample_rate_normalize`。官方文档里的 `--codec_model_path`（音频编解码器，
**独立仓库**）本适配器**不默认发送**：离线环境里必须指到本地，否则运行期会去
联网拉取。要用它就把 `tts.codec_model_path` 指向本地目录（见
`configs/default.yaml` 的注释），未知的标志名可用 `tts.codec_model_flag` 覆盖。

部署前的强制动作（**人工**，无法在离线环境自动化）：

1. 以你实际拉取的官方仓库**那个 revision** 的 `inference.py` 为准，逐项核对
   上面两份清单（README 与源码同看）；键名/标志名不一致时改
   `format_input_jsonl` 与 `TtsConfig`，不要靠猜；
2. 跑一次 `inference.py --help`，确认适配器发送的每个标志都存在（未知标志
   会让 CLI 直接失败）；
3. **上线前的批处理门槛**：用**两个性别相反的音色预设**（如 `m_calm` + `f_clear`）
   各生成一段，然后**人工试听**确认两段音色确有区分度、且性别与
   `speaker_gender1/2` 一致。听不出来或性别不对，就说明键名没被脚本读取——
   此时必须停下修契约，**不要**带着这个问题进入批处理。

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
| 必须中文 | Prompt 硬约束 + CJK 占比检测（整篇 0.85；逐幕只拦「基本不是中文」，阈值 0.5） |
| 5–15 分钟 | 三重保险：字数反算 + `max_new_tokens` 硬限 + ffprobe 实测校验（越界按偏差缩放字数目标重生成，最多 `max_retries` 次） |
| 双人对话 | `[S1]`/`[S2]` 结构强制 + 轮次均衡校验（各占 ≥25%） |
| 性别匹配 | 按性别选音色（构造保证）+ 配置层校验；音色性别由参考音频保证，冒烟测试人工试听确认 |

时长目标设为 8.5 分钟，距上下限各有充分余量。

逐幕的语言判据（0.5）与整篇的（0.85）是**两个不同的阈值**：0.85 是规范对整篇
交付物的要求，逐幕套用会误杀数字密集的主体幕（CJK 占比把阿拉伯数字算作
非中文）。0.85 仍用于给模型的重试提示——低一档时让它改写得更像中文，但不因此
判失败。

## 耗时与可观测性（RTF / 首字延迟）

**先说清楚架构能做到什么**：本系统逐幕合成，但 mp3 只在三幕全部完成后才发布，
所以**交付音频的「首字时间」等于全案耗时**——它不可能 ≤ 30 秒，任何相反的说法
都不成立。三幕交织真正买到的收益是「首幕音频就绪」提前（第一幕生成完即开始
合成，与后两幕的生成重叠）。

两者都由 `src/aiboke/pipeline.py` 的阶段耗时日志给出实测值
（`logging.getLogger("aiboke.pipeline")`，INFO 级）：

| 日志 | 含义 |
| --- | --- |
| `阶段耗时：第 N 幕文稿 …（目标约 X 字，实际 Y 字）` | 该幕的生成耗时与字数 |
| `阶段耗时：第 N 幕语音 …（N 轮，S 秒目标）` | 该幕的合成耗时 |
| `阶段耗时：拼接 … 响度归一化 … 转码 mp3 … 时长探测 …` | 音频后处理四步各自耗时 |
| `阶段耗时：封面 …` | 封面耗时 |
| `首幕音频就绪：T 秒（自本轮开始…）` | 首幕音频文件就绪的时刻（**不是**交付 mp3 的首字） |
| `全案耗时 T 秒：音频 D 秒，端到端 RTF R；交付物首字延迟 T 秒（…）` | 设计要求实测的两项指标，按实测值报 |

时长越界的重生成会额外记录 `按实测时长重生成：字数目标系数 a -> b`，便于事后
判断重生成是否真的收敛。**指标以这些日志为准**：设计文档里的 RTF ≤ 2 与
首字 ≤ 30s 是赛题基线，本仓库没有 GPU，无法在离线环境给出实测值。

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

约 **54GB**（未设 `HF_TOKEN`）到 **122GB**（设了 `HF_TOKEN`，MOSS-TTS-GGUF
整仓），挂载盘建议预留 **150GB**（还要留转换 GGUF 的临时空间）。设计文档写的
42–43GB 是按「量化后权重」算的，与实际仓库体积相差约 3 倍。

| 组件 | 磁盘 | 备注 |
|---|---|---|
| Qwen3.8-27B | ≈ 21 GB | 只取 `Q5_K_XL` 一个量化文件；整仓有 25 个量化版本（>400GB），故下载脚本带 `--include` 过滤 |
| MOSS-TTSD | ≈ 17 GB / 68 GB | 未设 `HF_TOKEN`：完整权重（transformers 后备）；设了：gated GGUF 整仓 |
| MOSS-Audio-Tokenizer | ≈ 14 GB（**不拉**） | ONNX **fp32** 权重；**本仓库没有任何代码路径消费它**，下载脚本已把它注释掉（仅 hybrid/ONNX 编解码方案需要）。备选的 transformers 后端要的是 HF 格式的编解码器，见下载脚本第 4 条的提示 |
| Z-Image-Turbo | ≈ 33 GB | 仓库**只提供 bf16/fp32 权重，没有预量化 FP8** |

### 显存（A100-40GB）

| 组件 | 精度 | 显存 | 来源 |
|---|---|---|---|
| Qwen3.8-27B | GGUF Q5_K_XL | ~19 GB | 设计估算 |
| MOSS-TTSD 8B | GGUF Q6_K | ~8–9 GB | 设计估算 |
| MOSS-Audio-Tokenizer | ONNX / GGUF | ~2–3 GB | 设计估算；native 路径实际加载的是转出来的 audio encoder/decoder GGUF，不是上面那份不下载的 ONNX |
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

**本项目代码尚未指定许可证**——未声明许可证即默认保留所有权利。若要开源分发，请先补充
`LICENSE` 文件；依赖项全部为 Apache-2.0，选 Apache-2.0 或 MIT 均无兼容性问题。

依赖的模型与数据全部为 Apache-2.0：

- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — 文稿生成
- [MOSS-TTSD](https://github.com/OpenMOSS/MOSS-TTSD) — 双人对话语音合成
- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) — 封面生成
- [AISHELL-3](https://www.openslr.org/93/) — 备选参考音色（Apache-2.0，允许商用）

`configs/podcast_styles.json` 中的长度标定表移植自
[uiuing/ai-podcast-workflow](https://github.com/uiuing/ai-podcast-workflow)（MIT License）。
