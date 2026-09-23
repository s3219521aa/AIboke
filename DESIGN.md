# AI 中文双人播客生成系统 — 设计文档

日期：2026-09-22
状态：待评审

---

## 1. 背景与目标

构建一个离线系统：输入一个中文商业故事主题，输出一期完整的中文双人对话播客，包含音频、封面与文稿三件产物。

### 1.1 输入

| 字段 | 说明 |
|---|---|
| `topic` | 商业故事主题，如"星巴克国内运营转移" |
| `speaker_gender1` | 主播 1 性别，`男` / `女` |
| `speaker_gender2` | 主播 2 性别，`男` / `女` |

### 1.2 输出

| 产物 | 规格 |
|---|---|
| `podcast.mp3` | mp3 音频，时长必须落在 5–15 分钟，建议 7–10 分钟 |
| `cover.png` | 1024×1024 播客封面 |
| `script.json` | `{"title": str, "content": [{"speaker": 1\|2, "text": str}, ...]}` |

### 1.3 运行环境约束

- 单卡 NVIDIA A100-40GB
- 4 vCPU / 8 GB 系统内存
- 全程离线，无外网（模型预先拉取后挂载）
- 交付形态为 Docker 镜像，模型以挂载方式提供

---

## 2. 评分结构与优先级

| 维度 | 分值 | 说明 |
|---|---|---|
| **0 分项** | 一旦触发则整案 0 分 | 见 2.1 |
| 文稿内容质量 | 40 | 内容准确性、商业叙事性 |
| 对话与表达质量 | 30 | 连贯性、互动性、口语化、词汇句式多样性 |
| 语音自然度 | 20 | 音色区分度/表现力、自然度 |
| 封面图 | 10 | 视觉效果与主题一致性 |

**关键推论：文稿相关占 70 分（40 + 30），语音占 20 分，封面占 10 分。**

这决定了投入优先级：

1. **最高**：四个 0 分门限的防御（触发即全盘归零）
2. **其次**：文稿质量（Prompt 工程 + 事实性控制 + 叙事结构）
3. **再次**：语音自然度（模型选型已由调研确定，主要是参数调优）
4. **最后**：封面（分值最低，但必须产出且与主题相关）

### 2.1 四个 0 分门限

| 门限 | 原文要求 |
|---|---|
| 语言 | 音频与文稿主要语言必须为中文，否则整案 0 分 |
| 时长 | 必须 ≥5 分钟且 ≤15 分钟，否则整案 0 分 |
| 形式 | 必须为双人对话，否则整案 0 分 |
| 性别 | 必须符合指定性别要求，否则整案 0 分 |

### 2.2 监测指标

| 指标 | 基线 |
|---|---|
| RTF | ≤ 2 |
| 音频首字返回时间 | ≤ 30 s |
| 失败率 | ≤ 10% |

---

## 3. 架构

### 3.1 总览

```
输入 (topic, gender1, gender2)
  │
  ├─ [阶段 1] 文稿生成 ──── Qwen3.8-27B (llama.cpp server)
  │            三幕式生成，逐幕产出
  │                 │
  │                 │ 每幕产完立即投递
  │                 ▼
  ├─ [阶段 2] 语音合成 ──── MOSS-TTSD 8B (llama.cpp 默认 / transformers 备选)
  │            逐幕合成 → 拼接为单条音轨
  │
  ├─ [阶段 3] 封面生成 ──── Z-Image-Turbo
  │
  └─ [门限校验] 四道 0 分门限
                  │
                  ├─ 通过 → 输出三件产物
                  └─ 不通过 → 带反馈重试（最多 2 次）
```

### 3.2 三幕式生成的动机

文稿不一次性生成，而是拆成三幕顺序产出：

| 幕 | 内容 | 目标字数占比 |
|---|---|---|
| 第一幕 | 开场铺垫、抛出悬念、点明为什么值得聊 | 20% |
| 第二幕 | 主体讲述：关键事件、核心人物、重要数据、行业背景 | 55% |
| 第三幕 | 分析总结：影响、启示、收尾 | 25% |

两个收益：

1. **强制叙事结构**。评分第 5 条要求"开场铺垫 → 主体讲述 → 分析总结"的结构，且明确反对"平铺直叙信息罗列"。分幕把结构约束进流程，而不是指望模型自觉。
2. **降低首字延迟**。第一幕一产出即可开始语音合成，与第二、三幕的生成重叠执行。这直接服务"音频首字返回时间 ≤ 30s"的基线。

### 3.3 数据流

```
CaseInput
  → [script_writer] → Transcript(title, turns[])   三幕累加
  → [gates.check_script]                           文稿侧门限
  → [tts] → 每幕音频 wav[]
  → [audio_utils] → podcast.mp3                    ffmpeg 拼接 + 响度归一化
  → [gates.check_audio]                            时长门限
  → [cover] → cover.png
  → Episode(podcast.mp3, cover.png, script.json)
```

---

## 4. 模型选型

| 组件 | 模型 | 精度 | 许可 | 获取来源 |
|---|---|---|---|---|
| 文稿生成 | Qwen3.8-27B | GGUF Q5_K_XL | Apache-2.0 | ModelScope |
| 对话语音 | MOSS-TTSD v1.0 (8B) | GGUF Q6_K | Apache-2.0 | ModelScope / GitCode 镜像 |
| 音频编解码 | MOSS-Audio-Tokenizer | ONNX | Apache-2.0 | ModelScope / hf-mirror |
| 音色设计 | MOSS-VoiceGenerator (1.7B) | bf16 | Apache-2.0 | ModelScope |
| 封面生成 | Z-Image-Turbo (6B) | FP8 | Apache-2.0 | ModelScope（`Tongyi-MAI/Z-Image-Turbo`） |

全部为 Apache-2.0，无商用风险。

### 4.1 为什么选 MOSS-TTSD

它自带的 TTSD-eval 中文评测数据直接可比：

| 模型 | ZH 音色相似度 | ZH 说话人归属准确率 | ZH WER |
|---|---|---|---|
| **MOSS-TTSD** | **0.7949** | **0.9587** | **0.0485** |
| VibeVoice 7B | 0.7590 | 0.9222 | 0.0570 |
| VibeVoice 1.5B | 0.7415 | 0.8798 | 0.0818 |
| Eleven V3（闭源） | 0.6970 | 0.9653 | 0.0363 |
| 豆包 Podcast（闭源） | 0.8034 | 0.9606 | 0.0472 |

开源模型中中文三项指标全面领先。更关键的是它是**原生对话模型**（1–5 人、处理轮流发言与交叠语音、单次最长 60 分钟），而不是"逐句合成再拼接"——后者会损失评分明确要求的"活人感"与"衔接自然"。

### 4.2 为什么选 Z-Image-Turbo

6B / 8 步出图 / 1024×1024 约 2.3 秒（RTX 4090）/ 显存 <16GB（FP8 约 8GB）/ Apache-2.0，且出自阿里通义，中文提示词理解好。

排除 Qwen-Image 的原因：20B 参数在 bf16 下即 40GB，单卡无法容纳，会与 TTS 争抢显存；其优势（中文文字渲染）对封面并非必需——封面反而应避免出现文字，因为文生图模型的文字渲染普遍不可靠。

---

## 5. 显存预算（A100-40GB）

| 组件 | 精度 | 显存 |
|---|---|---|
| Qwen3.8-27B | GGUF Q5_K_XL | ~19 GB |
| MOSS-TTSD 8B | GGUF Q6_K | ~8–9 GB |
| MOSS-Audio-Tokenizer | ONNX | ~2–3 GB |
| Z-Image-Turbo | FP8 | ~8 GB |
| **合计** | | **~37–39 GB** |

设计目标：**四个模型全部常驻，不做换入换出**。这是选择 llama.cpp 路线（而非 vLLM + AWQ）的核心收益——AWQ INT4 权重为 18.7–21GB，反而比 GGUF Q5_K_XL 更大，会把总量推过 40GB 从而被迫串行化。

若实测显存吃紧，降级顺序为：① LLM 降至 Q4_K_M（17GB，腾出 2GB）② 封面阶段串行化（卸载 LLM 后再加载 Z-Image）。

补充：Qwen3.8-27B 的 64 层中 48 层为 Gated DeltaNet，其记忆占用与上下文长度无关，仅 16 层维持常规 KV cache，因此长文稿生成的 KV cache 增量很小。

---

## 6. 时长控制（0 分门限之一）

要求 5–15 分钟，目标定在 **8.5 分钟（510 秒）**，距上下限各有充分余量。

### 6.1 三重保险

**第一重 — 脚本字数反算**

以语速常量为基准反算目标字数：

```
目标字数 = 目标秒数 × 每秒字数
默认每秒字数 = 3.33（即 200 字/分钟）
510 秒 × 3.33 ≈ 1700 字
```

该 200 字/分钟 来源于已移植的 `podcast_styles.json` 实测标定表（2000–3000 字 → 10–15 分钟）。**该常量必须通过 `scripts/calibrate_length.py` 在目标机器上用实际音色实测校准**，因为不同音色与语速设置会显著影响实际产出。

**第二重 — 生成长度硬限制**

MOSS-TTSD 官方换算：**1 秒音频 ≈ 12.5 tokens**。

```
max_new_tokens = 目标秒数 × 12.5 = 510 × 12.5 = 6375
```

**第三重 — 实测校验**

用 ffprobe 读取实际时长，若越界则按偏差调整目标字数后重生成（最多 2 次）。

### 6.2 容差设定

| 判定 | 条件 | 动作 |
|---|---|---|
| 合规 | 300 ≤ 时长 ≤ 900 秒 | 通过 |
| 偏离目标但合规 | 合规但距 510 秒超过 ±90 秒 | 通过，记录警告 |
| 越界 | <300 或 >900 秒 | 重生成 |

---

## 7. 性别与音色方案（0 分门限之一）

评分要求"生成的播客应符合指定性别要求"。性别由**构造保证**——音色按请求的性别选取，而非事后检测。

两条路径，主用第一条：

1. **MOSS-VoiceGenerator 文本造音色（默认）**
   输入自由文本描述（如"一位成熟稳重的男声，音色低沉有磁性，语速偏慢"），直接生成音色，**无需任何参考音频**。彻底绕开参考音频的版权问题，且性别与气质可精确控制。生成的音频作为 MOSS-TTSD 的 `prompt_audio` 参考。

2. **AISHELL-3 参考音频（备选）**
   85 小时 / 218 位说话人，**Apache-2.0 明确允许商用**。按性别各挑若干条作为克隆参考。

音色预设存于 `configs/voice_presets.json`，按 `男`/`女` 索引。两位主播使用不同预设以保证音色区分度（评分项）。

### 7.1 角色与性别的映射（固定约定）

两位主播的角色固定，性别按输入参数指定：

| speaker | 角色 | 性别来源 |
|---|---|---|
| 1 | 提问引导者 — 抛出问题、追问、替听众发问，负责推进节奏 | `speaker_gender1` |
| 2 | 讲述分析者 — 提供事实、数据、行业背景与深度分析 | `speaker_gender2` |

`speaker` 字段恒为 `1` 或 `2`，与 `speaker_gender1` / `speaker_gender2` 一一对应。当两位性别相同时（男男 / 女女），两人的**音色预设必须不同**，否则会违反"音色应有区分度"的评分要求。

### 7.2 性别正确性的验证边界

性别由构造保证（按参数选音色），但**预设文件的性别标注本身可能出错**。因此：

- `check_genders` 只校验请求的性别组合在预设库中存在对应条目（配置层校验），不尝试从音频反推性别
- 预设音色的性别正确性由 `smoke_test.py` 的人工试听环节确认，属一次性验证，不进入自动流程

---

## 8. 文稿质量设计（70 分）

### 8.1 Prompt 要点

系统提示词需覆盖以下约束：

- **角色**：资深中文播客制作人兼撰稿人
- **形式**：双人对话，口语化，具备真实播客的"活人感"
- **互动**：提问、回应、补充、质疑、感叹；自然插话（"对""没错""等等""我打断一下"）
- **结构**：开场铺垫 → 主体讲述 → 分析总结（由三幕式在流程层强制）
- **事实准确性**：仅使用广为人知、可核实的事实、数据与人物；**不确定的具体数字宁可不给**，改用定性表述（"大幅增长""翻了好几倍"）；严禁编造数据、事件、人物
- **表达**：避免书面语与机械播报；避免罗列式讲解（"第一…第二…第三…"）；词汇句式丰富，避免重复口头禅
- **字数**：按当前幕的目标字数生成

### 8.2 事实性策略

离线环境无法联网核查，因此策略是**降低编造概率**而非事后验证：

1. Prompt 层面明确禁止编造，并给出"不确定就不给具体数字"的替代路径
2. 主体幕生成后追加一次轻量自检调用：让模型标出自己不确定的具体断言，再据此改写或删除
3. 偏好广为人知的商业案例；对陌生主题倾向定性描述

### 8.3 输出格式

LLM 直接输出符合赛题 schema 的 JSON。优先使用 llama.cpp 的 JSON schema 约束解码从源头保证合法；若服务端不支持则回退到标记文本解析，并在解析失败时尝试修复。

---

## 9. 模块设计

```
AIboke/
├── README.md
├── DESIGN.md                    本方案的副本
├── requirements.txt
├── Dockerfile
├── .env.example
├── configs/
│   ├── default.yaml             主配置：模型路径、超参、目标时长
│   ├── podcast_styles.json      风格与长度标定（移植自 ai-podcast-workflow，MIT）
│   └── voice_presets.json       性别 → 音色预设
├── scripts/
│   ├── download_models.sh       ModelScope / hf-mirror 拉取
│   ├── build_llamacpp.sh        编译 upstream 与 OpenMOSS fork
│   ├── calibrate_length.py      语速标定
│   ├── smoke_test.py            可跑性验证
│   └── generate.py              CLI 入口
└── src/aiboke/
    ├── config.py                配置加载与校验
    ├── schema.py                CaseInput / Turn / Transcript / Episode
    ├── length.py                字↔秒换算、目标字数分配
    ├── gates.py                 四个 0 分门限校验
    ├── prompts.py               三幕式 Prompt 组装
    ├── llm_client.py            llama.cpp server (OpenAI 兼容) 客户端
    ├── script_writer.py         三幕生成 + 解析 + 带反馈重试
    ├── voices.py                性别 → 音色/参考音频解析
    ├── tts.py                   MOSS-TTSD 适配器（双后端）
    ├── cover.py                 Z-Image-Turbo 封面
    ├── audio_utils.py           ffmpeg 拼接 / 响度归一化 / mp3 编码
    └── pipeline.py              编排
```

### 9.1 模块职责

| 模块 | 职责 | 依赖 |
|---|---|---|
| `schema.py` | 定义全部数据结构，纯数据 | 无 |
| `length.py` | 字↔秒换算、三幕字数分配，纯函数 | 无 |
| `gates.py` | 四道门限的判定，返回结构化结果，纯函数 | `schema` |
| `prompts.py` | 组装三幕 Prompt，纯函数 | `schema` |
| `llm_client.py` | 封装 llama.cpp server 的 HTTP 调用 | 网络 |
| `script_writer.py` | 驱动三幕生成、解析、按门限反馈重试 | `llm_client`, `prompts`, `gates`, `length` |
| `voices.py` | 性别 → 音色预设解析与校验 | `config` |
| `tts.py` | MOSS-TTSD 调用，双后端可切换 | `voices` |
| `cover.py` | 主题 → 封面提示词 → 1024×1024 PNG | `llm_client`(扩写) |
| `audio_utils.py` | ffmpeg 封装：拼接、响度归一化、编码 | ffmpeg |
| `pipeline.py` | 编排三个阶段与门限校验 | 以上全部 |

`schema.py`、`length.py`、`gates.py`、`prompts.py` 为纯逻辑，无需 GPU 即可单元测试。

### 9.2 关键接口

```python
# schema.py
@dataclass(frozen=True)
class CaseInput:
    topic: str
    speaker_gender1: str   # "男" | "女"
    speaker_gender2: str

@dataclass(frozen=True)
class Turn:
    speaker: int           # 1 | 2
    text: str

@dataclass(frozen=True)
class Transcript:
    title: str
    turns: tuple[Turn, ...]

@dataclass(frozen=True)
class Episode:
    audio_path: Path
    cover_path: Path
    script_path: Path

# gates.py
@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    retry_hint: str | None = None   # 不通过时给 LLM 的反馈

def check_chinese(text: str, min_ratio: float = 0.85) -> GateResult
def check_duration(seconds: float, low: float = 300, high: float = 900) -> GateResult
def check_two_speakers(turns: Sequence[Turn], min_share: float = 0.25) -> GateResult
def check_genders(gender1: str, gender2: str, presets: VoicePresets) -> GateResult

# tts.py — 按幕调用，每次返回该幕的单条 wav
class TtsBackend(Protocol):
    def synthesize(self, turns: Sequence[Turn], voices: VoicePair,
                   target_seconds: float) -> Path: ...

class MossTtsLlamaCpp(TtsBackend): ...   # 默认
class MossTtsTransformers(TtsBackend): ...  # 备选
```

`pipeline` 对每一幕调用一次 `synthesize`，得到三个 wav 后再由 `audio_utils` 拼接为最终单条 mp3。`target_seconds` 为该幕的目标时长，按其在总字数中的占比折算，用于推导 `max_new_tokens`。

---

## 10. I/O 层

评测方的接口规范未知，因此按通用约定设计，并保留可替换的适配层。

**主入口：CLI**

```bash
python scripts/generate.py --input case.json --output-dir out/
```

`case.json`：
```json
{"topic": "星巴克国内运营转移", "speaker_gender1": "男", "speaker_gender2": "女"}
```

输出目录产出 `podcast.mp3`、`cover.png`、`script.json`。

**附加：HTTP 封装**

提供一层轻量 FastAPI 包装，接受同样的 JSON、返回产物路径，供需要常驻服务的评测方调用。两种调用方式共用同一 `pipeline`。

---

## 11. 音频后处理

1. 逐幕合成得到的 wav 按顺序拼接
2. 响度归一化至 **-16 LUFS**（播客通用标准），防止音量忽大忽小
3. 转码为 mp3
4. 幕间插入短暂自然停顿（约 300–500 ms），避免生硬切边

评分第 10 条要求"无突然爆音或断裂"，因此拼接处需做淡入淡出处理。

**不混入背景音乐**：赛题仅要求 mp3 音频，且第 10 条明确要求"无噪音杂音"。BGM 会压低语音清晰度、增加语音自然度失分风险，收益为负，故明确排除。

---

## 12. 错误处理

| 失败场景 | 处理 |
|---|---|
| LLM 输出非法 JSON | 约束解码预防 → 解析修复 → 重试一次 |
| 门限不通过 | 带具体反馈重试，最多 2 次（如"时长 4分20秒，需增加约 800 字"） |
| TTS 产出为空/异常 | 切换另一个后端重试 |
| **llama.cpp 静默乱码** | 启动冒烟测试拦截，见下 |
| 封面生成失败 | 退化为纯色/渐变底图 + 主题文字（保证产物存在，不因 10 分项导致整案失败） |

### 12.1 llama.cpp 静默乱码的专项防御

调研发现 llama.cpp 旧版本（< b10450）在 Qwen3.8 的 DeltaNet 层 CUDA 执行路径上有 bug，症状为：**模型正常加载、显存正常、速度正常、无任何报错，但输出全是乱码**。这是最危险的失败模式——会被误判为成功，而文稿占 70 分且可能触发"非中文则 0 分"。

防御措施：

1. 固定最低版本 b10450，且二进制与 `.so` 必须同版本（只换 `llama-server` 不换 `libggml-cuda.so` 仍会复现）
2. `smoke_test.py` 在启动时生成一句已知中文并校验其连贯性，**不通过则拒绝进入批处理**

---

## 13. 测试策略

### 13.1 单元测试（无需 GPU）

- `length.py`：字↔秒换算、三幕字数分配、容差判定边界
- `gates.py`：四道门限的通过/不通过边界（含中文占比 0.85 临界、时长 300/900 秒临界、说话人轮次均衡）
- `schema.py`：序列化/反序列化、schema 校验
- `prompts.py`：三幕 Prompt 组装、字数占位符替换
- JSON 解析与修复路径

### 13.2 冒烟测试（需 GPU）

- llama.cpp 版本与 `.so` 一致性
- Qwen3.8-27B 中文输出正确性（**校验内容，不只看加载成功**）
- MOSS-TTSD 产出 30 秒双人中文音频，两位说话人音色可区分
- Z-Image 产出 1024×1024 PNG
- **人工试听**：确认各音色预设的性别与气质符合 `voice_presets.json` 的标注（见 7.2），属一次性验证

### 13.3 端到端

3 个主题 × 4 种性别组合（男男 / 男女 / 女男 / 女女），校验三件产物齐全且四道门限全部通过。

### 13.4 性能验证

实测 RTF 与首字延迟，对比基线（RTF ≤ 2、首字 ≤ 30s）。

---

## 14. 部署

- Docker 镜像基于 CUDA runtime 基础镜像
- **模型不烘进镜像**，以挂载方式提供（符合赛题"支持挂载模型"）
- `MODELS_ROOT` 环境变量指定挂载点
- 镜像内含：编译好的两份 llama.cpp 二进制、Python 运行时、ffmpeg
- 系统内存仅 8GB，需避免任何模型走 CPU 路径；llama.cpp 以 `-ngl` 全量卸载至 GPU

---

## 15. 已知风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| MOSS-TTSD 的 llama.cpp 原生路径需编译 OpenMOSS fork 并转换 GGUF 格式 | 集成成本最高的一环 | 同时实现 transformers 后备 |
| 该 fork 可能未 rebase 到支持 Qwen3.8 的上游 master | 需编译两份 llama.cpp | 分离构建脚本，互不影响 |
| 无 A100 可供本地实测 | 第 5 节显存数字与吞吐均为推算值 | 交付冒烟测试，在目标机器上落实 |
| 语速常量 200 字/分钟为移植值 | 时长估算可能偏差 | 提供标定脚本；第三重实测校验兜底 |
| MOSS-TTSD 中文 WER 4.85% | 音频与文稿存在约 5% 字词偏差，而评分要求"音频无漏字错字" | 启用 `--text_normalize`；记录为已知偏差 |
| 评测方接口规范未知 | 可能需要改适配层 | I/O 层隔离为可替换适配器 |

---

## 16. 非目标（YAGNI）

明确不做以下内容：

- **背景音乐混音** — 负收益，见第 11 节
- **字幕文件（SRT/VTT）** — 赛题未要求
- **多集/连续剧集** — 赛题单个 case 独立
- **自评分系统** — 赛题由评测方打分，不重复造轮子
- **Web UI** — 无用户交互需求
- **流式音频输出协议** — 三幕式重叠已足够满足首字延迟，完整的流式协议复杂度不划算

---

## 17. 待确认事项

1. 评测方的容器调用方式与超时预算（当前按通用约定设计，见第 10 节）
2. 模型权重的实际挂载路径与磁盘容量。量化后总计约 42–43GB；若 MOSS-TTSD 的 "first-class" GGUF 需从完整 HF 权重转换，还需额外约 17GB 作为转换源，合计约 45–60GB。
3. MOSS-VoiceGenerator 预设音色的实际效果需在目标机器上试听确认
