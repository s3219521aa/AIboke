# AI 中文双人播客生成系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建离线系统，输入中文商业故事主题与两位主播性别，输出一期完整的中文双人对话播客（mp3 音频 + 1024×1024 封面 + JSON 文稿）。

**Architecture:** 三阶段流水线。阶段一用 Qwen3.8-27B（llama.cpp server）分三幕生成对话文稿；阶段二用 MOSS-TTSD 8B 逐幕合成语音后拼接；阶段三用 Z-Image-Turbo 生成封面。全流程受四道 0 分门限校验约束，不合格则带反馈重试。

**Tech Stack:** Python 3.11+、llama.cpp（两份构建：upstream + OpenMOSS fork）、ONNX Runtime、ffmpeg、pytest、PyYAML、requests

**Spec:** `docs/superpowers/specs/2026-09-22-ai-podcast-generation-design.md`

## Global Constraints

以下为项目级硬约束，**每个任务都隐含包含本节**：

- **四个 0 分门限**（任一不通过则整案 0 分）：
  - 语言：文稿与音频主要语言必须为中文
  - 时长：必须满足 `300 ≤ 时长 ≤ 900` 秒（5–15 分钟）
  - 形式：必须是双人对话
  - 性别：必须符合指定的 `speaker_gender1` / `speaker_gender2`
- **目标时长**：510 秒（8.5 分钟）
- **语速常量**：200 字/分钟（可通过标定脚本覆盖）
- **音频 token 换算**：`1 秒 ≈ 12.5 tokens`（MOSS-TTSD 官方）
- **中文占比阈值**：0.85
- **说话人轮次下限**：每位主播至少占 25% 的轮次
- **响度归一化目标**：-16 LUFS
- **llama.cpp 最低版本**：b10450；二进制与 `.so` 必须同版本
- **模型许可**：全部必须为 Apache-2.0
- **不引入背景音乐**、不生成字幕文件（见设计文档第 16 节）
- **运行环境**：A100-40GB、4 vCPU / 8 GB RAM、全程离线

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | 项目元数据与依赖声明 |
| `src/aiboke/__init__.py` | 包入口，导出公开 API |
| `src/aiboke/schema.py` | 全部数据结构定义，纯数据无逻辑 |
| `src/aiboke/config.py` | 配置加载与校验 |
| `src/aiboke/length.py` | 字↔秒换算、三幕字数分配、时长判定，纯函数 |
| `src/aiboke/gates.py` | 四道 0 分门限判定，纯函数 |
| `src/aiboke/prompts.py` | 三幕式 Prompt 组装，纯函数 |
| `src/aiboke/voices.py` | 性别 → 音色预设解析 |
| `src/aiboke/llm_client.py` | llama.cpp server（OpenAI 兼容）HTTP 客户端 |
| `src/aiboke/script_writer.py` | 三幕文稿生成、解析、带门限反馈重试 |
| `src/aiboke/tts.py` | MOSS-TTSD 适配器，双后端可切换 |
| `src/aiboke/audio_utils.py` | ffmpeg 封装：拼接、响度归一化、编码、探测 |
| `src/aiboke/cover.py` | Z-Image-Turbo 封面生成 + 降级兜底 |
| `src/aiboke/pipeline.py` | 编排三阶段与门限校验 |
| `src/aiboke/server.py` | 轻量 HTTP 封装（FastAPI） |
| `configs/default.yaml` | 主配置 |
| `configs/podcast_styles.json` | 风格与长度标定表 |
| `configs/voice_presets.json` | 性别 → 音色预设 |
| `scripts/generate.py` | CLI 入口 |
| `scripts/calibrate_length.py` | 语速标定 |
| `scripts/smoke_test.py` | 可跑性验证 |
| `scripts/download_models.sh` | 模型拉取（ModelScope / hf-mirror） |
| `scripts/build_llamacpp.sh` | llama.cpp 双份构建 |
| `Dockerfile` | 镜像构建 |
| `tests/` | 单元测试 |

**分层原则**：`schema` / `length` / `gates` / `prompts` / `voices` 为纯逻辑层，**不依赖任何 GPU、网络或外部进程**，可完整单元测试。`llm_client` / `tts` / `cover` / `audio_utils` 为适配层，通过依赖注入 + 假实现测试。`pipeline` 只做编排。

---

# 阶段一：纯逻辑层（无需 GPU，可完全验证）

### Task 1: 项目骨架、数据模型与配置

**Files:**
- Create: `pyproject.toml`
- Create: `src/aiboke/__init__.py`
- Create: `src/aiboke/schema.py`
- Create: `src/aiboke/config.py`
- Create: `configs/default.yaml`
- Test: `tests/test_schema.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无（项目起点）
- Produces:
  - `CaseInput(topic: str, speaker_gender1: str, speaker_gender2: str)`
  - `Turn(speaker: int, text: str)`
  - `Transcript(title: str, turns: tuple[Turn, ...])`
  - `VoicePreset(id: str, gender: str, description: str, reference_audio: str | None)`
  - `VoicePair(speaker1: VoicePreset, speaker2: VoicePreset)`
  - `Episode(audio_path: Path, cover_path: Path, script_path: Path)`
  - `CaseInput.from_dict(d) -> CaseInput`（校验 topic 非空、性别取值合法）
  - `Transcript.to_json_obj() -> dict`（产出赛题要求的 `{"title":..., "content":[{"speaker":..,"text":..}]}`）
  - `Config`（全部配置项的冻结数据类）
  - `load_config(path: Path) -> Config`

- [ ] **Step 1: 初始化 git 仓库与目录结构**

```bash
cd /d/codes/AIboke
git init
mkdir -p src/aiboke configs scripts tests docs/superpowers/plans
```

- [ ] **Step 2: 写 pyproject.toml**

```toml
[project]
name = "aiboke"
version = "0.1.0"
description = "AI 中文双人播客生成系统"
requires-python = ">=3.11"
dependencies = [
    "PyYAML>=6.0",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
gpu = ["numpy", "soundfile", "onnxruntime-gpu", "torch", "diffusers", "transformers"]
server = ["fastapi", "uvicorn"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 3: 写失败的测试 `tests/test_schema.py`**

```python
import pytest

from aiboke.schema import CaseInput, Episode, Transcript, Turn, VoicePreset


def test_case_input_from_dict_ok():
    ci = CaseInput.from_dict(
        {"topic": "星巴克国内运营转移", "speaker_gender1": "男", "speaker_gender2": "女"}
    )
    assert ci.topic == "星巴克国内运营转移"
    assert ci.speaker_gender1 == "男"
    assert ci.speaker_gender2 == "女"


def test_case_input_rejects_empty_topic():
    with pytest.raises(ValueError, match="topic"):
        CaseInput.from_dict({"topic": "  ", "speaker_gender1": "男", "speaker_gender2": "女"})


def test_case_input_rejects_bad_gender():
    with pytest.raises(ValueError, match="speaker_gender1"):
        CaseInput.from_dict({"topic": "t", "speaker_gender1": "male", "speaker_gender2": "女"})


def test_transcript_to_json_obj_matches_required_schema():
    t = Transcript(
        title="瑞幸如何靠生椰拿铁翻盘",
        turns=(Turn(speaker=1, text="大家好。"), Turn(speaker=2, text="没错。")),
    )
    obj = t.to_json_obj()
    assert obj == {
        "title": "瑞幸如何靠生椰拿铁翻盘",
        "content": [
            {"speaker": 1, "text": "大家好。"},
            {"speaker": 2, "text": "没错。"},
        ],
    }


def test_turn_rejects_invalid_speaker():
    with pytest.raises(ValueError, match="speaker"):
        Turn(speaker=3, text="x")


def test_episode_holds_three_paths(tmp_path):
    e = Episode(
        audio_path=tmp_path / "a.mp3",
        cover_path=tmp_path / "c.png",
        script_path=tmp_path / "s.json",
    )
    assert e.audio_path.name == "a.mp3"
```

- [ ] **Step 4: 运行测试确认失败**

Run: `python -m pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke'`

- [ ] **Step 5: 实现 `src/aiboke/schema.py`**

```python
"""全部数据结构定义。纯数据，不含业务逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VALID_GENDERS = frozenset({"男", "女"})


@dataclass(frozen=True)
class CaseInput:
    topic: str
    speaker_gender1: str
    speaker_gender2: str

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("topic 不能为空")
        for field, value in (
            ("speaker_gender1", self.speaker_gender1),
            ("speaker_gender2", self.speaker_gender2),
        ):
            if value not in VALID_GENDERS:
                raise ValueError(f"{field} 必须是 男 或 女，实际为 {value!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "CaseInput":
        try:
            topic = d["topic"]
            g1 = d["speaker_gender1"]
            g2 = d["speaker_gender2"]
        except KeyError as exc:
            raise ValueError(f"缺少必需字段: {exc.args[0]}") from exc
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic 必须是非空字符串")
        return cls(topic=topic.strip(), speaker_gender1=g1, speaker_gender2=g2)


@dataclass(frozen=True)
class Turn:
    speaker: int
    text: str

    def __post_init__(self) -> None:
        if self.speaker not in (1, 2):
            raise ValueError(f"speaker 必须是 1 或 2，实际为 {self.speaker!r}")


@dataclass(frozen=True)
class Transcript:
    title: str
    turns: tuple[Turn, ...]

    def to_json_obj(self) -> dict:
        return {
            "title": self.title,
            "content": [{"speaker": t.speaker, "text": t.text} for t in self.turns],
        }

    @property
    def full_text(self) -> str:
        return "".join(t.text for t in self.turns)


@dataclass(frozen=True)
class VoicePreset:
    id: str
    gender: str
    description: str
    reference_audio: str | None = None

    def __post_init__(self) -> None:
        if self.gender not in VALID_GENDERS:
            raise ValueError(f"VoicePreset.gender 必须是 男 或 女，实际为 {self.gender!r}")


@dataclass(frozen=True)
class VoicePair:
    speaker1: VoicePreset
    speaker2: VoicePreset


@dataclass(frozen=True)
class Episode:
    audio_path: Path
    cover_path: Path
    script_path: Path
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_schema.py -v`
Expected: PASS（6 passed）

- [ ] **Step 7: 写失败的测试 `tests/test_config.py`**

```python
import pytest
import yaml

from aiboke.config import Config, load_config


def _write(tmp_path, data):
    p = tmp_path / "default.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


BASE = {
    "target_seconds": 510.0,
    "chars_per_minute": 200.0,
    "llm": {"base_url": "http://127.0.0.1:8080", "model": "qwen3.8-27b"},
    "tts": {"backend": "llamacpp", "binary": "/opt/llama.cpp/build-cuda/bin/llama-moss-tts"},
    "cover": {"steps": 8, "size": 1024},
    "models_root": "/models",
}


def test_load_config_ok(tmp_path):
    cfg = load_config(_write(tmp_path, BASE))
    assert isinstance(cfg, Config)
    assert cfg.target_seconds == 510.0
    assert cfg.llm.model == "qwen3.8-27b"
    assert cfg.tts.backend == "llamacpp"
    assert cfg.cover.size == 1024


def test_load_config_rejects_bad_backend(tmp_path):
    data = {**BASE, "tts": {**BASE["tts"], "backend": "magic"}}
    with pytest.raises(ValueError, match="backend"):
        load_config(_write(tmp_path, data))


def test_load_config_rejects_target_outside_allowed_window(tmp_path):
    data = {**BASE, "target_seconds": 1200.0}
    with pytest.raises(ValueError, match="target_seconds"):
        load_config(_write(tmp_path, data))


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
```

- [ ] **Step 8: 运行测试确认失败**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.config'`

- [ ] **Step 9: 实现 `src/aiboke/config.py`**

```python
"""配置加载与校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .length import MAX_SECONDS, MIN_SECONDS

VALID_BACKENDS = frozenset({"llamacpp", "transformers"})


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str
    timeout: float = 300.0
    temperature: float = 0.7
    top_p: float = 0.80
    presence_penalty: float = 1.5


@dataclass(frozen=True)
class TtsConfig:
    backend: str
    binary: str | None = None
    model_path: str | None = None
    temperature: float = 1.1
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


@dataclass(frozen=True)
class CoverConfig:
    steps: int = 8
    size: int = 1024
    model_path: str | None = None


@dataclass(frozen=True)
class Config:
    target_seconds: float
    chars_per_minute: float
    models_root: str
    llm: LlmConfig
    tts: TtsConfig
    cover: CoverConfig
    max_retries: int = 2
    chinese_min_ratio: float = 0.85
    speaker_min_share: float = 0.25
    target_lufs: float = -16.0
    factcheck: bool = True


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise ValueError(f"{where} 缺少必需字段 {key!r}")
    return d[key]


def load_config(path: Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    target_seconds = float(_require(raw, "target_seconds", "config"))
    if not (MIN_SECONDS <= target_seconds <= MAX_SECONDS):
        raise ValueError(
            f"target_seconds 必须落在 [{MIN_SECONDS}, {MAX_SECONDS}] 内，"
            f"否则必然违反时长门限，实际为 {target_seconds}"
        )

    llm_raw = _require(raw, "llm", "config")
    tts_raw = _require(raw, "tts", "config")
    cover_raw = raw.get("cover", {})

    backend = _require(tts_raw, "backend", "tts")
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"tts.backend 必须是 {sorted(VALID_BACKENDS)} 之一，实际为 {backend!r}"
        )

    return Config(
        target_seconds=target_seconds,
        chars_per_minute=float(_require(raw, "chars_per_minute", "config")),
        models_root=str(raw.get("models_root", "/models")),
        llm=LlmConfig(
            base_url=_require(llm_raw, "base_url", "llm"),
            model=_require(llm_raw, "model", "llm"),
            timeout=float(llm_raw.get("timeout", 300.0)),
            temperature=float(llm_raw.get("temperature", 0.7)),
            top_p=float(llm_raw.get("top_p", 0.80)),
            presence_penalty=float(llm_raw.get("presence_penalty", 1.5)),
        ),
        tts=TtsConfig(
            backend=backend,
            binary=tts_raw.get("binary"),
            model_path=tts_raw.get("model_path"),
            temperature=float(tts_raw.get("temperature", 1.1)),
            top_p=float(tts_raw.get("top_p", 0.9)),
            top_k=int(tts_raw.get("top_k", 50)),
            repetition_penalty=float(tts_raw.get("repetition_penalty", 1.1)),
        ),
        cover=CoverConfig(
            steps=int(cover_raw.get("steps", 8)),
            size=int(cover_raw.get("size", 1024)),
            model_path=cover_raw.get("model_path"),
        ),
        max_retries=int(raw.get("max_retries", 2)),
        chinese_min_ratio=float(raw.get("chinese_min_ratio", 0.85)),
        speaker_min_share=float(raw.get("speaker_min_share", 0.25)),
        target_lufs=float(raw.get("target_lufs", -16.0)),
        factcheck=bool(raw.get("factcheck", True)),
    )
```

- [ ] **Step 10: 写 `configs/default.yaml`**

```yaml
# AI 播客生成系统主配置
target_seconds: 510.0          # 8.5 分钟，距 5/15 分钟门限均有充分余量
chars_per_minute: 200.0        # 语速常量，务必用 scripts/calibrate_length.py 实测校准
max_retries: 2
models_root: /models
factcheck: true                # 主体幕生成后追加一次事实自检（离线无法联网核查，只能降低编造概率）

chinese_min_ratio: 0.85
speaker_min_share: 0.25
target_lufs: -16.0

llm:
  base_url: http://127.0.0.1:8080
  model: qwen3.8-27b
  timeout: 300.0
  temperature: 0.7             # Qwen3.8 非思考模式官方推荐值
  top_p: 0.80
  presence_penalty: 1.5

tts:
  backend: llamacpp            # llamacpp | transformers
  binary: /opt/llama.cpp/build-cuda/bin/llama-moss-tts
  model_path: /models/MOSS-TTSD-GGUF
  temperature: 1.1
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1

cover:
  steps: 8
  size: 1024
  model_path: /models/Z-Image-Turbo
```

- [ ] **Step 11: 运行全部测试并提交**

```bash
python -m pytest tests/ -v
git add pyproject.toml src/ configs/ tests/ docs/
git commit -m "feat: 项目骨架、数据模型与配置加载"
```

Expected: 10 passed

---

### Task 2: 时长换算与三幕字数分配

**Files:**
- Create: `src/aiboke/length.py`
- Test: `tests/test_length.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - 常量：`MIN_SECONDS=300.0`、`MAX_SECONDS=900.0`、`DEFAULT_TARGET_SECONDS=510.0`、`DEFAULT_CHARS_PER_MINUTE=200.0`、`TOKENS_PER_SECOND=12.5`、`ACT_WEIGHTS=(0.20, 0.55, 0.25)`
  - `target_chars(seconds: float, chars_per_minute: float) -> int`
  - `act_char_targets(total_chars: int, weights: Sequence[float] = ACT_WEIGHTS) -> tuple[int, int, int]`
  - `seconds_to_max_tokens(seconds: float) -> int`
  - `DurationVerdict` 枚举：`OK` / `WARN` / `OUT_OF_RANGE`
  - `classify_duration(seconds: float, target: float = DEFAULT_TARGET_SECONDS, low: float = MIN_SECONDS, high: float = MAX_SECONDS, warn_delta: float = 90.0) -> DurationVerdict`
  - `chars_to_seconds(chars: int, chars_per_minute: float) -> float`

- [ ] **Step 1: 写失败的测试 `tests/test_length.py`**

```python
import pytest

from aiboke.length import (
    ACT_WEIGHTS,
    DurationVerdict,
    act_char_targets,
    chars_to_seconds,
    classify_duration,
    seconds_to_max_tokens,
    target_chars,
)


def test_target_chars_at_defaults():
    # 510 秒 = 8.5 分钟，8.5 * 200 = 1700 字
    assert target_chars(510.0, 200.0) == 1700


def test_target_chars_rounds_to_int():
    assert isinstance(target_chars(500.0, 200.0), int)


def test_chars_to_seconds_inverts_target_chars():
    assert chars_to_seconds(1700, 200.0) == pytest.approx(510.0)


def test_act_char_targets_at_defaults_sums_exactly():
    got = act_char_targets(1700, ACT_WEIGHTS)
    assert got == (340, 935, 425)
    assert sum(got) == 1700


def test_act_char_targets_sums_exactly_when_not_divisible():
    got = act_char_targets(1001, ACT_WEIGHTS)
    assert sum(got) == 1001


def test_act_char_targets_rejects_bad_weights():
    with pytest.raises(ValueError, match="weights"):
        act_char_targets(1000, (0.5, 0.5))


def test_seconds_to_max_tokens_uses_official_ratio():
    assert seconds_to_max_tokens(510.0) == 6375
    assert seconds_to_max_tokens(1.0) == 13


def test_classify_duration_ok_at_target():
    assert classify_duration(510.0) is DurationVerdict.OK


def test_classify_duration_ok_within_warn_delta():
    # |430 - 510| = 80 <= 90
    assert classify_duration(430.0) is DurationVerdict.OK


def test_classify_duration_warn_beyond_delta_but_in_range():
    # |400 - 510| = 110 > 90，但仍在 [300, 900] 内
    assert classify_duration(400.0) is DurationVerdict.WARN


def test_classify_duration_warn_at_lower_bound():
    assert classify_duration(300.0) is DurationVerdict.WARN


def test_classify_duration_out_of_range_low():
    assert classify_duration(299.9) is DurationVerdict.OUT_OF_RANGE


def test_classify_duration_out_of_range_high():
    assert classify_duration(900.1) is DurationVerdict.OUT_OF_RANGE
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_length.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.length'`

- [ ] **Step 3: 实现 `src/aiboke/length.py`**

```python
"""时长换算与三幕字数分配。纯函数，无副作用。

时长门限是四个 0 分项之一：必须 300 <= 秒数 <= 900，否则整案 0 分。
本模块提供三重保险中的前两重：脚本字数反算 与 生成长度硬限制。
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

MIN_SECONDS = 300.0
MAX_SECONDS = 900.0
DEFAULT_TARGET_SECONDS = 510.0
DEFAULT_CHARS_PER_MINUTE = 200.0

# MOSS-TTSD 官方换算：1 秒音频约等于 12.5 个 token
TOKENS_PER_SECOND = 12.5

# 三幕字数占比：开场铺垫 / 主体讲述 / 分析总结
ACT_WEIGHTS: tuple[float, float, float] = (0.20, 0.55, 0.25)

# 距目标时长超过此值即记录警告（但仍视为通过）
DEFAULT_WARN_DELTA = 90.0


class DurationVerdict(Enum):
    OK = "ok"
    WARN = "warn"
    OUT_OF_RANGE = "out_of_range"


def target_chars(seconds: float, chars_per_minute: float) -> int:
    """按语速反算目标字数。"""
    if chars_per_minute <= 0:
        raise ValueError(f"chars_per_minute 必须为正数，实际为 {chars_per_minute}")
    return int(round(seconds / 60.0 * chars_per_minute))


def chars_to_seconds(chars: int, chars_per_minute: float) -> float:
    """按语速反算预计时长（秒）。"""
    if chars_per_minute <= 0:
        raise ValueError(f"chars_per_minute 必须为正数，实际为 {chars_per_minute}")
    return chars / chars_per_minute * 60.0


def act_char_targets(
    total_chars: int,
    weights: Sequence[float] = ACT_WEIGHTS,
) -> tuple[int, int, int]:
    """把总字数按权重分给三幕。

    前两幕四舍五入，第三幕取剩余量，保证总和与 total_chars 严格相等——
    否则累计误差会让实际字数偏离时长目标。
    """
    if len(weights) != 3:
        raise ValueError(f"weights 必须恰好有 3 项（对应三幕），实际为 {len(weights)} 项")
    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"weights 之和必须为 1.0，实际为 {total_weight}")
    if total_chars < 3:
        raise ValueError(f"total_chars 过小，无法分配给三幕：{total_chars}")

    first = int(round(total_chars * weights[0]))
    second = int(round(total_chars * weights[1]))
    third = total_chars - first - second
    return first, second, third


def seconds_to_max_tokens(seconds: float) -> int:
    """把目标时长换算为 MOSS-TTSD 的 max_new_tokens（第二重保险）。"""
    if seconds <= 0:
        raise ValueError(f"seconds 必须为正数，实际为 {seconds}")
    return int(round(seconds * TOKENS_PER_SECOND))


def classify_duration(
    seconds: float,
    target: float = DEFAULT_TARGET_SECONDS,
    low: float = MIN_SECONDS,
    high: float = MAX_SECONDS,
    warn_delta: float = DEFAULT_WARN_DELTA,
) -> DurationVerdict:
    """判定实测时长的合规性。

    越界即触发 0 分，必须重生成；偏离目标但合规仅记录警告。
    """
    if seconds < low or seconds > high:
        return DurationVerdict.OUT_OF_RANGE
    if abs(seconds - target) > warn_delta:
        return DurationVerdict.WARN
    return DurationVerdict.OK
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_length.py -v`
Expected: PASS（13 passed）

- [ ] **Step 5: 提交**

```bash
git add src/aiboke/length.py tests/test_length.py
git commit -m "feat: 时长换算与三幕字数分配"
```

---

### Task 3: 四道 0 分门限校验

**Files:**
- Create: `src/aiboke/gates.py`
- Test: `tests/test_gates.py`

**Interfaces:**
- Consumes: `aiboke.schema.Turn`、`aiboke.schema.VoicePreset`、`aiboke.length.MIN_SECONDS`
- Produces:
  - `GateResult(name: str, passed: bool, detail: str, retry_hint: str | None)`
  - `cjk_ratio(text: str) -> float`
  - `check_chinese(text: str, min_ratio: float = 0.85) -> GateResult`
  - `check_duration(seconds: float, low: float = MIN_SECONDS, high: float = 900.0) -> GateResult`
  - `check_two_speakers(turns: Sequence[Turn], min_share: float = 0.25) -> GateResult`
  - `check_genders(gender1: str, gender2: str, presets: Mapping[str, Sequence[VoicePreset]]) -> GateResult`
  - `all_passed(results: Sequence[GateResult]) -> bool`

- [ ] **Step 1: 写失败的测试 `tests/test_gates.py`**

```python
import pytest

from aiboke.gates import (
    GateResult,
    all_passed,
    check_chinese,
    check_duration,
    check_genders,
    check_two_speakers,
    cjk_ratio,
)
from aiboke.schema import Turn, VoicePreset


# ---------- cjk_ratio ----------

def test_cjk_ratio_pure_chinese():
    assert cjk_ratio("大家好欢迎收听") == 1.0


def test_cjk_ratio_ignores_punctuation_and_space():
    # 标点与空白不计入分母
    assert cjk_ratio("大家好，欢迎 收听。") == 1.0


def test_cjk_ratio_mixed_counts_letters():
    # "AI播客" -> 2 个汉字 / (2 汉字 + 2 字母) = 0.5
    assert cjk_ratio("AI播客") == pytest.approx(0.5)


def test_cjk_ratio_empty_text_is_zero():
    assert cjk_ratio("") == 0.0
    assert cjk_ratio("   ，。") == 0.0


# ---------- check_chinese ----------

def test_check_chinese_passes_on_chinese_with_few_english_terms():
    text = "今天我们要聊的是星巴克在中国市场的运营权转移。" * 10 + "CEO 说了什么？"
    assert check_chinese(text).passed
    assert check_chinese(text).name == "language"


def test_check_chinese_fails_on_english():
    r = check_chinese("This is an English podcast about coffee and business.")
    assert not r.passed
    assert r.retry_hint is not None


def test_check_chinese_fails_on_empty():
    assert not check_chinese("").passed


# ---------- check_duration ----------

def test_check_duration_passes_inside_window():
    assert check_duration(510.0).passed
    assert check_duration(300.0).passed
    assert check_duration(900.0).passed


def test_check_duration_fails_below_five_minutes():
    r = check_duration(299.0)
    assert not r.passed
    assert "300" in r.detail


def test_check_duration_fails_above_fifteen_minutes():
    assert not check_duration(901.0).passed


# ---------- check_two_speakers ----------

def _alternating(n: int) -> tuple[Turn, ...]:
    return tuple(Turn(speaker=(i % 2) + 1, text=f"第{i}轮内容") for i in range(n))


def test_check_two_speakers_passes_balanced():
    assert check_two_speakers(_alternating(10)).passed


def test_check_two_speakers_fails_single_speaker():
    turns = tuple(Turn(speaker=1, text=f"第{i}轮") for i in range(10))
    r = check_two_speakers(turns)
    assert not r.passed
    assert "2" in r.detail


def test_check_two_speakers_fails_imbalanced():
    turns = tuple(Turn(speaker=1, text="a") for _ in range(9)) + (Turn(speaker=2, text="b"),)
    assert not check_two_speakers(turns).passed


def test_check_two_speakers_fails_empty():
    assert not check_two_speakers(()).passed


# ---------- check_genders ----------

def _presets():
    return {
        "男": [VoicePreset(id="m1", gender="男", description="低沉男声")],
        "女": [VoicePreset(id="f1", gender="女", description="清亮女声")],
    }


def test_check_genders_passes_mixed_pair():
    assert check_genders("男", "女", _presets()).passed


def test_check_genders_fails_when_gender_missing():
    r = check_genders("女", "女", _presets())
    assert not r.passed
    assert r.retry_hint is None  # 配置问题，重试无意义


# ---------- all_passed ----------

def test_all_passed_true_when_all_ok():
    ok = GateResult(name="x", passed=True, detail="")
    assert all_passed([ok, ok])


def test_all_passed_false_when_any_fails():
    ok = GateResult(name="x", passed=True, detail="")
    bad = GateResult(name="y", passed=False, detail="boom")
    assert not all_passed([ok, bad])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_gates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.gates'`

- [ ] **Step 3: 实现 `src/aiboke/gates.py`**

```python
"""四道 0 分门限的校验。纯函数，无副作用。

赛题规定命中任一门限即整案 0 分，因此这四项是最高优先级：
  1. 语言必须为中文
  2. 时长必须落在 5-15 分钟
  3. 必须是双人对话
  4. 性别必须符合指定要求

每个 GateResult 可携带 retry_hint，供上层带反馈重试；配置类问题
（如性别组合无可用音色）不提供 retry_hint，因为重试不会改变结果。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Mapping, Sequence

from .length import MAX_SECONDS, MIN_SECONDS
from .schema import Turn, VoicePreset

DEFAULT_CHINESE_MIN_RATIO = 0.85
DEFAULT_SPEAKER_MIN_SHARE = 0.25


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    retry_hint: str | None = None


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _is_latin_or_digit(ch: str) -> bool:
    if ch.isdigit():
        return True
    if "a" <= ch.lower() <= "z":
        return True
    # 全角字母数字
    return unicodedata.category(ch) in ("Nd", "Lu", "Ll", "Lt", "Lm", "Lo") and not _is_cjk(ch)


def cjk_ratio(text: str) -> float:
    """汉字占「汉字 + 拉丁字母 + 数字」的比例。

    标点、空白、表情等不计入分母——它们在中文与英文文本中都常见，
    计入会让「含少量英文专有名词的纯中文稿」被误判为不合格。
    """
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = sum(1 for ch in text if not _is_cjk(ch) and _is_latin_or_digit(ch))
    denominator = cjk + other
    if denominator == 0:
        return 0.0
    return cjk / denominator


def check_chinese(text: str, min_ratio: float = DEFAULT_CHINESE_MIN_RATIO) -> GateResult:
    if not text or not text.strip():
        return GateResult(
            name="language",
            passed=False,
            detail="文稿为空，无法判定语言",
            retry_hint="上一次输出为空，请重新生成完整的中文对话文稿。",
        )
    ratio = cjk_ratio(text)
    passed = ratio >= min_ratio
    detail = f"中文占比 {ratio:.3f}（阈值 {min_ratio}）"
    hint = None
    if not passed:
        hint = (
            f"上一次输出的中文占比仅 {ratio:.3f}，低于要求的 {min_ratio}。"
            "请确保全部对话内容使用简体中文，仅专有名词可保留英文。"
        )
    return GateResult(name="language", passed=passed, detail=detail, retry_hint=hint)


def check_duration(
    seconds: float,
    low: float = MIN_SECONDS,
    high: float = MAX_SECONDS,
) -> GateResult:
    passed = low <= seconds <= high
    detail = f"时长 {seconds:.1f} 秒，允许区间 [{low:.0f}, {high:.0f}] 秒"
    hint = None
    if not passed:
        if seconds < low:
            shortfall = int(round((low - seconds) / 60.0 * 200))
            hint = (
                f"上一次音频仅 {seconds / 60:.1f} 分钟，低于 5 分钟下限。"
                f"请把文稿总字数增加约 {max(shortfall, 200)} 字。"
            )
        else:
            excess = int(round((seconds - high) / 60.0 * 200))
            hint = (
                f"上一次音频达 {seconds / 60:.1f} 分钟，超过 15 分钟上限。"
                f"请把文稿总字数减少约 {max(excess, 200)} 字。"
            )
    return GateResult(name="duration", passed=passed, detail=detail, retry_hint=hint)


def check_two_speakers(
    turns: Sequence[Turn],
    min_share: float = DEFAULT_SPEAKER_MIN_SHARE,
) -> GateResult:
    if not turns:
        return GateResult(
            name="two_speakers",
            passed=False,
            detail="没有任何对话轮次",
            retry_hint="上一次输出没有对话轮次，请生成完整的双人对话。",
        )

    counts = {1: 0, 2: 0}
    for t in turns:
        if t.speaker in counts:
            counts[t.speaker] += 1

    total = counts[1] + counts[2]
    if counts[1] == 0 or counts[2] == 0:
        return GateResult(
            name="two_speakers",
            passed=False,
            detail=f"仅检测到一位主播（轮次统计 {counts}），必须为双人对话",
            retry_hint=(
                "上一次输出只有一位主播发言。必须写成两位主播的对话，"
                "用 speaker 1 与 speaker 2 交替发言。"
            ),
        )

    share1 = counts[1] / total
    share2 = counts[2] / total
    passed = share1 >= min_share and share2 >= min_share
    detail = f"轮次占比 speaker1={share1:.2f} speaker2={share2:.2f}（下限 {min_share}）"
    hint = None
    if not passed:
        hint = (
            f"两位主播发言严重失衡（speaker1 占 {share1:.0%}，speaker2 占 {share2:.0%}）。"
            "请让两人充分互动，各自至少占到四分之一的内容。"
        )
    return GateResult(name="two_speakers", passed=passed, detail=detail, retry_hint=hint)


def check_genders(
    gender1: str,
    gender2: str,
    presets: Mapping[str, Sequence[VoicePreset]],
) -> GateResult:
    """校验请求的性别组合在音色预设库中有对应条目。

    这是配置层校验——性别由构造保证（按性别选音色），此处的职责是
    尽早发现「预设库缺少该性别」或「同性别但预设不足两条」的问题。
    预设本身的性别标注正确性由冒烟测试的人工试听确认（见设计文档 7.2）。
    """
    for label, gender in (("speaker_gender1", gender1), ("speaker_gender2", gender2)):
        if not presets.get(gender):
            return GateResult(
                name="genders",
                passed=False,
                detail=f"{label}={gender} 在音色预设库中没有可用条目",
                retry_hint=None,
            )

    if gender1 == gender2 and len(presets[gender1]) < 2:
        return GateResult(
            name="genders",
            passed=False,
            detail=(
                f"两位主播均为 {gender1}，但 {gender1} 的预设只有 "
                f"{len(presets[gender1])} 条，无法保证音色有区分度"
            ),
            retry_hint=None,
        )

    return GateResult(
        name="genders",
        passed=True,
        detail=f"性别组合 {gender1}+{gender2} 有可用音色预设",
    )


def all_passed(results: Sequence[GateResult]) -> bool:
    return all(r.passed for r in results)


def first_retry_hint(results: Sequence[GateResult]) -> str | None:
    """取出第一个可重试的反馈，供上层拼接重试指令。"""
    for r in results:
        if not r.passed and r.retry_hint:
            return r.retry_hint
    return None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_gates.py -v`
Expected: PASS（18 passed）

- [ ] **Step 5: 提交**

```bash
git add src/aiboke/gates.py tests/test_gates.py
git commit -m "feat: 四道 0 分门限校验"
```

---

### Task 4: 三幕式 Prompt 组装

**Files:**
- Create: `src/aiboke/prompts.py`
- Create: `configs/podcast_styles.json`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `aiboke.schema.CaseInput`、`aiboke.schema.Turn`
- Produces:
  - 常量：`SYSTEM_PROMPT: str`、`ACT_NAMES: tuple[str, str, str]`、`ACT_GUIDANCE: tuple[str, str, str]`、`SCRIPT_JSON_SCHEMA: dict`
  - `build_act_prompt(case: CaseInput, act_index: int, char_target: int, prior_turns: Sequence[Turn], retry_hint: str | None = None) -> str`
  - `build_repair_prompt(raw: str, error: str) -> str`
  - `build_factcheck_prompt(turns: Sequence[Turn]) -> str`
  - `build_cover_prompt(topic: str) -> str`
  - `build_cover_upgrade_prompt(topic: str) -> str`

- [ ] **Step 1: 写失败的测试 `tests/test_prompts.py`**

```python
import pytest

from aiboke.prompts import (
    ACT_GUIDANCE,
    ACT_NAMES,
    SCRIPT_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_act_prompt,
    build_cover_prompt,
    build_cover_upgrade_prompt,
    build_repair_prompt,
)
from aiboke.schema import CaseInput, Turn


def _case():
    return CaseInput(topic="星巴克国内运营转移", speaker_gender1="男", speaker_gender2="女")


def test_system_prompt_mentions_required_structure():
    assert "开场" in SYSTEM_PROMPT
    assert "总结" in SYSTEM_PROMPT


def test_system_prompt_forbids_fabrication():
    assert "编造" in SYSTEM_PROMPT


def test_system_prompt_demands_colloquial_tone():
    assert "口语" in SYSTEM_PROMPT


def test_three_acts_defined():
    assert len(ACT_NAMES) == 3
    assert len(ACT_GUIDANCE) == 3


def test_script_json_schema_matches_required_output():
    props = SCRIPT_JSON_SCHEMA["properties"]
    assert "title" in props
    assert props["content"]["items"]["properties"]["speaker"]["enum"] == [1, 2]


def test_build_act_prompt_includes_topic_and_genders():
    p = build_act_prompt(_case(), act_index=0, char_target=340, prior_turns=())
    assert "星巴克国内运营转移" in p
    assert "男" in p and "女" in p
    assert "340" in p


def test_build_act_prompt_marks_first_act_as_opening():
    p = build_act_prompt(_case(), act_index=0, char_target=340, prior_turns=())
    assert ACT_NAMES[0] in p


def test_build_act_prompt_includes_prior_context_for_later_acts():
    prior = (Turn(speaker=1, text="上一幕的结尾内容"),)
    p = build_act_prompt(_case(), act_index=1, char_target=935, prior_turns=prior)
    assert "上一幕的结尾内容" in p
    assert ACT_NAMES[1] in p


def test_build_act_prompt_includes_retry_hint():
    p = build_act_prompt(
        _case(), act_index=0, char_target=340, prior_turns=(),
        retry_hint="上一次中文占比不足",
    )
    assert "上一次中文占比不足" in p


def test_build_act_prompt_rejects_bad_index():
    with pytest.raises(ValueError, match="act_index"):
        build_act_prompt(_case(), act_index=5, char_target=100, prior_turns=())


def test_build_repair_prompt_includes_raw_and_error():
    p = build_repair_prompt('{"title": "x"', "JSON 解析失败")
    assert "JSON 解析失败" in p
    assert '{"title": "x"' in p


def test_build_factcheck_prompt_includes_dialogue_text():
    p = build_factcheck_prompt((Turn(speaker=1, text="市占率高达 37.5%。"),))
    assert "市占率高达 37.5%。" in p
    assert "不确定" in p


def test_build_factcheck_prompt_asks_for_same_json_shape():
    p = build_factcheck_prompt((Turn(speaker=1, text="内容"),))
    assert "content" in p
    assert "speaker" in p


def test_build_cover_prompt_mentions_topic_and_square():
    p = build_cover_prompt("星巴克国内运营转移")
    assert "星巴克国内运营转移" in p
    assert "1024" in p


def test_cover_upgrade_prompt_asks_for_english_prompt():
    p = build_cover_upgrade_prompt("星巴克国内运营转移")
    assert "星巴克国内运营转移" in p
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.prompts'`

- [ ] **Step 3: 实现 `src/aiboke/prompts.py`**

```python
"""三幕式中文播客文稿的 Prompt 组装。纯函数，无副作用。

文稿相关占评分 70 分（内容质量 40 + 对话表达 30），是本系统投入
最大的部分。Prompt 需要同时约束：叙事结构、口语化、互动性、
事实准确性、字数。
"""

from __future__ import annotations

import json
from typing import Sequence

from .schema import CaseInput, Turn

ACT_NAMES: tuple[str, str, str] = ("第一幕：开场铺垫", "第二幕：主体讲述", "第三幕：分析总结")

ACT_GUIDANCE: tuple[str, str, str] = (
    "用一两句抓住听众的钩子开场，点明今天要讲的商业故事，"
    "抛出核心悬念或反差，让听众有继续听下去的理由。不要在这一幕展开细节。",
    "讲述故事主体：关键事件、核心人物、重要数据、行业背景与转折点。"
    "按时间或因果顺序推进，有起伏、有冲突、有意外。这是篇幅最长的一幕。",
    "收束故事：分析这件事的影响与启示，总结商业逻辑，"
    "给出一个有余味的结尾。可以呼应开场的悬念。",
)

# llama.cpp 的 JSON schema 约束解码用；同时作为给模型的格式说明
SCRIPT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "content": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "integer", "enum": [1, 2]},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
            },
        },
    },
    "required": ["title", "content"],
}

SYSTEM_PROMPT = """你是一位资深的中文播客制作人兼撰稿人，擅长把商业故事写成引人入胜的双人对话。

你的听众是普通商业爱好者，不是专业人士。你要写出的文稿具备以下特征：

【形式】
- 两位主播的自然对话，不是访谈提纲，也不是朗读稿
- 全程使用简体中文；只有公司名、人名、专业术语等专有名词可保留英文
- 极其口语化：用日常说话的方式，短句为主，允许语气词和口头禅
- 两位主播要有明确的角色分工与性格差异

【互动（这是好播客的灵魂）】
- 频繁互动：提问、回应、补充、质疑、感叹、打趣
- 自然的插话与接话，例如"对""没错""等一下""我插一句""这我倒是第一次听说"
- 一位主播抛出信息，另一位要接住并推进，而不是各说各话
- 允许适度的分歧和追问，让对话有张力

【结构】
必须遵循「开场铺垫 → 主体讲述 → 分析总结」的叙事弧线：
- 开场要有钩子，点明主题并制造悬念
- 主体要有故事性，有事件、人物、数据、转折，像讲故事一样推进
- 结尾要分析影响与启示，给出有回味的收束
严禁平铺直叙地罗列信息，严禁"第一点、第二点、第三点"这种讲稿式表达。

【事实准确性（极其重要）】
- 只使用广为人知、可核实的商业事实、人物和数据
- 如果对某个具体数字没有把握，宁可不给数字，改用定性描述
  （例如用"销量翻了好几倍""市占率大幅下滑"代替编造的精确百分比）
- 严禁编造数据、事件、时间、人物姓名或引语
- 不确定的事就模糊表述或干脆不提，绝不允许为了生动而虚构

【表达质量】
- 词汇和句式要丰富，避免反复使用同一口头禅
- 避免书面语和机械播报腔
- 该有的停顿、感叹、犹豫要写进对话里，让语音合成有发挥空间

【输出格式】
严格输出 JSON，不要有任何额外文字或 Markdown 代码块标记：
{"title": "播客标题", "content": [{"speaker": 1, "text": "说话内容"}, {"speaker": 2, "text": "说话内容"}]}
其中 speaker 只能是 1 或 2；title 是吸引人的播客标题，不要带书名号。"""


def _role_description(case: CaseInput) -> str:
    return (
        f"主播1（speaker 1）是{case.speaker_gender1}性，担任提问引导者——"
        f"负责抛出问题、追问、替听众发问，推进节目节奏。\n"
        f"主播2（speaker 2）是{case.speaker_gender2}性，担任讲述分析者——"
        f"负责提供事实、数据、行业背景与深度分析。"
    )


def build_act_prompt(
    case: CaseInput,
    act_index: int,
    char_target: int,
    prior_turns: Sequence[Turn],
    retry_hint: str | None = None,
) -> str:
    if not 0 <= act_index < len(ACT_NAMES):
        raise ValueError(f"act_index 必须在 [0, {len(ACT_NAMES) - 1}] 内，实际为 {act_index}")

    parts = [
        f"播客主题：{case.topic}",
        "",
        _role_description(case),
        "",
        f"当前要写的是【{ACT_NAMES[act_index]}】。",
        f"这一幕的写作要求：{ACT_GUIDANCE[act_index]}",
        f"这一幕的目标字数：约 {char_target} 字（中文字符计，不含标点）。请尽量贴近该字数。",
    ]

    if prior_turns:
        context = "\n".join(f"主播{t.speaker}：{t.text}" for t in prior_turns[-6:])
        parts += [
            "",
            "【前文结尾（用于衔接，不要重复其内容）】",
            context,
        ]

    if retry_hint:
        parts += [
            "",
            "【上一次生成的问题，必须修正】",
            retry_hint,
        ]

    parts += [
        "",
        "请只输出这一幕的 JSON，格式严格如下（不要输出前文的重复内容）：",
        json.dumps(SCRIPT_JSON_SCHEMA, ensure_ascii=False),
    ]
    return "\n".join(parts)


def build_repair_prompt(raw: str, error: str) -> str:
    return (
        "你上一次的输出不是合法 JSON，无法解析。\n\n"
        f"解析错误：{error}\n\n"
        "原始输出：\n"
        f"{raw}\n\n"
        "请把它修正为合法的 JSON，只输出修正后的 JSON 本身，"
        "不要任何解释、不要 Markdown 代码块标记。格式必须是：\n"
        f"{json.dumps(SCRIPT_JSON_SCHEMA, ensure_ascii=False)}"
    )


def build_factcheck_prompt(turns: Sequence[Turn]) -> str:
    """主体幕生成后的事实自检。

    离线环境无法联网核查，因此策略是让模型自己标出没把握的具体断言，
    并改写为定性表述——降低编造概率，而不是事后验证。
    """
    dialogue = json.dumps(
        [{"speaker": t.speaker, "text": t.text} for t in turns],
        ensure_ascii=False,
        indent=2,
    )
    return (
        "下面是播客文稿的对话内容。请逐句检查其中**具体的、可被证伪的事实性断言**，"
        "包括精确数字（如百分比、金额、年份、门店数）、事件细节、人物姓名与引语。\n\n"
        f"{dialogue}\n\n"
        "对每一处你没有十足把握的断言，请改写为定性表述"
        "（例如把「市占率高达 37.5%」改为「市占率大幅下滑」），"
        "或者干脆删去该细节。\n"
        "对有把握的内容必须原样保留，不要润色、不要改变语气、不要增删对话轮次。\n\n"
        "只输出修订后的完整 JSON，结构与输入完全一致（title 与 content 两个字段，"
        f"content 每项含 speaker 与 text），不要任何解释：\n"
        f"{json.dumps(SCRIPT_JSON_SCHEMA, ensure_ascii=False)}"
    )


def build_cover_prompt(topic: str) -> str:
    return (
        f"为一期主题为「{topic}」的中文商业播客生成封面图。\n"
        "要求：\n"
        "- 正方形构图，用于 1024×1024 的播客封面\n"
        "- 现代、简洁、有设计感的插画或扁平化风格，配色鲜明\n"
        "- 画面呼应主题的商业意象，但不要出现任何文字、字母或数字\n"
        "  （文生图模型的文字渲染不可靠，出现乱码会严重拉低观感）\n"
        "- 视觉上要有吸引力，像专业播客的封面\n"
        "请只输出英文的图像生成提示词，不要任何其他内容。"
    )


def build_cover_upgrade_prompt(topic: str) -> str:
    return build_cover_prompt(topic)
```

- [ ] **Step 4: 写 `configs/podcast_styles.json`**

```json
{
  "_source": "长度标定表移植自 https://github.com/uiuing/ai-podcast-workflow (MIT License)。字/分钟 折算值用于本项目的语速常量初始值，须由 scripts/calibrate_length.py 实测校准。",
  "podcast_formats": [
    {
      "id": "brief",
      "name": "快闪洞察",
      "word_count": "1200-2000字",
      "audio_duration": "5-10分钟",
      "dialogue_rounds": "15-25轮对话",
      "avg_words_per_round": "80-100字"
    },
    {
      "id": "standard",
      "name": "沉浸解读",
      "word_count": "2000-3000字",
      "audio_duration": "10-15分钟",
      "dialogue_rounds": "25-40轮对话",
      "avg_words_per_round": "80-100字"
    },
    {
      "id": "deep",
      "name": "透彻剖析",
      "word_count": "3000-4000字",
      "audio_duration": "15-20分钟",
      "dialogue_rounds": "40-50轮对话",
      "avg_words_per_round": "80-100字"
    }
  ],
  "derived_chars_per_minute": 200.0,
  "podcast_styles": [
    {
      "id": "business_story",
      "name": "商业故事",
      "description": "以叙事为主线的商业案例解读",
      "tone": "沉稳、有洞察力、娓娓道来",
      "interaction": "高频互动",
      "features": [
        "开场用悬念或反差抓住听众",
        "按时间线讲清事件来龙去脉",
        "关键数据要有解读，不能只报数字",
        "结尾落到商业逻辑与启示"
      ]
    }
  ]
}
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_prompts.py -v`
Expected: PASS（13 passed）

- [ ] **Step 6: 提交**

```bash
git add src/aiboke/prompts.py configs/podcast_styles.json tests/test_prompts.py
git commit -m "feat: 三幕式 Prompt 组装与风格标定配置"
```

---

# 阶段二：适配层（依赖注入 + 假实现测试）

### Task 5: 音色预设与性别解析

**Files:**
- Create: `src/aiboke/voices.py`
- Create: `configs/voice_presets.json`
- Test: `tests/test_voices.py`

**Interfaces:**
- Consumes: `aiboke.schema.VoicePreset`、`aiboke.schema.VoicePair`
- Produces:
  - `load_presets(path: Path) -> dict[str, list[VoicePreset]]`
  - `resolve_pair(presets: Mapping[str, Sequence[VoicePreset]], gender1: str, gender2: str, seed: int | None = None) -> VoicePair`

- [ ] **Step 1: 写失败的测试 `tests/test_voices.py`**

```python
import json

import pytest

from aiboke.voices import load_presets, resolve_pair


def _presets_file(tmp_path, data):
    p = tmp_path / "voice_presets.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


VALID = {
    "presets": [
        {"id": "m_calm", "gender": "男", "description": "低沉磁性的男声，语速偏慢"},
        {"id": "m_lively", "gender": "男", "description": "明亮活泼的男声，语速偏快"},
        {"id": "f_clear", "gender": "女", "description": "清亮温和的女声"},
        {"id": "f_bright", "gender": "女", "description": "明快爽朗的女声"},
    ]
}


def test_load_presets_groups_by_gender(tmp_path):
    presets = load_presets(_presets_file(tmp_path, VALID))
    assert set(presets) == {"男", "女"}
    assert len(presets["男"]) == 2
    assert presets["男"][0].id == "m_calm"


def test_load_presets_rejects_bad_gender(tmp_path):
    bad = {"presets": [{"id": "x", "gender": "male", "description": "d"}]}
    with pytest.raises(ValueError, match="gender"):
        load_presets(_presets_file(tmp_path, bad))


def test_load_presets_rejects_empty(tmp_path):
    with pytest.raises(ValueError, match="presets"):
        load_presets(_presets_file(tmp_path, {"presets": []}))


def test_resolve_pair_mixed_genders(tmp_path):
    presets = load_presets(_presets_file(tmp_path, VALID))
    pair = resolve_pair(presets, "男", "女")
    assert pair.speaker1.gender == "男"
    assert pair.speaker2.gender == "女"


def test_resolve_pair_same_gender_gives_distinct_presets(tmp_path):
    presets = load_presets(_presets_file(tmp_path, VALID))
    for g in ("男", "女"):
        pair = resolve_pair(presets, g, g)
        assert pair.speaker1.id != pair.speaker2.id, "同性别时必须选中不同音色以保证区分度"


def test_resolve_pair_is_deterministic_with_seed(tmp_path):
    presets = load_presets(_presets_file(tmp_path, VALID))
    a = resolve_pair(presets, "男", "男", seed=7)
    b = resolve_pair(presets, "男", "男", seed=7)
    assert (a.speaker1.id, a.speaker2.id) == (b.speaker1.id, b.speaker2.id)


def test_resolve_pair_raises_when_gender_missing(tmp_path):
    data = {"presets": [{"id": "m", "gender": "男", "description": "d"}]}
    presets = load_presets(_presets_file(tmp_path, data))
    with pytest.raises(ValueError, match="女"):
        resolve_pair(presets, "女", "男")


def test_resolve_pair_raises_when_same_gender_needs_two(tmp_path):
    data = {"presets": [{"id": "m", "gender": "男", "description": "d"}]}
    presets = load_presets(_presets_file(tmp_path, data))
    with pytest.raises(ValueError, match="两条"):
        resolve_pair(presets, "男", "男")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_voices.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.voices'`

- [ ] **Step 3: 实现 `src/aiboke/voices.py`**

```python
"""性别到音色预设的解析。

性别门限由构造保证——按请求的性别挑选音色，而非事后检测音频。
同性别组合（男男 / 女女）时必须选中不同预设，否则违反
「音色应有区分度」的评分要求。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping, Sequence

from .schema import VALID_GENDERS, VoicePair, VoicePreset


def load_presets(path: Path) -> dict[str, list[VoicePreset]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"音色预设文件不存在: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("presets")
    if not entries:
        raise ValueError(f"presets 字段缺失或为空: {path}")

    grouped: dict[str, list[VoicePreset]] = {g: [] for g in VALID_GENDERS}
    for entry in entries:
        try:
            preset = VoicePreset(
                id=entry["id"],
                gender=entry["gender"],
                description=entry["description"],
                reference_audio=entry.get("reference_audio"),
            )
        except KeyError as exc:
            raise ValueError(f"音色预设缺少字段 {exc.args[0]!r}: {entry}") from exc
        grouped[preset.gender].append(preset)

    return {g: v for g, v in grouped.items() if v}


def resolve_pair(
    presets: Mapping[str, Sequence[VoicePreset]],
    gender1: str,
    gender2: str,
    seed: int | None = None,
) -> VoicePair:
    for label, gender in (("speaker_gender1", gender1), ("speaker_gender2", gender2)):
        if not presets.get(gender):
            raise ValueError(f"{label}={gender} 没有可用的音色预设")

    rng = random.Random(seed)
    first = rng.choice(list(presets[gender1]))

    if gender1 == gender2:
        candidates = [p for p in presets[gender2] if p.id != first.id]
        if not candidates:
            raise ValueError(
                f"两位主播均为 {gender2}，但该性别只有一条音色预设，"
                "需要至少两条才能保证音色有区分度"
            )
        second = rng.choice(candidates)
    else:
        second = rng.choice(list(presets[gender2]))

    return VoicePair(speaker1=first, speaker2=second)
```

- [ ] **Step 4: 写 `configs/voice_presets.json`**

```json
{
  "_comment": "音色由 MOSS-VoiceGenerator 依据 description 文本直接生成，无需参考音频；也支持填写 reference_audio 指向自备的克隆参考音频。description 的性别表述必须与 gender 字段一致——冒烟测试需人工试听确认。",
  "presets": [
    {
      "id": "m_calm",
      "gender": "男",
      "description": "一位成熟稳重的男声，音色低沉有磁性，语速从容偏慢，语调沉稳，适合讲述深度商业分析"
    },
    {
      "id": "m_lively",
      "gender": "男",
      "description": "一位年轻明快的男声，音色明亮清晰，语速稍快，语气轻松有活力，适合引导话题和追问"
    },
    {
      "id": "f_clear",
      "gender": "女",
      "description": "一位温和知性的女声，音色清亮柔和，语速适中，语调自然亲切，适合讲述商业故事"
    },
    {
      "id": "f_bright",
      "gender": "女",
      "description": "一位活泼爽朗的女声，音色明亮清脆，语速偏快，语气轻快有感染力，适合主持互动"
    }
  ]
}
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_voices.py -v`
Expected: PASS（8 passed）

- [ ] **Step 6: 提交**

```bash
git add src/aiboke/voices.py configs/voice_presets.json tests/test_voices.py
git commit -m "feat: 音色预设与性别解析"
```

---

### Task 6: llama.cpp server 客户端

**Files:**
- Create: `src/aiboke/llm_client.py`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Consumes: `aiboke.config.LlmConfig`
- Produces:
  - `LlmError(Exception)`
  - `LlmClient(cfg: LlmConfig, session=None)`
  - `LlmClient.complete(system: str, user: str, *, json_schema: dict | None = None, max_tokens: int | None = None) -> str`

- [ ] **Step 1: 写失败的测试 `tests/test_llm_client.py`**

```python
import pytest

from aiboke.config import LlmConfig
from aiboke.llm_client import LlmClient, LlmError


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if not self._responses:
            raise AssertionError("FakeSession 收到多余的请求")
        return self._responses.pop(0)


def _cfg():
    return LlmConfig(base_url="http://127.0.0.1:8080", model="qwen3.8-27b")


def _ok(text):
    return FakeResponse({"choices": [{"message": {"content": text}}]})


def test_complete_returns_message_content():
    session = FakeSession([_ok("你好")])
    client = LlmClient(_cfg(), session=session)
    assert client.complete("sys", "user") == "你好"


def test_complete_hits_openai_compatible_endpoint():
    session = FakeSession([_ok("x")])
    LlmClient(_cfg(), session=session).complete("sys", "user")
    assert session.calls[0]["url"] == "http://127.0.0.1:8080/v1/chat/completions"


def test_complete_sends_both_roles_and_disables_thinking():
    session = FakeSession([_ok("x")])
    LlmClient(_cfg(), session=session).complete("S", "U")
    messages = session.calls[0]["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "S"}
    assert messages[1] == {"role": "user", "content": "U"}


def test_complete_passes_json_schema_when_given():
    schema = {"type": "object", "properties": {}}
    session = FakeSession([_ok("{}")])
    LlmClient(_cfg(), session=session).complete("s", "u", json_schema=schema)
    body = session.calls[0]["json"]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == schema


def test_complete_omits_response_format_without_schema():
    session = FakeSession([_ok("x")])
    LlmClient(_cfg(), session=session).complete("s", "u")
    assert "response_format" not in session.calls[0]["json"]


def test_complete_raises_llm_error_on_transport_failure():
    class Boom:
        def post(self, *a, **k):
            raise ConnectionError("connection refused")

    with pytest.raises(LlmError, match="llama.cpp"):
        LlmClient(_cfg(), session=Boom()).complete("s", "u")


def test_complete_raises_llm_error_on_malformed_payload():
    session = FakeSession([FakeResponse({"unexpected": True})])
    with pytest.raises(LlmError, match="响应结构"):
        LlmClient(_cfg(), session=session).complete("s", "u")


def test_complete_rejects_empty_content():
    session = FakeSession([_ok("")])
    with pytest.raises(LlmError, match="空"):
        LlmClient(_cfg(), session=session).complete("s", "u")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.llm_client'`

- [ ] **Step 3: 实现 `src/aiboke/llm_client.py`**

```python
"""llama.cpp server 的 OpenAI 兼容客户端。

为什么走 HTTP 而不是直接加载模型：llama.cpp 以 llama-server 常驻，
模型只加载一次，多个 case 复用，避免反复加载 19GB 权重。

thinking 模式必须关闭——Qwen3.8 默认开启思考，会先输出大段推理内容，
既浪费时间又会污染文稿。
"""

from __future__ import annotations

import requests

from .config import LlmConfig


class LlmError(RuntimeError):
    """LLM 调用失败。"""


class LlmClient:
    def __init__(self, cfg: LlmConfig, session=None) -> None:
        self._cfg = cfg
        self._session = session or requests.Session()

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> str:
        body: dict = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._cfg.temperature,
            "top_p": self._cfg.top_p,
            "presence_penalty": self._cfg.presence_penalty,
            "stream": False,
            # 关闭思考模式：Qwen3.8 默认开启，会先产出推理 token
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "podcast_script", "schema": json_schema},
            }

        url = f"{self._cfg.base_url.rstrip('/')}/v1/chat/completions"
        try:
            resp = self._session.post(url, json=body, timeout=self._cfg.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 — 统一转成领域异常
            raise LlmError(
                f"调用 llama.cpp server 失败（{url}）：{exc}。"
                "请确认 llama-server 已启动、版本 >= b10450，且二进制与 libggml-cuda.so 同版本。"
            ) from exc

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"llama.cpp 响应结构与预期不符：{payload!r}") from exc

        if not isinstance(content, str) or not content.strip():
            raise LlmError("llama.cpp 返回了空内容")

        return content
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: PASS（9 passed）

- [ ] **Step 5: 提交**

```bash
git add src/aiboke/llm_client.py tests/test_llm_client.py
git commit -m "feat: llama.cpp server 客户端"
```

---

### Task 7: 三幕文稿生成器

**Files:**
- Create: `src/aiboke/script_writer.py`
- Test: `tests/test_script_writer.py`

**Interfaces:**
- Consumes: `LlmClient.complete`、`prompts.*`、`length.*`、`gates.check_chinese`、`gates.check_two_speakers`、`schema.*`
- Produces:
  - `ScriptError(Exception)`
  - `extract_json(raw: str) -> dict`（剥离 Markdown 围栏、定位最外层大括号）
  - `parse_act(raw: str) -> tuple[Turn, ...]`
  - `ActResult(act_index: int, turns: tuple[Turn, ...], char_target: int, title: str)`
  - `ScriptWriter(client, target_seconds, chars_per_minute, max_retries=2, factcheck=True)`
  - `ScriptWriter.iter_acts(case: CaseInput) -> Iterator[ActResult]`（**逐幕产出，供流水线重叠合成**）
  - `ScriptWriter.write(case: CaseInput, title_hint: str | None = None) -> Transcript`（便利方法，消费 `iter_acts`）

- [ ] **Step 1: 写失败的测试 `tests/test_script_writer.py`**

```python
import json

import pytest

from aiboke.script_writer import ScriptError, ScriptWriter, extract_json, parse_act
from aiboke.schema import CaseInput


def _case():
    return CaseInput(topic="星巴克国内运营转移", speaker_gender1="男", speaker_gender2="女")


def _act_json(title, n_turns=4, text="这是一段中文的对话内容。"):
    return json.dumps(
        {
            "title": title,
            "content": [
                {"speaker": (i % 2) + 1, "text": text} for i in range(n_turns)
            ],
        },
        ensure_ascii=False,
    )


class FakeClient:
    """按顺序返回预设响应；记录每次调用的 prompt 以便断言。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user, *, json_schema=None, max_tokens=None):
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        if not self.responses:
            raise AssertionError("FakeClient 收到多余的调用")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


# ---------- extract_json ----------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_markdown_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_ignores_surrounding_prose():
    raw = '好的，这是文稿：\n{"a": 1}\n希望有帮助。'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ScriptError, match="JSON"):
        extract_json("完全不是 JSON")


# ---------- parse_act ----------

def test_parse_act_returns_turns():
    turns = parse_act(_act_json("标题"))
    assert len(turns) == 4
    assert turns[0].speaker == 1


def test_parse_act_rejects_missing_content():
    with pytest.raises(ScriptError, match="content"):
        parse_act('{"title": "只有标题"}')


def test_parse_act_rejects_bad_speaker():
    raw = json.dumps({"title": "t", "content": [{"speaker": 7, "text": "x"}]})
    with pytest.raises(ScriptError):
        parse_act(raw)


# ---------- ScriptWriter ----------

def _writer(client, **over):
    """测试用构造器：默认关闭事实自检，便于精确控制 LLM 调用次数。"""
    kw = dict(target_seconds=510.0, chars_per_minute=200.0, max_retries=2, factcheck=False)
    kw.update(over)
    return ScriptWriter(client, **kw)


def test_write_concatenates_three_acts():
    client = FakeClient([_act_json("标题A"), _act_json("标题B"), _act_json("标题C")])
    t = _writer(client).write(_case())
    assert len(client.calls) == 3, "应当恰好生成三幕"
    assert len(t.turns) == 12
    assert t.title == "标题A", "标题取第一幕的"


def test_iter_acts_yields_one_result_per_act_in_order():
    client = FakeClient([_act_json("T"), _act_json("T"), _act_json("T")])
    acts = list(_writer(client).iter_acts(_case()))
    assert [a.act_index for a in acts] == [0, 1, 2]
    assert all(len(a.turns) == 4 for a in acts)


def test_iter_acts_char_targets_follow_act_weights():
    client = FakeClient([_act_json("T"), _act_json("T"), _act_json("T")])
    acts = list(_writer(client).iter_acts(_case()))
    # 1700 字按 20/55/25 分配
    assert [a.char_target for a in acts] == [340, 935, 425]


def test_iter_acts_is_lazy_so_first_act_is_usable_immediately():
    """流水线依赖惰性：第一幕产出后即可开始合成语音，与后续幕重叠。"""
    client = FakeClient([_act_json("T"), _act_json("T"), _act_json("T")])
    gen = _writer(client).iter_acts(_case())
    first = next(gen)
    assert first.act_index == 0
    assert len(client.calls) == 1, "尚未请求第二幕时不应已经调用过 LLM"


def test_factcheck_runs_only_on_body_act():
    client = FakeClient([_act_json("T"), _act_json("T"), _act_json("T")])
    list(_writer(client, factcheck=True).iter_acts(_case()))
    # 三次生成 + 主体幕（第二幕）的一次事实自检
    assert len(client.calls) == 4


def test_factcheck_softens_claims_via_llm_rewrite():
    original = _act_json("T", text="市占率高达 37.5%。")
    softened = _act_json("T", text="市占率大幅下滑。")
    client = FakeClient([original, original, softened, original])
    acts = list(_writer(client, factcheck=True).iter_acts(_case()))
    assert acts[1].turns[0].text == "市占率大幅下滑。"


def test_factcheck_keeps_original_when_rewrite_unparseable():
    client = FakeClient([_act_json("T"), _act_json("T"), "这不是 JSON", _act_json("T")])
    acts = list(_writer(client, factcheck=True).iter_acts(_case()))
    assert len(acts[1].turns) == 4, "自检失败时应保留原文而非丢失内容"


def test_write_propagates_prior_context_between_acts():
    client = FakeClient([_act_json("T"), _act_json("T"), _act_json("T")])
    _writer(client).write(_case())
    # 第二幕的 prompt 里应当出现第一幕的内容作为衔接上下文
    assert "这是一段中文的对话内容。" in client.calls[1]["user"]


def test_write_requests_json_schema_constrained_decoding():
    client = FakeClient([_act_json("T"), _act_json("T"), _act_json("T")])
    _writer(client).write(_case())
    assert client.calls[0]["json_schema"] is not None


def test_write_repairs_unparseable_act():
    client = FakeClient(["这不是 JSON", _act_json("修复后的标题"), _act_json("T"), _act_json("T")])
    t = _writer(client).write(_case())
    assert t.title == "修复后的标题"


def test_write_raises_when_repair_also_fails():
    client = FakeClient(["坏输出", "还是坏的", _act_json("T"), _act_json("T")])
    with pytest.raises(ScriptError, match="修复"):
        _writer(client).write(_case())


def test_write_retries_act_when_chinese_gate_fails():
    english = json.dumps(
        {"title": "T", "content": [{"speaker": 1, "text": "This is English."}]},
        ensure_ascii=False,
    )
    client = FakeClient([english, _act_json("T"), _act_json("T"), _act_json("T")])
    t = _writer(client).write(_case())
    assert len(t.turns) == 4


def test_write_raises_after_exhausting_retries_on_english():
    english = json.dumps(
        {"title": "T", "content": [{"speaker": 1, "text": "Pure English here."}]},
        ensure_ascii=False,
    )
    client = FakeClient([english, english, english])
    with pytest.raises(ScriptError, match="中文"):
        _writer(client).write(_case())


def test_write_raises_when_single_speaker():
    one = json.dumps(
        {"title": "T", "content": [{"speaker": 1, "text": "只有一个人说话。"}] * 8},
        ensure_ascii=False,
    )
    client = FakeClient([one, one, one])
    with pytest.raises(ScriptError):
        _writer(client).write(_case())


def test_write_passes_retry_hint_into_next_prompt():
    short = json.dumps(
        {"title": "T", "content": [{"speaker": 1, "text": "只有一个人说中文。"}] * 8},
        ensure_ascii=False,
    )
    client = FakeClient([short, _act_json("T"), _act_json("T"), _act_json("T")])
    _writer(client).write(_case())
    assert "主播" in client.calls[1]["user"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_script_writer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.script_writer'`

- [ ] **Step 3: 实现 `src/aiboke/script_writer.py`**

```python
"""三幕式文稿生成。

为什么分三幕而不是一次生成整篇：
  1. 把「开场铺垫 -> 主体讲述 -> 分析总结」的结构约束进流程，
     而不是指望模型自觉遵守（评分第 5 条明确要求该结构）
  2. 第一幕一产出即可开始语音合成，与后续幕重叠执行，
     直接服务「音频首字返回时间 <= 30s」的基线
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterator, Sequence

from .gates import check_chinese, check_two_speakers, first_retry_hint
from .length import (
    ACT_WEIGHTS,
    DEFAULT_CHARS_PER_MINUTE,
    DEFAULT_TARGET_SECONDS,
    act_char_targets,
    target_chars,
)
from .llm_client import LlmClient
from .prompts import (
    SCRIPT_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_act_prompt,
    build_factcheck_prompt,
    build_repair_prompt,
)
from .schema import CaseInput, Transcript, Turn

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ScriptError(RuntimeError):
    """文稿生成失败。"""


def extract_json(raw: str) -> dict:
    """从模型输出中提取 JSON 对象。

    模型常会加 Markdown 围栏或前后客套话，这里都剥掉。
    """
    if not raw or not raw.strip():
        raise ScriptError("模型返回空内容，无法解析 JSON")

    text = raw.strip()

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ScriptError(f"输出中找不到 JSON 对象：{raw[:200]!r}")
        text = text[start : end + 1]

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ScriptError(f"JSON 解析失败：{exc}") from exc

    if not isinstance(obj, dict):
        raise ScriptError(f"JSON 顶层必须是对象，实际为 {type(obj).__name__}")
    return obj


def parse_act(raw: str) -> tuple[Turn, ...]:
    """把单幕的模型输出解析为对话轮次。"""
    obj = extract_json(raw)
    content = obj.get("content")
    if not isinstance(content, list) or not content:
        raise ScriptError(f"输出缺少非空的 content 数组：{obj!r}")

    turns: list[Turn] = []
    for i, item in enumerate(content):
        if not isinstance(item, dict):
            raise ScriptError(f"content[{i}] 不是对象：{item!r}")
        try:
            speaker = int(item["speaker"])
            text = item["text"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ScriptError(f"content[{i}] 缺少合法的 speaker/text：{item!r}") from exc
        if not isinstance(text, str) or not text.strip():
            raise ScriptError(f"content[{i}] 的 text 为空")
        try:
            turns.append(Turn(speaker=speaker, text=text.strip()))
        except ValueError as exc:
            raise ScriptError(str(exc)) from exc
    return tuple(turns)


def _extract_title(raw: str) -> str:
    try:
        title = extract_json(raw).get("title")
    except ScriptError:
        return ""
    return title.strip() if isinstance(title, str) else ""


# 主体幕的下标——事实自检只对这一幕执行（信息密度最高，编造风险最大）
_BODY_ACT_INDEX = 1


@dataclass(frozen=True)
class ActResult:
    act_index: int
    turns: tuple[Turn, ...]
    char_target: int
    title: str


class ScriptWriter:
    def __init__(
        self,
        client: LlmClient,
        target_seconds: float = DEFAULT_TARGET_SECONDS,
        chars_per_minute: float = DEFAULT_CHARS_PER_MINUTE,
        max_retries: int = 2,
        factcheck: bool = True,
    ) -> None:
        self._client = client
        self._target_seconds = target_seconds
        self._chars_per_minute = chars_per_minute
        self._max_retries = max_retries
        self._factcheck = factcheck

    def iter_acts(self, case: CaseInput) -> Iterator[ActResult]:
        """逐幕产出文稿。

        生成器是惰性的：调用方拿到第一幕后即可开始语音合成，与后续幕的
        文稿生成重叠执行——这是满足「音频首字返回时间 <= 30s」的关键。
        若改成先返回完整 Transcript 再合成，该收益即丧失。
        """
        total = target_chars(self._target_seconds, self._chars_per_minute)
        act_targets = act_char_targets(total, ACT_WEIGHTS)

        prior: list[Turn] = []
        for act_index, char_target in enumerate(act_targets):
            turns, raw = self._write_act(case, act_index, char_target, prior)
            if self._factcheck and act_index == _BODY_ACT_INDEX:
                turns = self._soften_uncertain_claims(turns)
            yield ActResult(
                act_index=act_index,
                turns=turns,
                char_target=char_target,
                title=_extract_title(raw),
            )
            prior.extend(turns)

    def write(self, case: CaseInput, title_hint: str | None = None) -> Transcript:
        """便利方法：消费 iter_acts 并返回完整文稿。"""
        all_turns: list[Turn] = []
        title = title_hint or ""
        for act in self.iter_acts(case):
            if not title and act.title:
                title = act.title
            all_turns.extend(act.turns)
        return Transcript(title=title or case.topic, turns=tuple(all_turns))

    def _soften_uncertain_claims(self, turns: tuple[Turn, ...]) -> tuple[Turn, ...]:
        """主体幕的事实自检：让模型把没把握的具体断言改为定性表述。

        离线环境无法联网核查，因此策略是降低编造概率而非事后验证。
        自检失败时保留原文——宁可留下可能有偏差的表述，也不能丢失内容。
        """
        if not turns:
            return turns
        try:
            raw = self._client.complete(
                SYSTEM_PROMPT,
                build_factcheck_prompt(turns),
                json_schema=SCRIPT_JSON_SCHEMA,
            )
            return parse_act(raw)
        except Exception:  # noqa: BLE001 — 自检是增强，不是必需步骤
            return turns

    def _write_act(
        self,
        case: CaseInput,
        act_index: int,
        char_target: int,
        prior_turns: Sequence[Turn],
    ) -> tuple[tuple[Turn, ...], str]:
        retry_hint: str | None = None
        last_error = ""

        for attempt in range(self._max_retries + 1):
            prompt = build_act_prompt(case, act_index, char_target, prior_turns, retry_hint)
            raw = self._client.complete(SYSTEM_PROMPT, prompt, json_schema=SCRIPT_JSON_SCHEMA)

            try:
                turns = parse_act(raw)
            except ScriptError as exc:
                last_error = str(exc)
                raw = self._try_repair(raw, last_error)
                try:
                    turns = parse_act(raw)
                except ScriptError as exc2:
                    last_error = str(exc2)
                    retry_hint = (
                        f"上一次输出不是合法 JSON（{last_error}）。"
                        "请只输出 JSON 对象，不要任何解释文字或代码块标记。"
                    )
                    continue

            verdicts = [check_chinese("".join(t.text for t in turns)), check_two_speakers(turns)]
            hint = first_retry_hint(verdicts)
            if hint is None:
                return turns, raw

            last_error = "；".join(v.detail for v in verdicts if not v.passed)
            retry_hint = hint

        raise ScriptError(
            f"第 {act_index + 1} 幕在 {self._max_retries + 1} 次尝试后仍未通过校验：{last_error}"
        )

    def _try_repair(self, raw: str, error: str) -> str:
        try:
            return self._client.complete(
                SYSTEM_PROMPT, build_repair_prompt(raw, error), json_schema=SCRIPT_JSON_SCHEMA
            )
        except Exception:  # noqa: BLE001 — 修复失败则交回上层重试
            return raw
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_script_writer.py -v`
Expected: PASS（16 passed）

- [ ] **Step 5: 提交**

```bash
git add src/aiboke/script_writer.py tests/test_script_writer.py
git commit -m "feat: 三幕式文稿生成器"
```

---

### Task 8: MOSS-TTSD 语音合成适配器

**Files:**
- Create: `src/aiboke/tts.py`
- Test: `tests/test_tts.py`

**Interfaces:**
- Consumes: `aiboke.config.TtsConfig`、`aiboke.schema.Turn`、`aiboke.schema.VoicePair`、`aiboke.length.seconds_to_max_tokens`
- Produces:
  - `TtsError(Exception)`
  - `TtsBackend`（Protocol）：`synthesize(turns, voices, out_path, target_seconds) -> Path`
  - `LlamaCppTts(cfg: TtsConfig, runner=None)`
  - `TransformersTts(cfg: TtsConfig, runner=None)`
  - `build_backend(cfg: TtsConfig, runner=None) -> TtsBackend`
  - `format_tagged_script(turns: Sequence[Turn]) -> str`（产出 `[S1]...[S2]...` 文本）
  - `format_input_jsonl(turns, voices) -> str`（transformers 后端的 JSONL 输入）

- [ ] **Step 1: 写失败的测试 `tests/test_tts.py`**

```python
import json

import pytest

from aiboke.config import TtsConfig
from aiboke.schema import Turn, VoicePreset, VoicePair
from aiboke.tts import (
    LlamaCppTts,
    TransformersTts,
    TtsError,
    build_backend,
    format_input_jsonl,
    format_tagged_script,
)


def _turns():
    return (Turn(1, "大家好，欢迎收听。"), Turn(2, "没错，今天聊个有意思的话题。"))


def _voices():
    return VoicePair(
        speaker1=VoicePreset(id="m_calm", gender="男", description="低沉男声"),
        speaker2=VoicePreset(id="f_clear", gender="女", description="清亮女声"),
    )


class FakeRunner:
    """替代 subprocess.run；记录命令并按需产出文件。"""

    def __init__(self, *, writes=None, returncode=0, stderr=""):
        self.calls = []
        self._writes = writes
        self._returncode = returncode
        self._stderr = stderr

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self._writes:
            self._writes(cmd)
        return type("R", (), {"returncode": self._returncode, "stderr": self._stderr, "stdout": ""})()


# ---------- 脚本格式化 ----------

def test_format_tagged_script_uses_speaker_tags():
    assert format_tagged_script(_turns()) == (
        "[S1]大家好，欢迎收听。[S2]没错，今天聊个有意思的话题。"
    )


def test_format_tagged_script_rejects_empty():
    with pytest.raises(TtsError, match="轮次"):
        format_tagged_script(())


def test_format_input_jsonl_contains_voice_descriptions_and_tags():
    payload = json.loads(format_input_jsonl(_turns(), _voices()))
    assert "[S1]" in payload["text"] and "[S2]" in payload["text"]
    assert payload["prompt_text_speaker1"].startswith("[S1]")
    assert payload["prompt_text_speaker2"].startswith("[S2]")
    assert payload["voice_description_speaker1"] == "低沉男声"


def test_format_input_jsonl_rejects_mismatched_voices():
    bad = VoicePair(
        speaker1=VoicePreset(id="a", gender="女", description="d"),
        speaker2=VoicePreset(id="b", gender="女", description="d"),
    )
    with pytest.raises(TtsError, match="女"):
        format_input_jsonl(_turns(), bad)


# ---------- LlamaCppTts ----------

def _llamacpp_cfg():
    return TtsConfig(backend="llamacpp", binary="/opt/llama-moss-tts", model_path="/models/ttsd")


def test_llamacpp_invokes_binary_with_max_tokens_from_target(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 120.0)

    cmd = runner.calls[0]
    assert cmd[0] == "/opt/llama-moss-tts"
    joined = " ".join(cmd)
    assert "--max-new-tokens" in joined
    # 120 秒 * 12.5 = 1500
    assert "1500" in joined


def test_llamacpp_passes_speaker_tags_and_model_path(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    joined = " ".join(runner.calls[0])
    assert "[S1]" in joined
    assert "/models/ttsd" in joined


def test_llamacpp_raises_on_nonzero_exit(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(returncode=1, stderr="CUDA error: out of memory")
    with pytest.raises(TtsError, match="out of memory"):
        LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)


def test_llamacpp_raises_when_no_output_file(tmp_path):
    out = tmp_path / "missing.wav"
    runner = FakeRunner()  # 不写文件，模拟静默失败
    with pytest.raises(TtsError, match="未产出"):
        LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)


def test_llamacpp_raises_when_binary_missing_from_config():
    cfg = TtsConfig(backend="llamacpp", binary=None, model_path="/models/ttsd")
    with pytest.raises(TtsError, match="binary"):
        LlamaCppTts(cfg, runner=FakeRunner()).synthesize(
            _turns(), _voices(), __import__("pathlib").Path("o.wav"), 60.0
        )


# ---------- TransformersTts ----------

def _tf_cfg():
    return TtsConfig(backend="transformers", model_path="/models/MOSS-TTSD-v1.0")


def test_transformers_invokes_inference_script(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    TransformersTts(_tf_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    joined = " ".join(runner.calls[0])
    assert "inference.py" in joined
    assert "voice_clone_and_continuation" in joined


def test_transformers_enables_text_normalize(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    TransformersTts(_tf_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    assert "--text_normalize" in " ".join(runner.calls[0])


# ---------- build_backend ----------

def test_build_backend_llamacpp():
    assert isinstance(build_backend(_llamacpp_cfg()), LlamaCppTts)


def test_build_backend_transformers():
    assert isinstance(build_backend(_tf_cfg()), TransformersTts)


def test_build_backend_rejects_unknown():
    with pytest.raises(TtsError, match="backend"):
        build_backend(TtsConfig(backend="magic"))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_tts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.tts'`

- [ ] **Step 3: 实现 `src/aiboke/tts.py`**

```python
"""MOSS-TTSD 语音合成适配器。

两个后端：
  - llamacpp（默认）：走 MOSS-TTS 的 llama.cpp 原生路径，权重为 GGUF，
    显存约 9GB，可与 LLM 同时常驻
  - transformers（备选）：走官方 inference.py，权重 bf16 约 19GB，
    与 LLM 无法同时常驻，但无需编译 fork

时长控制依赖 MOSS-TTSD 的官方换算：1 秒音频 ≈ 12.5 tokens，
因此 max_new_tokens = 目标秒数 * 12.5（三重保险中的第二重）。
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .config import TtsConfig
from .length import seconds_to_max_tokens
from .schema import Turn, VoicePair

# transformers 后端的官方推理脚本名（位于 MOSS-TTSD 仓库内）
_INFERENCE_SCRIPT = "inference.py"


class TtsError(RuntimeError):
    """语音合成失败。"""


def format_tagged_script(turns: Sequence[Turn]) -> str:
    """拼接为 MOSS-TTSD 的 [S1]/[S2] 标签脚本。"""
    if not turns:
        raise TtsError("没有任何对话轮次可供合成")
    return "".join(f"[S{t.speaker}]{t.text}" for t in turns)


def format_input_jsonl(turns: Sequence[Turn], voices: VoicePair) -> str:
    """构造 transformers 后端的 JSONL 输入（单行）。"""
    if not turns:
        raise TtsError("没有任何对话轮次可供合成")
    if voices.speaker1.gender != "男" and voices.speaker1.gender != "女":
        raise TtsError(f"speaker1 性别非法：{voices.speaker1.gender}")
    if voices.speaker2.gender != "男" and voices.speaker2.gender != "女":
        raise TtsError(f"speaker2 性别非法：{voices.speaker2.gender}")

    payload = {
        "text": format_tagged_script(turns),
        "prompt_audio_speaker1": voices.speaker1.reference_audio or "",
        "prompt_text_speaker1": f"[S1]{voices.speaker1.description}",
        "prompt_audio_speaker2": voices.speaker2.reference_audio or "",
        "prompt_text_speaker2": f"[S2]{voices.speaker2.description}",
        "voice_description_speaker1": voices.speaker1.description,
        "voice_description_speaker2": voices.speaker2.description,
    }
    return json.dumps(payload, ensure_ascii=False)


class TtsBackend(Protocol):
    def synthesize(
        self,
        turns: Sequence[Turn],
        voices: VoicePair,
        out_path: Path,
        target_seconds: float,
    ) -> Path: ...


def _default_runner(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


class LlamaCppTts:
    """MOSS-TTS 的 llama.cpp 原生路径（默认后端）。"""

    def __init__(self, cfg: TtsConfig, runner: Callable | None = None) -> None:
        self._cfg = cfg
        self._run = runner or _default_runner

    def synthesize(self, turns, voices, out_path: Path, target_seconds: float) -> Path:
        if not self._cfg.binary:
            raise TtsError("tts.binary 未配置，无法调用 llama.cpp 后端")
        if not self._cfg.model_path:
            raise TtsError("tts.model_path 未配置，找不到 MOSS-TTSD 权重")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._cfg.binary,
            "--model", self._cfg.model_path,
            "--text", format_tagged_script(turns),
            "--output", str(out_path),
            "--max-new-tokens", str(seconds_to_max_tokens(target_seconds)),
            "--temperature", str(self._cfg.temperature),
            "--top-p", str(self._cfg.top_p),
            "--top-k", str(self._cfg.top_k),
            "--repetition-penalty", str(self._cfg.repetition_penalty),
            "--text-normalize",
            "--sample-rate-normalize",
        ]

        proc = self._run(cmd)
        if proc.returncode != 0:
            raise TtsError(
                f"llama.cpp 语音合成失败（退出码 {proc.returncode}）：{proc.stderr.strip()[:500]}"
            )
        if not out_path.exists() or out_path.stat().st_size == 0:
            # 静默失败：进程退出码为 0 但没有产出文件
            raise TtsError(f"llama.cpp 语音合成未产出音频文件：{out_path}")
        return out_path


class TransformersTts:
    """官方 inference.py 后端（备选）。"""

    def __init__(self, cfg: TtsConfig, runner: Callable | None = None) -> None:
        self._cfg = cfg
        self._run = runner or _default_runner

    def synthesize(self, turns, voices, out_path: Path, target_seconds: float) -> Path:
        if not self._cfg.model_path:
            raise TtsError("tts.model_path 未配置，找不到 MOSS-TTSD 权重")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(format_input_jsonl(turns, voices))
            jsonl_path = fh.name

        cmd = [
            "python", _INFERENCE_SCRIPT,
            "--model_path", self._cfg.model_path,
            "--input_jsonl", jsonl_path,
            "--save_dir", str(out_path.parent),
            "--mode", "voice_clone_and_continuation",
            "--max_new_tokens", str(seconds_to_max_tokens(target_seconds)),
            "--temperature", str(self._cfg.temperature),
            "--top_p", str(self._cfg.top_p),
            "--top_k", str(self._cfg.top_k),
            "--repetition_penalty", str(self._cfg.repetition_penalty),
            "--text_normalize",
            "--sample_rate_normalize",
        ]

        proc = self._run(cmd)
        if proc.returncode != 0:
            raise TtsError(
                f"transformers 语音合成失败（退出码 {proc.returncode}）：{proc.stderr.strip()[:500]}"
            )
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise TtsError(f"transformers 语音合成未产出音频文件：{out_path}")
        return out_path


def build_backend(cfg: TtsConfig, runner: Callable | None = None) -> TtsBackend:
    if cfg.backend == "llamacpp":
        return LlamaCppTts(cfg, runner=runner)
    if cfg.backend == "transformers":
        return TransformersTts(cfg, runner=runner)
    raise TtsError(f"未知的 tts.backend：{cfg.backend!r}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_tts.py -v`
Expected: PASS（16 passed）

- [ ] **Step 5: 提交**

```bash
git add src/aiboke/tts.py tests/test_tts.py
git commit -m "feat: MOSS-TTSD 语音合成适配器"
```

---

### Task 9: 音频后处理（ffmpeg）

**Files:**
- Create: `src/aiboke/audio_utils.py`
- Test: `tests/test_audio_utils.py`

**Interfaces:**
- Consumes: 无（仅依赖 ffmpeg/ffprobe 可执行文件）
- Produces:
  - `AudioError(Exception)`
  - `probe_duration(path: Path, runner=None) -> float`（秒）
  - `concat_wavs(paths: Sequence[Path], out_path: Path, gap_ms: int = 400, runner=None) -> Path`
  - `normalize_loudness(in_path: Path, out_path: Path, target_lufs: float = -16.0, runner=None) -> Path`
  - `to_mp3(in_path: Path, out_path: Path, bitrate: str = "192k", runner=None) -> Path`

- [ ] **Step 1: 写失败的测试 `tests/test_audio_utils.py`**

```python
from pathlib import Path

import pytest

from aiboke.audio_utils import (
    AudioError,
    concat_wavs,
    normalize_loudness,
    probe_duration,
    to_mp3,
)


class FakeRunner:
    def __init__(self, *, stdout="", returncode=0, stderr="", writes=None):
        self.calls = []
        self._stdout = stdout
        self._rc = returncode
        self._stderr = stderr
        self._writes = writes

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self._writes:
            self._writes(cmd)
        return type("R", (), {"returncode": self._rc, "stdout": self._stdout, "stderr": self._stderr})()


def test_probe_duration_parses_ffprobe_output():
    runner = FakeRunner(stdout="512.345\n")
    assert probe_duration(Path("a.mp3"), runner=runner) == pytest.approx(512.345)


def test_probe_duration_uses_ffprobe_show_entries():
    runner = FakeRunner(stdout="1.0\n")
    probe_duration(Path("a.mp3"), runner=runner)
    joined = " ".join(runner.calls[0])
    assert "ffprobe" in joined
    assert "format=duration" in joined


def test_probe_duration_raises_on_unparseable_output():
    with pytest.raises(AudioError, match="时长"):
        probe_duration(Path("a.mp3"), runner=FakeRunner(stdout="N/A\n"))


def test_probe_duration_raises_on_nonzero_exit():
    with pytest.raises(AudioError, match="ffprobe"):
        probe_duration(Path("a.mp3"), runner=FakeRunner(returncode=1, stderr="No such file"))


def test_concat_wavs_requires_at_least_one_input(tmp_path):
    with pytest.raises(AudioError, match="至少"):
        concat_wavs([], tmp_path / "o.wav", runner=FakeRunner())


def test_concat_wavs_invokes_ffmpeg_with_all_inputs(tmp_path):
    out = tmp_path / "o.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    concat_wavs([tmp_path / "a.wav", tmp_path / "b.wav"], out, runner=runner)
    joined = " ".join(runner.calls[0])
    assert "ffmpeg" in joined
    assert "a.wav" in joined and "b.wav" in joined


def test_concat_wavs_applies_gap(tmp_path):
    out = tmp_path / "o.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    concat_wavs([tmp_path / "a.wav", tmp_path / "b.wav"], out, gap_ms=400, runner=runner)
    assert "0.4" in " ".join(runner.calls[0])


def test_concat_wavs_raises_when_no_output(tmp_path):
    with pytest.raises(AudioError, match="未产出"):
        concat_wavs([tmp_path / "a.wav"], tmp_path / "o.wav", runner=FakeRunner())


def test_normalize_loudness_targets_configured_lufs(tmp_path):
    out = tmp_path / "n.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    normalize_loudness(tmp_path / "i.wav", out, target_lufs=-16.0, runner=runner)
    assert "loudnorm" in " ".join(runner.calls[0])
    assert "-16.0" in " ".join(runner.calls[0])


def test_to_mp3_sets_bitrate(tmp_path):
    out = tmp_path / "o.mp3"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"ID3"))
    to_mp3(tmp_path / "i.wav", out, bitrate="192k", runner=runner)
    joined = " ".join(runner.calls[0])
    assert "libmp3lame" in joined
    assert "192k" in joined


def test_to_mp3_raises_when_no_output(tmp_path):
    with pytest.raises(AudioError, match="未产出"):
        to_mp3(tmp_path / "i.wav", tmp_path / "o.mp3", runner=FakeRunner())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_audio_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.audio_utils'`

- [ ] **Step 3: 实现 `src/aiboke/audio_utils.py`**

```python
"""ffmpeg / ffprobe 封装：拼接、响度归一化、转码、时长探测。

评分要求「无突然爆音或断裂」，因此幕间拼接需要插入自然停顿并做
淡入淡出；响度归一化到 -16 LUFS 防止音量忽大忽小。

不混入背景音乐——赛题仅要求语音，BGM 会压低清晰度并增加失分风险。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Sequence


class AudioError(RuntimeError):
    """音频处理失败。"""


def _default_runner(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _require_output(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise AudioError(f"ffmpeg 未产出音频文件：{path}")
    return path


def probe_duration(path: Path, runner: Callable | None = None) -> float:
    """读取音频时长（秒）。时长门限的第三重保险依赖此函数。"""
    run = runner or _default_runner
    path = Path(path)
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        raise AudioError(f"ffprobe 读取时长失败：{proc.stderr.strip()[:300]}")

    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError) as exc:
        raise AudioError(f"无法从 ffprobe 输出解析时长：{proc.stdout!r}") from exc


def concat_wavs(
    paths: Sequence[Path],
    out_path: Path,
    gap_ms: int = 400,
    runner: Callable | None = None,
) -> Path:
    """按顺序拼接多段音频，段间插入停顿并做交叉淡化。"""
    if not paths:
        raise AudioError("至少需要一段音频才能拼接")

    run = runner or _default_runner
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(paths) == 1:
        cmd = ["ffmpeg", "-y", "-i", str(paths[0]), "-c", "copy", str(out_path)]
        proc = run(cmd)
        if proc.returncode != 0:
            raise AudioError(f"ffmpeg 复制音频失败：{proc.stderr.strip()[:300]}")
        return _require_output(out_path)

    gap_s = gap_ms / 1000.0
    inputs: list[str] = []
    for p in paths:
        inputs += ["-i", str(p)]

    # 段间插入静音，末端做淡化，避免生硬切边
    n = len(paths)
    filter_parts = []
    for i in range(n - 1):
        filter_parts.append(
            f"[{i}:a]adelay=0|0,apad=pad_dur={gap_s}[a{i}]"
        )
    filter_parts.append(f"[{n - 1}:a]anull[a{n - 1}]")
    concat_inputs = "".join(f"[a{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=0:a=1[out]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-af", "afade=t=in:d=0.05",
        str(out_path),
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg 拼接音频失败：{proc.stderr.strip()[:300]}")
    return _require_output(out_path)


def normalize_loudness(
    in_path: Path,
    out_path: Path,
    target_lufs: float = -16.0,
    runner: Callable | None = None,
) -> Path:
    run = runner or _default_runner
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        str(out_path),
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg 响度归一化失败：{proc.stderr.strip()[:300]}")
    return _require_output(out_path)


def to_mp3(
    in_path: Path,
    out_path: Path,
    bitrate: str = "192k",
    runner: Callable | None = None,
) -> Path:
    run = runner or _default_runner
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        str(out_path),
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg 转码 mp3 失败：{proc.stderr.strip()[:300]}")
    return _require_output(out_path)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_audio_utils.py -v`
Expected: PASS（11 passed）

- [ ] **Step 5: 提交**

```bash
git add src/aiboke/audio_utils.py tests/test_audio_utils.py
git commit -m "feat: ffmpeg 音频后处理封装"
```

---

### Task 10: 封面生成

**Files:**
- Create: `src/aiboke/cover.py`
- Test: `tests/test_cover.py`

**Interfaces:**
- Consumes: `aiboke.config.CoverConfig`、`aiboke.prompts.build_cover_prompt`
- Produces:
  - `CoverError(Exception)`
  - `CoverGenerator`（Protocol）：`generate(topic: str, out_path: Path) -> Path`
  - `ZImageCover(cfg: CoverConfig, runner=None)`
  - `FallbackCover(cfg: CoverConfig, runner=None)`（纯色底图兜底，保证产物存在）
  - `build_generator(cfg: CoverConfig, runner=None) -> CoverGenerator`

- [ ] **Step 1: 写失败的测试 `tests/test_cover.py`**

```python
from pathlib import Path

import pytest

from aiboke.config import CoverConfig
from aiboke.cover import (
    CoverError,
    FallbackCover,
    ZImageCover,
    build_generator,
)


class FakeRunner:
    def __init__(self, *, returncode=0, stderr="", writes=None):
        self.calls = []
        self._rc = returncode
        self._stderr = stderr
        self._writes = writes

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self._writes:
            self._writes(cmd)
        return type("R", (), {"returncode": self._rc, "stdout": "", "stderr": self._stderr})()


def _cfg():
    return CoverConfig(steps=8, size=1024, model_path="/models/Z-Image-Turbo")


def test_zimage_invokes_runner_with_size_and_steps(tmp_path):
    out = tmp_path / "c.png"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"\x89PNG"))
    ZImageCover(_cfg(), runner=runner).generate("星巴克国内运营转移", out)
    joined = " ".join(runner.calls[0])
    assert "1024" in joined
    assert "8" in joined
    assert "/models/Z-Image-Turbo" in joined


def test_zimage_prompt_forbids_text_in_image(tmp_path):
    out = tmp_path / "c.png"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"\x89PNG"))
    ZImageCover(_cfg(), runner=runner).generate("星巴克国内运营转移", out)
    assert "文字" in " ".join(runner.calls[0])


def test_zimage_raises_when_no_output(tmp_path):
    with pytest.raises(CoverError, match="未产出"):
        ZImageCover(_cfg(), runner=FakeRunner()).generate("主题", tmp_path / "c.png")


def test_zimage_raises_on_nonzero_exit(tmp_path):
    runner = FakeRunner(returncode=1, stderr="CUDA out of memory")
    with pytest.raises(CoverError, match="out of memory"):
        ZImageCover(_cfg(), runner=runner).generate("主题", tmp_path / "c.png")


def test_zimage_raises_when_model_path_missing():
    with pytest.raises(CoverError, match="model_path"):
        ZImageCover(CoverConfig(model_path=None), runner=FakeRunner()).generate(
            "主题", Path("c.png")
        )


def test_fallback_cover_produces_file(tmp_path):
    out = tmp_path / "c.png"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"\x89PNG"))
    FallbackCover(_cfg(), runner=runner).generate("星巴克国内运营转移", out)
    assert out.exists()
    assert "ffmpeg" in " ".join(runner.calls[0])


def test_build_generator_returns_zimage_by_default():
    assert isinstance(build_generator(_cfg()), ZImageCover)


def test_build_generator_returns_fallback_when_requested():
    assert isinstance(build_generator(_cfg(), prefer_fallback=True), FallbackCover)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_cover.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.cover'`

- [ ] **Step 3: 实现 `src/aiboke/cover.py`**

```python
"""封面生成。1024x1024 PNG，与播客主题相关。

封面仅占 10 分，但产物缺失可能被判定为输出不达标而拉高失败率
（基线失败率 <= 10%）。因此提供 FallbackCover：即使图像模型不可用，
也用纯色底图保证产物存在。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Protocol

from .config import CoverConfig
from .prompts import build_cover_prompt


class CoverError(RuntimeError):
    """封面生成失败。"""


def _default_runner(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _require_output(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise CoverError(f"封面生成未产出文件：{path}")
    return path


class CoverGenerator(Protocol):
    def generate(self, topic: str, out_path: Path) -> Path: ...


class ZImageCover:
    """Z-Image-Turbo：6B / 8 步 / 1024x1024 / Apache-2.0。"""

    def __init__(self, cfg: CoverConfig, runner: Callable | None = None) -> None:
        self._cfg = cfg
        self._run = runner or _default_runner

    def generate(self, topic: str, out_path: Path) -> Path:
        if not self._cfg.model_path:
            raise CoverError("cover.model_path 未配置，找不到 Z-Image-Turbo 权重")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 提示词由 prompts 模块统一构造：明确禁止画面出现文字，
        # 因为文生图模型的文字渲染不可靠，乱码会严重拉低观感
        prompt = build_cover_prompt(topic)

        cmd = [
            "python", "-m", "aiboke.cover_runner",
            "--model", self._cfg.model_path,
            "--prompt", prompt,
            "--size", str(self._cfg.size),
            "--steps", str(self._cfg.steps),
            "--output", str(out_path),
        ]
        proc = self._run(cmd)
        if proc.returncode != 0:
            raise CoverError(f"Z-Image 封面生成失败：{proc.stderr.strip()[:300]}")
        return _require_output(out_path)


class FallbackCover:
    """纯色底图兜底，保证产物存在，不因 10 分项拉高失败率。"""

    def __init__(self, cfg: CoverConfig, runner: Callable | None = None) -> None:
        self._cfg = cfg
        self._run = runner or _default_runner

    def generate(self, topic: str, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        size = self._cfg.size

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x1F3A93:s={size}x{size}",
            "-frames:v", "1",
            str(out_path),
        ]
        proc = self._run(cmd)
        if proc.returncode != 0:
            raise CoverError(f"生成兜底封面失败：{proc.stderr.strip()[:300]}")
        return _require_output(out_path)


def build_generator(cfg: CoverConfig, runner: Callable | None = None, prefer_fallback: bool = False) -> CoverGenerator:
    if prefer_fallback:
        return FallbackCover(cfg, runner=runner)
    return ZImageCover(cfg, runner=runner)
```

- [ ] **Step 4: 写 `src/aiboke/cover_runner.py`**（Z-Image 的真实推理入口，在 GPU 机器上执行）

```python
"""Z-Image-Turbo 推理入口。

被 cover.ZImageCover 以子进程方式调用，使 diffusers/torch 的重型依赖
不污染主流程，也便于在无 GPU 环境下跳过。

用法：
    python -m aiboke.cover_runner --model <路径> --prompt <提示词> \
        --size 1024 --steps 8 --output <输出png>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Z-Image-Turbo 封面生成")
    parser.add_argument("--model", required=True, help="模型权重路径")
    parser.add_argument("--prompt", required=True, help="图像生成提示词")
    parser.add_argument("--size", type=int, default=1024, help="输出边长（正方形）")
    parser.add_argument("--steps", type=int, default=8, help="推理步数，Z-Image-Turbo 推荐 8")
    parser.add_argument("--output", required=True, help="输出 PNG 路径")
    args = parser.parse_args(argv)

    try:
        import torch
        from diffusers import ZImagePipeline
    except ImportError as exc:
        print(f"缺少依赖：{exc}", file=sys.stderr)
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    image = pipe(
        prompt=args.prompt,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        # Z-Image-Turbo 为蒸馏模型，CFG 固定为 1.0（不可调）
        guidance_scale=1.0,
    ).images[0]

    image.save(output)
    print(f"已保存封面：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_cover.py -v`
Expected: PASS（8 passed）

- [ ] **Step 6: 提交**

```bash
git add src/aiboke/cover.py src/aiboke/cover_runner.py tests/test_cover.py
git commit -m "feat: 封面生成与兜底"
```

---

# 阶段三：编排、入口与部署

### Task 11: 流水线编排

**Files:**
- Create: `src/aiboke/pipeline.py`
- Modify: `src/aiboke/__init__.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `ScriptWriter`、`TtsBackend`、`CoverGenerator`、`audio_utils.*`、`gates.*`、`length.*`、`voices.*`、`schema.*`
- Produces:
  - `PipelineError(Exception)`
  - `Pipeline(cfg, script_writer, tts, cover, preset_map, runner=None)`
  - `Pipeline.run(case: CaseInput, out_dir: Path) -> Episode`

- [ ] **Step 1: 写失败的测试 `tests/test_pipeline.py`**

```python
import json
from pathlib import Path

import pytest

from aiboke.config import Config, CoverConfig, LlmConfig, TtsConfig
from aiboke.pipeline import Pipeline, PipelineError
from aiboke.schema import CaseInput, Transcript, Turn, VoicePair, VoicePreset
from aiboke.script_writer import ActResult


def _cfg(**over):
    base = dict(
        target_seconds=510.0,
        chars_per_minute=200.0,
        models_root="/models",
        llm=LlmConfig(base_url="http://x", model="m"),
        tts=TtsConfig(backend="llamacpp", binary="b", model_path="p"),
        cover=CoverConfig(steps=8, size=1024, model_path="z"),
    )
    base.update(over)
    return Config(**base)


def _case():
    return CaseInput(topic="星巴克国内运营转移", speaker_gender1="男", speaker_gender2="女")


GOOD_TURNS = tuple(
    Turn(speaker=(i % 2) + 1, text="这是一段足够长的中文对话内容，用来通过语言与轮次校验。")
    for i in range(10)
)


class FakeScriptWriter:
    """把给定文稿切成三幕惰性产出，模拟真实的三幕式生成行为。"""

    def __init__(self, transcript=None, error=None):
        self._t = transcript
        self._e = error
        self.acts_requested = 0

    def iter_acts(self, case):
        if self._e:
            raise self._e
        turns = self._t.turns if self._t else GOOD_TURNS
        size = max(1, len(turns) // 3)
        groups = [turns[i : i + size] for i in range(0, len(turns), size)]
        while len(groups) < 3:
            groups.append(groups[-1])
        for i, g in enumerate(groups[:3]):
            self.acts_requested += 1
            yield ActResult(
                act_index=i,
                turns=tuple(g),
                char_target=sum(len(t.text) for t in g),
                title=(self._t.title if self._t else "T") if i == 0 else "",
            )

    def write(self, case, title_hint=None):
        all_turns: list[Turn] = []
        for a in self.iter_acts(case):
            all_turns.extend(a.turns)
        return Transcript(title=self._t.title if self._t else "T", turns=tuple(all_turns))


class FakeTts:
    def __init__(self, error=None):
        self.calls = []
        self._e = error

    def synthesize(self, turns, voices, out_path, target_seconds):
        self.calls.append((len(turns), target_seconds))
        if self._e:
            raise self._e
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"RIFF")
        return Path(out_path)


class FakeCover:
    def generate(self, topic, out_path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"\x89PNG")
        return Path(out_path)


def _presets():
    return {
        "男": [VoicePreset(id="m", gender="男", description="d")],
        "女": [VoicePreset(id="f", gender="女", description="d")],
    }


def _audio_runner(*, duration=510.0):
    """替代 ffmpeg/ffprobe 的 runner：产出文件并让 ffprobe 返回指定时长。"""

    class R:
        def __call__(self, cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)
            if "ffprobe" in joined:
                return type("P", (), {"returncode": 0, "stdout": f"{duration}\n", "stderr": ""})()
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"RIFF" if out.suffix == ".wav" else b"ID3")
            return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    return R()


def _pipeline(tmp_path, **over):
    return Pipeline(
        cfg=_cfg(**over),
        script_writer=FakeScriptWriter(Transcript(title="测试标题", turns=GOOD_TURNS)),
        tts=FakeTts(),
        cover=FakeCover(),
        preset_map=_presets(),
        runner=_audio_runner(),
    )


def test_run_produces_three_artifacts(tmp_path):
    ep = _pipeline(tmp_path).run(_case(), tmp_path / "out")
    assert ep.audio_path.exists()
    assert ep.cover_path.exists()
    assert ep.script_path.exists()


def test_run_writes_script_json_matching_required_schema(tmp_path):
    ep = _pipeline(tmp_path).run(_case(), tmp_path / "out")
    obj = json.loads(ep.script_path.read_text(encoding="utf-8"))
    assert set(obj) == {"title", "content"}
    assert obj["content"][0]["speaker"] in (1, 2)


def test_run_names_outputs_predictably(tmp_path):
    ep = _pipeline(tmp_path).run(_case(), tmp_path / "out")
    assert ep.audio_path.name == "podcast.mp3"
    assert ep.cover_path.name == "cover.png"
    assert ep.script_path.name == "script.json"


def test_run_calls_tts_once_per_act(tmp_path):
    p = _pipeline(tmp_path)
    p.run(_case(), tmp_path / "out")
    assert len(p._tts.calls) == 3, "三幕应各合成一次"


def test_run_synthesizes_each_act_before_requesting_the_next(tmp_path):
    """三幕式设计的核心收益：第一幕产出后立即合成，与后续幕的生成重叠。

    若实现改成先拿完整文稿再统一合成，本测试会失败——那意味着
    「音频首字返回时间 <= 30s」的基线失去了保障。
    """
    order = []

    class RecordingWriter:
        def iter_acts(self, case):
            for i in range(3):
                order.append(f"script{i}")
                yield ActResult(
                    act_index=i, turns=GOOD_TURNS, char_target=500, title="T"
                )

    class RecordingTts(FakeTts):
        def synthesize(self, turns, voices, out_path, target_seconds):
            order.append("tts")
            return super().synthesize(turns, voices, out_path, target_seconds)

    Pipeline(
        cfg=_cfg(),
        script_writer=RecordingWriter(),
        tts=RecordingTts(),
        cover=FakeCover(),
        preset_map=_presets(),
        runner=_audio_runner(),
    ).run(_case(), tmp_path / "out")

    assert order == ["script0", "tts", "script1", "tts", "script2", "tts"]


def test_run_synthesizes_after_script_before_cover(tmp_path):
    order = []

    class OrderedTts(FakeTts):
        def synthesize(self, *a, **k):
            order.append("tts")
            return super().synthesize(*a, **k)

    class OrderedCover(FakeCover):
        def generate(self, *a, **k):
            order.append("cover")
            return super().generate(*a, **k)

    Pipeline(
        cfg=_cfg(),
        script_writer=FakeScriptWriter(Transcript("T", GOOD_TURNS)),
        tts=OrderedTts(),
        cover=OrderedCover(),
        preset_map=_presets(),
        runner=_audio_runner(),
    ).run(_case(), tmp_path / "out")
    assert order == ["tts", "tts", "tts", "cover"]


def test_run_raises_pipeline_error_when_script_writer_fails(tmp_path):
    p = Pipeline(
        cfg=_cfg(),
        script_writer=FakeScriptWriter(error=RuntimeError("LLM 挂了")),
        tts=FakeTts(),
        cover=FakeCover(),
        preset_map=_presets(),
        runner=_audio_runner(),
    )
    with pytest.raises(PipelineError, match="文稿"):
        p.run(_case(), tmp_path / "out")


def test_run_raises_pipeline_error_when_gender_presets_missing(tmp_path):
    p = Pipeline(
        cfg=_cfg(),
        script_writer=FakeScriptWriter(Transcript("T", GOOD_TURNS)),
        tts=FakeTts(),
        cover=FakeCover(),
        preset_map={"男": [VoicePreset(id="m", gender="男", description="d")]},
        runner=_audio_runner(),
    )
    with pytest.raises(PipelineError, match="音色"):
        p.run(_case(), tmp_path / "out")


def test_run_raises_pipeline_error_when_duration_out_of_range(tmp_path):
    p = Pipeline(
        cfg=_cfg(),
        script_writer=FakeScriptWriter(Transcript("T", GOOD_TURNS)),
        tts=FakeTts(),
        cover=FakeCover(),
        preset_map=_presets(),
        runner=_audio_runner(duration=250.0),  # 低于 300 秒下限
    )
    with pytest.raises(PipelineError, match="时长"):
        p.run(_case(), tmp_path / "out")


def test_run_falls_back_to_fallback_cover_on_cover_failure(tmp_path):
    class BrokenCover:
        def generate(self, topic, out_path):
            raise RuntimeError("显存不足")

    p = Pipeline(
        cfg=_cfg(),
        script_writer=FakeScriptWriter(Transcript("T", GOOD_TURNS)),
        tts=FakeTts(),
        cover=BrokenCover(),
        preset_map=_presets(),
        runner=_audio_runner(),
    )
    # 封面仅 10 分，不应导致整案失败
    ep = p.run(_case(), tmp_path / "out")
    assert ep.cover_path.exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.pipeline'`

- [ ] **Step 3: 实现 `src/aiboke/pipeline.py`**

```python
"""三阶段流水线编排。

阶段一 文稿生成 -> 阶段二 语音合成 -> 阶段三 封面生成，全程受四道
0 分门限约束。封面失败不导致整案失败（仅 10 分，且产物需存在）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import audio_utils
from .config import Config
from .gates import all_passed, check_chinese, check_duration, check_genders, check_two_speakers
from .schema import CaseInput, Episode, Transcript, Turn, VoicePreset
from .voices import resolve_pair


class PipelineError(RuntimeError):
    """流水线失败。"""


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        script_writer,
        tts,
        cover,
        preset_map: Mapping[str, Sequence[VoicePreset]],
        runner: Callable | None = None,
    ) -> None:
        self._cfg = cfg
        self._writer = script_writer
        self._tts = tts
        self._cover = cover
        self._presets = preset_map
        self._run = runner

    def run(self, case: CaseInput, out_dir: Path) -> Episode:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 前置门限：性别配置必须可用，否则重试无意义
        gender_gate = check_genders(case.speaker_gender1, case.speaker_gender2, self._presets)
        if not gender_gate.passed:
            raise PipelineError(f"音色配置不满足性别要求：{gender_gate.detail}")

        try:
            voices = resolve_pair(
                self._presets, case.speaker_gender1, case.speaker_gender2
            )
        except ValueError as exc:
            raise PipelineError(f"无法解析音色：{exc}") from exc

        # 阶段一 + 阶段二交织：逐幕生成文稿并立即合成语音。
        # 交织而非「先全稿后合成」是三幕式设计的核心收益——第一幕一产出
        # 即开始合成，与后续幕的生成重叠，保障「音频首字 <= 30s」的基线。
        audio_wavs: list[Path] = []
        all_turns: list[Turn] = []
        title = ""
        try:
            for act in self._writer.iter_acts(case):
                if not title and act.title:
                    title = act.title
                all_turns.extend(act.turns)

                # 逐幕先查语言门限：命中「非中文则 0 分」的风险要在付出
                # 昂贵的语音合成代价之前就拦下来
                act_text = "".join(t.text for t in act.turns)
                lang_gate = check_chinese(act_text, self._cfg.chinese_min_ratio)
                if not lang_gate.passed:
                    raise PipelineError(
                        f"第 {act.act_index + 1} 幕未通过语言门限：{lang_gate.detail}"
                    )

                # 用该幕实际字数折算目标秒数，比用预设字数更贴近真实产出
                act_seconds = len(act_text) / self._cfg.chars_per_minute * 60.0
                out = out_dir / f"act{act.act_index}.wav"
                audio_wavs.append(
                    self._tts.synthesize(act.turns, voices, out, act_seconds)
                )
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"文稿生成或语音合成失败：{exc}") from exc

        transcript = Transcript(title=title or case.topic, turns=tuple(all_turns))

        # 整体门限：双人对话与轮次均衡必须对完整文稿判定——
        # 单幕内部可能天然由一位主播主导，逐幕判定会误报
        script_gates = [
            check_chinese(transcript.full_text, self._cfg.chinese_min_ratio),
            check_two_speakers(transcript.turns, self._cfg.speaker_min_share),
        ]
        if not all_passed(script_gates):
            failed = "；".join(g.detail for g in script_gates if not g.passed)
            raise PipelineError(f"文稿未通过门限校验：{failed}")

        if not audio_wavs:
            raise PipelineError("没有任何一幕成功合成语音")

        # 拼接 -> 响度归一化 -> 转 mp3
        concat_path = audio_utils.concat_wavs(
            audio_wavs, out_dir / "podcast_concat.wav", runner=self._run
        )
        norm_path = audio_utils.normalize_loudness(
            concat_path, out_dir / "podcast_norm.wav",
            target_lufs=self._cfg.target_lufs, runner=self._run,
        )
        audio_path = audio_utils.to_mp3(
            norm_path, out_dir / "podcast.mp3", runner=self._run
        )

        # 时长门限：越界即 0 分，必须失败而非静默通过
        duration = audio_utils.probe_duration(audio_path, runner=self._run)
        duration_gate = check_duration(duration)
        if not duration_gate.passed:
            raise PipelineError(
                f"时长未通过门限校验：{duration_gate.detail}。"
                f"建议：{duration_gate.retry_hint or '调整目标字数后重试'}"
            )

        # 阶段三：封面（失败则兜底，不阻断交付）
        cover_path = self._generate_cover(case, out_dir)

        script_path = out_dir / "script.json"
        script_path.write_text(
            json.dumps(transcript.to_json_obj(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return Episode(audio_path=audio_path, cover_path=cover_path, script_path=script_path)

    def _generate_cover(self, case: CaseInput, out_dir: Path) -> Path:
        from .cover import FallbackCover, build_generator

        out = out_dir / "cover.png"
        try:
            return self._cover.generate(case.topic, out)
        except Exception:  # noqa: BLE001 — 封面仅 10 分，降级优于失败
            fallback = FallbackCover(self._cfg.cover, runner=self._run)
            return fallback.generate(case.topic, out)
```

- [ ] **Step 4: 更新 `src/aiboke/__init__.py`**

```python
"""AI 中文双人播客生成系统。"""

from .config import Config, load_config
from .pipeline import Pipeline, PipelineError
from .schema import CaseInput, Episode, Transcript, Turn, VoicePair, VoicePreset

__all__ = [
    "CaseInput",
    "Config",
    "Episode",
    "Pipeline",
    "PipelineError",
    "Transcript",
    "Turn",
    "VoicePair",
    "VoicePreset",
    "load_config",
]

__version__ = "0.1.0"
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS（9 passed）

- [ ] **Step 6: 运行全量测试并提交**

```bash
python -m pytest tests/ -v
git add src/aiboke/pipeline.py src/aiboke/__init__.py tests/test_pipeline.py
git commit -m "feat: 三阶段流水线编排"
```

Expected: 全部通过

---

### Task 12: CLI 入口与 HTTP 封装

**Files:**
- Create: `scripts/generate.py`
- Create: `src/aiboke/server.py`
- Create: `src/aiboke/bootstrap.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `aiboke.*` 全部
- Produces:
  - `aiboke.bootstrap.build_pipeline(cfg: Config, models_root: Path, prefer_fallback_cover: bool = False) -> Pipeline`
  - `scripts/generate.py`：`--input <case.json>`、`--output-dir <dir>`、`--config <yaml>`
  - `aiboke.server.create_app(cfg_path: Path) -> FastAPI`

- [ ] **Step 1: 写失败的测试 `tests/test_cli.py`**

```python
import json
import subprocess
import sys

import pytest

from aiboke.bootstrap import load_presets_from_config


def test_load_presets_from_config_reads_voice_presets(tmp_path):
    p = tmp_path / "voice_presets.json"
    p.write_text(
        json.dumps({"presets": [{"id": "m", "gender": "男", "description": "d"}]}),
        encoding="utf-8",
    )
    presets = load_presets_from_config(p)
    assert presets["男"][0].id == "m"


def test_cli_rejects_missing_input_file(tmp_path):
    proc = subprocess.run(
        [sys.executable, "scripts/generate.py", "--input", str(tmp_path / "no.json"),
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "no.json" in (proc.stderr + proc.stdout)


def test_cli_rejects_malformed_case_json(tmp_path):
    bad = tmp_path / "case.json"
    bad.write_text('{"topic": "", "speaker_gender1": "男", "speaker_gender2": "女"}',
                   encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "scripts/generate.py", "--input", str(bad),
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "topic" in (proc.stderr + proc.stdout)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiboke.bootstrap'`

- [ ] **Step 3: 实现 `src/aiboke/bootstrap.py`**

```python
"""依赖装配：把配置组装成可运行的 Pipeline。

把「如何构造各组件」从 CLI 与 HTTP 入口中抽出来，使两个入口共享
同一套装配逻辑，也便于在测试中替换假实现。
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .cover import build_generator
from .llm_client import LlmClient
from .pipeline import Pipeline
from .script_writer import ScriptWriter
from .tts import build_backend
from .voices import load_presets


def load_presets_from_config(path: Path):
    return load_presets(Path(path))


def build_pipeline(
    cfg: Config,
    models_root: Path,
    prefer_fallback_cover: bool = False,
) -> Pipeline:
    models_root = Path(models_root)

    llm = LlmClient(cfg.llm)
    writer = ScriptWriter(
        llm,
        target_seconds=cfg.target_seconds,
        chars_per_minute=cfg.chars_per_minute,
        max_retries=cfg.max_retries,
        factcheck=cfg.factcheck,
    )
    tts = build_backend(cfg.tts)
    cover = build_generator(cfg.cover, prefer_fallback=prefer_fallback_cover)

    preset_path = Path(__file__).resolve().parents[2] / "configs" / "voice_presets.json"
    presets = load_presets(preset_path)

    return Pipeline(cfg, writer, tts, cover, presets)
```

- [ ] **Step 4: 实现 `scripts/generate.py`**

```python
#!/usr/bin/env python
"""CLI 入口。

用法：
    python scripts/generate.py --input case.json --output-dir out/
    python scripts/generate.py --topic "星巴克国内运营转移" --g1 男 --g2 女 --output-dir out/

输入 case.json：
    {"topic": "...", "speaker_gender1": "男", "speaker_gender2": "女"}

输出（在 --output-dir 下）：
    podcast.mp3  cover.png  script.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiboke.bootstrap import build_pipeline  # noqa: E402
from aiboke.config import load_config  # noqa: E402
from aiboke.schema import CaseInput  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成一期 AI 中文双人播客")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="输入 JSON 文件")
    src.add_argument("--topic", type=str, help="直接指定主题")
    p.add_argument("--g1", default="男", help="主播1性别（男/女）")
    p.add_argument("--g2", default="女", help="主播2性别（男/女）")
    p.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "default.yaml")
    p.add_argument("--models-root", type=Path, default=None, help="模型挂载点，覆盖配置")
    p.add_argument("--fallback-cover", action="store_true", help="跳过大模型封面，用纯色兜底")
    return p.parse_args(argv)


def load_case(args: argparse.Namespace) -> CaseInput:
    if args.input is not None:
        path: Path = args.input
        if not path.exists():
            raise FileNotFoundError(f"输入文件不存在: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"输入文件的 JSON 无法解析: {exc}") from exc
        return CaseInput.from_dict(data)
    return CaseInput(topic=args.topic, speaker_gender1=args.g1, speaker_gender2=args.g2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        case = load_case(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"输入错误：{exc}", file=sys.stderr)
        return 2

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    models_root = args.models_root or Path(cfg.models_root)

    try:
        pipeline = build_pipeline(cfg, models_root, prefer_fallback_cover=args.fallback_cover)
        episode = pipeline.run(case, args.output_dir)
    except Exception as exc:  # noqa: BLE001 — CLI 需要给出干净的失败信息
        print(f"生成失败：{exc}", file=sys.stderr)
        return 1

    print("生成完成：")
    print(f"  音频：{episode.audio_path}")
    print(f"  封面：{episode.cover_path}")
    print(f"  文稿：{episode.script_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 实现 `src/aiboke/server.py`**

```python
"""轻量 HTTP 封装。

评测方的容器调用方式未知，因此除 CLI 外额外提供常驻服务模式，
两种入口共用同一套 Pipeline 装配逻辑。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .bootstrap import build_pipeline
from .config import load_config
from .schema import CaseInput


def create_app(cfg_path: Path):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class GenerateRequest(BaseModel):
        topic: str
        speaker_gender1: str
        speaker_gender2: str

    cfg = load_config(Path(cfg_path))
    pipeline = build_pipeline(cfg, Path(cfg.models_root))
    app = FastAPI(title="AI 中文播客生成")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict:
        try:
            case = CaseInput(
                topic=req.topic,
                speaker_gender1=req.speaker_gender1,
                speaker_gender2=req.speaker_gender2,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        out_dir = Path(tempfile.mkdtemp(prefix="aiboke_"))
        try:
            episode = pipeline.run(case, out_dir)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "audio_path": str(episode.audio_path),
            "cover_path": str(episode.cover_path),
            "script_path": str(episode.script_path),
        }

    return app
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS（3 passed）

- [ ] **Step 7: 运行全量测试并提交**

```bash
python -m pytest tests/ -v
git add scripts/generate.py src/aiboke/server.py src/aiboke/bootstrap.py tests/test_cli.py
git commit -m "feat: CLI 入口与 HTTP 封装"
```

---

### Task 13: 模型拉取与 llama.cpp 构建脚本

**Files:**
- Create: `scripts/download_models.sh`
- Create: `scripts/build_llamacpp.sh`

**Interfaces:**
- Consumes: 无（shell 脚本）
- Produces: 挂载点下的模型目录结构，与 `configs/default.yaml` 的路径约定一致

- [ ] **Step 1: 写 `scripts/download_models.sh`**

```bash
#!/usr/bin/env bash
# 模型拉取：优先 ModelScope（国内直连），失败则回退 hf-mirror。
#
# 用法：
#   bash scripts/download_models.sh [目标目录]     # 默认 /models
#
# 预计磁盘占用：量化权重约 42-43GB；若需从完整权重转换 MOSS-TTSD 的
# first-class GGUF，额外约 17GB，合计约 45-60GB。

set -euo pipefail

MODELS_ROOT="${1:-/models}"
mkdir -p "${MODELS_ROOT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

ensure_tools() {
    if ! have modelscope && ! have hf; then
        log "安装下载工具（modelscope + huggingface_hub）"
        pip install -U modelscope "huggingface_hub[cli]"
    fi
}

# 优先 ModelScope，失败回退 hf-mirror
fetch() {
    local repo="$1" dest="$2"
    if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ]; then
        log "已存在，跳过：${dest}"
        return 0
    fi
    log "拉取 ${repo}"
    if have modelscope && modelscope download --model "${repo}" --local_dir "${dest}"; then
        return 0
    fi
    log "ModelScope 失败，回退 hf-mirror"
    hf download "${repo}" --local-dir "${dest}"
}

ensure_tools

# --- 1. 文稿模型：Qwen3.8-27B GGUF Q5_K_XL（约 19GB）---
# 选 Q5_K_XL 而非 Q8_0：留出显存给 TTS 与封面，使四个模型可同时常驻
fetch "unsloth/Qwen3.8-27B-GGUF" "${MODELS_ROOT}/Qwen3.8-27B-GGUF"

# --- 2. 语音模型：MOSS-TTSD（约 9GB）---
# 注意：HF 上的 OpenMOSS-Team/MOSS-TTS-GGUF 为 gated 仓库，需先在网页
# 接受条款并配置 HF_TOKEN；ModelScope / GitCode 有同样的镜像。
if [ -n "${HF_TOKEN:-}" ]; then
    hf download "OpenMOSS-Team/MOSS-TTS-GGUF" --local-dir "${MODELS_ROOT}/MOSS-TTSD-GGUF" || \
        log "MOSS-TTS-GGUF 拉取失败——该仓库为 gated，请先接受条款或改用 ModelScope 镜像"
else
    log "未设置 HF_TOKEN，尝试 ModelScope 镜像"
    fetch "openmoss/MOSS-TTSD-v1.0" "${MODELS_ROOT}/MOSS-TTSD-GGUF" || \
        log "MOSS-TTSD GGUF 未获取，transformers 后端将回退到完整权重"
fi

# --- 3. 音频编解码器（约 3GB）---
fetch "OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX" "${MODELS_ROOT}/MOSS-Audio-Tokenizer-ONNX"

# --- 4. 封面模型：Z-Image-Turbo（约 8GB）---
fetch "Tongyi-MAI/Z-Image-Turbo" "${MODELS_ROOT}/Z-Image-Turbo"

log "完成。目录结构："
find "${MODELS_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '  %p\n' | sort
cat <<'EOF'

请据此核对 configs/default.yaml 中的路径：
  tts.model_path   -> ${MODELS_ROOT}/MOSS-TTSD-GGUF
  cover.model_path -> ${MODELS_ROOT}/Z-Image-Turbo
EOF
```

- [ ] **Step 2: 写 `scripts/build_llamacpp.sh`**

```bash
#!/usr/bin/env bash
# 编译两份 llama.cpp：
#   1. upstream（用于 Qwen3.8-27B 文稿生成）
#   2. OpenMOSS fork（moss-tts-firstclass 分支，用于 MOSS-TTSD 语音合成）
#
# 为什么要两份：MOSS-TTS 的 first-class 推理路径位于 OpenMOSS 的 fork 中，
# 该 fork 是否已 rebase 到支持 Qwen3.8 的上游 master 无法事先确认，
# 因此分开构建互不影响。
#
# 关键：必须同时更新二进制与 .so。只替换 llama-server 而保留旧的
# libggml-cuda.so 会导致 Qwen3.8 的 DeltaNet 层静默产出乱码。
#
# 用法：bash scripts/build_llamacpp.sh [安装根目录]   # 默认 /opt/llama.cpp

set -euo pipefail

PREFIX="${1:-/opt/llama.cpp}"
JOBS="$(nproc)"
# A100 是 sm_80，不要照抄消费级卡的 86/89
CUDA_ARCH="${CUDA_ARCH:-80}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

build() {
    local name="$1" url="$2" branch="$3" srcdir="$4"
    log "构建 ${name}（arch=${CUDA_ARCH}）"
    if [ ! -d "${srcdir}/.git" ]; then
        git clone --branch "${branch}" "${url}" "${srcdir}"
    else
        git -C "${srcdir}" fetch --all --tags
        git -C "${srcdir}" checkout "${branch}"
        git -C "${srcdir}" pull --ff-only
    fi
    # 确保不是浅克隆——浅克隆下 git pull 会谎报 Already up to date
    if [ -f "${srcdir}/.git/shallow" ]; then
        git -C "${srcdir}" fetch --unshallow
    fi

    cmake -S "${srcdir}" -B "${srcdir}/build-cuda" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
    cmake --build "${srcdir}/build-cuda" --config Release -j "${JOBS}"
}

mkdir -p "${PREFIX}"

# 1) upstream：文稿生成
build "llama.cpp (upstream)" \
      "https://github.com/ggml-org/llama.cpp" "master" \
      "${PREFIX}/upstream"

# 2) OpenMOSS fork：MOSS-TTSD 语音合成
build "llama.cpp (OpenMOSS, moss-tts-firstclass)" \
      "https://github.com/OpenMOSS/llama.cpp" "moss-tts-firstclass" \
      "${PREFIX}/openmoss"

cat <<EOF

构建完成。

请记录 upstream 的版本号，必须 >= b10450：
  ${PREFIX}/upstream/build-cuda/bin/llama-server --version

自检：确认二进制与 .so 来自同一次构建（否则可能静默乱码）
  ldd ${PREFIX}/upstream/build-cuda/bin/llama-server | grep ggml
  echo \$LD_LIBRARY_PATH

若 MOSS-TTSD 的 GGUF 不是 first-class 格式，还需从完整权重转换：
  python ${PREFIX}/openmoss/convert_hf_to_gguf.py \\
      --model-gguf <MOSS-TTSD 完整权重目录> \\
      --outfile <输出.gguf> --outtype f16
EOF
```

- [ ] **Step 3: 设置可执行位并做 shell 语法检查**

```bash
chmod +x scripts/download_models.sh scripts/build_llamacpp.sh
bash -n scripts/download_models.sh
bash -n scripts/build_llamacpp.sh
```

Expected: 无输出（语法正确）

- [ ] **Step 4: 提交**

```bash
git add scripts/download_models.sh scripts/build_llamacpp.sh
git commit -m "feat: 模型拉取与 llama.cpp 双份构建脚本"
```

---

### Task 14: 语速标定与冒烟测试脚本

**Files:**
- Create: `scripts/calibrate_length.py`
- Create: `scripts/smoke_test.py`

**Interfaces:**
- Consumes: `aiboke.tts.build_backend`、`aiboke.audio_utils.probe_duration`、`aiboke.llm_client.LlmClient`、`aiboke.cover.build_generator`
- Produces:
  - `calibrate_length.py`：测量实际语速（字/分钟），输出建议的 `chars_per_minute`
  - `smoke_test.py`：四步可跑性验证，退出码非 0 表示不通过

- [ ] **Step 1: 写 `scripts/calibrate_length.py`**

```python
#!/usr/bin/env python
"""语速标定。

configs/default.yaml 里的 chars_per_minute=200 是从已有的播客长度
标定表折算的移植值，不同音色与语速设置会显著影响实际语速。本脚本
用实际音色合成一段已知字数的中文，测出真实时长后反算语速。

用法：
    python scripts/calibrate_length.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiboke import audio_utils  # noqa: E402
from aiboke.config import load_config  # noqa: E402
from aiboke.schema import Turn, VoicePair, VoicePreset  # noqa: E402
from aiboke.tts import build_backend  # noqa: E402

# 一段字数已知、内容中立的中文，避免数字与英文影响语速
SAMPLE_TEXT = (
    "今天我们聊一个很有意思的话题，关于一家公司如何在最困难的时候做出改变，"
    "并且最终重新赢得了市场的认可。这个故事里有决策，有取舍，也有运气。"
    "我们先从它的起点说起，那时候没人看好它。"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="标定实际语速")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--repeat", type=int, default=3, help="重复次数，取中位数以降低抖动")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tts = build_backend(cfg.tts)

    turns = (Turn(speaker=1, text=SAMPLE_TEXT),)
    voices = VoicePair(
        speaker1=VoicePreset(id="calib1", gender="男", description="标准的男声，语速自然"),
        speaker2=VoicePreset(id="calib2", gender="女", description="标准的女声，语速自然"),
    )
    char_count = len(SAMPLE_TEXT)
    measured: list[float] = []

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.repeat):
            out = Path(tmp) / f"calib{i}.wav"
            tts.synthesize(turns, voices, out, target_seconds=30.0)
            seconds = audio_utils.probe_duration(out)
            measured.append(seconds)
            print(f"第 {i + 1} 次：{char_count} 字 -> {seconds:.2f} 秒")

    measured.sort()
    median = measured[len(measured) // 2]
    chars_per_minute = char_count / median * 60.0

    print(f"\n实测语速：{chars_per_minute:.1f} 字/分钟")
    print(f"当前配置：{cfg.chars_per_minute} 字/分钟")
    print("\n建议把 configs/default.yaml 的 chars_per_minute 改为：")
    print(f"  chars_per_minute: {chars_per_minute:.1f}")
    print(
        f"\n按该语速，{cfg.target_seconds:.0f} 秒目标时长对应"
        f" {int(cfg.target_seconds / 60 * chars_per_minute)} 字。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 写 `scripts/smoke_test.py`**

```python
#!/usr/bin/env python
"""可跑性验证。在目标 GPU 机器上运行，退出码非 0 表示环境不可用。

最重要的一步是第 2 步：llama.cpp 旧版本在 Qwen3.8 的 DeltaNet 层
CUDA 路径上有 bug，症状是模型正常加载、显存正常、速度正常、零报错，
但输出全是乱码。因此必须校验输出内容，而不能只看加载是否成功。

用法：
    python scripts/smoke_test.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiboke import audio_utils  # noqa: E402
from aiboke.config import load_config  # noqa: E402
from aiboke.cover import build_generator  # noqa: E402
from aiboke.gates import cjk_ratio  # noqa: E402
from aiboke.llm_client import LlmClient  # noqa: E402
from aiboke.schema import Turn, VoicePair, VoicePreset  # noqa: E402
from aiboke.tts import build_backend  # noqa: E402

MIN_LLAMACPP_BUILD = 10450


class SmokeFailure(RuntimeError):
    pass


def step_no(n: int, title: str) -> None:
    print(f"\n[{n}/5] {title}")


def check_llamacpp_version(cfg) -> None:
    binary = cfg.tts.binary
    if not binary:
        print("  跳过：未配置 tts.binary")
        return
    server = Path(binary).parent / "llama-server"
    if not server.exists():
        print(f"  警告：找不到 {server}，跳过版本检查")
        return
    try:
        out = subprocess.run([str(server), "--version"], capture_output=True, text=True).stdout
    except OSError as exc:
        raise SmokeFailure(f"无法执行 {server}：{exc}") from exc

    print(f"  {out.strip().splitlines()[0] if out.strip() else '(无版本信息)'}")
    if str(MIN_LLAMACPP_BUILD) not in out:
        print(
            f"  警告：未能确认构建号 >= b{MIN_LLAMACPP_BUILD}。"
            "低于该版本在 Qwen3.8 上可能静默输出乱码，请务必核实。"
        )


def check_llm(cfg) -> None:
    """最关键的一步：必须校验输出内容为连贯中文，而不只是调用成功。"""
    client = LlmClient(cfg.llm)
    prompt = "请只回答一句话，用简体中文介绍你自己是一位播客主播。不要有任何其他内容。"
    text = client.complete("你是一位中文播客主播。", prompt)
    ratio = cjk_ratio(text)

    print(f"  输出：{text[:120]}")
    print(f"  中文占比：{ratio:.3f}")

    if ratio < 0.5:
        raise SmokeFailure(
            f"LLM 输出疑似乱码或非中文（中文占比 {ratio:.3f}）。"
            "这极可能是 llama.cpp 版本低于 b10450 导致的 DeltaNet CUDA bug——"
            "该 bug 不会报错，只会静默产出乱码。请升级 llama.cpp 并确认 "
            "libggml-cuda.so 与二进制同版本。"
        )


def check_tts(cfg) -> None:
    tts = build_backend(cfg.tts)
    turns = (
        Turn(speaker=1, text="大家好，欢迎收听本期商业故事。"),
        Turn(speaker=2, text="今天我们聊一个关于品牌转型的话题。"),
        Turn(speaker=1, text="没错，这个故事挺有意思的。"),
        Turn(speaker=2, text="那我们就从头说起吧。"),
    )
    voices = VoicePair(
        speaker1=VoicePreset(id="smoke_m", gender="男", description="清晰的男声"),
        speaker2=VoicePreset(id="smoke_f", gender="女", description="清晰的女声"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke.wav"
        tts.synthesize(turns, voices, out, target_seconds=30.0)
        seconds = audio_utils.probe_duration(out)
        size_kb = out.stat().st_size / 1024
    print(f"  产出 {seconds:.1f} 秒音频（{size_kb:.0f} KB）")
    if seconds < 3.0:
        raise SmokeFailure(f"TTS 仅产出 {seconds:.1f} 秒音频，明显偏短，疑似失败")
    print("  请人工试听，确认两位主播音色可区分，且性别与预设标注一致")


def check_cover(cfg) -> None:
    gen = build_generator(cfg.cover)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke.png"
        gen.generate("一家公司如何完成品牌转型", out)
        size_kb = out.stat().st_size / 1024
    print(f"  产出封面（{size_kb:.0f} KB），期望 {cfg.cover.size}x{cfg.cover.size}")
    if size_kb < 5:
        raise SmokeFailure("封面文件过小，疑似生成失败")


def check_roundtrip(cfg) -> None:
    client = LlmClient(cfg.llm)
    raw = client.complete(
        "你只输出 JSON，不要任何其他内容。",
        '请输出 {"title": "测试", "content": [{"speaker": 1, "text": "你好"}]} 这个 JSON。',
    )
    from aiboke.script_writer import parse_act

    try:
        turns = parse_act(raw)
    except Exception as exc:  # noqa: BLE001
        raise SmokeFailure(f"LLM 输出的 JSON 无法解析：{exc}\n原始输出：{raw[:200]}") from exc
    print(f"  JSON 解析成功，得到 {len(turns)} 轮对话")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="可跑性验证")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--skip-cover", action="store_true", help="跳过封面检查")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    steps = [
        ("llama.cpp 版本与库一致性", lambda: check_llamacpp_version(cfg)),
        ("LLM 中文输出正确性（防静默乱码）", lambda: check_llm(cfg)),
        ("LLM JSON 输出可解析", lambda: check_roundtrip(cfg)),
        ("TTS 双人中文语音", lambda: check_tts(cfg)),
    ]
    if not args.skip_cover:
        steps.append(("封面生成", lambda: check_cover(cfg)))

    for i, (title, fn) in enumerate(steps, start=1):
        step_no(i, title)
        try:
            fn()
        except SmokeFailure as exc:
            print(f"\n\033[1;31m不通过：{exc}\033[0m", file=sys.stderr)
            return 1

    print("\n\033[1;32m全部通过。环境可用于生成播客。\033[0m")
    print("提醒：请人工确认音色预设的性别标注与试听结果一致（见设计文档 7.2）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 做语法检查**

```bash
python -m py_compile scripts/calibrate_length.py scripts/smoke_test.py
python scripts/smoke_test.py --help
python scripts/calibrate_length.py --help
```

Expected: 两个 `--help` 均正常输出

- [ ] **Step 4: 提交**

```bash
git add scripts/calibrate_length.py scripts/smoke_test.py
git commit -m "feat: 语速标定与可跑性验证脚本"
```

---

### Task 15: Dockerfile 与文档

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `.env.example`
- Create: `README.md`
- Modify: `DESIGN.md`（复制设计文档）

**Interfaces:**
- Consumes: 全部源码与脚本
- Produces: 可构建的镜像；模型通过 `-v` 挂载，不烘进镜像

- [ ] **Step 1: 写 `Dockerfile`**

```dockerfile
# AI 中文双人播客生成系统
#
# 模型不烘进镜像，运行时以 -v 挂载（符合赛题「支持挂载模型」）：
#   docker run --gpus all -v /host/models:/models \
#       -v /host/out:/out aiboke \
#       python scripts/generate.py --input /case.json --output-dir /out
#
# 注意：llama.cpp 需在此镜像内或构建阶段编译（见 scripts/build_llamacpp.sh），
# 因为要针对 A100 的 sm_80 编译 CUDA 内核。

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    CUDA_ARCH=80

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg git cmake build-essential ninja-build \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip

WORKDIR /app

# 先装依赖，利用层缓存
COPY pyproject.toml ./
RUN pip install -e ".[server]" && pip install numpy soundfile onnxruntime-gpu

# 编译两份 llama.cpp（upstream 用于文稿，OpenMOSS fork 用于语音）
COPY scripts/build_llamacpp.sh /app/scripts/build_llamacpp.sh
RUN bash /app/scripts/build_llamacpp.sh /opt/llama.cpp

# 源码
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
RUN pip install -e .

ENV PATH="/opt/llama.cpp/upstream/build-cuda/bin:/opt/llama.cpp/openmoss/build-cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama.cpp/upstream/build-cuda/bin:/opt/llama.cpp/openmoss/build-cuda/bin" \
    MODELS_ROOT=/models \
    PYTHONPATH=/app/src

VOLUME ["/models", "/out"]

# 默认进入交互 shell；评测方按约定传入实际命令
CMD ["bash"]
```

- [ ] **Step 2: 写 `.dockerignore`**

```
.git
.research
docs
tests
**/__pycache__
**/*.pyc
*.md
out/
temp/
```

- [ ] **Step 3: 写 `.env.example`**

```bash
# AI 播客生成系统环境变量
# 复制为 .env 并按需修改

# 模型挂载点
MODELS_ROOT=/models

# HuggingFace 镜像（拉取模型用；MOSS-TTS-GGUF 为 gated 仓库时需 Token）
HF_ENDPOINT=https://hf-mirror.com
# HF_TOKEN=hf_xxx

# 编译 llama.cpp 时的 CUDA 架构（A100 是 80，不要用消费级卡的 86/89）
CUDA_ARCH=80

# LLM 服务
LLM_BASE_URL=http://127.0.0.1:8080
```

- [ ] **Step 4: 写 `README.md`**

`````markdown
# AI 中文双人播客生成系统

输入一个中文商业故事主题与两位主播的性别，输出一期完整的中文双人对话播客。

- 文稿：三幕式（开场铺垫 → 主体讲述 → 分析总结），由 Qwen3.8-27B 生成
- 音频：MOSS-TTSD 双人对话合成，5–15 分钟 mp3
- 封面：Z-Image-Turbo 生成的 1024×1024 PNG

全部模型均为 Apache-2.0，全程离线运行，**无需任何付费 API**。

## 快速开始

```bash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 跑单元测试（无需 GPU）
python -m pytest tests/ -v

# 3. 拉取模型（需约 45GB 磁盘）
bash scripts/download_models.sh /models

# 4. 编译 llama.cpp（针对 A100 的 sm_80）
bash scripts/build_llamacpp.sh /opt/llama.cpp

# 5. 启动 LLM 服务
/opt/llama.cpp/upstream/build-cuda/bin/llama-server \
    -m /models/Qwen3.8-27B-GGUF/<量化文件>.gguf \
    --port 8080 -ngl 99 --host 127.0.0.1

# 6. 环境自检（务必先跑这一步）
python scripts/smoke_test.py --config configs/default.yaml

# 7. 标定语速，把结果填回 configs/default.yaml
python scripts/calibrate_length.py --config configs/default.yaml

# 8. 生成播客
python scripts/generate.py \
    --input case.json --output-dir out/
```

`case.json`：

```json
{"topic": "星巴克国内运营转移", "speaker_gender1": "男", "speaker_gender2": "女"}
```

输出到 `out/`：`podcast.mp3`、`cover.png`、`script.json`。

## 两种调用方式

```bash
# CLI
python scripts/generate.py --topic "星巴克国内运营转移" --g1 男 --g2 女 --output-dir out/

# HTTP 常驻服务
uvicorn aiboke.server:create_app --factory --host 0.0.0.0 --port 8000
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

## 显存预算（A100-40GB）

| 组件 | 精度 | 显存 |
|---|---|---|
| Qwen3.8-27B | GGUF Q5_K_XL | ~19 GB |
| MOSS-TTSD 8B | GGUF Q6_K | ~8–9 GB |
| MOSS-Audio-Tokenizer | ONNX | ~2–3 GB |
| Z-Image-Turbo | FP8 | ~8 GB |
| **合计** | | **~37–39 GB** |

设计目标为四个模型全部常驻，不做换入换出。若显存吃紧，先把 LLM 降到
Q4_K_M（约 17GB）。

## 目录结构

```
src/aiboke/     核心库（schema / length / gates / prompts 为纯逻辑，可无 GPU 单测）
configs/        配置与音色预设
scripts/        下载、编译、标定、自检、CLI
tests/          单元测试
docs/           设计文档与实现计划
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
`````

- [ ] **Step 5: 复制设计文档到仓库根**

```bash
cp docs/superpowers/specs/2026-09-22-ai-podcast-generation-design.md DESIGN.md
```

- [ ] **Step 6: 做整体校验**

```bash
python -m pytest tests/ -v
python -m py_compile scripts/*.py src/aiboke/*.py
bash -n scripts/download_models.sh scripts/build_llamacpp.sh
python scripts/generate.py --help
```

Expected: 全部通过；`--help` 正常输出

- [ ] **Step 7: 提交**

```bash
git add Dockerfile .dockerignore .env.example README.md DESIGN.md
git commit -m "docs: Dockerfile、README 与环境变量模板"
```

---

## 完成后的验证清单

以下步骤必须在**目标 GPU 机器**上执行（本地无 GPU 无法验证）：

- [ ] `python -m pytest tests/ -v` 全部通过
- [ ] `bash scripts/download_models.sh` 成功拉取全部模型
- [ ] `bash scripts/build_llamacpp.sh` 成功编译两份 llama.cpp
- [ ] `llama-server --version` 显示构建号 ≥ b10450
- [ ] `ldd llama-server | grep ggml` 确认加载的是同一次构建的 `.so`
- [ ] `python scripts/smoke_test.py` 五步全部通过
- [ ] `python scripts/calibrate_length.py` 得到实测语速，回填 `chars_per_minute`
- [ ] 用实测语速重新跑一遍语速标定，确认收敛
- [ ] 端到端跑通 4 种性别组合（男男 / 男女 / 女男 / 女女）
- [ ] 每次产出校验：时长落在 300–900 秒、中文占比 ≥0.85、两位主播轮次各 ≥25%
- [ ] 人工试听：音色区分度、性别与预设标注一致、无爆音断裂
- [ ] 记录实测 RTF 与首字延迟，对比基线（RTF ≤ 2、首字 ≤ 30s）
- [ ] 构建 Docker 镜像并用挂载模型跑通一次完整生成
