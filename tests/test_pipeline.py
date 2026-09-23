"""三阶段流水线编排的测试。

全部外部边界（LLM 文稿、TTS 后端、封面、ffmpeg/ffprobe）都是假实现，
不启动任何真实子进程、不发起任何网络调用。

门限的检查时机是本模块的重点：性别/音色门在昂贵工作之前，语言门在该幕
合成之前，双人门在整篇文稿之后，时长门在 mp3 产出之后。
"""

import json
import logging
from pathlib import Path

import pytest

from aiboke.config import Config, CoverConfig, LlmConfig, TtsConfig
from aiboke.cover import CoverError
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


def test_run_checks_language_gate_before_synthesizing_that_act(tmp_path):
    """语言门必须在该幕合成之前拦住，而不是等到整篇或事后。

    这是四道 0 分门限之一，且合成是整条流水线里最贵的一步：命中英文的幕
    若先合成再被整篇门限拦下，白烧一次 TTS，首字延迟也被拖长。
    第 2 幕（turns[3:6]）写成英文，第 1 幕的中文合成应当已经发生。
    """
    english = Turn(speaker=1, text="This whole act is written in plain English.")
    turns = tuple(english if 3 <= i < 6 else t for i, t in enumerate(GOOD_TURNS))
    p = Pipeline(
        cfg=_cfg(),
        script_writer=FakeScriptWriter(Transcript("T", turns)),
        tts=FakeTts(),
        cover=FakeCover(),
        preset_map=_presets(),
        runner=_audio_runner(),
    )
    with pytest.raises(PipelineError, match="语言"):
        p.run(_case(), tmp_path / "out")
    assert len(p._tts.calls) == 1, "第 2 幕不该在被拦下之前合成语音"


def test_run_raises_when_whole_transcript_has_single_speaker(tmp_path):
    """双人门只能对完整文稿判定，且不通过时必须失败。

    单幕内部可能天然由一位主播主导，因此这个门不能随幕检查——它对整篇
    判定，落在所有幕之后、拼接之前。全篇只有 speaker 1 时必须报错，
    并且不得留下 script.json 之类会让人误以为成功的产物。
    """
    single = tuple(Turn(speaker=1, text=t.text) for t in GOOD_TURNS)
    p = Pipeline(
        cfg=_cfg(),
        script_writer=FakeScriptWriter(Transcript("T", single)),
        tts=FakeTts(),
        cover=FakeCover(),
        preset_map=_presets(),
        runner=_audio_runner(),
    )
    with pytest.raises(PipelineError, match="门限"):
        p.run(_case(), tmp_path / "out")
    assert not (tmp_path / "out" / "script.json").exists()


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


def test_run_falls_back_to_fallback_cover_on_oserror(tmp_path):
    """封面抛裸 OSError 时同样要走兜底，不能只看 CoverError。

    ZImageCover 的尺寸校验直接 read_bytes()，文件不可读/被删时逃逸的是
    OSError；解释器缺失时是 FileNotFoundError。这类失败若只按 CoverError
    捕获就会绕过兜底，把「产物缺失」变成静默的交付缺陷（基线失败率上升）。
    """

    class OSErrorCover:
        def generate(self, topic, out_path):
            raise OSError("[WinError 5] 拒绝访问")

    p = Pipeline(
        cfg=_cfg(),
        script_writer=FakeScriptWriter(Transcript("T", GOOD_TURNS)),
        tts=FakeTts(),
        cover=OSErrorCover(),
        preset_map=_presets(),
        runner=_audio_runner(),
    )
    ep = p.run(_case(), tmp_path / "out")
    assert ep.cover_path.exists()


def test_run_propagates_when_fallback_cover_also_fails(tmp_path):
    """兜底封面也失败时必须响亮失败，而不是留下一个不存在的封面路径。

    兜底自己也要调 ffmpeg；连它都起不来，说明 cover.png 已不可能产出。
    静默返回路径会让调用方以为三件产物齐全，属于「伪装成成功」。
    """

    class BrokenCover:
        def generate(self, topic, out_path):
            raise RuntimeError("显存不足")

    class MissingFfmpeg:
        """ffprobe 与音频编码正常，唯独封面那一次 ffmpeg 找不到。"""

        def __call__(self, cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)
            if "ffprobe" in joined:
                return type("P", (), {"returncode": 0, "stdout": "510.0\n", "stderr": ""})()
            out = Path(cmd[-1])
            if out.suffix == ".png":
                raise FileNotFoundError("[WinError 2] 系统找不到指定的文件：ffmpeg")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"RIFF" if out.suffix == ".wav" else b"ID3")
            return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    p = Pipeline(
        cfg=_cfg(),
        script_writer=FakeScriptWriter(Transcript("T", GOOD_TURNS)),
        tts=FakeTts(),
        cover=BrokenCover(),
        preset_map=_presets(),
        runner=MissingFfmpeg(),
    )
    with pytest.raises(CoverError, match="无法启动"):
        p.run(_case(), tmp_path / "out")


def test_run_warns_when_duration_in_range_but_far_from_target(tmp_path, caplog):
    """合规但远离目标的时长必须通过并记录警告。

    880 秒仍在 [300, 900] 内，但距 510 秒目标 370 秒，远超 90 秒容忍带。
    设计 §6.2 要求这一档「通过 + 警告」：门限是硬约束，警告是给操作者的
    信号，把警告升级成失败会白白丢掉一个合规的案子。
    """
    p = Pipeline(
        cfg=_cfg(),
        script_writer=FakeScriptWriter(Transcript("T", GOOD_TURNS)),
        tts=FakeTts(),
        cover=FakeCover(),
        preset_map=_presets(),
        runner=_audio_runner(duration=880.0),
    )
    with caplog.at_level(logging.WARNING, logger="aiboke.pipeline"):
        ep = p.run(_case(), tmp_path / "out")

    assert ep.audio_path.exists()
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("880.0" in m and "510.0" in m for m in messages), messages


def test_run_does_not_warn_when_duration_near_target(tmp_path, caplog):
    """贴近目标时长时不得记录时长警告，否则警告会退化成噪声。"""
    p = Pipeline(
        cfg=_cfg(),
        script_writer=FakeScriptWriter(Transcript("T", GOOD_TURNS)),
        tts=FakeTts(),
        cover=FakeCover(),
        preset_map=_presets(),
        runner=_audio_runner(duration=510.0),
    )
    with caplog.at_level(logging.WARNING, logger="aiboke.pipeline"):
        p.run(_case(), tmp_path / "out")

    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == []
