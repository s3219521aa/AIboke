"""预检脚本（smoke_test / calibrate_length）的测试。

这两个脚本是「环境是否可用于生成」的强制门槛，所以它们必须走**生产路径**
取音色：读仓库里真实的 configs/voice_presets.json，再按性别解析
（bootstrap.load_presets_from_config -> voices.resolve_pair）。手搓一套没有
参考音频的假预设会让冒烟测试绕开「音色性别由参考音频保证」这条 0 分门限的
唯一真实来源——一个生产跑不起来的环境也能通过冒烟。这里钉住四件事：

  1. 默认路径确实读的是随仓库发布的预设文件；
  2. 内置合成预设只走显式开关（--no-reference-audio）；
  3. --skip-tts / --skip-cover 真的移除了对应步骤；
  4. 有步骤被跳过时**不得**宣称环境可用。

两个脚本都不是包内模块，按路径加载（与 tests/test_cli.py 的 CLI 加载方式一致）。
"""

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_PRESETS = _REPO_ROOT / "configs" / "voice_presets.json"


def _load_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"aiboke_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shipped_ids():
    raw = json.loads(SHIPPED_PRESETS.read_text(encoding="utf-8"))
    return {entry["id"] for entry in raw["presets"]}


# ---------- C1：音色必须来自随仓库发布的预设 ----------

def test_smoke_test_derives_voices_from_shipped_presets():
    """冒烟测试的默认音色 = 真实预设文件里按性别取出的那两条。

    手搓预设（旧行为）会让「预设文件本身坏掉/缺某一性别」永远测不出来，
    而生产用的正是这个文件。
    """
    smoke = _load_script("smoke_test")
    voices = smoke.resolve_smoke_voices(SHIPPED_PRESETS, allow_synthetic=False)

    shipped = _shipped_ids()
    assert voices.speaker1.gender == "男"
    assert voices.speaker2.gender == "女"
    assert voices.speaker1.id in shipped
    assert voices.speaker2.id in shipped
    # 默认路径不带参时用的就是仓库里的那个文件
    assert smoke.DEFAULT_PRESETS_PATH == SHIPPED_PRESETS


def test_calibrate_length_derives_voices_from_shipped_presets():
    """语速标定同样必须用真实音色：语速是音色相关的。"""
    calibrate = _load_script("calibrate_length")
    voices = calibrate.resolve_calibration_voices(SHIPPED_PRESETS, allow_synthetic=False)

    shipped = _shipped_ids()
    assert voices.speaker1.gender == "男"
    assert voices.speaker2.gender == "女"
    assert voices.speaker1.id in shipped
    assert voices.speaker2.id in shipped
    assert calibrate.DEFAULT_PRESETS_PATH == SHIPPED_PRESETS


def test_smoke_test_synthetic_voices_only_behind_explicit_flag():
    """内置合成预设必须由开关显式启用，且确实没有参考音频。

    它的存在意义是「音色尚未生成时先验证其余环境」，不是生产路径。
    """
    smoke = _load_script("smoke_test")
    voices = smoke.resolve_smoke_voices(SHIPPED_PRESETS, allow_synthetic=True)

    assert (voices.speaker1.id, voices.speaker2.id) == ("smoke_m", "smoke_f")
    assert voices.speaker1.reference_audio is None
    assert voices.speaker2.reference_audio is None


def test_smoke_test_wraps_broken_presets_into_smoke_failure(tmp_path):
    """预设文件缺某一性别时要报可读的不通过，而不是裸 ValueError 穿透。"""
    smoke = _load_script("smoke_test")
    broken = tmp_path / "voice_presets.json"
    broken.write_text(
        json.dumps({"presets": [{"id": "m", "gender": "男", "description": "d"}]}),
        encoding="utf-8",
    )
    with pytest.raises(smoke.SmokeFailure, match="女"):
        smoke.resolve_smoke_voices(broken, allow_synthetic=False)


def test_smoke_test_reports_missing_presets_file(tmp_path):
    smoke = _load_script("smoke_test")
    with pytest.raises(smoke.SmokeFailure, match="无法读取"):
        smoke.resolve_smoke_voices(tmp_path / "不存在.json", allow_synthetic=False)


# ---------- 参考音频采样率（M4：不一致不会被拦下）----------

def _voices_with_refs(tmp_path):
    """一对带参考音频的音色（音频内容无关紧要：探测函数在测试里被替换）。"""
    a = tmp_path / "m.wav"
    b = tmp_path / "f.wav"
    for p in (a, b):
        p.write_bytes(b"RIFF")
    from aiboke.schema import VoicePair, VoicePreset

    return VoicePair(
        speaker1=VoicePreset(id="m_calm", gender="男", description="d", reference_audio=str(a)),
        speaker2=VoicePreset(id="f_clear", gender="女", description="d", reference_audio=str(b)),
    )


def test_smoke_test_flags_reference_audio_that_is_not_24k(tmp_path, monkeypatch, capsys):
    """非 24kHz 的参考音频必须报「不通过」并给出转换命令。

    采样率不一致时 ffmpeg 会静默重采样（不报错），音色被无声地改变——没有
    这道核对，这个问题要到人工试听时才发现，甚至发现不了。
    """
    smoke = _load_script("smoke_test")
    monkeypatch.setattr(smoke, "probe_sample_rate", lambda path: 16000)
    voices = _voices_with_refs(tmp_path)

    with pytest.raises(smoke.SmokeFailure) as excinfo:
        smoke.check_reference_rates(voices)
    assert "16000" in str(excinfo.value)
    assert "ffmpeg" in str(excinfo.value) or "重采样" in str(excinfo.value)


def test_smoke_test_accepts_24k_reference_audio(tmp_path, monkeypatch, capsys):
    smoke = _load_script("smoke_test")
    monkeypatch.setattr(smoke, "probe_sample_rate", lambda path: 24000)
    smoke.check_reference_rates(_voices_with_refs(tmp_path))
    assert "24000 Hz" in capsys.readouterr().out


def test_smoke_test_degrades_to_a_reminder_when_ffprobe_is_unavailable(
    tmp_path, monkeypatch, capsys
):
    """读不出采样率（无 ffprobe）时只提醒，不阻断——核对项交人工。"""
    smoke = _load_script("smoke_test")
    monkeypatch.setattr(smoke, "probe_sample_rate", lambda path: None)
    smoke.check_reference_rates(_voices_with_refs(tmp_path))
    assert "人工确认" in capsys.readouterr().out



class _Args:
    """build_steps 只看这几个开关，不必构造完整的 argparse 命名空间。"""

    def __init__(self, skip_tts=False, skip_cover=False, no_reference_audio=False,
                 voice_presets=SHIPPED_PRESETS):
        self.skip_tts = skip_tts
        self.skip_cover = skip_cover
        self.no_reference_audio = no_reference_audio
        self.voice_presets = voice_presets


def _titles(steps):
    return [title for title, _ in steps]


def test_smoke_test_skip_flags_remove_the_matching_steps():
    smoke = _load_script("smoke_test")

    assert _titles(smoke.build_steps(None, _Args())) == [
        "llama.cpp 版本与库一致性",
        "LLM 中文输出正确性（防静默乱码）",
        "LLM JSON 输出可解析",
        "TTS 双人中文语音",
        "封面生成",
    ]

    no_tts = _titles(smoke.build_steps(None, _Args(skip_tts=True)))
    assert not any("TTS" in t for t in no_tts)
    assert "封面生成" in no_tts, "跳过 TTS 后封面步骤必须仍能执行（这正是缺陷所在）"

    no_cover = _titles(smoke.build_steps(None, _Args(skip_cover=True)))
    assert "封面生成" not in no_cover
    assert any("TTS" in t for t in no_cover)


def test_smoke_test_parses_new_flags():
    smoke = _load_script("smoke_test")
    args = smoke.parse_args(["--skip-tts", "--no-reference-audio"])
    assert args.skip_tts is True
    assert args.no_reference_audio is True
    assert args.voice_presets == SHIPPED_PRESETS


def test_skipped_steps_are_not_reported_as_a_usable_environment(monkeypatch, capsys):
    """跳过步骤后不得打印「全部通过，环境可用」——那正是最贵的误判。"""
    smoke = _load_script("smoke_test")
    monkeypatch.setattr(
        smoke, "build_steps", lambda cfg, args: [("占位步骤", lambda: None)]
    )

    code = smoke.main(["--config", str(_REPO_ROOT / "configs" / "default.yaml"), "--skip-tts"])

    assert code == 0
    out = capsys.readouterr().out
    assert "跳过的步骤未验证" in out
    assert "环境可用于生成播客" not in out


def test_full_run_reports_all_passed(monkeypatch, capsys):
    """没跳过任何步骤时照旧宣称「全部通过」。"""
    smoke = _load_script("smoke_test")
    monkeypatch.setattr(
        smoke, "build_steps", lambda cfg, args: [("占位步骤", lambda: None)]
    )

    code = smoke.main(["--config", str(_REPO_ROOT / "configs" / "default.yaml")])

    assert code == 0
    assert "全部通过。环境可用于生成播客。" in capsys.readouterr().out


def test_smoke_test_failing_step_returns_nonzero(monkeypatch, capsys):
    """步骤失败仍要以退出码 1 + 一行「不通过」收场（预检的既有契约）。"""
    smoke = _load_script("smoke_test")

    def _boom():
        raise smoke.SmokeFailure("演示失败")

    monkeypatch.setattr(smoke, "build_steps", lambda cfg, args: [("占位步骤", _boom)])
    code = smoke.main(["--config", str(_REPO_ROOT / "configs" / "default.yaml")])
    assert code == 1
    assert "不通过：演示失败" in capsys.readouterr().err
