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
