import pytest

from aiboke.prompts import (
    ACT_GUIDANCE,
    ACT_NAMES,
    SCRIPT_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_act_prompt,
    build_cover_image_prompt,
    build_cover_prompt,
    build_cover_upgrade_prompt,
    build_factcheck_prompt,
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


# ---------- 直接给文生图模型的视觉提示词（Task 10 的封面用这条） ----------

def test_build_cover_image_prompt_is_english_and_keeps_topic():
    p = build_cover_image_prompt("星巴克国内运营转移")
    assert "星巴克国内运营转移" in p
    # 除主题原样保留外，描述部分应当是英文（给图像模型用）
    assert p.replace("星巴克国内运营转移", "").isascii()


def test_build_cover_image_prompt_forbids_letters_and_numbers():
    p = build_cover_image_prompt("星巴克国内运营转移")
    assert "no text" in p.lower()
    assert "no letters" in p.lower() or "no words" in p.lower()
    assert "no numbers" in p.lower()


def test_build_cover_image_prompt_is_not_an_llm_instruction():
    """它是画面描述，不是「请让 LLM 输出提示词」的元指令。"""
    p = build_cover_image_prompt("星巴克国内运营转移")
    assert "请只输出" not in p
    assert "要求：" not in p
    assert "为一期主题为" not in p
