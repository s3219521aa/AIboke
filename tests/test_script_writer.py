"""三幕式文稿生成器的测试。

相对 brief 的三处假数据调整（断言本身只在 --1-- 处改动，原因写在测试内）：

  1. 每条假响应按它对应那一幕的目标字数构造。测试配置下三幕目标为
     340/935/425 字，它们的 [0.6x, 1.4x] 区间互不重叠——第二幕的下限
     561 字已经高于第一幕的上限 476 字——所以不存在一份「通用长度」的
     假文稿能同时落在三幕的字数区间内。`_acts()` 负责按幕取长度。
  2. 带数字的事实断言必须选汉字占比 >= 0.85 的句子。brief 用的
     「市占率高达 37.5%。」占比只有 0.625，会被语言门判为不合格并触发
     一次与事实自检无关的重试，测试也就不再测它本意要测的东西；改用
     NOISY_NUMBER（0.875）。
  3. 事实自检需要一次额外调用，对应的假响应要补上。
  4. 坏 JSON 会带反馈重试，每次尝试消耗「生成 + 修复」两条响应。
"""

import json
import logging

import pytest

from aiboke.gates import cjk_ratio
from aiboke.script_writer import ScriptError, ScriptWriter, extract_json, parse_act
from aiboke.schema import CaseInput

_DEFAULT_SENTENCE = "这是一段中文的对话内容。"

# 带阿拉伯数字的事实断言（汉字占比 0.875，刚好过 0.85 的语言门）——
# 用它构造事实自检的假数据，既保留「可被证伪的精确数字」这一被测对象，
# 又不会因为汉字占比过低而触发无关的语言门重试。
NOISY_NUMBER = "有数据显示，它的市占率一度高达 37.5%，但随后快速回落。"

# 1700 字（510 秒 x 200 字/分钟）按 20/55/25 分配给三幕
_ACT_TARGETS = (340, 935, 425)


def _case():
    return CaseInput(topic="星巴克国内运营转移", speaker_gender1="男", speaker_gender2="女")


def _filler(n_chars, sentence=_DEFAULT_SENTENCE):
    """把一句中文重复并截断到恰好 n_chars 字。"""
    if n_chars <= 0:
        raise ValueError("n_chars 必须为正数")
    return (sentence * (n_chars // len(sentence) + 1))[:n_chars]


def _act_json(title, n_turns=4, text=_DEFAULT_SENTENCE, chars=None):
    """构造单幕的假响应。

    chars 给定时把 text 重复到该幕的目标字数并均分到各轮，使整幕总字数
    落在建议区间内（见模块开头第 1 条）。
    """
    if chars is not None:
        text = _filler(max(chars // n_turns, 1), text)
    return json.dumps(
        {
            "title": title,
            "content": [
                {"speaker": (i % 2) + 1, "text": text} for i in range(n_turns)
            ],
        },
        ensure_ascii=False,
    )


def _acts(*titles):
    """三幕各自的假响应，按各幕的目标字数构造。"""
    if not titles:
        titles = ("T", "T", "T")
    return [_act_json(t, chars=c) for t, c in zip(titles, _ACT_TARGETS)]


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


def test_extract_json_ignores_trailing_prose_after_json():
    """JSON 之后的客套话也要剥掉——模型经常在对象后面补一句收尾。"""
    assert extract_json('{"a": 1}\nhope this helps') == {"a": 1}
    assert extract_json('{"a": 1}\n希望有帮助。') == {"a": 1}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ScriptError, match="JSON"):
        extract_json("完全不是 JSON")


def test_extract_json_rejects_non_object():
    with pytest.raises(ScriptError, match="对象"):
        extract_json("[1, 2, 3]")


# ---------- parse_act ----------

def test_parse_act_returns_turns():
    turns = parse_act(_act_json("标题"))
    assert len(turns) == 4
    assert turns[0].speaker == 1


def test_parse_act_rejects_missing_content():
    with pytest.raises(ScriptError, match="content"):
        parse_act('{"title": "只有标题"}')


def test_parse_act_rejects_empty_content():
    with pytest.raises(ScriptError, match="content"):
        parse_act('{"title": "标题", "content": []}')


def test_parse_act_rejects_bad_speaker():
    raw = json.dumps({"title": "t", "content": [{"speaker": 7, "text": "x"}]})
    with pytest.raises(ScriptError):
        parse_act(raw)


def test_parse_act_rejects_non_integer_speaker():
    raw = json.dumps({"title": "t", "content": [{"speaker": "甲", "text": "x"}]})
    with pytest.raises(ScriptError):
        parse_act(raw)


# ---------- ScriptWriter ----------

def _writer(client, **over):
    """测试用构造器：默认关闭事实自检，便于精确控制 LLM 调用次数。"""
    kw = dict(target_seconds=510.0, chars_per_minute=200.0, max_retries=2, factcheck=False)
    kw.update(over)
    return ScriptWriter(client, **kw)


def test_write_concatenates_three_acts():
    client = FakeClient(_acts("标题A", "标题B", "标题C"))
    t = _writer(client).write(_case())
    assert len(client.calls) == 3, "应当恰好生成三幕"
    assert len(t.turns) == 12
    assert t.title == "标题A", "标题取第一幕的"


def test_iter_acts_yields_one_result_per_act_in_order():
    client = FakeClient(_acts())
    acts = list(_writer(client).iter_acts(_case()))
    assert [a.act_index for a in acts] == [0, 1, 2]
    assert all(len(a.turns) == 4 for a in acts)


def test_iter_acts_char_targets_follow_act_weights():
    client = FakeClient(_acts())
    acts = list(_writer(client).iter_acts(_case()))
    # 1700 字按 20/55/25 分配
    assert [a.char_target for a in acts] == [340, 935, 425]


def test_iter_acts_is_lazy_so_first_act_is_usable_immediately():
    """流水线依赖惰性：第一幕产出后即可开始合成语音，与后续幕重叠。"""
    client = FakeClient(_acts())
    gen = _writer(client).iter_acts(_case())
    first = next(gen)
    assert first.act_index == 0
    assert len(client.calls) == 1, "尚未请求第二幕时不应已经调用过 LLM"


def test_factcheck_runs_only_on_body_act():
    # 调用顺序：第一幕、第二幕、主体幕事实自检、第三幕——故需要四条假响应
    client = FakeClient(
        [
            _act_json("T", chars=340),
            _act_json("T", chars=935),
            _act_json("T", chars=935),
            _act_json("T", chars=425),
        ]
    )
    list(_writer(client, factcheck=True).iter_acts(_case()))
    assert len(client.calls) == 4


def test_factcheck_softens_claims_via_llm_rewrite():
    client = FakeClient(
        [
            _act_json("T", chars=340, text=NOISY_NUMBER),
            _act_json("T", chars=935, text=NOISY_NUMBER),
            _act_json("T", text="市占率大幅下滑。"),
            _act_json("T", chars=425, text=NOISY_NUMBER),
        ]
    )
    acts = list(_writer(client, factcheck=True).iter_acts(_case()))
    assert acts[1].turns[0].text == "市占率大幅下滑。"


def test_factcheck_keeps_original_when_rewrite_unparseable():
    client = FakeClient(
        [
            _act_json("T", chars=340),
            _act_json("T", chars=935),
            "这不是 JSON",
            _act_json("T", chars=425),
        ]
    )
    acts = list(_writer(client, factcheck=True).iter_acts(_case()))
    assert len(acts[1].turns) == 4, "自检失败时应保留原文而非丢失内容"


def test_factcheck_rewrite_in_english_is_rejected(caplog):
    """自检可能把主体幕改成英文——那是 0 分项，必须退回通过门限的原文。"""
    english = json.dumps(
        {
            "title": "T",
            "content": [
                {"speaker": (i % 2) + 1, "text": "This act is now in English."}
                for i in range(4)
            ],
        },
        ensure_ascii=False,
    )
    body = _act_json("T", chars=935)
    client = FakeClient(
        [_act_json("T", chars=340), body, english, _act_json("T", chars=425)]
    )
    with caplog.at_level(logging.WARNING, logger="aiboke.script_writer"):
        acts = list(_writer(client, factcheck=True).iter_acts(_case()))
    assert acts[1].turns == parse_act(body), "应当保留通过门限的原文"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "未通过门限" in messages and "中文" in messages, messages


def test_factcheck_rewrite_with_single_speaker_is_rejected(caplog):
    """重写塌成单人独白同样是 0 分项，必须退回原文。"""
    one_speaker = json.dumps(
        {
            "title": "T",
            "content": [{"speaker": 1, "text": "只有一个人说话，其余照旧。"}] * 8,
        },
        ensure_ascii=False,
    )
    body = _act_json("T", chars=935)
    client = FakeClient(
        [_act_json("T", chars=340), body, one_speaker, _act_json("T", chars=425)]
    )
    with caplog.at_level(logging.WARNING, logger="aiboke.script_writer"):
        acts = list(_writer(client, factcheck=True).iter_acts(_case()))
    assert acts[1].turns == parse_act(body), "应当保留通过门限的原文"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "未通过门限" in messages and "主播" in messages, messages


def test_write_propagates_prior_context_between_acts():
    client = FakeClient(_acts())
    _writer(client).write(_case())
    # 第二幕的 prompt 里应当出现第一幕的内容作为衔接上下文
    assert _DEFAULT_SENTENCE in client.calls[1]["user"]


def test_write_requests_json_schema_constrained_decoding():
    client = FakeClient(_acts())
    _writer(client).write(_case())
    assert client.calls[0]["json_schema"] is not None


def test_write_repairs_unparseable_act():
    client = FakeClient(
        [
            "这不是 JSON",
            _act_json("修复后的标题", chars=340),
            _act_json("T", chars=935),
            _act_json("T", chars=425),
        ]
    )
    t = _writer(client).write(_case())
    assert t.title == "修复后的标题"


def test_write_raises_when_repair_also_fails():
    # 每次尝试消耗两条响应：生成 + 修复。max_retries=2 即三次尝试，
    # 所以要耗尽重试必须给 6 条不可解析的响应——只给 4 条的话，第二轮
    # 就会拿到 _act_json("T") 而成功，测试便不再测到「修复也失败」。
    client = FakeClient(["坏输出", "还是坏的"] * 3)
    with pytest.raises(ScriptError, match="修复"):
        _writer(client).write(_case())
    assert len(client.calls) == 6, "三次尝试各消耗一次生成与一次修复"


def test_write_recovers_when_repair_fails_but_next_attempt_succeeds():
    """坏 JSON 不早失败：带反馈重试后仍要能拿到合格的一幕（P12）。"""
    client = FakeClient(["坏输出", "还是坏的"] + _acts())
    t = _writer(client).write(_case())
    assert len(client.calls) == 5, "首次尝试消耗生成+修复，之后三幕各一次"
    assert len(t.turns) == 12
    assert "JSON" in client.calls[2]["user"], "重试 prompt 要带上格式反馈"


def test_repair_call_failure_is_logged(caplog):
    """修复调用本身抛异常也必须留痕：端点挂掉与「又一段坏 JSON」表象相同。"""
    client = FakeClient(
        [
            "坏输出",
            RuntimeError("server 挂了"),
            _act_json("T", chars=340),
            _act_json("T", chars=935),
            _act_json("T", chars=425),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="aiboke.script_writer"):
        t = _writer(client).write(_case())
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "RuntimeError" in messages and "server 挂了" in messages, messages
    assert len(t.turns) == 12, "修复失败后带反馈重试仍应拿到完整文稿"


def test_write_retries_act_when_chinese_gate_fails():
    english = json.dumps(
        {"title": "T", "content": [{"speaker": 1, "text": "This is English."}]},
        ensure_ascii=False,
    )
    client = FakeClient([english] + _acts())
    t = _writer(client).write(_case())
    # 第一幕重试一次后通过，共 4 次调用；三幕各 4 轮（brief 原文写的是 4 轮，
    # 与它自己的 test_write_concatenates_three_acts（12 轮）矛盾）
    assert len(client.calls) == 4
    assert len(t.turns) == 12


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
    client = FakeClient([short] + _acts())
    _writer(client).write(_case())
    assert "主播" in client.calls[1]["user"]


# ---------- 单幕字数建议（P10） ----------

def test_write_retries_when_act_is_far_below_target_length():
    """字数远低于目标时，重试 prompt 里要带上扩充指令。"""
    short = _act_json("T")  # 48 字，远低于第一幕 340 字的目标
    client = FakeClient([short] + _acts())
    acts = list(_writer(client).iter_acts(_case()))
    retry_prompt = client.calls[1]["user"]
    assert "上一幕只写了约 48 字" in retry_prompt
    assert "扩充" in retry_prompt
    assert len(acts[0].turns) == 4, "重试通过后应当采用第二次的输出"
    assert len(client.calls) == 4


def test_write_retries_when_act_far_exceeds_target_length():
    """字数远超目标时，重试 prompt 里要带上压缩指令。"""
    too_long = _act_json("T", chars=1000)  # 第一幕目标 340，上限 476
    client = FakeClient([too_long] + _acts())
    acts = list(_writer(client).iter_acts(_case()))
    assert "压缩" in client.calls[1]["user"]
    assert len(acts[0].turns) == 4


def test_length_only_failure_is_accepted_after_retries_are_exhausted():
    """字数只是建议：耗尽重试后仍返回该幕，绝不升级成整案失败。"""
    short = _act_json("T")  # 48 字，三次尝试都不达标
    client = FakeClient([short, short, short])
    gen = _writer(client).iter_acts(_case())
    first = next(gen)
    assert len(client.calls) == 3, "max_retries=2 即三次尝试"
    assert first.act_index == 0
    assert len(first.turns) == 4, "应当返回这一幕而不是抛异常"
    assert first.title == "T"


def test_gate_failure_is_fatal_while_length_failure_is_not():
    """语言门是 0 分项，耗尽重试必须报错；字数不是，两者收尾方式不同。"""
    english = json.dumps(
        {"title": "T", "content": [{"speaker": 1, "text": "Pure English here."}]},
        ensure_ascii=False,
    )
    with pytest.raises(ScriptError, match="中文"):
        next(_writer(FakeClient([english, english, english])).iter_acts(_case()))


# ---------- 逐幕语言门的两个阈值（C2）----------

def _ratio_act_json(title, cjk, digits, n_turns=4):
    """构造一「幕」假响应，其汉字占比恰为 cjk/(cjk+digits)。

    cjk_ratio 的分母只计汉字与拉丁字母/数字（标点、空白不计），所以
    「汉字 × cjk + 阿拉伯数字 × digits」的文本正好命中目标占比。
    """
    per_turn = (cjk + digits) // n_turns
    cjk_per_turn = per_turn * cjk // (cjk + digits)
    text = "中" * cjk_per_turn + "1" * (per_turn - cjk_per_turn)
    return json.dumps(
        {
            "title": title,
            "content": [{"speaker": (i % 2) + 1, "text": text} for i in range(n_turns)],
        },
        ensure_ascii=False,
    )


def test_low_chinese_ratio_act_is_accepted_after_retries(caplog):
    """中文占比 0.80 的幕不得让 writer 失败——它远高于灾难性阈值。

    0.85 是规范对**整篇**交付物的要求，逐幕套用会误杀数字密集的主体幕
    （cjk_ratio 把阿拉伯数字算作非中文；20/55/25 权重下主体幕 0.75 对应全篇
    0.854）。pipeline 的逐幕判据是 CATASTROPHIC_LANGUAGE_RATIO = 0.5，writer
    若仍用 0.85 先失败，那条裁定就永远不可达：一个全稿 0.89 的案子会死在
    主体幕上。
    """
    low = _ratio_act_json("T", cjk=272, digits=68)  # 0.80，恰好 340 字
    turns = parse_act(low)
    ratio = cjk_ratio("".join(t.text for t in turns))
    assert ratio == pytest.approx(0.80)
    assert ratio < 0.85

    client = FakeClient([low, low, low] + _acts()[1:])
    with caplog.at_level(logging.WARNING, logger="aiboke.script_writer"):
        acts = list(_writer(client).iter_acts(_case()))

    assert len(client.calls) == 5, "第一幕重试两次后接受，后两幕各一次"
    assert acts[0].turns == turns, "接受的是模型产出的这一幕，而不是丢弃它"
    # 提示仍按整篇阈值（0.85）措辞：模型该被推着去写中文，只是不再据此判失败
    assert "0.85" in client.calls[1]["user"]
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "建议性偏差" in messages and "0.800" in messages, messages


def test_act_below_catastrophic_threshold_still_fails():
    """几乎不是中文的幕（占比 0.20）仍是 0 分项，耗尽重试必须报错。"""
    bad = _ratio_act_json("T", cjk=68, digits=272)  # 0.20
    turns = parse_act(bad)
    assert cjk_ratio("".join(t.text for t in turns)) == pytest.approx(0.20)

    with pytest.raises(ScriptError, match="中文"):
        next(_writer(FakeClient([bad, bad, bad])).iter_acts(_case()))


def test_writer_and_pipeline_share_one_catastrophic_threshold():
    """两处逐幕判据必须是同一个常量。

    各写一份数字的下场是其中一处静默失效：writer 用 0.85 时 pipeline 的
    0.5 永远不可达（C2 的成因），反过来则会让整幕英文悄悄溜到下游。
    """
    import inspect

    from aiboke import gates, pipeline

    default = inspect.signature(ScriptWriter.__init__).parameters[
        "chinese_catastrophic_ratio"
    ].default
    assert default == pipeline.CATASTROPHIC_LANGUAGE_RATIO
    assert default == gates.DEFAULT_CATASTROPHIC_CHINESE_RATIO


# ---------- char_scale（时长越界后的重生成，I1）----------

def test_iter_acts_scales_char_targets_by_char_scale():
    """char_scale 按比例缩放三幕目标。

    pipeline 在实测时长越界后按 target/measured 缩放字数目标并整段重跑
    （设计 §6.1 的第三重保险）；缩放若没传到字数上，重生成就是白跑一遍。
    """
    client = FakeClient([_act_json("T", chars=c) for c in (425, 1169, 531)])
    acts = list(_writer(client).iter_acts(_case(), char_scale=1.25))
    # 1700 字 × 1.25 = 2125，按 20/55/25 分配
    assert [a.char_target for a in acts] == [425, 1169, 531]


def test_iter_acts_rejects_non_positive_char_scale():
    with pytest.raises(ValueError, match="char_scale"):
        next(_writer(FakeClient([])).iter_acts(_case(), char_scale=0))


# ---------- 事实自检的退化必须可见（P11） ----------

def test_factcheck_failure_is_logged_and_keeps_original(caplog):
    client = FakeClient(
        [
            _act_json("T", chars=340),
            _act_json("T", chars=935),
            "这不是 JSON",
            _act_json("T", chars=425),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="aiboke.script_writer"):
        acts = list(_writer(client, factcheck=True).iter_acts(_case()))
    assert len(acts[1].turns) == 4, "自检失败不得丢内容"
    messages = [r.getMessage() for r in caplog.records]
    assert any("事实自检失败" in m and "ScriptError" in m for m in messages), messages


def test_factcheck_logs_exception_type_and_message(caplog):
    client = FakeClient(
        [
            _act_json("T", chars=340),
            _act_json("T", chars=935),
            RuntimeError("server 挂了"),
            _act_json("T", chars=425),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="aiboke.script_writer"):
        acts = list(_writer(client, factcheck=True).iter_acts(_case()))
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "RuntimeError" in messages and "server 挂了" in messages
    assert len(acts[1].turns) == 4, "自检抛异常时也必须保留原文"
