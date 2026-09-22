"""三幕式文稿生成器的测试。

相对 brief 的三处假数据调整（断言本身只在 --1-- 处改动，原因写在测试内）：

  1. 每条假响应按它对应那一幕的目标字数构造。测试配置下三幕目标为
     340/935/425 字，它们的 [0.6x, 1.4x] 区间互不重叠——第二幕的下限
     561 字已经高于第一幕的上限 476 字——所以不存在一份「通用长度」的
     假文稿能同时落在三幕的字数区间内。`_acts()` 负责按幕取长度。
  2. 含阿拉伯数字的句子（如「市占率高达 37.5%。」）汉字占比只有 0.56，
     会被语言门判为不合格并触发重试，因此事实自检用句改为全角中文。
  3. 事实自检需要一次额外调用，对应的假响应要补上。
"""

import json
import logging

import pytest

from aiboke.script_writer import ScriptError, ScriptWriter, extract_json, parse_act
from aiboke.schema import CaseInput

_DEFAULT_SENTENCE = "这是一段中文的对话内容。"

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
    claim = "市占率高达百分之三十七点五。"
    client = FakeClient(
        [
            _act_json("T", chars=340, text=claim),
            _act_json("T", chars=935, text=claim),
            _act_json("T", text="市占率大幅下滑。"),
            _act_json("T", chars=425, text=claim),
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
    client = FakeClient(["坏输出", "还是坏的", _act_json("T"), _act_json("T")])
    with pytest.raises(ScriptError, match="修复"):
        _writer(client).write(_case())


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
