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

# 逐幕（而非整篇）语言门的**硬失败**阈值。0.85 是规范对整篇交付物的要求，
# 逐幕套用会误杀：cjk_ratio 把阿拉伯数字算作非中文，数字密集的主体幕可能
# 低于 0.85 而全稿仍在 0.85 以上（20/55/25 权重下，主体幕 0.75 对应全篇
# 0.854）。因此逐幕只拦「这一幕基本不是中文」的灾难性情况（整幕英文的
# 幻觉输出），其余交整篇门限。
#
# 定义在这里而不是各处各写一份：script_writer（逐幕接受/拒绝）与 pipeline
# （逐幕提前中止）用的是**同一个判据**，两处取值不同会让其中一处静默失效。
DEFAULT_CATASTROPHIC_CHINESE_RATIO = 0.5


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
