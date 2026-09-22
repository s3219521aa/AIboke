"""时长换算与三幕字数分配。纯函数，无副作用。

时长门限是四个 0 分项之一：必须 300 <= 秒数 <= 900，否则整案 0 分。
本模块提供三重保险中的前两重：脚本字数反算 与 生成长度硬限制。
"""

from __future__ import annotations

import math
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
    return math.ceil(seconds * TOKENS_PER_SECOND)


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
