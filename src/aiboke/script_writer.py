"""三幕式文稿生成。

为什么分三幕而不是一次生成整篇：
  1. 把「开场铺垫 -> 主体讲述 -> 分析总结」的结构约束进流程，
     而不是指望模型自觉遵守（评分第 5 条明确要求该结构）
  2. 第一幕一产出即可开始语音合成，与后续幕的生成重叠执行，
     缩短「首幕音频就绪」的时间（注意：交付的 mp3 要等三幕全部完成才
     发布，所以它并不缩短交付音频的首字时间——见 pipeline 模块头部）

校验的优先级分三层：
  1. 语言门与双人门对应赛题的两个 0 分项，耗尽重试即报错。语言门逐幕用
     **灾难性**阈值（gates.DEFAULT_CATASTROPHIC_CHINESE_RATIO），与
     pipeline 的逐幕判据共享同一个常量；整篇阈值（0.85）只用来生成重试
     提示，让模型知道该写中文，不作接受/拒绝的判据。
  2. 中文占比低于整篇阈值、单幕字数偏离目标：带反馈重试，但耗尽重试后
     仍然接受该幕——让它们升级成整案失败是更坏的结果。字数对应的时长
     门限由下游单独把关，中文占比由整篇门限把关。
  3. 输出不可解析：先让模型拿着自己的原文修复一次，再失败才重试整个生成。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterator, Sequence

from .gates import (
    DEFAULT_CATASTROPHIC_CHINESE_RATIO,
    DEFAULT_CHINESE_MIN_RATIO,
    DEFAULT_SPEAKER_MIN_SHARE,
    GateResult,
    all_passed,
    check_chinese,
    check_two_speakers,
    cjk_ratio,
    first_retry_hint,
)
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

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# 单幕字数的容忍区间：偏离即提示重写，但不作为失败条件
_LENGTH_LOW_RATIO = 0.6
_LENGTH_HIGH_RATIO = 1.4


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

    # 无条件按最外层大括号切片：模型既会在前面加客套话，也会在 JSON 之后
    # 补一句「希望有帮助」。只在开头不是 { 时才切片会漏掉后一种情况。
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


def _turns_text(turns: Sequence[Turn]) -> str:
    return "".join(t.text for t in turns)


def _act_chars(turns: Sequence[Turn]) -> int:
    return len(_turns_text(turns))


def _length_hint(actual: int, char_target: int) -> str | None:
    """单幕字数是否偏离目标过多；偏离则给出改写指令。

    返回 None 表示字数在容忍区间内。这只是建议性检查——详见模块开头
    关于两类校验优先级差异的说明。
    """
    if actual < _LENGTH_LOW_RATIO * char_target:
        return (
            f"上一幕只写了约 {actual} 字，目标是约 {char_target} 字。"
            "请把这一幕扩充到接近目标字数，仍有内容可讲，不要草草收尾。"
        )
    if actual > _LENGTH_HIGH_RATIO * char_target:
        return (
            f"上一幕写了约 {actual} 字，超出目标 {char_target} 字较多。"
            "请压缩到接近目标字数，删去重复和离题的枝节。"
        )
    return None


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
        chinese_min_ratio: float = DEFAULT_CHINESE_MIN_RATIO,
        speaker_min_share: float = DEFAULT_SPEAKER_MIN_SHARE,
        chinese_catastrophic_ratio: float = DEFAULT_CATASTROPHIC_CHINESE_RATIO,
    ) -> None:
        self._client = client
        self._target_seconds = target_seconds
        self._chars_per_minute = chars_per_minute
        self._max_retries = max_retries
        self._factcheck = factcheck
        self._chinese_min_ratio = chinese_min_ratio
        self._speaker_min_share = speaker_min_share
        self._chinese_catastrophic_ratio = chinese_catastrophic_ratio

    def _gate_verdicts(self, turns: Sequence[Turn]) -> list[GateResult]:
        """逐幕的 0 分门限判定——接受/拒绝只看这里。

        生成与事实自检共用同一个入口，保证两处的阈值不会走偏——两个门
        都对应 0 分项，事实自检的重写同样必须过这两关。

        语言门用**灾难性**阈值而非整篇的 0.85：0.85 是规范对整篇交付物的
        要求，逐幕套用会误杀数字密集的主体幕（cjk_ratio 把阿拉伯数字算作
        非中文），而这类误杀正是整案失败的主要来源之一。0.85 只在
        `_advisory` 里作为重试提示使用。
        """
        return [
            check_chinese(_turns_text(turns), self._chinese_catastrophic_ratio),
            check_two_speakers(turns, self._speaker_min_share),
        ]

    def _advisory(self, turns: Sequence[Turn], char_target: int) -> tuple[str | None, str]:
        """建议性偏差的反馈：低于整篇阈值的中文占比、偏离目标的单幕字数。

        两者都不改变接受/拒绝的判据（0 分门限已在 _gate_verdicts 里判定），
        只在还有重试名额时把模型往更好的方向推，耗尽重试后接受该幕。语言
        优先于字数：它对应 0 分项，字数只对应时长，而时长在下游另有硬门限。

        返回 (重试提示, 失败原因)；提示为 None 表示没有建议性偏差。
        """
        text = _turns_text(turns)
        strict = check_chinese(text, self._chinese_min_ratio)
        if strict.retry_hint is not None:
            # 提示按整篇阈值（0.85）措辞：模型该知道的仍然是「要写中文」，
            # 只是我们不再据此判失败
            return (
                strict.retry_hint,
                f"中文占比 {cjk_ratio(text):.3f} 低于整篇阈值 {self._chinese_min_ratio}",
            )

        actual = _act_chars(turns)
        length_hint = _length_hint(actual, char_target)
        if length_hint is not None:
            return length_hint, f"字数约 {actual} 字，偏离目标 {char_target} 字较多"
        return None, ""

    def iter_acts(self, case: CaseInput, char_scale: float = 1.0) -> Iterator[ActResult]:
        """逐幕产出文稿。

        生成器是惰性的：调用方拿到第一幕后即可开始语音合成，与后续幕的
        文稿生成重叠执行，从而缩短「首幕音频就绪」的时间（交付 mp3 要等
        三幕全部完成，见 pipeline 模块头部）。若改成先返回完整 Transcript
        再合成，该收益即丧失。

        char_scale 缩放三幕的字数目标，供 pipeline 在实测时长越界后按偏差
        重生成（设计 §6.1 的第三重保险）：时长只有在语音合成之后才测得到，
        这条反馈无法在单幕内部消费。
        """
        if char_scale <= 0:
            raise ValueError(f"char_scale 必须为正数，实际为 {char_scale}")

        total = target_chars(self._target_seconds * char_scale, self._chars_per_minute)
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
        自检是增强而非必需步骤：失败或退化时一律保留原文——宁可留下可能有
        偏差的表述，也不能丢失内容，更不能让重写把 0 分项搞砸。主体幕占全篇
        55%，重写若变成英文或塌成单人独白，下游没有任何环节能兜住，所以这里
        要按与生成时相同的门限再校验一次。两种情况都必须留下日志：这一步保护
        的是内容准确性，悄悄退化会让问题无从察觉。
        """
        if not turns:
            return turns
        try:
            raw = self._client.complete(
                SYSTEM_PROMPT,
                build_factcheck_prompt(turns),
                json_schema=SCRIPT_JSON_SCHEMA,
            )
            rewritten = parse_act(raw)
        except Exception as exc:  # noqa: BLE001 — 自检是增强，不是必需步骤
            logger.warning(
                "主体幕事实自检失败，保留原文（内容准确性可能受影响）：%s: %s",
                type(exc).__name__,
                exc,
            )
            return turns

        verdicts = self._gate_verdicts(rewritten)
        if not all_passed(verdicts):
            logger.warning(
                "事实自检的重写未通过门限，保留原文：%s",
                "；".join(v.detail for v in verdicts if not v.passed),
            )
            return turns
        return rewritten

    def _write_act(
        self,
        case: CaseInput,
        act_index: int,
        char_target: int,
        prior_turns: Sequence[Turn],
    ) -> tuple[tuple[Turn, ...], str]:
        """生成单幕，最多尝试 max_retries + 1 次。

        带反馈重试的三类不合格：输出不可解析（每轮先多花一次修复调用）、
        0 分门限不过关（语言门用灾难性阈值、双人门）、以及建议性偏差
        （中文占比低于整篇阈值、字数偏离目标）。每一轮先判 0 分门限——它们
        对应 0 分项，反馈更关键，优先送给模型；过了才看建议性偏差。
        收尾方式不同：0 分门限不过关、或始终拿不回可解析输出时报错；只有
        建议性偏差时接受该幕，不升级成整案失败。
        """
        retry_hint: str | None = None
        last_error = ""
        advisory_ok: tuple[tuple[Turn, ...], str] | None = None

        for _attempt in range(self._max_retries + 1):
            prompt = build_act_prompt(case, act_index, char_target, prior_turns, retry_hint)
            raw = self._client.complete(SYSTEM_PROMPT, prompt, json_schema=SCRIPT_JSON_SCHEMA)

            try:
                turns = parse_act(raw)
            except ScriptError as exc:
                # 坏 JSON 先让模型拿着自己的原文改正一次；修复也失败则带反馈重试
                # 整个生成——每次失败只多花一次修复调用，而多一次生成机会在
                # 离线环境里比早失败更有价值（整案失败是仅次于 0 分的坏结果）。
                repaired = self._try_repair(raw, str(exc))
                try:
                    turns = parse_act(repaired)
                except ScriptError as exc2:
                    last_error = f"输出不是合法 JSON，修复后仍无法解析：{exc2}"
                    advisory_ok = None
                    retry_hint = (
                        f"上一次输出不是合法 JSON（{exc2}）。"
                        "请只输出 JSON 对象，不要任何解释文字或代码块标记。"
                    )
                    continue
                raw = repaired

            verdicts = self._gate_verdicts(turns)
            hint = first_retry_hint(verdicts)
            if hint is not None:
                last_error = "；".join(v.detail for v in verdicts if not v.passed)
                advisory_ok = None
                retry_hint = hint
                continue

            hint, reason = self._advisory(turns, char_target)
            if hint is None:
                return turns, raw

            last_error = reason
            advisory_ok = (turns, raw)
            retry_hint = hint

        if advisory_ok is not None:
            logger.warning(
                "第 %d 幕在 %d 次尝试后仍有建议性偏差（目标 %d 字），"
                "接受该幕以免整案失败：%s",
                act_index + 1,
                self._max_retries + 1,
                char_target,
                last_error,
            )
            return advisory_ok

        raise ScriptError(
            f"第 {act_index + 1} 幕在 {self._max_retries + 1} 次尝试后仍未通过校验：{last_error}"
        )

    def _try_repair(self, raw: str, error: str) -> str:
        try:
            return self._client.complete(
                SYSTEM_PROMPT, build_repair_prompt(raw, error), json_schema=SCRIPT_JSON_SCHEMA
            )
        except Exception as exc:  # noqa: BLE001 — 修复失败则交回上层重试
            # 端点挂掉与「模型又输出了一段坏 JSON」表象相同，不记日志就分不清
            logger.warning(
                "JSON 修复调用失败，交回上层带反馈重试：%s: %s", type(exc).__name__, exc
            )
            return raw
