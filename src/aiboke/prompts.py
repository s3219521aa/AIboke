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
- 自然的插话与接话，例如“对”“没错”“等一下”“我插一句”“这我倒是第一次听说”
- 一位主播抛出信息，另一位要接住并推进，而不是各说各话
- 允许适度的分歧和追问，让对话有张力

【结构】
必须遵循「开场铺垫 → 主体讲述 → 分析总结」的叙事弧线：
- 开场要有钩子，点明主题并制造悬念
- 主体要有故事性，有事件、人物、数据、转折，像讲故事一样推进
- 结尾要分析影响与启示，给出有回味的收束
严禁平铺直叙地罗列信息，严禁“第一点、第二点、第三点”这种讲稿式表达。

【事实准确性（极其重要）】
- 只使用广为人知、可核实的商业事实、人物和数据
- 如果对某个具体数字没有把握，宁可不给数字，改用定性描述
  （例如用“销量翻了好几倍”“市占率大幅下滑”代替编造的精确百分比）
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
        "对每一处你没有十足把握、不确定的断言，请改写为定性表述"
        "（例如把「市占率高达 37.5%」改为「市占率大幅下滑」），"
        "或者干脆删去该细节。\n"
        "对有把握的内容必须原样保留，不要润色、不要改变语气、不要增删对话轮次。\n\n"
        "只输出修订后的完整 JSON **对象**，必须含 title 与 content 两个字段"
        "（content 为对话数组，每项含 speaker 与 text）。"
        "title 请沿用原标题或按内容概括。不要任何解释：\n"
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
