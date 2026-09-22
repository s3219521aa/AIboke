"""llama.cpp server 的 OpenAI 兼容客户端。

为什么走 HTTP 而不是直接加载模型：llama.cpp 以 llama-server 常驻，
模型只加载一次，多个 case 复用，避免反复加载 19GB 权重。

thinking 模式必须关闭——Qwen3.8 默认开启思考，会先输出大段推理内容，
既浪费时间又会污染文稿。
"""

from __future__ import annotations

import requests

from .config import LlmConfig


class LlmError(RuntimeError):
    """LLM 调用失败。"""


class LlmClient:
    def __init__(self, cfg: LlmConfig, session=None) -> None:
        self._cfg = cfg
        self._session = session or requests.Session()

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> str:
        body: dict = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._cfg.temperature,
            "top_p": self._cfg.top_p,
            "presence_penalty": self._cfg.presence_penalty,
            "stream": False,
            # 关闭思考模式：Qwen3.8 默认开启，会先产出推理 token
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "podcast_script", "schema": json_schema},
            }

        url = f"{self._cfg.base_url.rstrip('/')}/v1/chat/completions"
        try:
            resp = self._session.post(url, json=body, timeout=self._cfg.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 — 统一转成领域异常
            raise LlmError(
                f"调用 llama.cpp server 失败（{url}）：{exc}。"
                "请确认 llama-server 已启动、版本 >= b10450，且二进制与 libggml-cuda.so 同版本。"
            ) from exc

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"llama.cpp 响应结构与预期不符：{payload!r}") from exc

        if not isinstance(content, str) or not content.strip():
            raise LlmError("llama.cpp 返回了空内容")

        return content
