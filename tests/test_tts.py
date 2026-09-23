"""MOSS-TTSD 语音合成适配器的测试。

相对 brief 的一处调整：

  原 `test_format_input_jsonl_rejects_mismatched_voices` 用 女+女 音色对
  作为「非法输入」，该前提是错的：同性别两位主播（男男/女女）是合法配置，
  音色区分度由 voices.resolve_pair 保证。删除该测试，替换为
  `test_format_input_jsonl_accepts_same_gender_pair`，防止未来加上的过度校验
  把合法配置判死。

全部测试通过注入 FakeRunner 替代 subprocess，不启动任何真实进程。
"""

import json

import pytest

from aiboke.config import TtsConfig
from aiboke.schema import Turn, VoicePreset, VoicePair
from aiboke.tts import (
    LlamaCppTts,
    TransformersTts,
    TtsError,
    build_backend,
    format_input_jsonl,
    format_tagged_script,
)


def _turns():
    return (Turn(1, "大家好，欢迎收听。"), Turn(2, "没错，今天聊个有意思的话题。"))


def _voices():
    return VoicePair(
        speaker1=VoicePreset(id="m_calm", gender="男", description="低沉男声"),
        speaker2=VoicePreset(id="f_clear", gender="女", description="清亮女声"),
    )


class FakeRunner:
    """替代 subprocess.run；记录命令并按需产出文件。"""

    def __init__(self, *, writes=None, returncode=0, stderr=""):
        self.calls = []
        self._writes = writes
        self._returncode = returncode
        self._stderr = stderr

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self._writes:
            self._writes(cmd)
        return type("R", (), {"returncode": self._returncode, "stderr": self._stderr, "stdout": ""})()


# ---------- 脚本格式化 ----------

def test_format_tagged_script_uses_speaker_tags():
    assert format_tagged_script(_turns()) == (
        "[S1]大家好，欢迎收听。[S2]没错，今天聊个有意思的话题。"
    )


def test_format_tagged_script_rejects_empty():
    with pytest.raises(TtsError, match="轮次"):
        format_tagged_script(())


def test_format_input_jsonl_contains_voice_descriptions_and_tags():
    payload = json.loads(format_input_jsonl(_turns(), _voices()))
    assert "[S1]" in payload["text"] and "[S2]" in payload["text"]
    assert payload["prompt_text_speaker1"].startswith("[S1]")
    assert payload["prompt_text_speaker2"].startswith("[S2]")
    assert payload["voice_description_speaker1"] == "低沉男声"


def test_format_input_jsonl_accepts_same_gender_pair():
    """同性别的两位主播是合法配置（男男/女女），不得被拒绝。

    音色区分度由 voices.resolve_pair 保证（同性别时选取不同预设），
    本函数只负责格式化，不应重复该职责。
    """
    pair = VoicePair(
        speaker1=VoicePreset(id="f_clear", gender="女", description="清亮女声"),
        speaker2=VoicePreset(id="f_bright", gender="女", description="明快女声"),
    )
    payload = json.loads(format_input_jsonl(_turns(), pair))
    assert payload["voice_description_speaker1"] == "清亮女声"
    assert payload["voice_description_speaker2"] == "明快女声"


# ---------- LlamaCppTts ----------

def _llamacpp_cfg():
    return TtsConfig(backend="llamacpp", binary="/opt/llama-moss-tts", model_path="/models/ttsd")


def test_llamacpp_invokes_binary_with_max_tokens_from_target(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 120.0)

    cmd = runner.calls[0]
    assert cmd[0] == "/opt/llama-moss-tts"
    joined = " ".join(cmd)
    assert "--max-new-tokens" in joined
    # 120 秒 * 12.5 = 1500
    assert "1500" in joined


def test_llamacpp_passes_speaker_tags_and_model_path(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    joined = " ".join(runner.calls[0])
    assert "[S1]" in joined
    assert "/models/ttsd" in joined


def test_llamacpp_raises_on_nonzero_exit(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(returncode=1, stderr="CUDA error: out of memory")
    with pytest.raises(TtsError, match="out of memory"):
        LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)


def test_llamacpp_raises_when_no_output_file(tmp_path):
    out = tmp_path / "missing.wav"
    runner = FakeRunner()  # 不写文件，模拟静默失败
    with pytest.raises(TtsError, match="未产出"):
        LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)


def test_llamacpp_raises_when_binary_missing_from_config():
    cfg = TtsConfig(backend="llamacpp", binary=None, model_path="/models/ttsd")
    with pytest.raises(TtsError, match="binary"):
        LlamaCppTts(cfg, runner=FakeRunner()).synthesize(
            _turns(), _voices(), __import__("pathlib").Path("o.wav"), 60.0
        )


# ---------- TransformersTts ----------

def _tf_cfg():
    return TtsConfig(backend="transformers", model_path="/models/MOSS-TTSD-v1.0")


def test_transformers_invokes_inference_script(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    TransformersTts(_tf_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    joined = " ".join(runner.calls[0])
    assert "inference.py" in joined
    assert "voice_clone_and_continuation" in joined


def test_transformers_enables_text_normalize(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    TransformersTts(_tf_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    assert "--text_normalize" in " ".join(runner.calls[0])


# ---------- build_backend ----------

def test_build_backend_llamacpp():
    assert isinstance(build_backend(_llamacpp_cfg()), LlamaCppTts)


def test_build_backend_transformers():
    assert isinstance(build_backend(_tf_cfg()), TransformersTts)


def test_build_backend_rejects_unknown():
    with pytest.raises(TtsError, match="backend"):
        build_backend(TtsConfig(backend="magic"))
