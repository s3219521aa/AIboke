"""MOSS-TTSD 语音合成适配器的测试。

相对 brief 的调整（均在实现里落地）：

  1. 删除 `test_format_input_jsonl_rejects_mismatched_voices`：它把 女+女 音色对
     当作非法输入，但同性别双主播是合法配置（音色区分度由 voices.resolve_pair
     保证）。替换为 `test_format_input_jsonl_accepts_same_gender_pair`。
  2. 默认夹具 `_voices()` 带上了 reference_audio：llama.cpp 后端的文本里只有
     [S1]/[S2] 轮次标签、标签不含性别，因此该后端被要求必须拿到参考音频，
     否则报错（见 F1）。要测「缺参考音频必须报错」用 `_voices_without_refs()`。

全部测试通过注入 FakeRunner 替代 subprocess，不启动任何真实进程。
"""

import json
import sys
from pathlib import Path

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
    """默认音色对：带参考音频。

    llama.cpp 后端只有 [S1]/[S2] 轮次标签（不含性别），参考音频是它唯一能
    据以控制音色的条件，所以「正常输入」就应当带参考音频。
    """
    return VoicePair(
        speaker1=VoicePreset(
            id="m_calm", gender="男", description="低沉男声",
            reference_audio="/voices/m_calm.wav",
        ),
        speaker2=VoicePreset(
            id="f_clear", gender="女", description="清亮女声",
            reference_audio="/voices/f_clear.wav",
        ),
    )


def _voices_without_refs():
    """只有音色描述、没有参考音频——llama.cpp 后端必须拒绝这种输入。"""
    return VoicePair(
        speaker1=VoicePreset(id="m_calm", gender="男", description="低沉男声"),
        speaker2=VoicePreset(id="f_clear", gender="女", description="清亮女声"),
    )


class FakeRunner:
    """替代 subprocess.run；记录命令并按需产出文件或抛出异常。"""

    def __init__(self, *, writes=None, returncode=0, stderr="", raises=None):
        self.calls = []
        self._writes = writes
        self._returncode = returncode
        self._stderr = stderr
        self._raises = raises

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self._raises:
            raise self._raises
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


def _writes_file(path):
    return lambda cmd: path.write_bytes(b"RIFF")


def test_llamacpp_invokes_binary_with_max_tokens_from_target(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 120.0)

    cmd = runner.calls[0]
    assert cmd[0] == "/opt/llama-moss-tts"
    joined = " ".join(cmd)
    assert "--max-new-tokens" in joined
    # 120 秒 * 12.5 = 1500
    assert "1500" in joined


def test_llamacpp_passes_speaker_tags_and_model_path(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    joined = " ".join(runner.calls[0])
    assert "[S1]" in joined
    assert "/models/ttsd" in joined


def test_llamacpp_passes_reference_audio_and_reference_text(tmp_path):
    """llama.cpp 后端必须把音色条件传给子进程。

    文本里的 [S1]/[S2] 只标明「这是谁说的」，不含性别；不传参考音频就等于
    放弃对音色性别的控制，会直接违反「性别符合要求」这条 0 分门限。
    """
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)

    cmd = runner.calls[0]
    assert "--reference-audio" in cmd
    assert "S1=/voices/m_calm.wav" in cmd
    assert "S2=/voices/f_clear.wav" in cmd
    assert "--reference-text" in cmd
    assert "S1=低沉男声" in cmd
    assert "S2=清亮女声" in cmd


def test_llamacpp_reference_flag_names_come_from_config(tmp_path):
    """上游 fork 的标志拼写未经上机核对，运维要能在 config.yaml 里改而不用改代码。"""
    cfg = TtsConfig(
        backend="llamacpp",
        binary="/opt/llama-moss-tts",
        model_path="/models/ttsd",
        reference_audio_flag="--voice-ref",
        reference_text_flag="--voice-ref-text",
    )
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    LlamaCppTts(cfg, runner=runner).synthesize(_turns(), _voices(), out, 60.0)

    cmd = runner.calls[0]
    assert "--voice-ref" in cmd and "S1=/voices/m_calm.wav" in cmd
    assert "--voice-ref-text" in cmd
    assert "--reference-audio" not in cmd


def test_llamacpp_rejects_voices_without_reference_audio(tmp_path):
    """缺参考音频时必须报错，并指出两条出路——不得静默合成出与性别无关的音色。"""
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    with pytest.raises(TtsError) as excinfo:
        LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(
            _turns(), _voices_without_refs(), out, 60.0
        )

    msg = str(excinfo.value)
    assert "speaker1" in msg and "speaker2" in msg
    assert "reference_audio" in msg
    assert "configs/voice_presets.json" in msg       # 出路一：预生成音色
    assert "transformers" in msg                     # 出路二：换后端
    assert runner.calls == []                        # 失败必须发生在起进程之前


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
            _turns(), _voices(), Path("o.wav"), 60.0
        )


def test_llamacpp_wraps_missing_binary_into_ttserror(tmp_path):
    """二进制不在 PATH 上时 subprocess 抛 FileNotFoundError，必须归一为 TtsError。

    否则异常类型会穿透调用方的 `except TtsError`，重试/兜底逻辑全部失效。
    """
    out = tmp_path / "act0.wav"
    runner = FakeRunner(raises=FileNotFoundError("[WinError 2] 系统找不到指定的文件"))
    with pytest.raises(TtsError, match="无法启动"):
        LlamaCppTts(_llamacpp_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)


# ---------- TransformersTts ----------

def _tf_cfg():
    return TtsConfig(backend="transformers", model_path="/models/MOSS-TTSD-v1.0")


def test_transformers_invokes_inference_script(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    TransformersTts(_tf_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    joined = " ".join(runner.calls[0])
    assert "inference.py" in joined
    assert "voice_clone_and_continuation" in joined


def test_transformers_uses_current_interpreter_and_absolute_script(tmp_path):
    """解释器用 sys.executable，脚本用绝对路径——不能依赖 PATH 与 cwd。"""
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    TransformersTts(_tf_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)

    cmd = runner.calls[0]
    assert cmd[0] == sys.executable
    assert Path(cmd[1]).is_absolute()
    assert Path(cmd[1]).name == "inference.py"


def test_transformers_honors_configured_inference_script(tmp_path):
    """配置里给出绝对路径时原样使用（部署时脚本在 checkout 的任意位置）。"""
    script = tmp_path / "MOSS-TTSD" / "inference.py"
    cfg = TtsConfig(
        backend="transformers",
        model_path="/models/MOSS-TTSD-v1.0",
        inference_script=str(script),
    )
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    TransformersTts(cfg, runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    assert runner.calls[0][1] == str(script.resolve())


def test_transformers_enables_text_normalize(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    TransformersTts(_tf_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    assert "--text_normalize" in " ".join(runner.calls[0])


def test_transformers_claims_output_written_under_another_name(tmp_path):
    """官方 inference.py 不接受输出文件名，只在 save_dir 下自行取名。

    父进程必须能在子进程成功退出后认领那个文件，否则每次调用都会误报失败。
    """
    out = tmp_path / "act0.wav"
    produced = tmp_path / "MOSS-TTSD_output_0000.wav"

    def _write_own_name(cmd):
        produced.write_bytes(b"RIFF-official")

    runner = FakeRunner(writes=_write_own_name)
    result = TransformersTts(_tf_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)

    assert result == out
    assert out.read_bytes() == b"RIFF-official"
    assert not produced.exists()  # 是移动，不是复制


def test_transformers_does_not_claim_preexisting_wav(tmp_path):
    """认领的只能是「本次新产生」的文件。

    否则某次静默失败会把上一幕的产物搬过来冒充本幕结果——那比直接报错更糟。
    """
    stale = tmp_path / "act0.wav"
    stale.write_bytes(b"RIFF-previous-act")
    out = tmp_path / "act1.wav"

    runner = FakeRunner()  # 子进程退出码 0，但什么都没写
    with pytest.raises(TtsError, match="未产出"):
        TransformersTts(_tf_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    assert not out.exists()


def test_transformers_wraps_missing_interpreter_into_ttserror(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(raises=FileNotFoundError("[WinError 2] 系统找不到指定的文件"))
    with pytest.raises(TtsError, match="无法启动"):
        TransformersTts(_tf_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)


# ---------- build_backend ----------

def test_build_backend_llamacpp():
    assert isinstance(build_backend(_llamacpp_cfg()), LlamaCppTts)


def test_build_backend_transformers():
    assert isinstance(build_backend(_tf_cfg()), TransformersTts)


def test_build_backend_rejects_unknown():
    with pytest.raises(TtsError, match="backend"):
        build_backend(TtsConfig(backend="magic"))
