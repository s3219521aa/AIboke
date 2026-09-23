"""MOSS-TTSD 语音合成适配器的测试。

相对 brief 的调整（均在实现里落地）：

  1. 删除 `test_format_input_jsonl_rejects_mismatched_voices`：它把 女+女 音色对
     当作非法输入，但同性别双主播是合法配置（音色区分度由 voices.resolve_pair
     保证）。替换为 `test_format_input_jsonl_accepts_same_gender_pair`。
  2. 默认夹具 `_voices()` 带上了 reference_audio：llama.cpp 后端的文本里只有
     [S1]/[S2] 轮次标签、标签不含性别，因此该后端被要求必须拿到参考音频，
     否则报错（见 F1）。要测「缺参考音频必须报错」用 `_voices_without_refs()`。
  3. Ruling P23：`LlamaCppTts` 的命令行按 fork 的源码/文档改写——`-m` +
     `--audio-encoder-model`/`--audio-decoder-model` + `--text` +
     `--reference-audio`（单个拼接文件）+ `--wav-out` + `--max-new-tokens`。
     因此本文件的假 runner 分两个：TTS 进程一个，拼接参考音频的 ffmpeg 一个。

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
    """native 后端的正常配置：backbone 文件 + 必需的 audio decoder。

    decoder 属于 native 路径的必需参数（没有它二进制无法把 token 还原成
    wav），缺了会在起进程前报 TtsError——所以「正常输入」里必须有它。
    「缺 decoder 必须报错」与「fork 不需要时可关掉校验」另有两组用例。
    """
    return TtsConfig(
        backend="llamacpp",
        binary="/opt/llama-moss-tts",
        model_path="/models/ttsd",
        audio_decoder_model="/models/dec.gguf",
    )


def _writes_file(path):
    return lambda cmd: path.write_bytes(b"RIFF")


def _fake_concat_runner():
    """假的 ffmpeg runner：把拼接结果写到命令的最后一个参数上。"""
    return FakeRunner(writes=lambda cmd: Path(cmd[-1]).write_bytes(b"RIFF-combined"))


def _llamacpp(cfg, runner, concat=None):
    """构造 LlamaCppTts，默认给一个会产出拼接文件的假 ffmpeg runner。

    llama-moss-tts 只接受一个 --reference-audio，所以每次合成前都会先拼一次
    参考音频；测试里不能真跑 ffmpeg。
    """
    return LlamaCppTts(cfg, runner=runner, concat_runner=concat or _fake_concat_runner())


def test_llamacpp_invokes_binary_with_max_tokens_from_target(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    _llamacpp(_llamacpp_cfg(), runner).synthesize(_turns(), _voices(), out, 120.0)

    cmd = runner.calls[0]
    assert cmd[0] == "/opt/llama-moss-tts"
    joined = " ".join(cmd)
    assert "--max-new-tokens" in joined
    # 120 秒 * 12.5 = 1500
    assert "1500" in joined


def test_llamacpp_passes_speaker_tags_and_model_path(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    _llamacpp(_llamacpp_cfg(), runner).synthesize(_turns(), _voices(), out, 60.0)
    cmd = runner.calls[0]
    joined = " ".join(cmd)
    assert "[S1]" in joined
    # 模型用 -m 传，值就是 tts.model_path（native 路径下应是 backbone GGUF 文件）
    assert cmd[cmd.index("-m") + 1] == "/models/ttsd"


def test_llamacpp_concatenates_both_speakers_into_single_reference_audio(tmp_path):
    """llama-moss-tts 只接受一个 --reference-audio。

    两位主播的参考音频必须先按 S1 → S2 的顺序拼成一个文件；直接传两个
    `S1=…/S2=…` 值（旧实现）会让 CLI 报未知取值，而少传任何一个都会丢掉
    「性别必须符合」这条 0 分门限的音色条件。
    """
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    concat = _fake_concat_runner()
    _llamacpp(_llamacpp_cfg(), runner, concat=concat).synthesize(
        _turns(), _voices(), out, 60.0
    )

    # ffmpeg 收到两段参考音频，顺序是 speaker1 -> speaker2
    # （用 Path 比较，避免 Windows 的分隔符差异）
    ff_paths = [Path(a) for a in concat.calls[0]]
    s1, s2 = Path("/voices/m_calm.wav"), Path("/voices/f_clear.wav")
    assert s1 in ff_paths and s2 in ff_paths
    assert ff_paths.index(s1) < ff_paths.index(s2)

    # 送给二进制的只有一个 --reference-audio，值是拼接产物而非原始文件
    cmd = runner.calls[0]
    assert cmd.count("--reference-audio") == 1
    combined = cmd[cmd.index("--reference-audio") + 1]
    assert Path(combined).name == "reference.wav"
    assert combined not in ("/voices/m_calm.wav", "/voices/f_clear.wav")
    # 临时文件随临时目录一起清理
    assert not Path(combined).exists()


def test_llamacpp_sends_only_flags_the_binary_actually_has(tmp_path):
    """未知标志会让 CLI 直接失败，因此不能把它们当提示发出去。

    权威标志表来自 fork 源码 tools/tts/run-moss-tts-delay.cpp 与
    docs/moss-tts-firstclass-e2e_zh.md。
    """
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    _llamacpp(_llamacpp_cfg(), runner).synthesize(_turns(), _voices(), out, 60.0)
    cmd = runner.calls[0]

    forbidden = {
        "--output",              # 该二进制是 --wav-out
        "--temperature",         # 只有 --text-temperature / --audio-temperature
        "--top-p", "--top-k",    # 只有 --text-top-p / --audio-top-p 等分通道形式
        "--repetition-penalty",  # 只有 --audio-repetition-penalty
        "--reference-text",      # 该二进制没有这个标志
        "--text-normalize", "--sample-rate-normalize",  # 属官方 inference.py
    }
    assert forbidden.isdisjoint(cmd)
    assert cmd[cmd.index("--wav-out") + 1] == str(out)


def test_llamacpp_passes_audio_encoder_and_decoder_when_configured(tmp_path):
    """native 路径要三个 GGUF：backbone（-m）+ encoder + decoder。"""
    cfg = TtsConfig(
        backend="llamacpp",
        binary="/opt/llama-moss-tts",
        model_path="/models/backbone.gguf",
        audio_encoder_model="/models/enc.gguf",
        audio_decoder_model="/models/dec.gguf",
    )
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    _llamacpp(cfg, runner).synthesize(_turns(), _voices(), out, 60.0)
    cmd = runner.calls[0]
    assert cmd[cmd.index("--audio-encoder-model") + 1] == "/models/enc.gguf"
    assert cmd[cmd.index("--audio-decoder-model") + 1] == "/models/dec.gguf"


def test_llamacpp_allows_fork_without_decoder_when_check_is_disabled(tmp_path):
    """fork 不需要 decoder 时，可显式关掉校验——此时不发送该参数。

    这是「未配置就静默不发送」的唯一合法形态：必须由配置明确说出「本 fork
    不需要 decoder」，而不是让默认配置悄悄走到那条路上。
    """
    cfg = TtsConfig(
        backend="llamacpp",
        binary="/opt/llama-moss-tts",
        model_path="/models/ttsd",
        require_audio_decoder=False,
    )
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    _llamacpp(cfg, runner).synthesize(_turns(), _voices(), out, 60.0)
    cmd = runner.calls[0]
    assert "--audio-encoder-model" not in cmd
    assert "--audio-decoder-model" not in cmd


def test_llamacpp_requires_audio_decoder_before_starting_process(tmp_path):
    """native 路径缺 audio decoder 必须在起进程前报错，并给出可操作的出路。

    旧实现「未配置就不发送」把这条要求留给了二进制：起进程、加载 backbone、
    跑完整轮生成长达十几分钟，最后以一个含糊的 CLI 报错收场。decoder 是
    上游 fork 明列的必需参数，缺失在本地就能判定。
    """
    cfg = TtsConfig(
        backend="llamacpp", binary="/opt/llama-moss-tts", model_path="/models/ttsd"
    )
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    concat = _fake_concat_runner()
    with pytest.raises(TtsError) as excinfo:
        _llamacpp(cfg, runner, concat=concat).synthesize(_turns(), _voices(), out, 60.0)

    msg = str(excinfo.value)
    assert "audio_decoder_model" in msg
    assert "require_audio_decoder" in msg          # 出路二：显式关闭校验
    assert "convert_moss_audio_tokenizer_split_to_gguf.py" in msg  # 出路一：转换 GGUF
    assert runner.calls == []                      # 失败必须发生在起进程之前
    assert concat.calls == []                      # 连参考音频都不该开始拼接


def test_llamacpp_flag_names_come_from_config(tmp_path):
    """上机核对后若要改拼写，运维在 config.yaml 里改即可，不必动代码。"""
    cfg = TtsConfig(
        backend="llamacpp",
        binary="/opt/llama-moss-tts",
        model_path="/models/ttsd",
        model_flag="--backbone",
        output_flag="--out-wav",
        audio_encoder_flag="--enc",
        audio_decoder_flag="--dec",
        reference_audio_flag="--voice-ref",
        reference_text_flag="--voice-ref-text",
        audio_encoder_model="/models/enc.gguf",
        audio_decoder_model="/models/dec.gguf",
    )
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    _llamacpp(cfg, runner).synthesize(_turns(), _voices(), out, 60.0)

    cmd = runner.calls[0]
    assert cmd[cmd.index("--backbone") + 1] == "/models/ttsd"
    assert cmd[cmd.index("--out-wav") + 1] == str(out)
    assert cmd[cmd.index("--enc") + 1] == "/models/enc.gguf"
    assert cmd[cmd.index("--dec") + 1] == "/models/dec.gguf"
    assert cmd[cmd.index("--voice-ref") + 1].endswith("reference.wav")
    # 启用 reference-text 时发送的是合并后的 [S1]…[S2]… 文本
    assert cmd[cmd.index("--voice-ref-text") + 1] == "[S1]低沉男声[S2]清亮女声"
    assert "--reference-audio" not in cmd
    assert "--wav-out" not in cmd


def test_llamacpp_reference_text_flag_is_off_by_default(tmp_path):
    """默认不发 reference-text：llama-moss-tts 没有这个标志（源码参数表为准）。"""
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    _llamacpp(_llamacpp_cfg(), runner).synthesize(_turns(), _voices(), out, 60.0)
    assert "--reference-text" not in runner.calls[0]


def test_llamacpp_rejects_voices_without_reference_audio(tmp_path):
    """缺参考音频时必须报错，并指出两条出路——不得静默合成出与性别无关的音色。"""
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    concat = _fake_concat_runner()
    with pytest.raises(TtsError) as excinfo:
        _llamacpp(_llamacpp_cfg(), runner, concat=concat).synthesize(
            _turns(), _voices_without_refs(), out, 60.0
        )

    msg = str(excinfo.value)
    assert "speaker1" in msg and "speaker2" in msg
    assert "reference_audio" in msg
    assert "configs/voice_presets.json" in msg       # 出路一：预生成音色
    assert "transformers" in msg                     # 出路二：换后端
    assert runner.calls == []                        # 失败必须发生在起进程之前
    assert concat.calls == []                        # 连拼接都不该开始


def test_llamacpp_wraps_concat_failure_into_ttserror(tmp_path):
    """拼接参考音频失败（缺 ffmpeg 等）必须归一为 TtsError。

    否则 AudioError 会绕过调用方的 `except TtsError`，重试/兜底全部失效。
    """
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    concat = FakeRunner(raises=FileNotFoundError("[WinError 2] 找不到 ffmpeg"))
    with pytest.raises(TtsError, match="拼接"):
        _llamacpp(_llamacpp_cfg(), runner, concat=concat).synthesize(
            _turns(), _voices(), out, 60.0
        )
    assert runner.calls == []   # 拼接失败就不该起 TTS 进程


def test_llamacpp_raises_on_nonzero_exit(tmp_path):
    out = tmp_path / "act0.wav"
    runner = FakeRunner(returncode=1, stderr="CUDA error: out of memory")
    with pytest.raises(TtsError, match="out of memory"):
        _llamacpp(_llamacpp_cfg(), runner).synthesize(_turns(), _voices(), out, 60.0)


def test_llamacpp_raises_when_no_output_file(tmp_path):
    out = tmp_path / "missing.wav"
    runner = FakeRunner()  # 不写文件，模拟静默失败
    with pytest.raises(TtsError, match="未产出"):
        _llamacpp(_llamacpp_cfg(), runner).synthesize(_turns(), _voices(), out, 60.0)


def test_llamacpp_raises_when_binary_missing_from_config():
    cfg = TtsConfig(backend="llamacpp", binary=None, model_path="/models/ttsd")
    with pytest.raises(TtsError, match="binary"):
        _llamacpp(cfg, FakeRunner()).synthesize(_turns(), _voices(), Path("o.wav"), 60.0)


def test_llamacpp_wraps_missing_binary_into_ttserror(tmp_path):
    """二进制不在 PATH 上时 subprocess 抛 FileNotFoundError，必须归一为 TtsError。

    否则异常类型会穿透调用方的 `except TtsError`，重试/兜底逻辑全部失效。
    """
    out = tmp_path / "act0.wav"
    runner = FakeRunner(raises=FileNotFoundError("[WinError 2] 系统找不到指定的文件"))
    with pytest.raises(TtsError, match="无法启动"):
        _llamacpp(_llamacpp_cfg(), runner).synthesize(_turns(), _voices(), out, 60.0)


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


def test_transformers_passes_codec_model_path_when_configured(tmp_path):
    """官方 inference.py 的音频编解码器是**独立仓库**（--codec_model_path）。

    离线环境里不指到本地就只能在运行期联网拉取。该标志的上机拼写由配置给出，
    未配置时不发送（见下一条）。
    """
    cfg = TtsConfig(
        backend="transformers",
        model_path="/models/MOSS-TTSD-v1.0",
        codec_model_path="/models/MOSS-Audio-Tokenizer",
    )
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    TransformersTts(cfg, runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    cmd = runner.calls[0]
    assert cmd[cmd.index("--codec_model_path") + 1] == "/models/MOSS-Audio-Tokenizer"


def test_transformers_omits_codec_model_path_when_unset(tmp_path):
    """未配置就不发送，由官方脚本自己的默认值决定——不发未知/空值。"""
    out = tmp_path / "act0.wav"
    runner = FakeRunner(writes=_writes_file(out))
    TransformersTts(_tf_cfg(), runner=runner).synthesize(_turns(), _voices(), out, 60.0)
    assert "--codec_model_path" not in runner.calls[0]


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
