import pytest
import yaml

from aiboke.config import Config, load_config


def _write(tmp_path, data):
    p = tmp_path / "default.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


BASE = {
    "target_seconds": 510.0,
    "chars_per_minute": 200.0,
    "llm": {"base_url": "http://127.0.0.1:8080", "model": "qwen3.8-27b"},
    "tts": {"backend": "llamacpp", "binary": "/opt/llama.cpp/build-cuda/bin/llama-moss-tts"},
    "cover": {"steps": 8, "size": 1024},
    "models_root": "/models",
}


def test_load_config_ok(tmp_path):
    cfg = load_config(_write(tmp_path, BASE))
    assert isinstance(cfg, Config)
    assert cfg.target_seconds == 510.0
    assert cfg.llm.model == "qwen3.8-27b"
    assert cfg.tts.backend == "llamacpp"
    assert cfg.cover.size == 1024


def test_load_config_rejects_bad_backend(tmp_path):
    data = {**BASE, "tts": {**BASE["tts"], "backend": "magic"}}
    with pytest.raises(ValueError, match="backend"):
        load_config(_write(tmp_path, data))


def test_load_config_rejects_target_outside_allowed_window(tmp_path):
    data = {**BASE, "target_seconds": 1200.0}
    with pytest.raises(ValueError, match="target_seconds"):
        load_config(_write(tmp_path, data))


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_reads_native_tts_paths_and_flag_names(tmp_path):
    """P23 新增的字段：两个附加 GGUF 的路径与全部标志名。"""
    data = {
        **BASE,
        "tts": {
            **BASE["tts"],
            "audio_encoder_model": "/models/enc.gguf",
            "audio_decoder_model": "/models/dec.gguf",
            "model_flag": "--backbone",
            "output_flag": "--out-wav",
            "audio_encoder_flag": "--enc",
            "audio_decoder_flag": "--dec",
            "reference_audio_flag": "--voice-ref",
            "reference_text_flag": "--voice-ref-text",
        },
    }
    cfg = load_config(_write(tmp_path, data))
    assert cfg.tts.audio_encoder_model == "/models/enc.gguf"
    assert cfg.tts.audio_decoder_model == "/models/dec.gguf"
    assert cfg.tts.model_flag == "--backbone"
    assert cfg.tts.output_flag == "--out-wav"
    assert cfg.tts.audio_encoder_flag == "--enc"
    assert cfg.tts.audio_decoder_flag == "--dec"
    assert cfg.tts.reference_audio_flag == "--voice-ref"
    assert cfg.tts.reference_text_flag == "--voice-ref-text"


def test_load_config_defaults_reference_text_flag_to_empty(tmp_path):
    """默认不发 reference-text：llama-moss-tts 没有这个标志（fork 源码参数表为准）。"""
    cfg = load_config(_write(tmp_path, BASE))
    assert cfg.tts.reference_text_flag == ""
    assert cfg.tts.model_flag == "-m"
    assert cfg.tts.output_flag == "--wav-out"
    assert cfg.tts.audio_encoder_model is None
    assert cfg.tts.audio_decoder_model is None
    # decoder 是 native 路径的必需参数，缺失由适配器报错；校验本身默认开启
    assert cfg.tts.require_audio_decoder is True
    assert cfg.tts.codec_model_path is None
    assert cfg.tts.codec_model_flag == "--codec_model_path"


# ---------- 「存在但为空」的配置段 ----------

def test_load_config_rejects_bare_key_section_from_yaml_text(tmp_path):
    """`tts:`（键在、值为空）必须报 ValueError，而不是 TypeError。

    YAML 把它解析成 None，旧实现直接取键会抛 TypeError——它不是 CLI/HTTP
    入口捕获的异常类型，于是操作员看到的是一段 traceback，而不是
    「配置错误：tts 不能为空」这一行干净信息。
    """
    p = tmp_path / "default.yaml"
    p.write_text(
        "target_seconds: 510.0\nchars_per_minute: 200.0\n"
        "llm:\n  base_url: http://127.0.0.1:8080\n  model: m\n"
        "tts:\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tts"):
        load_config(p)


@pytest.mark.parametrize("key", ["llm", "tts", "cover"])
def test_load_config_rejects_null_section(tmp_path, key):
    """三段共同的约定：给了键就必须给内容，null 是配置错误。"""
    data = {**BASE, key: None}
    with pytest.raises(ValueError, match=key):
        load_config(_write(tmp_path, data))


def test_load_config_rejects_empty_file(tmp_path):
    """空文件同样是用法错误，给出可读信息而不是「缺少 target_seconds」。"""
    p = tmp_path / "default.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="空"):
        load_config(p)


def test_load_config_rejects_non_mapping_document(tmp_path):
    """顶层是列表（YAML 写成了流水账）时报配置错误，不抛 TypeError。"""
    p = tmp_path / "default.yaml"
    p.write_text("- target_seconds\n- 510\n", encoding="utf-8")
    with pytest.raises(ValueError, match="映射"):
        load_config(p)
