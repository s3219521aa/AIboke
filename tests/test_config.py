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
