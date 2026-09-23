"""CLI 入口、依赖装配与 HTTP 封装的测试。

这里的用例刻意不碰真实模型：只覆盖「输入校验必须先于任何模型工作」的
两条早退路径、装配层是否把配置阈值真正递给 ScriptWriter，以及两个入口
如何把流水线的异常契约（PipelineError / CoverError）翻译给调用方。
不启动子进程解码、不发网络请求、不依赖 GPU 或 llama.cpp 服务。
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aiboke import bootstrap
from aiboke.bootstrap import load_presets_from_config
from aiboke.config import Config, CoverConfig, LlmConfig, TtsConfig
from aiboke.cover import CoverError
from aiboke.pipeline import PipelineError

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_presets_from_config_reads_voice_presets(tmp_path):
    p = tmp_path / "voice_presets.json"
    p.write_text(
        json.dumps({"presets": [{"id": "m", "gender": "男", "description": "d"}]}),
        encoding="utf-8",
    )
    presets = load_presets_from_config(p)
    assert presets["男"][0].id == "m"


def test_cli_rejects_missing_input_file(tmp_path):
    proc = subprocess.run(
        [sys.executable, "scripts/generate.py", "--input", str(tmp_path / "no.json"),
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    assert proc.returncode != 0
    assert "no.json" in (proc.stderr + proc.stdout)


def test_cli_rejects_malformed_case_json(tmp_path):
    bad = tmp_path / "case.json"
    bad.write_text('{"topic": "", "speaker_gender1": "男", "speaker_gender2": "女"}',
                   encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "scripts/generate.py", "--input", str(bad),
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    assert proc.returncode != 0
    assert "topic" in (proc.stderr + proc.stdout)


def test_cli_rejects_non_object_case_json(tmp_path):
    """顶层不是 JSON 对象时必须干净地报错。

    schema.CaseInput.from_dict 对非对象会抛 TypeError（无法从 dict 取值），
    它不是 main 里捕获的异常类型，放任穿透就是给操作员一段 traceback。
    """
    bad = tmp_path / "case.json"
    bad.write_text('["星巴克国内运营转移"]', encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "scripts/generate.py", "--input", str(bad),
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    assert proc.returncode != 0
    err = proc.stderr + proc.stdout
    assert "JSON 对象" in err
    assert "Traceback" not in err


def _config(**over) -> Config:
    base = dict(
        target_seconds=510.0,
        chars_per_minute=200.0,
        models_root="/models",
        llm=LlmConfig(base_url="http://127.0.0.1:8080", model="m"),
        tts=TtsConfig(backend="llamacpp", binary="b", model_path="p"),
        cover=CoverConfig(steps=8, size=1024, model_path="z"),
    )
    base.update(over)
    return Config(**base)


def test_build_pipeline_forwards_gate_thresholds(tmp_path, monkeypatch):
    """两道 0 分门限的阈值必须从 Config 真正传到 ScriptWriter。

    ScriptWriter 的同名参数默认值恰好等于 gates 里的常量，漏传不会改变
    默认行为——只会让 configs/default.yaml 里的修改静默失效。因此这里
    用与默认值不同的数字（0.95 / 0.4）钉住这条线，漏传即 KeyError。
    """
    captured: dict = {}

    class CapturingWriter:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(bootstrap, "ScriptWriter", CapturingWriter)
    cfg = _config(chinese_min_ratio=0.95, speaker_min_share=0.4)

    bootstrap.build_pipeline(cfg, tmp_path / "models")

    assert captured["chinese_min_ratio"] == 0.95
    assert captured["speaker_min_share"] == 0.4


def _load_cli_module():
    """把 scripts/generate.py 当模块加载，以便在主进程内驱动 main()。"""
    path = _REPO_ROOT / "scripts" / "generate.py"
    spec = importlib.util.spec_from_file_location("aiboke_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FailingPipeline:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def run(self, case, out_dir):
        raise self._exc


@pytest.mark.parametrize(
    "exc, prefix",
    [
        (PipelineError("文稿生成或语音合成失败：连接被拒绝"), "生成失败："),
        (CoverError("主封面与兜底封面均失败"), "封面生成失败"),
    ],
)
def test_cli_reports_failures_without_traceback(exc, prefix, tmp_path, monkeypatch, capsys):
    """失败信息要区分「生成失败」与「封面失败且兜底也失败」。"""
    module = _load_cli_module()
    monkeypatch.setattr(module, "build_pipeline", lambda *a, **k: _FailingPipeline(exc))

    code = module.main(
        ["--topic", "星巴克国内运营转移", "--output-dir", str(tmp_path / "out")]
    )

    assert code != 0
    err = capsys.readouterr().err
    assert err.startswith(prefix)
    assert str(exc) in err
    assert "Traceback" not in err


def test_http_endpoint_reports_failures_as_500(monkeypatch):
    """两个入口共用同一异常契约：HTTP 一律 500，封面失败不另设状态码。"""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from aiboke import server

    monkeypatch.setattr(
        server,
        "build_pipeline",
        lambda *a, **k: _FailingPipeline(CoverError("主封面与兜底封面均失败")),
    )
    app = server.create_app(_REPO_ROOT / "configs" / "default.yaml")
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}

    # 请求体本身不合法仍是 422，不能与生成失败混为一谈
    bad = client.post(
        "/generate",
        json={"topic": "星巴克国内运营转移", "speaker_gender1": "中", "speaker_gender2": "女"},
    )
    assert bad.status_code == 422

    resp = client.post(
        "/generate",
        json={"topic": "星巴克国内运营转移", "speaker_gender1": "男", "speaker_gender2": "女"},
    )
    assert resp.status_code == 500
    assert "封面生成失败" in resp.json()["detail"]
