import json

import pytest

from aiboke.voices import load_presets, resolve_pair


def _presets_file(tmp_path, data):
    p = tmp_path / "voice_presets.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


VALID = {
    "presets": [
        {"id": "m_calm", "gender": "男", "description": "低沉磁性的男声，语速偏慢"},
        {"id": "m_lively", "gender": "男", "description": "明亮活泼的男声，语速偏快"},
        {"id": "f_clear", "gender": "女", "description": "清亮温和的女声"},
        {"id": "f_bright", "gender": "女", "description": "明快爽朗的女声"},
    ]
}


def test_load_presets_groups_by_gender(tmp_path):
    presets = load_presets(_presets_file(tmp_path, VALID))
    assert set(presets) == {"男", "女"}
    assert len(presets["男"]) == 2
    assert presets["男"][0].id == "m_calm"


def test_load_presets_rejects_bad_gender(tmp_path):
    bad = {"presets": [{"id": "x", "gender": "male", "description": "d"}]}
    with pytest.raises(ValueError, match="gender"):
        load_presets(_presets_file(tmp_path, bad))


def test_load_presets_rejects_empty(tmp_path):
    with pytest.raises(ValueError, match="presets"):
        load_presets(_presets_file(tmp_path, {"presets": []}))


def test_resolve_pair_mixed_genders(tmp_path):
    presets = load_presets(_presets_file(tmp_path, VALID))
    pair = resolve_pair(presets, "男", "女")
    assert pair.speaker1.gender == "男"
    assert pair.speaker2.gender == "女"


def test_resolve_pair_same_gender_gives_distinct_presets(tmp_path):
    presets = load_presets(_presets_file(tmp_path, VALID))
    for g in ("男", "女"):
        pair = resolve_pair(presets, g, g)
        assert pair.speaker1.id != pair.speaker2.id, "同性别时必须选中不同音色以保证区分度"


def test_resolve_pair_is_deterministic_with_seed(tmp_path):
    presets = load_presets(_presets_file(tmp_path, VALID))
    a = resolve_pair(presets, "男", "男", seed=7)
    b = resolve_pair(presets, "男", "男", seed=7)
    assert (a.speaker1.id, a.speaker2.id) == (b.speaker1.id, b.speaker2.id)


def test_resolve_pair_raises_when_gender_missing(tmp_path):
    data = {"presets": [{"id": "m", "gender": "男", "description": "d"}]}
    presets = load_presets(_presets_file(tmp_path, data))
    with pytest.raises(ValueError, match="女"):
        resolve_pair(presets, "女", "男")


def test_resolve_pair_raises_when_same_gender_needs_two(tmp_path):
    data = {"presets": [{"id": "m", "gender": "男", "description": "d"}]}
    presets = load_presets(_presets_file(tmp_path, data))
    with pytest.raises(ValueError, match="两条"):
        resolve_pair(presets, "男", "男")


def test_load_presets_buckets_by_declared_gender_not_by_id(tmp_path):
    """钉住不变式：桶键来自 preset 自身的 gender 字段。

    check_genders 只校验「请求的性别在预设库中有条目」，发现不了「桶里装了
    另一种性别的音色」。该不一致在 load_presets 里结构上不可能发生——本测试
    把它显式钉住，以防将来改为按 JSON 键分组或手工维护映射时静默退化。
    """
    data = {
        "presets": [
            {"id": "m_looking", "gender": "女", "description": "清亮女声"},
            {"id": "f_looking", "gender": "男", "description": "低沉男声"},
        ]
    }
    presets = load_presets(_presets_file(tmp_path, data))
    assert [p.id for p in presets["女"]] == ["m_looking"]
    assert [p.id for p in presets["男"]] == ["f_looking"]
