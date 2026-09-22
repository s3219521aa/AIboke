import pytest

from aiboke.schema import CaseInput, Episode, Transcript, Turn, VoicePreset


def test_case_input_from_dict_ok():
    ci = CaseInput.from_dict(
        {"topic": "星巴克国内运营转移", "speaker_gender1": "男", "speaker_gender2": "女"}
    )
    assert ci.topic == "星巴克国内运营转移"
    assert ci.speaker_gender1 == "男"
    assert ci.speaker_gender2 == "女"


def test_case_input_rejects_empty_topic():
    with pytest.raises(ValueError, match="topic"):
        CaseInput.from_dict({"topic": "  ", "speaker_gender1": "男", "speaker_gender2": "女"})


def test_case_input_rejects_bad_gender():
    with pytest.raises(ValueError, match="speaker_gender1"):
        CaseInput.from_dict({"topic": "t", "speaker_gender1": "male", "speaker_gender2": "女"})


def test_transcript_to_json_obj_matches_required_schema():
    t = Transcript(
        title="瑞幸如何靠生椰拿铁翻盘",
        turns=(Turn(speaker=1, text="大家好。"), Turn(speaker=2, text="没错。")),
    )
    obj = t.to_json_obj()
    assert obj == {
        "title": "瑞幸如何靠生椰拿铁翻盘",
        "content": [
            {"speaker": 1, "text": "大家好。"},
            {"speaker": 2, "text": "没错。"},
        ],
    }


def test_turn_rejects_invalid_speaker():
    with pytest.raises(ValueError, match="speaker"):
        Turn(speaker=3, text="x")


def test_episode_holds_three_paths(tmp_path):
    e = Episode(
        audio_path=tmp_path / "a.mp3",
        cover_path=tmp_path / "c.png",
        script_path=tmp_path / "s.json",
    )
    assert e.audio_path.name == "a.mp3"
