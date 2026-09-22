import pytest

from aiboke.length import (
    ACT_WEIGHTS,
    DurationVerdict,
    act_char_targets,
    chars_to_seconds,
    classify_duration,
    seconds_to_max_tokens,
    target_chars,
)


def test_target_chars_at_defaults():
    # 510 秒 = 8.5 分钟，8.5 * 200 = 1700 字
    assert target_chars(510.0, 200.0) == 1700


def test_target_chars_rounds_to_int():
    assert isinstance(target_chars(500.0, 200.0), int)


def test_chars_to_seconds_inverts_target_chars():
    assert chars_to_seconds(1700, 200.0) == pytest.approx(510.0)


def test_act_char_targets_at_defaults_sums_exactly():
    got = act_char_targets(1700, ACT_WEIGHTS)
    assert got == (340, 935, 425)
    assert sum(got) == 1700


def test_act_char_targets_sums_exactly_when_not_divisible():
    got = act_char_targets(1001, ACT_WEIGHTS)
    assert sum(got) == 1001


def test_act_char_targets_rejects_bad_weights():
    with pytest.raises(ValueError, match="weights"):
        act_char_targets(1000, (0.5, 0.5))


def test_seconds_to_max_tokens_uses_official_ratio():
    assert seconds_to_max_tokens(510.0) == 6375
    assert seconds_to_max_tokens(1.0) == 13


def test_classify_duration_ok_at_target():
    assert classify_duration(510.0) is DurationVerdict.OK


def test_classify_duration_ok_within_warn_delta():
    # |430 - 510| = 80 <= 90
    assert classify_duration(430.0) is DurationVerdict.OK


def test_classify_duration_warn_beyond_delta_but_in_range():
    # |400 - 510| = 110 > 90，但仍在 [300, 900] 内
    assert classify_duration(400.0) is DurationVerdict.WARN


def test_classify_duration_warn_at_lower_bound():
    assert classify_duration(300.0) is DurationVerdict.WARN


def test_classify_duration_out_of_range_low():
    assert classify_duration(299.9) is DurationVerdict.OUT_OF_RANGE


def test_classify_duration_out_of_range_high():
    assert classify_duration(900.1) is DurationVerdict.OUT_OF_RANGE
