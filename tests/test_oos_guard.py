"""Critical: OOS guard must prevent any data load past IS_END."""

from __future__ import annotations

import os
from datetime import date

import pytest

from data.splits import (
    IS_END,
    OOS_START,
    OOSAccessError,
    assert_is_only,
    clip_to_is,
    is_oos_unlocked,
)


def test_assert_blocks_end_past_is_end():
    os.environ.pop("OOS_UNLOCKED", None)
    with pytest.raises(OOSAccessError):
        assert_is_only(date(2018, 1, 1), date(2025, 1, 1))


def test_assert_blocks_start_past_is_end():
    os.environ.pop("OOS_UNLOCKED", None)
    with pytest.raises(OOSAccessError):
        assert_is_only(OOS_START, date(2025, 6, 1))


def test_assert_passes_within_is():
    os.environ.pop("OOS_UNLOCKED", None)
    assert_is_only(date(2018, 1, 1), IS_END)
    assert_is_only(date(2020, 6, 1), date(2022, 6, 1))


def test_unlock_allows_oos():
    os.environ["OOS_UNLOCKED"] = "true"
    try:
        assert is_oos_unlocked()
        # Should NOT raise after unlock
        assert_is_only(date(2018, 1, 1), date(2025, 1, 1))
    finally:
        os.environ.pop("OOS_UNLOCKED", None)


def test_clip_to_is_truncates_when_locked():
    os.environ.pop("OOS_UNLOCKED", None)
    s, e = clip_to_is(date(2018, 1, 1), date(2025, 12, 31))
    assert s == date(2018, 1, 1)
    assert e == IS_END


def test_clip_to_is_passthrough_when_unlocked():
    os.environ["OOS_UNLOCKED"] = "true"
    try:
        s, e = clip_to_is(date(2018, 1, 1), date(2025, 12, 31))
        assert s == date(2018, 1, 1)
        assert e == date(2025, 12, 31)
    finally:
        os.environ.pop("OOS_UNLOCKED", None)
