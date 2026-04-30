"""IS/OOS split definitions and OOS data-access guard.

Hard rule (per global CLAUDE.md ETF2.0 v5i discipline + user instruction
2026-04-28): OOS data must NOT be loaded or referenced during parameter
selection. This module enforces that at the data-layer level — any
adapter call that would touch dates after `IS_END` raises
`OOSAccessError` unless explicitly unlocked.

Unlock procedure: set environment variable ``OOS_UNLOCKED=true`` AFTER
IS results are recorded and approved. The unlock is process-local, not
persistent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Split:
    name: str
    start: date
    end: date


IS_TRAIN = Split("is_train", date(2018, 1, 1), date(2022, 6, 30))
IS_VALIDATE = Split("is_validate", date(2022, 7, 1), date(2023, 12, 31))
OOS_TEST = Split("oos_test", date(2024, 1, 1), date(2026, 4, 28))

IS_START: date = IS_TRAIN.start
IS_END: date = IS_VALIDATE.end          # 2023-12-31
OOS_START: date = OOS_TEST.start        # 2024-01-01
OOS_END: date = OOS_TEST.end

TRAIN_START: date = IS_TRAIN.start      # 2018-01-01
TRAIN_END: date = IS_TRAIN.end          # 2022-06-30
VALIDATE_START: date = IS_VALIDATE.start  # 2022-07-01
VALIDATE_END: date = IS_VALIDATE.end    # 2023-12-31


class OOSAccessError(RuntimeError):
    """Raised when code attempts to access OOS data without unlock."""


def is_oos_unlocked() -> bool:
    return os.environ.get("OOS_UNLOCKED", "").lower() in {"1", "true", "yes"}


def assert_is_only(start: date, end: date) -> None:
    """Validate that a date range is strictly within IS bounds.

    Raises OOSAccessError if `end > IS_END` and the OOS guard is locked.
    """
    if end > IS_END and not is_oos_unlocked():
        raise OOSAccessError(
            f"Refusing to load data with end={end} > IS_END={IS_END}. "
            f"This date range covers OOS territory. Unlock with "
            f"`OOS_UNLOCKED=true` ONLY after IS results are recorded "
            f"and parameter selection is frozen. See data/splits.py."
        )
    if start > IS_END and not is_oos_unlocked():
        raise OOSAccessError(
            f"Refusing to load data with start={start} > IS_END={IS_END}. "
            f"This entire range is OOS. Unlock via `OOS_UNLOCKED=true` "
            f"only after IS results are frozen."
        )


def clip_to_is(start: date, end: date) -> tuple[date, date]:
    """Return (start, end) clipped to [IS_START, IS_END] when guard is locked.

    Use for *automatic* truncation in cases where it is safe to silently
    drop OOS dates (e.g. building a universe panel for IS-only research).
    Returns the original range when the OOS guard is unlocked.
    """
    if is_oos_unlocked():
        return start, end
    return max(start, IS_START), min(end, IS_END)
