"""Cooperative deadlines for resumable local downloads."""

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


class WorkStopped(KeyboardInterrupt):
    """Leave provider retry loops without misreporting an unavailable session."""


class JobControl:
    def __init__(self, stop_at: datetime | None = None, stop_file: Path | None = None):
        if stop_at is not None and stop_at.tzinfo is None:
            raise ValueError("stop deadline must be timezone-aware")
        self.stop_at = stop_at
        self.stop_file = stop_file

    def check(self) -> None:
        if self.stop_file is not None and self.stop_file.exists():
            raise WorkStopped("operator stop file requested a checkpoint")
        if self.stop_at is not None and datetime.now(UTC) >= self.stop_at:
            raise WorkStopped("download deadline reached")

    def sleep(self, seconds: float) -> None:
        self.check()
        if (
            self.stop_at is not None
            and datetime.now(UTC) + timedelta(seconds=seconds) >= self.stop_at
        ):
            raise WorkStopped("provider retry would exceed download deadline; resume later")
        remaining = max(0.0, seconds)
        while remaining > 0:
            part = min(remaining, 1.0)
            time.sleep(part)
            remaining -= part
            self.check()
