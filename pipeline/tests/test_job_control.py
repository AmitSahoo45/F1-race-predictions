from datetime import UTC, datetime, timedelta

import pytest
from f1forecast.job_control import JobControl, WorkStopped


def test_retry_that_cannot_finish_before_shutdown_stops_without_waiting(monkeypatch):
    monkeypatch.setattr(
        "f1forecast.job_control.time.sleep", lambda _: pytest.fail("slept past budget")
    )
    with pytest.raises(WorkStopped, match="retry"):
        JobControl(stop_at=datetime.now(UTC) + timedelta(minutes=1)).sleep(3601)


def test_operator_stop_file_interrupts_checkpointable_work(tmp_path):
    flag = tmp_path / "stop.flag"
    control = JobControl(stop_file=flag)
    control.check()
    flag.touch()
    with pytest.raises(WorkStopped, match="operator"):
        control.check()
