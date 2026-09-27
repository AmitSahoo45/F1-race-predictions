"""Prepare reproducible historical feature/label tables from the local archive."""

from datetime import timedelta
from pathlib import Path

import pandas as pd

from .archive import load_summaries
from .features import build_feature_rows, completed_session_rows
from .training import attach_results


def prepare_training_data(archive_dir: str | Path, destination: str | Path) -> pd.DataFrame:
    summaries = load_summaries(archive_dir)
    if summaries.empty:
        raise ValueError("archive is empty; ingest historical sessions first")
    finalized = completed_session_rows(summaries)
    entries = []
    for _, results in finalized.loc[finalized.session_type.isin(["Q", "R"])].groupby("session_id"):
        session = results.iloc[0]
        target = "qualifying" if session.session_type == "Q" else "race"
        cutoff = pd.Timestamp(session.session_start) - timedelta(minutes=30)
        # Historical roster proxy is the last eligible earlier session, never the target result.
        eligible = finalized.loc[
            finalized.event_id.eq(session.event_id)
            & pd.to_datetime(finalized.session_end, utc=True).le(cutoff)
            & finalized.session_type.isin(
                ["FP1", "FP2", "FP3"] if target == "qualifying" else ["Q"]
            )
        ]
        if eligible.empty:
            continue
        last_id = eligible.sort_values("session_end").iloc[-1].session_id
        for entrant in eligible.loc[eligible.session_id.eq(last_id)].itertuples():
            entries.append(
                {
                    "event_id": session.event_id,
                    "season": int(session.season),
                    "circuit_id": session.circuit_id,
                    "driver_id": entrant.driver_id,
                    "team_id": entrant.team_id,
                    "target": target,
                    "target_session_id": session.session_id,
                    "cutoff": cutoff,
                }
            )
    if not entries:
        raise ValueError("no target has a prior eligible session roster; ingest practice, Q and R")
    features = build_feature_rows(summaries, pd.DataFrame(entries), mode="reconstructed")
    labeled = attach_results(features, summaries)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(destination, index=False)
    return labeled
