"""Observed identities and completion-gated provisional entry lists."""

from pathlib import Path

import pandas as pd

from .archive import load_event_snapshots
from .contracts import Driver, Event, TargetState
from .features import completed_session_rows


def drivers_from_snapshots(snapshots: list) -> list[Driver]:
    """Use observed provider names/codes/teams, including practice substitutes."""
    drivers: dict[str, Driver] = {}
    for _, summary, tables in sorted(snapshots, key=lambda item: item[0]["session_start"]):
        results = tables.get("results", pd.DataFrame())
        for row in summary.to_dict("records"):
            code = str(row["driver_code"])
            matches = (
                results.loc[results["Abbreviation"].eq(code)]
                if "Abbreviation" in results
                else pd.DataFrame()
            )
            observed = matches.iloc[0].to_dict() if not matches.empty else {}
            color = str(observed.get("TeamColor", "")).lstrip("#")
            if len(color) != 6 or any(c not in "0123456789abcdefABCDEF" for c in color):
                color = "b6c4cd"
            name = observed.get("FullName")
            team = observed.get("TeamName")
            drivers[str(row["driver_id"])] = Driver(
                id=str(row["driver_id"]),
                code=code,
                name=name if isinstance(name, str) and name else code,
                team=team if isinstance(team, str) and team else str(row["team_id"]),
                color="#" + color,
            )
    return sorted(drivers.values(), key=lambda driver: driver.code)


def infer_roster(
    event: Event, target: TargetState, summaries: pd.DataFrame, archive_dir: str | Path
) -> tuple[list[Driver], pd.Timestamp | None]:
    if summaries.empty:
        return [], None
    complete = completed_session_rows(summaries)
    current = complete.loc[
        complete.event_id.eq(event.id)
        & complete.session_type.isin(["Q"] if target.target == "race" else ["FP1", "FP2", "FP3"])
        & pd.to_datetime(complete.session_end, utc=True).le(target.cutoff_at)
        & pd.to_datetime(complete.available_at, utc=True).le(target.cutoff_at)
    ]
    if current.empty:
        return [], None
    session_id = current.sort_values("session_end").iloc[-1].session_id
    latest = current.loc[current.session_id.eq(session_id)]
    snapshots = [
        item
        for item in load_event_snapshots(archive_dir, event.id)
        if item[0]["session_id"] == session_id
    ]
    known = {d.id: d for d in event.entrants}
    known.update({d.id: d for d in drivers_from_snapshots(snapshots)})
    roster = []
    for row in latest.to_dict("records"):
        identifier = str(row["driver_id"])
        if identifier in known:
            roster.append(known[identifier])
            continue
        # Preserve the observed code when an archive has summaries only.
        code = str(row.get("driver_code") or identifier[:3]).upper()
        roster.append(
            Driver(
                id=identifier,
                code=code,
                name=identifier.replace("_", " ").title(),
                team=str(row["team_id"]),
                color="#b6c4cd",
            )
        )
    return roster, pd.to_datetime(latest.available_at, utc=True).max()
