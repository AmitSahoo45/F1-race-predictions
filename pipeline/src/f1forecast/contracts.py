"""The single source of truth for the static Python / TypeScript interface."""

from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator


def as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UTCDate = Annotated[AwareDatetime, AfterValidator(as_utc)]
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$")]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Target = Literal["qualifying", "race"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class Driver(Contract):
    id: Identifier
    code: str
    name: str
    team: str
    color: Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]


class Session(Contract):
    id: Identifier
    kind: Literal["FP1", "FP2", "FP3", "Q", "SQ", "SS", "S", "R"]
    start: UTCDate
    end: UTCDate

    @model_validator(mode="after")
    def chronological(self):
        if self.end <= self.start:
            raise ValueError("session end must follow start")
        return self


class TargetState(Contract):
    target: Target
    session_id: Identifier
    cutoff_at: UTCDate
    state: Literal["scheduled", "unavailable", "issued", "reconstructed"]
    reason: str | None = None
    forecast_id: Identifier | None = None
    entrant_ids: list[Identifier] = Field(default_factory=list)
    roster_source: str | None = None
    roster_observed_at: UTCDate | None = None


class Event(Contract):
    id: Identifier
    season: int = Field(ge=2018, le=2100)
    round: int = Field(ge=1, le=40)
    name: str
    circuit: str
    country: str
    timezone: str
    sessions: list[Session]
    entrants: list[Driver]
    targets: list[TargetState]

    @model_validator(mode="after")
    def check_event(self):
        ZoneInfo(self.timezone)
        for items in (self.sessions, self.entrants):
            if len({item.id for item in items}) != len(items):
                raise ValueError("duplicate session or entrant identity")
        if len({t.target for t in self.targets}) != len(self.targets):
            raise ValueError("duplicate prediction target")
        sessions = {session.id: session for session in self.sessions}
        for target in self.targets:
            if len(target.entrant_ids) != len(set(target.entrant_ids)) or not set(target.entrant_ids).issubset({d.id for d in self.entrants}):
                raise ValueError("target roster must contain unique known entrants")
            if target.roster_observed_at and target.roster_observed_at > target.cutoff_at:
                raise ValueError("target roster was not known before cutoff")
            session = sessions.get(target.session_id)
            if session is None or session.kind != {"race": "R", "qualifying": "Q"}[target.target]:
                raise ValueError("target must refer to its qualifying or race session")
            if target.cutoff_at != session.start - timedelta(minutes=30):
                raise ValueError("forecast cutoff must be 30 minutes before session")
            if target.state in ("issued", "reconstructed") and not target.forecast_id:
                raise ValueError("issued or reconstructed target requires a forecast")
            if target.state == "unavailable" and not target.reason:
                raise ValueError("unavailable target requires a reason")
        return self


class Coverage(Contract):
    available_sessions: list[str]
    missing_sessions: list[str]
    summary: str
    flags: list[str]


class Explanation(Contract):
    group: str
    contribution: Finite


class DriverPrediction(Contract):
    driver_id: Identifier
    rank: int = Field(ge=1)
    expected_position: Finite = Field(ge=1)
    p10: int = Field(ge=1)
    p90: int = Field(ge=1)
    win_probability: Probability
    podium_probability: Probability
    top10_probability: Probability
    position_probabilities: list[Probability]
    explanations: list[Explanation]


class Forecast(Contract):
    schema_version: Literal["1.0"] = "1.0"
    id: Identifier
    event_id: Identifier
    target: Target
    kind: Literal["demonstration", "reconstructed", "issued"]
    generated_at: UTCDate
    cutoff_at: UTCDate
    training_cutoff: UTCDate
    model_version: Identifier
    model_kind: Literal["baseline", "catboost"]
    sample_count: Literal[20000] = 20000
    seed: int
    coverage: Coverage
    input_snapshot_ids: list[str] = Field(default_factory=list)
    drivers: list[DriverPrediction] = Field(min_length=2)

    @model_validator(mode="after")
    def consistent_distribution(self):
        if self.training_cutoff >= self.cutoff_at:
            raise ValueError("training cutoff must precede prediction cutoff")
        if self.kind == "issued" and self.generated_at > self.cutoff_at:
            raise ValueError("issued forecast generated after cutoff")
        n = len(self.drivers)
        if len({d.driver_id for d in self.drivers}) != n:
            raise ValueError("duplicate forecast driver")
        if sorted(d.rank for d in self.drivers) != list(range(1, n + 1)):
            raise ValueError("forecast ranks must be a complete permutation")
        for row in self.drivers:
            probs = row.position_probabilities
            if len(probs) != n or abs(sum(probs) - 1) > 1e-6:
                raise ValueError("each driver needs a normalized complete position distribution")
            if not 1 <= row.p10 <= row.p90 <= n:
                raise ValueError("invalid position interval")
            expected = sum((i + 1) * p for i, p in enumerate(probs))
            if abs(row.expected_position - expected) > 1e-5:
                raise ValueError("expected position disagrees with distribution")
            if any(abs(a - b) > 1e-6 for a, b in [
                (row.win_probability, probs[0]),
                (row.podium_probability, sum(probs[:3])),
                (row.top10_probability, sum(probs[:10])),
            ]):
                raise ValueError("outcome marginal disagrees with position distribution")
        if any(abs(sum(d.position_probabilities[i] for d in self.drivers) - 1) > 1e-6
               for i in range(n)):
            raise ValueError("each finishing position must have one occupant")
        return self


class TelemetryPoint(Contract):
    distance_m: Finite = Field(ge=0)
    speed_kph: Finite = Field(ge=0, le=450)
    throttle_pct: Finite = Field(ge=0, le=100)
    brake: bool


class Stint(Contract):
    compound: str
    laps: int = Field(ge=1)
    pace_s: Finite = Field(gt=0)


class DriverAnalysis(Contract):
    driver_id: Identifier
    practice_pace_s: Finite | None
    long_run_pace_s: Finite | None
    stints: list[Stint]
    telemetry: list[TelemetryPoint]


class ActualResult(Contract):
    target: Target
    driver_id: Identifier
    position: int | None = Field(default=None, ge=1)
    status: str


class Analysis(Contract):
    event_id: Identifier
    kind: Literal["demonstration", "observed"]
    source: str
    circuit_points: list[tuple[Finite, Finite]]
    drivers: list[DriverAnalysis]
    actuals: list[ActualResult]


class Metrics(Contract):
    probability_loss: Finite = Field(ge=0)
    position_mae: Finite = Field(ge=0)
    win_brier: Finite = Field(ge=0)
    podium_brier: Finite = Field(ge=0)
    top10_brier: Finite = Field(ge=0)


class CalibrationBin(Contract):
    outcome: Literal["win", "podium", "top10"]
    predicted: Probability
    observed: Probability
    count: int = Field(ge=0)


class ConfidenceInterval(Contract):
    metric: str
    low: Finite
    high: Finite


class SeasonEvaluation(Contract):
    season: int
    target: Target
    event_count: int = Field(ge=1)
    kind: Literal["reconstructed", "prospective"]
    metrics: Metrics
    confidence_intervals: list[ConfidenceInterval] = Field(default_factory=list)
    calibration: list[CalibrationBin] = Field(default_factory=list)


class ModelComparison(Contract):
    name: str
    target: Target
    event_count: int = Field(ge=1)
    metrics: Metrics
    promoted: bool


class Evaluation(Contract):
    status: Literal["pending", "evaluated"]
    summary: str
    seasons: list[SeasonEvaluation]
    comparison: list[ModelComparison]
    limitations: list[str]

    @model_validator(mode="after")
    def evidence(self):
        if self.status == "pending" and (self.seasons or self.comparison):
            raise ValueError("pending evaluation cannot claim measured results")
        if self.status == "evaluated" and not self.seasons:
            raise ValueError("evaluated status needs event evidence")
        return self


class SiteData(Contract):
    schema_version: Literal["1.0"] = "1.0"
    generated_at: UTCDate
    demo_notice: str | None
    events: list[Event]
    forecasts: list[Forecast]
    analyses: list[Analysis]
    evaluation: Evaluation

    @model_validator(mode="after")
    def referential_integrity(self):
        events = {e.id: e for e in self.events}
        forecasts = {f.id: f for f in self.forecasts}
        if len(events) != len(self.events) or len(forecasts) != len(self.forecasts):
            raise ValueError("duplicate artifact identity")
        if (any(f.kind == "demonstration" for f in self.forecasts) or any(
                a.kind == "demonstration" for a in self.analyses)) and not self.demo_notice:
            raise ValueError("synthetic demonstrations require a visible disclosure")
        for f in self.forecasts:
            if f.event_id not in events:
                raise ValueError("forecast references an unknown event")
            entrants = {d.id for d in events[f.event_id].entrants}
            if not {d.driver_id for d in f.drivers}.issubset(entrants):
                raise ValueError("forecast references an unknown driver")
        for event in self.events:
            for target in event.targets:
                if target.forecast_id:
                    f = forecasts.get(target.forecast_id)
                    if not f or (f.event_id, f.target, f.cutoff_at) != (
                            event.id, target.target, target.cutoff_at):
                        raise ValueError("manifest forecast reference is inconsistent")
                    if target.state == "issued" and f.kind != "issued":
                        raise ValueError("only prospective forecasts can be marked issued")
        for analysis in self.analyses:
            if analysis.event_id not in events:
                raise ValueError("analysis references an unknown event")
        return self
