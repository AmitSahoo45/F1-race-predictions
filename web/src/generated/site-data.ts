// Generated from schemas/site.schema.json. Do not edit by hand.

export type SchemaVersion = "1.0";
export type GeneratedAt = string;
export type DemoNotice = string | null;
export type Id = string;
export type Season = number;
export type Round = number;
export type Name = string;
export type Circuit = string;
export type Country = string;
export type Timezone = string;
export type Id1 = string;
export type Kind = "FP1" | "FP2" | "FP3" | "Q" | "SQ" | "SS" | "S" | "R";
export type Start = string;
export type End = string;
export type Sessions = Session[];
export type Id2 = string;
export type Code = string;
export type Name1 = string;
export type Team = string;
export type Color = string;
export type Entrants = Driver[];
export type Target = "qualifying" | "race";
export type SessionId = string;
export type CutoffAt = string;
export type State = "scheduled" | "unavailable" | "issued" | "reconstructed";
export type Reason = string | null;
export type ForecastId = string | null;
export type EntrantIds = string[];
export type RosterSource = string | null;
export type RosterObservedAt = string | null;
export type Targets = TargetState[];
export type Events = Event[];
export type SchemaVersion1 = "1.0";
export type Id3 = string;
export type EventId = string;
export type Target1 = "qualifying" | "race";
export type Kind1 = "demonstration" | "reconstructed" | "issued";
export type GeneratedAt1 = string;
export type CutoffAt1 = string;
export type TrainingCutoff = string;
export type ModelVersion = string;
export type ModelKind = "baseline" | "catboost";
export type SampleCount = 20000;
export type Seed = number;
export type AvailableSessions = string[];
export type MissingSessions = string[];
export type Summary = string;
export type Flags = string[];
export type InputSnapshotIds = string[];
/**
 * @minItems 2
 */
export type Drivers = [DriverPrediction, DriverPrediction, ...DriverPrediction[]];
export type DriverId = string;
export type Rank = number;
export type ExpectedPosition = number;
export type P10 = number;
export type P90 = number;
export type WinProbability = number;
export type PodiumProbability = number;
export type Top10Probability = number;
export type PositionProbabilities = number[];
export type Group = string;
export type Contribution = number;
export type Explanations = Explanation[];
export type Forecasts = Forecast[];
export type EventId1 = string;
export type Kind2 = "demonstration" | "observed";
export type Source = string;
export type CircuitPoints = [unknown, unknown][];
export type DriverId1 = string;
export type PracticePaceS = number | null;
export type LongRunPaceS = number | null;
export type Compound = string;
export type Laps = number;
export type PaceS = number;
export type Stints = Stint[];
export type DistanceM = number;
export type SpeedKph = number;
export type ThrottlePct = number;
export type Brake = boolean;
export type Telemetry = TelemetryPoint[];
export type Drivers1 = DriverAnalysis[];
export type Target2 = "qualifying" | "race";
export type DriverId2 = string;
export type Position = number | null;
export type Status = string;
export type Actuals = ActualResult[];
export type Analyses = Analysis[];
export type Status1 = "pending" | "evaluated";
export type Summary1 = string;
export type Season1 = number;
export type Target3 = "qualifying" | "race";
export type EventCount = number;
export type Kind3 = "reconstructed" | "prospective";
export type ProbabilityLoss = number;
export type PositionMae = number;
export type WinBrier = number;
export type PodiumBrier = number;
export type Top10Brier = number;
export type Metric = string;
export type Low = number;
export type High = number;
export type ConfidenceIntervals = ConfidenceInterval[];
export type Outcome = "win" | "podium" | "top10";
export type Predicted = number;
export type Observed = number;
export type Count = number;
export type Calibration = CalibrationBin[];
export type Seasons = SeasonEvaluation[];
export type Name2 = string;
export type Target4 = "qualifying" | "race";
export type EventCount1 = number;
export type Promoted = boolean;
export type Comparison = ModelComparison[];
export type Limitations = string[];

export interface SiteData {
  schema_version?: SchemaVersion;
  generated_at: GeneratedAt;
  demo_notice: DemoNotice;
  events: Events;
  forecasts: Forecasts;
  analyses: Analyses;
  evaluation: Evaluation;
}
export interface Event {
  id: Id;
  season: Season;
  round: Round;
  name: Name;
  circuit: Circuit;
  country: Country;
  timezone: Timezone;
  sessions: Sessions;
  entrants: Entrants;
  targets: Targets;
}
export interface Session {
  id: Id1;
  kind: Kind;
  start: Start;
  end: End;
}
export interface Driver {
  id: Id2;
  code: Code;
  name: Name1;
  team: Team;
  color: Color;
}
export interface TargetState {
  target: Target;
  session_id: SessionId;
  cutoff_at: CutoffAt;
  state: State;
  reason?: Reason;
  forecast_id?: ForecastId;
  entrant_ids?: EntrantIds;
  roster_source?: RosterSource;
  roster_observed_at?: RosterObservedAt;
}
export interface Forecast {
  schema_version?: SchemaVersion1;
  id: Id3;
  event_id: EventId;
  target: Target1;
  kind: Kind1;
  generated_at: GeneratedAt1;
  cutoff_at: CutoffAt1;
  training_cutoff: TrainingCutoff;
  model_version: ModelVersion;
  model_kind: ModelKind;
  sample_count?: SampleCount;
  seed: Seed;
  coverage: Coverage;
  input_snapshot_ids?: InputSnapshotIds;
  drivers: Drivers;
}
export interface Coverage {
  available_sessions: AvailableSessions;
  missing_sessions: MissingSessions;
  summary: Summary;
  flags: Flags;
}
export interface DriverPrediction {
  driver_id: DriverId;
  rank: Rank;
  expected_position: ExpectedPosition;
  p10: P10;
  p90: P90;
  win_probability: WinProbability;
  podium_probability: PodiumProbability;
  top10_probability: Top10Probability;
  position_probabilities: PositionProbabilities;
  explanations: Explanations;
}
export interface Explanation {
  group: Group;
  contribution: Contribution;
}
export interface Analysis {
  event_id: EventId1;
  kind: Kind2;
  source: Source;
  circuit_points: CircuitPoints;
  drivers: Drivers1;
  actuals: Actuals;
}
export interface DriverAnalysis {
  driver_id: DriverId1;
  practice_pace_s: PracticePaceS;
  long_run_pace_s: LongRunPaceS;
  stints: Stints;
  telemetry: Telemetry;
}
export interface Stint {
  compound: Compound;
  laps: Laps;
  pace_s: PaceS;
}
export interface TelemetryPoint {
  distance_m: DistanceM;
  speed_kph: SpeedKph;
  throttle_pct: ThrottlePct;
  brake: Brake;
}
export interface ActualResult {
  target: Target2;
  driver_id: DriverId2;
  position?: Position;
  status: Status;
}
export interface Evaluation {
  status: Status1;
  summary: Summary1;
  seasons: Seasons;
  comparison: Comparison;
  limitations: Limitations;
}
export interface SeasonEvaluation {
  season: Season1;
  target: Target3;
  event_count: EventCount;
  kind: Kind3;
  metrics: Metrics;
  confidence_intervals?: ConfidenceIntervals;
  calibration?: Calibration;
}
export interface Metrics {
  probability_loss: ProbabilityLoss;
  position_mae: PositionMae;
  win_brier: WinBrier;
  podium_brier: PodiumBrier;
  top10_brier: Top10Brier;
}
export interface ConfidenceInterval {
  metric: Metric;
  low: Low;
  high: High;
}
export interface CalibrationBin {
  outcome: Outcome;
  predicted: Predicted;
  observed: Observed;
  count: Count;
}
export interface ModelComparison {
  name: Name2;
  target: Target4;
  event_count: EventCount1;
  metrics: Metrics;
  promoted: Promoted;
}
