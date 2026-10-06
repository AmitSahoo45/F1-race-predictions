// Generated from schemas/telemetry.schema.json. Do not edit by hand.

export type EventId = string;
export type DriverId = string;
export type SessionId = string;
/**
 * @minItems 1
 */
export type Points = [TelemetryPoint, ...TelemetryPoint[]];
export type DistanceM = number;
export type SpeedKph = number;
export type ThrottlePct = number;
export type Brake = boolean;
export type Traces = TelemetryTrace[];

/**
 * Selected public traces, published beside site.json and loaded on demand.
 */
export interface EventTelemetry {
  event_id: EventId;
  traces: Traces;
}
export interface TelemetryTrace {
  driver_id: DriverId;
  session_id: SessionId;
  points: Points;
}
export interface TelemetryPoint {
  distance_m: DistanceM;
  speed_kph: SpeedKph;
  throttle_pct: ThrottlePct;
  brake: Brake;
}
