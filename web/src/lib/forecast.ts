export type LifecycleTarget = { state: string; cutoff_at: string; reason?: string | null };
export type LifecycleView = { label: string; detail: string; tone: 'neutral' | 'accent' | 'warning' };

export function deriveForecastState(target: LifecycleTarget, hasResult: boolean, now: number, kind?: string, sessionStart?: string): LifecycleView {
  if (kind === 'demonstration') {
    return { label: 'Demonstration', detail: 'Synthetic example only; this is not an issued forecast or historical backtest.', tone: 'warning' };
  }
  if (target.state === 'unavailable') {
    return { label: 'Not issued', detail: target.reason || 'Required inputs were unavailable.', tone: 'warning' };
  }
  if (target.state === 'reconstructed') {
    return { label: 'Reconstructed', detail: 'Created after the event for historical analysis.', tone: 'warning' };
  }
  if (target.state === 'issued') {
    if (hasResult) return { label: 'Reconciled', detail: 'Forecast shown alongside the observed result.', tone: 'accent' };
    if (sessionStart && now < Date.parse(sessionStart)) return { label: 'Issued', detail: 'Forecast is frozen ahead of the target session.', tone: 'accent' };
    return { label: 'Awaiting result', detail: 'Issued forecast is frozen; result pending.', tone: 'accent' };
  }
  const minutes = Math.max(0, Math.ceil((Date.parse(target.cutoff_at) - now) / 60000));
  if (minutes === 0) {
    return { label: 'Not issued', detail: 'The forecast cutoff passed without a published forecast.', tone: 'warning' };
  }
  return { label: 'Scheduled', detail: `Forecast cutoff in ${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`, tone: 'neutral' };
}

export function formatProbability(probability: number): string {
  if (probability < 0.01) return '<1%';
  if (probability > 0.99) return '>99%';
  return `${Math.round(probability * 100)}%`;
}

export function formatFrequency(probability: number, target: 'qualifying' | 'race'): string {
  const sessions = target === 'qualifying' ? 'simulated qualifying sessions' : 'simulated races';
  if (probability < 0.01) return `fewer than 1 in 100 ${sessions}`;
  if (probability < 0.3) return `about 1 in ${Math.round(1 / probability)} ${sessions}`;
  if (probability < 0.95) return `about ${Math.round(probability * 10)} in 10 ${sessions}`;
  if (probability <= 0.99) return `more than 9 in 10 ${sessions}`;
  return `more than 99 in 100 ${sessions}`;
}

export function formatLapTime(seconds: number): string {
  const milliseconds = Math.round(seconds * 1000);
  const minutes = Math.floor(milliseconds / 60000);
  return `${minutes}:${((milliseconds % 60000) / 1000).toFixed(3).padStart(6, '0')}`;
}

export function readComparison(query: string, entrants: string[]): [string, string] | null {
  const params = new URLSearchParams(query);
  const a = params.get('a');
  const b = params.get('b');
  return a && b && a !== b && entrants.includes(a) && entrants.includes(b) ? [a, b] : null;
}

export function writeComparison(query: string, pair: [string, string]): string {
  const params = new URLSearchParams(query);
  params.set('a', pair[0]);
  params.set('b', pair[1]);
  return params.toString();
}

export type ScorePrediction = { driver_id: string; expected_position: number; win_probability: number; podium_probability: number; top10_probability: number };
export type ScoreActual = { driver_id: string; position?: number | null; status: string };
export type EventScores = { count: number; positionMae: number; winBrier: number; podiumBrier: number; top10Brier: number };

export function calculateEventScores(predictions: ScorePrediction[], actuals: ScoreActual[]): EventScores | null {
  const count = predictions.length;
  if (!count || actuals.length !== count) return null;
  const results = new Map(actuals.map((actual) => [actual.driver_id, actual]));
  if (results.size !== count || new Set(predictions.map((item) => item.driver_id)).size !== count) return null;
  if (actuals.some((actual) => ['DNS', 'DNQ', 'DSQ'].includes(actual.status.toUpperCase()))) return null;
  const positions = predictions.map((prediction) => results.get(prediction.driver_id)?.position);
  if (positions.some((position) => typeof position !== 'number' || !Number.isInteger(position) || position < 1 || position > count)) return null;
  if (new Set(positions).size !== count) return null;
  const squared = (probability: number, outcome: boolean) => (probability - Number(outcome)) ** 2;
  const sum = { positionMae: 0, winBrier: 0, podiumBrier: 0, top10Brier: 0 };
  predictions.forEach((prediction, index) => {
    const position = positions[index] as number;
    sum.positionMae += Math.abs(prediction.expected_position - position);
    sum.winBrier += squared(prediction.win_probability, position === 1);
    sum.podiumBrier += squared(prediction.podium_probability, position <= 3);
    sum.top10Brier += squared(prediction.top10_probability, position <= 10);
  });
  return { count, positionMae: sum.positionMae / count, winBrier: sum.winBrier / count, podiumBrier: sum.podiumBrier / count, top10Brier: sum.top10Brier / count };
}
