import { describe, expect, it } from 'vitest';
import { calculateEventScores, deriveForecastState, formatProbability, readComparison, writeComparison } from './forecast';

describe('forecast lifecycle', () => {
  const now = Date.parse('2026-09-26T10:00:00Z');
  const cutoff = '2026-09-26T11:00:00Z';

  it('shows the remaining countdown before an unissued cutoff', () => {
    expect(deriveForecastState({ state: 'scheduled', cutoff_at: cutoff, reason: null }, false, now))
      .toEqual({ label: 'Scheduled', detail: 'Forecast cutoff in 1h 00m', tone: 'neutral' });
  });

  it('marks a passed cutoff with no artifact as unavailable', () => {
    expect(deriveForecastState({ state: 'scheduled', cutoff_at: cutoff, reason: null }, false, Date.parse('2026-09-26T11:01:00Z')).label)
      .toBe('Not issued');
  });

  it('distinguishes an issued forecast awaiting its result from a reconciled one', () => {
    const target = { state: 'issued', cutoff_at: cutoff, reason: null };
    expect(deriveForecastState(target, false, now).label).toBe('Awaiting result');
    expect(deriveForecastState(target, true, now).label).toBe('Reconciled');
  });

  it('labels a frozen forecast issued before the target session begins', () => {
    const target = { state: 'issued', cutoff_at: cutoff, reason: null };
    expect(deriveForecastState(target, false, now, 'issued', '2026-09-26T12:00:00Z').label).toBe('Issued');
    expect(deriveForecastState(target, false, Date.parse('2026-09-26T12:01:00Z'), 'issued', '2026-09-26T12:00:00Z').label).toBe('Awaiting result');
  });

  it('keeps historical reconstruction distinct even after results exist', () => {
    expect(deriveForecastState({ state: 'reconstructed', cutoff_at: cutoff, reason: null }, true, now).label)
      .toBe('Reconstructed');
  });

  it('labels synthetic demonstration artifacts without implying a historical backtest', () => {
    expect(deriveForecastState({ state: 'reconstructed', cutoff_at: cutoff, reason: null }, false, now, 'demonstration').label)
      .toBe('Demonstration');
  });
});

describe('public number formatting', () => {
  it('does not imply absolute certainty from rounded probabilities', () => {
    expect(formatProbability(0.004)).toBe('<1%');
    expect(formatProbability(0.996)).toBe('>99%');
    expect(formatProbability(0.246)).toBe('25%');
  });
});

describe('shareable driver comparison', () => {
  const entrants = ['VER', 'NOR', 'LEC'];
  it('accepts two distinct known drivers and preserves other query parameters', () => {
    expect(readComparison('?target=race&a=VER&b=NOR', entrants)).toEqual(['VER', 'NOR']);
    expect(writeComparison('?target=race&a=VER&b=NOR', ['LEC', 'NOR'])).toBe('target=race&a=LEC&b=NOR');
  });
  it('rejects an unknown or repeated driver pair', () => {
    expect(readComparison('?a=VER&b=VER', entrants)).toBeNull();
    expect(readComparison('?a=VER&b=HAM', entrants)).toBeNull();
  });
});

describe('complete event reconciliation', () => {
  const predictions = [
    { driver_id: 'a', expected_position: 1.2, win_probability: .8, podium_probability: 1, top10_probability: 1 },
    { driver_id: 'b', expected_position: 2.1, win_probability: .15, podium_probability: 1, top10_probability: 1 },
    { driver_id: 'c', expected_position: 2.7, win_probability: .05, podium_probability: 1, top10_probability: 1 },
  ];
  it('scores all matched classified positions, including a retired driver with a result position', () => {
    const actuals = [
      { driver_id: 'a', position: 1, status: 'classified' },
      { driver_id: 'b', position: 3, status: 'retired' },
      { driver_id: 'c', position: 2, status: 'classified' },
    ];
    const scores = calculateEventScores(predictions, actuals);
    expect(scores?.count).toBe(3);
    expect(scores?.positionMae).toBeCloseTo(.6);
    expect(scores?.winBrier).toBeCloseTo(0.021666666666666667);
    expect(scores?.podiumBrier).toBe(0);
    expect(scores?.top10Brier).toBe(0);
  });
  it('withholds full-event scores when a driver is DNS or a classification is duplicated', () => {
    expect(calculateEventScores(predictions, [
      { driver_id: 'a', position: 1, status: 'classified' },
      { driver_id: 'b', position: null, status: 'DNS' },
      { driver_id: 'c', position: 2, status: 'classified' },
    ])).toBeNull();
    expect(calculateEventScores(predictions, [
      { driver_id: 'a', position: 1, status: 'classified' },
      { driver_id: 'b', position: 1, status: 'classified' },
      { driver_id: 'c', position: 2, status: 'classified' },
    ])).toBeNull();
    expect(calculateEventScores(predictions, [
      { driver_id: 'a', position: 1, status: 'classified' },
      { driver_id: 'b', position: 3, status: 'DNS' },
      { driver_id: 'c', position: 2, status: 'classified' },
    ])).toBeNull();
  });
});
