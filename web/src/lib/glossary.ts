export const GLOSSARY = {
  'probability-loss': {
    term: 'Probability loss',
    definition: 'How surprised a forecast was by the full observed order: its Plackett–Luce negative log-likelihood divided by that of a random order. 1.0 matches guessing; lower is better.',
  },
  'position-mae': {
    term: 'Position MAE',
    definition: 'Mean absolute error: the average gap, in places, between each driver’s expected and actual position. Lower is better.',
  },
  'brier-score': {
    term: 'Brier score',
    definition: 'The mean squared difference between a forecast probability and the outcome, counted as 1 if it happened and 0 if not. 0 is perfect; lower is better.',
  },
  'plackett-luce': {
    term: 'Plackett–Luce',
    definition: 'A model of complete finishing orders. It fills positions one at a time, choosing each driver in proportion to their strength among those not yet placed.',
  },
  'long-run-pace': {
    term: 'Long-run pace',
    definition: 'The median lap time over practice runs of at least three consecutive usable laps within one tyre stint: an indicator of race pace.',
  },
  calibration: {
    term: 'Calibration',
    definition: 'Whether forecast probabilities match how often things happen: outcomes given 30% should occur about 30% of the time.',
  },
  reconstruction: {
    term: 'Reconstruction',
    definition: 'A forecast rebuilt after the event from sessions that ended before its cutoff, for historical evaluation. It was not published before the session.',
  },
} as const;

export type GlossaryId = keyof typeof GLOSSARY;
