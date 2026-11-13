// polymarket-nvidia-ceo-forecast-01 entry (Node 22, standard library only).
//
// Quantile forecast of a one-minute Yes-price grid over the next
// `horizon_minutes` (240) minutes, for every market of a single mutually
// exclusive Polymarket event ("Next CEO of NVIDIA?").
//
// Per market, the model is persistence plus three disciplined adjustments:
//
//   1. Center — the last price, tilted a little toward a short EMA of the
//      recent window. The tilt is scaled down as realized volatility rises,
//      because a volatile market's EMA is a stale estimate of where it trades.
//
//   2. Cross-market sum reversion — the event's candidate markets are mutually
//      exclusive, so their Yes prices should sum to 1. When the sum S at the
//      checkpoint is off by d = S - 1, each market is nudged by
//      -SUM_PULL * d / n_markets (capped), then clamped to [0, 1] so a nudge
//      can never move a price outside the valid range by more than a rounding
//      step. The persistence baseline ignores this signal entirely.
//
//   3. Spread — per-market realized vol per sqrt-minute from the recent 120
//      one-minute diffs, estimated robustly (MAD * sqrt(pi/2) blended with an
//      EWMA standard deviation), damped because sampled diffs over-state the
//      diffusion, floored and capped, and widened as sigma * sqrt(h) like a
//      random walk. Markets pinned near 0 or 1 get their outward side narrowed:
//      a price at 0.01 cannot fall 0.05.
//
// The reference to beat is the house persistence baseline:
//   q_tau(h) = clamp(last + z_tau * 0.0025 * sqrt(h))
// i.e. a fixed 0.0025 per sqrt-minute spread centred on the last price. This
// entry keeps that centre and replaces only the fixed spread with a per-market
// estimate, plus the two corrections below that the baseline ignores.
//
// Output contract: one `<id>.out` per `<id>.in`, each
//   {"forecasts":[{"id": <market id>, "q": [[q05,q25,q50,q75,q95], ...]}]}
// with exactly one row per market, `horizon_minutes` rows per market and five
// finite values per row in [0, 1], non-decreasing, rounded to 6 decimals.
// The program is deterministic: it reads only the case files, uses no clock or
// randomness, and processes files in sorted name order.
//
// Outside the grader's container:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal quantile at each level (six decimals). */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

// --- Model parameters. Fixed in advance; never fitted to answer data. ---

/** Horizon used when a case omits `horizon_minutes`. */
const DEFAULT_HORIZON = 240;
/** Recent window (minutes) for the EMA level and the vol estimate. */
const RECENT = 120;
/** Halflife (minutes) of the fast EMA used for the center tilt. */
const EMA_HALFLIFE = 20;
/** Halflife (minutes) of the EWMA vol estimate. */
const EWMA_HALFLIFE = 30;
/** Weight on the MAD estimate; (1 - this) goes to the EWMA estimate. */
const MAD_WEIGHT = 0.5;
/** Damping of sampled diffs toward the diffusion sigma (grid conservatism). */
const SIGMA_SCALE = 0.85;
/** Realized-vol floor, price units per sqrt-minute (~0.2 cent / sqrt(min)). */
const SIGMA_FLOOR = 0.002;
/** Realized-vol cap, price units per sqrt-minute (~2 cents / sqrt(min)). */
const SIGMA_CAP = 0.02;
/** Fraction of the EMA gap kept at zero vol (shrinks further as vol rises). */
const TILT = 0.25;
/** Vol (per sqrt-minute) at which the EMA tilt is halved. */
const TILT_VOL_REF = 0.004;
/** Shrink applied to the (-mean deviation / n_markets) sum correction. */
const SUM_PULL = 0.3;
/** Absolute cap on the sum correction, in price units (~0.8 cents). */
const SUM_CAP = 0.008;
/** Band near 0 and 1 inside which the outward spread side is narrowed. */
const PIN_BAND = 0.06;
/** Scale applied to the outward spread side of a pinned market. */
const PIN_SHRINK = 0.6;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes?: number;
  quantiles?: number[];
  markets: CaseMarket[];
}

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);
/** Clamp to [0,1] and round to 6 decimals; `+ 0` turns -0 into 0. */
const price6 = (x: number): number => Math.round(clamp01(x) * 1e6) / 1e6 + 0;

/** Clamp x into [lo, hi]. */
const clamp = (x: number, lo: number, hi: number): number => (x < lo ? lo : x > hi ? hi : x);

/** The trailing `n` prices of a grid (the whole grid when it is shorter). */
function recentWindow(prices: number[], n: number): number[] {
  return prices.slice(Math.max(0, prices.length - n));
}

/**
 * Realized vol per sqrt-minute from the recent one-minute diffs: a blend of
 * the robust MAD estimate and an EWMA standard deviation. Returns the undamped,
 * unclamped estimate so callers can reason about the raw volatility.
 */
function realizedVol(prices: number[]): number {
  const window = recentWindow(prices, RECENT + 1);
  if (window.length < 3) return SIGMA_FLOOR;

  const diffs: number[] = [];
  for (let i = 1; i < window.length; i++) diffs.push((window[i] as number) - (window[i - 1] as number));

  // Robust scale: mean |diff| * sqrt(pi/2) equals sigma for Gaussian diffs and
  // is far less sensitive to one-off jumps than a plain standard deviation.
  const meanAbs = diffs.reduce((a, d) => a + Math.abs(d), 0) / diffs.length;
  const mad = meanAbs * Math.sqrt(Math.PI / 2);

  // EWMA standard deviation, so a recent regime change dominates the window.
  const lambda = Math.pow(0.5, 1 / EWMA_HALFLIFE);
  let weight = 0;
  let sum = 0;
  let sumSq = 0;
  for (const d of diffs) {
    weight = lambda * weight + 1;
    sum = lambda * sum + d;
    sumSq = lambda * sumSq + d * d;
  }
  const mean = sum / weight;
  const variance = Math.max(0, sumSq / weight - mean * mean);
  const ewma = Math.sqrt(variance);

  return MAD_WEIGHT * mad + (1 - MAD_WEIGHT) * ewma;
}

/** Fast EMA of the recent window, expressed as a level in price units. */
function emaLevel(prices: number[]): number {
  const window = recentWindow(prices, RECENT);
  if (window.length === 0) return 0;
  const lambda = Math.pow(0.5, 1 / EMA_HALFLIFE);
  let ema = window[0] as number;
  for (let i = 1; i < window.length; i++) ema = lambda * ema + (1 - lambda) * (window[i] as number);
  return ema;
}

/**
 * Per-market center shift implied by the event sum. Mutually exclusive markets
 * should sum to ~1; when they sum to S, each is off by about (S - 1) / n and we
 * pull a fraction of that back, capped so the nudge stays a couple of cents at
 * most.
 */
function sumCorrection(sumLast: number, nMarkets: number): number {
  if (nMarkets <= 0) return 0;
  const dev = (sumLast - 1) / nMarkets;
  return clamp(SUM_PULL * dev, -SUM_CAP, SUM_CAP);
}

/** The forecast rows of one market, with the case-level sum correction. */
function forecast(m: CaseMarket, horizon: number, centerShift: number): number[][] {
  const p = m.prices;
  if (p.length === 0) throw new Error(`market ${m.id} has no prices`);
  const last = p[p.length - 1] as number;

  // --- Spread: robust realized vol, damped, floored, capped. ---
  const vol = realizedVol(p);
  const sigma = clamp(vol * SIGMA_SCALE, SIGMA_FLOOR, SIGMA_CAP);

  // --- Center: persistence, small EMA tilt scaled down by volatility. ---
  const tilt = TILT / (1 + vol / TILT_VOL_REF);
  const mu = clamp01(last + tilt * (emaLevel(p) - last) - centerShift);

  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const s = sigma * Math.sqrt(h);
    const row: number[] = [];
    for (const z of Z) {
      // Pinned markets: narrow the side that would leave [0, 1].
      let scale = 1;
      if (z < 0 && last < PIN_BAND) scale = PIN_SHRINK;
      if (z > 0 && last > 1 - PIN_BAND) scale = PIN_SHRINK;
      row.push(price6(mu + z * s * scale));
    }
    for (let i = 1; i < row.length; i++) {
      if ((row[i] as number) < (row[i - 1] as number)) row[i] = row[i - 1] as number;
    }
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Sorted so two runs over the same inputs write the same bytes in the same order.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();

for (const file of inputs) {
  const caseId = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const horizon = c.horizon_minutes ?? DEFAULT_HORIZON;

  if (c.quantiles !== undefined && (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j]))) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  }

  // Event-level signal: the last price of every market of this case.
  const sumLast = c.markets.reduce((acc, m) => acc + (m.prices[m.prices.length - 1] ?? 0), 0);
  const shift = sumCorrection(sumLast, c.markets.length);

  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, horizon, shift) }));
  writeFileSync(join(OUTPUT_DIR, `${caseId}.out`), `${JSON.stringify({ forecasts })}\n`);
}
