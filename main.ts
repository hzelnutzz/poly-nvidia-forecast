// polymarket-nvidia-ceo-forecast-01 entry (Node 22, standard library only).
//
// Model: vol-scaled mean-reversion around a smoothed level, shrunk toward the
// last price as a function of recent volatility.
//
// For each market, from its 1-minute price grid ending at the checkpoint `at`:
//
//   1. Smoothed level: a fast EMA of the recent prices, anchored on the last
//      price so the median forecast cannot drift far from what the market
//      actually trades.
//
//   2. Mean-reversion pull: the smoothed level is blended with the last price
//      by a shrinkage factor that tightens when volatility is low (prices are
//      sticky, persistence wins) and loosens when volatility is high (momentum
//      beyond the tick is noise, the smoothed level is the better bet):
//         mu = last + (1 - shrink) * (level - last),  shrink = clamp(v / SHRINK_VOL_MAX, SHRINK_MIN, 1)
//      where v is the realized vol per sqrt-minute over the recent window.
//
//   3. Spread: realized vol per square-root minute, floored and capped, widened
//      with horizon like a random walk:
//         sigma_h = clamp(v, SIGMA_FLOOR, SIGMA_CAP) * sqrt(h)
//         q_tau(h) = clamp(mu + z_tau * sigma_h, 0, 1)
//
// Realized vol v = std of the recent 1-minute price diffs (simple diffs in
// price units, robust enough on this forward-filled grid), damped by a factor
// < 1 because sampled diffs on a coarse grid over-roughen the diffusion.
//
// Output format is unchanged from the house baseline:
//   q_tau(h) rows for h = 1..horizon_minutes, five quantiles each, clamped to
//   [0,1], rounded to 6 decimals, non-decreasing in tau by construction.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

// --- Model parameters (fixed in advance, not tuned on any answer data) ---
/** Recent window (minutes) for EMA and realized vol. */
const RECENT = 120;
/** EMA halflife in minutes within that window. */
const HALFLIFE = 20;
/** Realized-vol floor / cap, in price units per sqrt-minute. */
const SIGMA_FLOOR = 0.0020;
const SIGMA_CAP = 0.02;
/** Dampening of sampled diffs toward the diffusion sigma. */
const SIGMA_SCALE = 0.85;
/** Shrinkage of the smoothed level toward the last price: tighter when calm. */
const SHRINK_MIN = 0.8;
/** Vol level (per sqrt-minute) at which the level is fully the last price. */
const SHRINK_VOL_MAX = 0.02;
/** Minutes over which the EMA is re-anchored toward the last price. */
const ANCHOR_MINUTES = 45;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Realized vol per sqrt-minute: std of recent 1-minute diffs, damped. */
function realizedVol(prices: number[]): number {
  const n = Math.min(RECENT, prices.length - 1);
  if (n < 2) return SIGMA_FLOOR;
  const diffs: number[] = [];
  for (let i = prices.length - n; i < prices.length; i++) diffs.push((prices[i] as number) - (prices[i - 1] as number));
  const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  const varr = diffs.reduce((a, d) => a + (d - mean) * (d - mean), 0) / (diffs.length - 1);
  return Math.sqrt(Math.max(0, varr)) * SIGMA_SCALE;
}

/** The vol-scaled mean-reversion forecast of one market. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const p = m.prices;
  const last = p[p.length - 1] as number;

  // Realized vol and its implied shrinkage toward the last price.
  const v = realizedVol(p);
  const shrink = Math.min(1, Math.max(SHRINK_MIN, v / SHRINK_VOL_MAX));

  // Fast EMA over the recent window, decayed per minute.
  const lambda = Math.pow(0.5, 1 / HALFLIFE);
  const start = Math.max(1, p.length - RECENT);
  let ema = p[start - 1] as number;
  let w = 1;
  for (let i = start; i < p.length; i++) {
    ema = lambda * ema + (1 - lambda) * (p[i] as number);
    w = lambda * w + 1;
  }
  ema /= w;

  // Anchor the level on the last price: keep only part of the EMA's distance.
  const rho = 0.3;
  const level = last + rho * (ema - last);

  // Mean-reversion center: blend level with last by the vol-dependent shrink.
  const mu = last + (1 - shrink) * (level - last);

  const sigma = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, v));
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const s = sigma * Math.sqrt(h);
    rows.push(Z.map((z) => price6(mu + z * s)));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
