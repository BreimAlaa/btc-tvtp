# Bitcoin volatility regimes with a time-varying-transition-probability HMM

Companion repository for the Master's final degree project *Regime detection in Bitcoin volatility
with a TVTP-HMM and a regime-conditioned, volatility-scaled allocation rule*.

Research question: does letting the HMM transition probabilities depend on an observable covariate
improve regime forecasting and downstream trading performance for BTC, in a statistically
significant sense, relative to a constant-transition HMM?

The mathematics (Hamilton filter, Kim smoother, Baum-Welch, Viterbi, the TVTP extension and its
GEM M-step, forecast outputs) is derived in `docs/methodology_derivation.md`; every equation number
referenced in the code comments points there.

## Layout

```
src/tvtp_hmm/
  core.py          filter (3.1-3.5), smoother (4.2-4.3), Viterbi (5.6), EM/GEM (5.2-5.3, 6.5-6.8),
                   direct ML with numerical Hessian (5.4-5.5), forecasts (7.2), LR test (6.9)
  simulate.py      simulators for the tests and the Monte Carlo study (exogenous AR(1) and internal log-RV covariates)
  data.py          Coin Metrics loader, returns, covariates, walk-forward windows, train-window standardisation
  walkforward.py   rolling refits, causal out-of-sample forecasts, per-window LR / BIC
  strategy.py      allocation rules, cost model, performance metrics
  stats_tests.py   Diebold-Mariano (Newey-West, HLN), QLIKE/MSE, PSR, deflated Sharpe, stationary bootstrap
tests/test_core.py unit tests (a)-(e) of the methodology, Student-t emission tests, cross-checks against hmmlearn and statsmodels
scripts/run_synthetic.py   parameter recovery, CI coverage, LR size/power
scripts/run_btc.py         full real-data pipeline, writes results/btc/ (checkpointed walk-forward, --reuse, --K, --skip_fullsample, --no_polish)
scripts/make_paper_tables.py   regenerates paper/generated/*.tex from the result folders
reproduce.sh                   full reproduction, tests to PDF
paper/main.tex, refs.bib       the write-up (compiled PDF included)
docs/methodology_derivation.md
results/synthetic/, results/btc/   generated tables, figures, summary.md
```

## Reproduce

```bash
pip install -r requirements.txt
pytest -q                                   # 21 tests, ~5 s (pytest picks up src/ from pyproject.toml)
ruff check .                                # lint; config in pyproject.toml
python scripts/run_synthetic.py             # ~3 min (numba)
python scripts/run_btc.py                   # ~5 min; downloads data/raw/coinmetrics_btc.csv on first run
python scripts/run_btc.py --reuse           # re-run evaluation on cached walk-forward forecasts
./reproduce.sh                              # everything above plus the two robustness runs and the paper, ~45 min
```

Long runs checkpoint every walk-forward window under `<out>/checkpoints/`, so an interrupted run resumes where it stopped.
A completed run is replayed from those checkpoints as well: delete `<out>/checkpoints/` to re-estimate the walk-forward from scratch
(about three minutes of CPU time for `results/btc`; a fresh run reproduces the shipped files bit for bit).

Options: `--first_test`, `--train_len` (default 1460 days), `--test_len` (refit every 91 days),
`--rv_window` (10), `--K` (2), `--emission` (gaussian | student_t), `--target_vol` (0.20), `--w_max` (1.0), `--cost_bps` (10), `--gamma` (2, Kelly rule), `--B` (2000 bootstrap draws).
Seeds are fixed; `numba` JIT compiles on first call.

## Data

Coin Metrics Community Network Data, `csv/btc.csv` from https://github.com/coinmetrics/data
(CC BY-NC 4.0, academic use). Daily series 2014-02-01 to 2026-05-23 (T = 4495 after warm-up).
`PriceUSD` is the Coin Metrics reference rate as of 00:00 UTC of the following day, i.e. the close of the UTC day,
so the day-t return and the day-t exchange flows cover the same interval.

Covariates (both measurable at the close of day t; the transition into day t uses z_{t-1}):

- `logrv`: log of the 10-day realised volatility of daily log returns (internal covariate).
- `exflow`: (exchange inflow - outflow) / 30-day mean total exchange flow, from `FlowInExNtv`, `FlowOutExNtv`
  (on-chain exchange net-flow covariate). Funding rates are not in this dataset.

## Headline results

**Synthetic verification** (`results/synthetic/summary.md`): 200 replications, T = 2000, exogenous
AR(1) covariate. All eight parameters recovered with negligible bias, Wald 95% CI coverage 0.93 to
0.965, mean standard error equal to the Monte Carlo standard deviation. LR test (6.9) under H0 rejects
at 10 / 5 / 1 percent in 0.10 / 0.05 / 0.01 of replications (KS test against chi-square(2), p = 0.07);
power against the alternative is 1.0. Same conclusions with the internal log-RV covariate (100 replications,
coverage 0.89 to 0.95, larger standard errors as expected).

**Full-sample fits, K = 2** (`results/btc/fullsample_fits.csv`):

| | HMM | TVTP log-RV | TVTP ex-flow | TVTP both |
|---|---|---|---|---|
| log-likelihood | 9364.2 | 9463.3 | 9367.0 | 9466.9 |
| LR vs HMM (df) | | 198.3 (2), p < 1e-40 | 5.7 (2), p = 0.058 | 205.6 (4) |
| beta1 slope (calm persistence) | | -0.914 (0.120) | -0.205 (0.143) | |
| beta2 slope (turbulent persistence) | | +1.251 (0.224) | +0.173 (0.117) | |

Signs match the hypothesis: rising realised volatility makes the calm regime less persistent and the
turbulent regime more persistent (`fig_transition_response.png`). Regime volatilities 30% and 105%
annualised. BIC prefers K = 3 for the constant HMM; K = 2 is kept for the main comparison so that the
TVTP effect is isolated (the K = 3 TVTP fit is also reported: LR = 163.6 on 6 df).

**Walk-forward, 34 windows, out-of-sample 2018-01-31 to 2026-05-23** (`results/btc/summary.md`):

- TVTP log-RV rejects the constant HMM by LR in 34/34 windows and has lower BIC in 34/34; ex-flow in 0/34.
- Variance forecasts vs squared-return proxy, Diebold-Mariano against the constant HMM:
  QLIKE t = -2.27 (p = 0.023), MSE t = -2.55 (p = 0.011), predictive log score t = -4.62 (p < 1e-5).
  Against an EWMA(0.94) reference the TVTP model is better in point but not significantly (QLIKE p = 0.22);
  the constant HMM is indistinguishable from EWMA.
- Primary rule (exposure = P(calm), 20% vol target, cap 1, 10 bps): Sharpe 0.61 (TVTP) vs 0.39 (HMM),
  turnover 9.2 vs 16.5 per year. Bootstrap 95% CI for the Sharpe difference [-0.05, 0.49], p = 0.11.
  PSR(> 0) = 0.96, deflated Sharpe = 0.84 over 54 trials.
- Robustness rule (capped Kelly on mixture forecasts, gamma 2): Sharpe 0.76 (TVTP) vs 0.52 (HMM) vs 0.71 buy-and-hold;
  Sharpe-difference CI [0.00, 0.47], p = 0.05; deflated Sharpe 0.92; max drawdown -55% vs -77% for buy-and-hold.
- The TVTP advantage widens with transaction costs (Sharpe gap 0.16 at 0 bps, 0.46 at 50 bps) because
  its regime probabilities are more stable.

**K = 3 robustness** (`results/btc_k3/`, `python scripts/run_btc.py --K 3 --skip_fullsample --no_polish --n_restarts 3 --out results/btc_k3`):
TVTP still rejects the constant HMM by LR in 34/34 windows (lower BIC in 29/34) and still improves the
predictive log score (DM t = -4.05, p < 0.001) and MSE (p = 0.03), but not QLIKE (p = 0.15). The trading gain
disappears: the three-regime constant HMM is already a much better trading model than the two-regime one
(Sharpe 0.54 vs 0.39 proportional, 0.68 vs 0.52 Kelly) and adding time variation changes it by -0.03 to -0.05
with bootstrap intervals centred on zero. The intermediate regime does the job the covariate does at K = 2.

**Student-t robustness** (`results/btc_t/`, `python scripts/run_btc.py --emission student_t --out results/btc_t`):
regime-specific degrees of freedom estimated by an ECM step inside EM. The constant t-HMM is the best full-sample
model of all (log-likelihood 9537.9, BIC -19009, nu = 2.7 and 4.1), and TVTP on top of it is still significant in
the full sample (LR = 30.6, slopes -1.05 / +1.52). Out of sample it adds nothing: LR rejects in 62% of windows, slopes are
unstable across windows, the t-HMM's log score (2.1395) already beats the Gaussian TVTP model (2.1196) and the t-TVTP
model does not improve it (DM p = 0.60); no trading gain (Sharpe 0.43 vs 0.53 proportional, 0.56 vs 0.52 Kelly).
Ranking of the three fixes to the two-regime Gaussian HMM by out-of-sample log score: t emission, then time variation, then a third regime.

## How to read the evidence

Against the two-regime Gaussian HMM, forecast-quality gains from the TVTP extension are large, consistent across
windows and significant, and trading gains are economically meaningful (+0.22 to +0.25 Sharpe) and borderline
significant (p = 0.05 to 0.11 depending on the rule). Against richer baselines the gains shrink or vanish: a third
regime keeps the density gain and loses the trading gain; Student-t emissions leave nothing for time variation to add
out of sample. Time-varying persistence is a real, well-identified full-sample feature of the data whose out-of-sample
value is largely a substitute for a properly specified emission distribution. No long-only, volatility-targeted rule beats
buy-and-hold on Sharpe over this sample, which is dominated by two bull runs; the regime rules do
cut drawdowns substantially. The on-chain ex-flow covariate carries no out-of-sample information in this
specification. These statements are what the paper should claim; nothing stronger is supported.

Known limitations: single asset, daily reference-rate prices rather than exchange OHLC (no range-based
RV), squared returns as the realised-variance proxy, a K = 2 Gaussian main specification when BIC favours K = 3 and,
more strongly, Student-t emissions (both robustness runs above), and a Student-t variance forecast that is fragile
when nu is close to 2.

## Paper

`paper/main.tex` is the write-up (14 pages compiled with the bibliography, 11pt Times, 43 references, limit 15 pages). Every tabulated
number and every number quoted in the text comes from `python scripts/make_paper_tables.py`, which writes
`paper/generated/macros.tex` and one file per table from the result folders; the only hand-written numbers are the
daily returns and filtered probabilities in the event-dating paragraph, which are in `results/btc/event_daily_detail.csv`.
Compile with `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`. The title block is complete (sole author, no supervisor).
