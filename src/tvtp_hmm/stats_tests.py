"""
Roadmap step 7: significance tests.

- Diebold & Mariano (1995) test on loss differentials with a Newey-West variance
  and the Harvey, Leybourne & Newbold (1997) small-sample correction.
- Volatility-forecast losses MSE and QLIKE (Patton 2011); QLIKE is robust to a
  noisy proxy such as the squared return.
- Probabilistic Sharpe ratio and deflated Sharpe ratio (Bailey & Lopez de Prado 2014).
- Stationary bootstrap (Politis & Romano 1994) confidence intervals for the
  difference in annualised Sharpe ratio between two strategies.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

ANN = 365


# ----------------------------------------------------------------------------
# Loss functions for variance forecasts h_{t|t-1} against a proxy of sigma_t^2
# ----------------------------------------------------------------------------
def loss_mse(h: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    return (h - proxy) ** 2


def loss_qlike(h: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    h = np.clip(h, 1e-12, None)
    return np.log(h) + proxy / h


# ----------------------------------------------------------------------------
# Diebold-Mariano
# ----------------------------------------------------------------------------
def newey_west_variance(d: np.ndarray, bandwidth: int) -> float:
    d = d - d.mean()
    T = len(d)
    v = np.dot(d, d) / T
    for lag in range(1, bandwidth + 1):
        w = 1.0 - lag / (bandwidth + 1.0)
        v += 2.0 * w * np.dot(d[lag:], d[:-lag]) / T
    return float(v)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1, bandwidth: int | None = None,
            small_sample: bool = True) -> dict:
    """H0: E[loss_a - loss_b] = 0. Negative statistic favours model a.

    Bandwidth defaults to max(horizon-1, floor(T^(1/3))) to be safe against residual
    autocorrelation in the loss differential.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    T = len(d)
    if bandwidth is None:
        bandwidth = max(horizon - 1, int(np.floor(T ** (1 / 3))))
    if T < 3:
        raise ValueError("Diebold-Mariano test needs at least three loss differentials")
    var = newey_west_variance(d, bandwidth)
    if var <= 0:   # identical forecasts: no evidence either way
        return {"dm_stat": 0.0, "p_value": 1.0, "mean_diff": float(d.mean()), "T": T, "bandwidth": bandwidth}
    stat = d.mean() / np.sqrt(var / T)
    if small_sample:  # Harvey-Leybourne-Newbold correction
        stat *= np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
        p = 2 * stats.t.sf(abs(stat), df=T - 1)
    else:
        p = 2 * stats.norm.sf(abs(stat))
    return {"dm_stat": float(stat), "p_value": float(p), "mean_diff": float(d.mean()), "T": T, "bandwidth": bandwidth}


# ----------------------------------------------------------------------------
# Sharpe ratio inference
# ----------------------------------------------------------------------------
def sharpe(r: np.ndarray, ann: int = ANN) -> float:
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else np.nan


def probabilistic_sharpe_ratio(r: np.ndarray, sr_benchmark_ann: float = 0.0, ann: int = ANN) -> dict:
    """PSR = Prob(true SR > benchmark), Bailey & Lopez de Prado (2012/2014). All SRs per period."""
    r = np.asarray(r, float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    sr_b = sr_benchmark_ann / np.sqrt(ann)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr_b) * np.sqrt(T - 1) / denom
    return {"psr": float(stats.norm.cdf(z)), "z": float(z), "sr_ann": float(sr * np.sqrt(ann)), "skew": float(g3), "kurt": float(g4)}


def deflated_sharpe_ratio(r: np.ndarray, trial_sharpes_ann: np.ndarray, ann: int = ANN) -> dict:
    """DSR = PSR evaluated at the expected maximum SR of N independent trials.

    trial_sharpes_ann: annualised Sharpe ratios of every strategy variant that was tried
    (the selected strategy included). Their number N and variance define the benchmark
        SR0 = sqrt(V[SR]) * ((1-gamma) Z(1 - 1/N) + gamma Z(1 - 1/(N e)))
    with gamma the Euler-Mascheroni constant.
    """
    tr = np.asarray(trial_sharpes_ann, float) / np.sqrt(ann)   # per-period
    N = len(tr)
    if N < 2:
        return {**probabilistic_sharpe_ratio(r, 0.0, ann), "sr0_ann": 0.0, "n_trials": N, "dsr": np.nan}
    gamma = 0.5772156649015329
    v = tr.var(ddof=1)
    sr0 = np.sqrt(v) * ((1 - gamma) * stats.norm.ppf(1 - 1 / N) + gamma * stats.norm.ppf(1 - 1 / (N * np.e)))
    out = probabilistic_sharpe_ratio(r, sr0 * np.sqrt(ann), ann)
    out.update({"dsr": out.pop("psr"), "sr0_ann": float(sr0 * np.sqrt(ann)), "n_trials": N})
    return out


# ----------------------------------------------------------------------------
# Stationary bootstrap
# ----------------------------------------------------------------------------
def stationary_bootstrap_indices(T: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty(T, dtype=int)
    idx[0] = rng.integers(T)
    restart = rng.random(T) < p
    for t in range(1, T):
        idx[t] = rng.integers(T) if restart[t] else (idx[t - 1] + 1) % T
    return idx


def bootstrap_sharpe_diff(r_a: np.ndarray, r_b: np.ndarray, B: int = 2000, mean_block: float = 20,
                          ann: int = ANN, seed: int = 0, level: float = 0.95) -> dict:
    """Paired stationary bootstrap of SR(a) - SR(b) and mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    r_a, r_b = np.asarray(r_a, float), np.asarray(r_b, float)
    T = len(r_a)
    d_sr = np.empty(B)
    d_mu = np.empty(B)
    for b in range(B):
        idx = stationary_bootstrap_indices(T, mean_block, rng)
        d_sr[b] = sharpe(r_a[idx], ann) - sharpe(r_b[idx], ann)
        d_mu[b] = (r_a[idx].mean() - r_b[idx].mean()) * ann
    obs = sharpe(r_a, ann) - sharpe(r_b, ann)
    lo, hi = np.quantile(d_sr, [(1 - level) / 2, 1 - (1 - level) / 2])
    p_two = 2 * min((d_sr <= 0).mean(), (d_sr >= 0).mean())
    return {"sr_diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_two_sided": float(p_two),
            "mean_diff_ann": float((r_a.mean() - r_b.mean()) * ann),
            "mean_diff_ci": tuple(map(float, np.quantile(d_mu, [(1 - level) / 2, 1 - (1 - level) / 2]))),
            "B": B, "mean_block": mean_block}
