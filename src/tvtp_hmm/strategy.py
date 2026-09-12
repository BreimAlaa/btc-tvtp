"""
Roadmap step 6: regime-conditioned, volatility-scaled allocation.

At the close of day t-1 the model delivers xi_{t|t-1} (regime probabilities for
day t) and h_{t|t-1} (variance forecast for day t). The position held over
(t-1, t] is

    w_t = e_t * min( w_max, sigma_target / sqrt(ann * h_{t|t-1}) ),
    e_t = sum_j xi_{t|t-1}(j) a_j,   a_j = (K-1-j)/(K-1)   (a = 1 calm ... 0 turbulent)

Long-only by construction (0 <= w_t <= w_max). Realised strategy return

    r_t = w_t R_t - c |w_t - w_{t-1}|

with R_t the simple return and c the proportional cost per unit turnover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 365  # BTC trades every calendar day


def regime_exposure(xi_pred: np.ndarray) -> np.ndarray:
    K = xi_pred.shape[1]
    a = (K - 1 - np.arange(K)) / max(K - 1, 1)
    return xi_pred @ a


def weights(xi_pred: np.ndarray, h: np.ndarray, target_vol: float = 0.20, w_max: float = 1.0,
            regime_conditioned: bool = True, vol_scaled: bool = True, ann: int = ANN) -> np.ndarray:
    """Primary rule: exposure = P(calm), scaled to a volatility target."""
    e = regime_exposure(xi_pred) if regime_conditioned else np.ones(len(h))
    if vol_scaled:
        scale = np.minimum(w_max, target_vol / np.sqrt(np.clip(ann * h, 1e-12, None)))
    else:
        scale = np.full(len(h), w_max)
    return e * scale


def weights_kelly(m: np.ndarray, h: np.ndarray, gamma: float = 1.0, w_max: float = 1.0) -> np.ndarray:
    """Robustness rule: capped fractional Kelly / mean-variance weight on the mixture forecasts,

        w_t = clip( m_{t|t-1} / (gamma * h_{t|t-1}), 0, w_max ).

    Regime conditioning enters through the mixture mean and variance of eq. (7.2)
    rather than through P(calm) directly (Ang & Bekaert 2002 style)."""
    return np.clip(m / (gamma * np.clip(h, 1e-12, None)), 0.0, w_max)


def backtest(w: np.ndarray, ret: np.ndarray, cost: float = 0.0010) -> pd.DataFrame:
    w = np.asarray(w, float)
    w_prev = np.concatenate([[0.0], w[:-1]])
    turnover = np.abs(w - w_prev)
    r = w * ret - cost * turnover
    return pd.DataFrame({"w": w, "turnover": turnover, "r": r})


def max_drawdown(r: np.ndarray) -> float:
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1).min())


def metrics(r: np.ndarray, turnover: np.ndarray | None = None, ann: int = ANN) -> dict:
    r = np.asarray(r, float)
    mu, sd = r.mean(), r.std(ddof=1)
    downside = r[r < 0].std(ddof=1) if (r < 0).sum() > 1 else np.nan
    out = {
        "ann_return": mu * ann,
        "ann_vol": sd * np.sqrt(ann),
        "sharpe": mu / sd * np.sqrt(ann) if sd > 0 else np.nan,
        "sortino": mu / downside * np.sqrt(ann) if downside and downside > 0 else np.nan,
        "max_drawdown": max_drawdown(r),
        "final_wealth": float(np.prod(1 + r)),
        "hit_rate": float((r > 0).mean()),
        "n_days": int(len(r)),
    }
    out["calmar"] = out["ann_return"] / abs(out["max_drawdown"]) if out["max_drawdown"] < 0 else np.nan
    if turnover is not None:
        out["ann_turnover"] = float(np.mean(turnover) * ann)
    return out
