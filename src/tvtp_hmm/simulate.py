"""Simulate from the (TVTP-)HMM. Used by the tests and by scripts/run_synthetic.py."""
from __future__ import annotations

import numpy as np
from .core import HMMParams, ergodic_distribution


def simulate_exog_ar1(params: HMMParams, T: int, rng: np.random.Generator,
                      phi: float = 0.9, z_sd: float = 1.0, burn: int = 200):
    """TVTP-HMM with an exogenous AR(1) covariate z_t (standardised to unit variance).

    Returns y (T,), Xlag (T, d), S (T,), z (T,).
    Xlag[t] = (1, z_{t-1}); the transition into t uses z_{t-1}.
    """
    K, d = params.K, params.d
    n = T + burn
    innov_sd = z_sd * np.sqrt(1 - phi ** 2)
    z = np.empty(n)
    z[0] = rng.normal(0, z_sd)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + rng.normal(0, innov_sd)
    S = np.empty(n, dtype=int)
    y = np.empty(n)
    x0 = np.zeros(d); x0[0] = 1.0
    def draw(j):
        if params.nu is None:
            return rng.normal(params.mu[j], params.sigma[j])
        return params.mu[j] + params.sigma[j] * rng.standard_t(params.nu[j])
    S[0] = rng.choice(K, p=ergodic_distribution(params.trans_at(x0)))
    y[0] = draw(S[0])
    for t in range(1, n):
        x = np.array([1.0, z[t - 1]]) if d == 2 else x0
        P = params.trans_at(x)
        S[t] = rng.choice(K, p=P[S[t - 1]])
        y[t] = draw(S[t])
    y, S, z = y[burn:], S[burn:], z[burn:]
    Xlag = np.ones((T, d))
    if d == 2:
        Xlag[1:, 1] = z[:-1]
        Xlag[0, 1] = z[0]
    return y, Xlag, S, z


def simulate_internal_rv(params: HMMParams, T: int, rng: np.random.Generator,
                         window: int = 10, burn: int = 300):
    """TVTP-HMM whose covariate is the standardised log realised volatility of past returns.

    z_t = ln sqrt(mean(y_{t-window+1..t}^2)), then standardised with the simulated
    sample moments; the transition into t uses z_{t-1}. Feedback loop is simulated exactly.
    Returns y, Xlag, S, z (all length T), plus the (mean, sd) used for standardisation.
    """
    K = params.K
    n = T + burn
    S = np.empty(n, dtype=int)
    y = np.empty(n)
    zraw = np.zeros(n)
    x0 = np.array([1.0, 0.0])
    S[0] = rng.choice(K, p=ergodic_distribution(params.trans_at(x0)))
    y[0] = rng.normal(params.mu[S[0]], params.sigma[S[0]])
    # standardisation constants: use the unconditional moments of log RV under the
    # ergodic mixture as a fixed reference so the model is well defined online
    m_ref, s_ref = _logrv_reference(params, window, rng)
    for t in range(1, n):
        lo = max(0, t - window)
        zraw[t - 1] = np.log(np.sqrt(np.mean(y[lo:t] ** 2)) + 1e-12)
        z_prev = (zraw[t - 1] - m_ref) / s_ref
        P = params.trans_at(np.array([1.0, z_prev]))
        S[t] = rng.choice(K, p=P[S[t - 1]])
        y[t] = rng.normal(params.mu[S[t]], params.sigma[S[t]])
    zraw[n - 1] = np.log(np.sqrt(np.mean(y[n - window:] ** 2)) + 1e-12)
    z = (zraw - m_ref) / s_ref
    y, S, z = y[burn:], S[burn:], z[burn:]
    Xlag = np.ones((T, 2))
    Xlag[1:, 1] = z[:-1]
    Xlag[0, 1] = z[0]
    return y, Xlag, S, z, (m_ref, s_ref)


def _logrv_reference(params: HMMParams, window: int, rng: np.random.Generator, n: int = 20000):
    """Reference mean/sd of log RV from a long constant-transition simulation at x = (1, 0)."""
    K = params.K
    P = params.trans_at(np.array([1.0, 0.0]))
    pi = ergodic_distribution(P)
    S = np.empty(n, dtype=int)
    S[0] = rng.choice(K, p=pi)
    for t in range(1, n):
        S[t] = rng.choice(K, p=P[S[t - 1]])
    y = rng.normal(params.mu[S], params.sigma[S])
    rv = np.sqrt(np.convolve(y ** 2, np.ones(window) / window, mode="valid"))
    lrv = np.log(rv + 1e-12)
    return float(lrv.mean()), float(lrv.std())
