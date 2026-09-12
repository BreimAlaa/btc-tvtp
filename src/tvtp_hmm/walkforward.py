"""
Walk-forward protocol (Section 7 of the methodology).

For each window: estimate on the training segment, then run the Hamilton filter
over training + test with the parameters held fixed, carrying xi_{t|t} across
the boundary. Rows belonging to the test segment give
    xi_{t|t-1}   regime forecast made at the close of t-1
    h_{t|t-1}    variance forecast, eq. (7.2)
    ln f(y_t | F_{t-1})   predictive log score, eq. (3.3)
which are legitimate out-of-sample quantities because the filter is causal and
the covariate standardisation uses training-window moments only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import HMMParams, FitResult, fit_em, fit_em_restarts, fit_direct, run_filter, moment_forecasts, lr_test
from .data import Standardizer, design_matrix, walk_forward_windows


@dataclass
class ModelSpec:
    name: str
    covariates: list[str] = field(default_factory=list)
    K: int = 2

    @property
    def d(self) -> int:
        return 1 + len(self.covariates)


def fit_spec(y: np.ndarray, Xlag: np.ndarray, spec: ModelSpec, warm: HMMParams | None = None,
             n_restarts: int = 5, polish: bool = True, seed: int = 0, **kw) -> FitResult:
    """Fit one spec. TVTP models are warm-started at the baseline solution with zero slopes
    (Section 6.5 pipeline) and additionally tried from random restarts; best likelihood wins."""
    best = fit_em_restarts(y, Xlag, spec.K, n_restarts=n_restarts, seed=seed, **kw)
    if warm is not None and warm.K == spec.K and Xlag.shape[1] > warm.d:
        beta = np.zeros((spec.K, spec.K - 1, Xlag.shape[1]))
        beta[:, :, : warm.d] = warm.beta
        r = fit_em(y, Xlag, spec.K, init=HMMParams(warm.mu, warm.sigma, beta, None if warm.nu is None else warm.nu.copy()), **kw)
        if r.loglik > best.loglik:
            best = r
    if polish:
        try:
            r = fit_direct(y, Xlag, best.params)
            if np.isfinite(r.loglik) and r.loglik > best.loglik:
                r.history = best.history
                best = r
        except Exception:  # keep the EM solution if the optimizer misbehaves
            pass
    return best


def forecast_segment(params: HMMParams, y_all: np.ndarray, Xlag_all: np.ndarray, start: int) -> dict:
    """Filter over the whole segment, return forecasts for rows >= start.

    The initial distribution is the ergodic distribution at the mean regressor of the TRAINING rows only
    (Section 6.3), so nothing from the test segment enters the filter before its own date."""
    pi = params.initial_distribution(Xlag_all[:start]) if start > 1 else None
    xi_pred, xi_filt, ll, trans, log_eta = run_filter(params, y_all, Xlag_all, pi=pi)
    m, h = moment_forecasts(params, xi_pred)
    return {"xi_pred": xi_pred[start:], "xi_filt": xi_filt[start:], "h": h[start:], "m": m[start:],
            "logscore": ll[start:], "p_stay": np.array([np.diag(trans[t]) for t in range(start, len(y_all))])}


def walk_forward(df: pd.DataFrame, specs: list[ModelSpec], train_len: int = 1460, test_len: int = 91,
                 first_test: int | None = None, n_restarts: int = 5, polish: bool = True, verbose: bool = True,
                 checkpoint_dir: str | None = None, student_t: bool = False):
    """Fit every spec on every window. specs[0] must be the constant-transition baseline
    (used as warm start and as the restricted model in the per-window LR test).

    Returns (forecasts, windows): forecasts[spec.name] is a DataFrame over the OOS dates,
    windows is a DataFrame with per-window fit statistics for every spec.
    """
    y = df["y"].to_numpy(float)
    n = len(df)
    if not specs or specs[0].covariates:
        raise ValueError("specs[0] must be the constant-transition baseline (no covariates)")
    start = train_len if first_test is None else max(first_test, train_len)
    if start >= n:
        raise ValueError(f"no test window: data has {n} rows but the first test index would be {start} "
                         f"(train_len={train_len}, first_test={first_test})")
    rows = {s.name: [] for s in specs}
    win_rows = []
    import os
    import pickle
    for wi, w in enumerate(walk_forward_windows(n, train_len, test_len, first_test)):
        ck = os.path.join(checkpoint_dir, f"window_{wi:03d}.pkl") if checkpoint_dir else None
        if ck and os.path.exists(ck):
            saved = pickle.load(open(ck, "rb"))
            for k, v in saved["rows"].items():
                rows[k].append(v)
            win_rows.extend(saved["win_rows"])
            if verbose:
                print(f"window {wi:3d} loaded from checkpoint", flush=True)
            continue
        saved = {"rows": {}, "win_rows": []}
        y_tr = y[w.train_start:w.train_end]
        y_seg = y[w.train_start:w.test_end]
        idx_test = df.index[w.train_end:w.test_end]
        base_fit = None
        fits = {}
        for spec in specs:
            std = None
            if spec.covariates:
                std = Standardizer().fit(df.iloc[w.train_start:w.train_end][[c + "_lag" for c in spec.covariates]].to_numpy(float))
            X_tr = design_matrix(df.iloc[w.train_start:w.train_end], spec.covariates, std)
            X_seg = design_matrix(df.iloc[w.train_start:w.test_end], spec.covariates, std)
            fit = fit_spec(y_tr, X_tr, spec, warm=base_fit.params if base_fit else None,
                           n_restarts=n_restarts, polish=polish, seed=wi, student_t=student_t)
            if base_fit is None:
                base_fit = fit
            fits[spec.name] = fit
            fc = forecast_segment(fit.params, y_seg, X_seg, w.train_end - w.train_start)
            frame = pd.DataFrame(index=idx_test)
            for k in range(spec.K):
                frame[f"xi{k}"] = fc["xi_pred"][:, k]
                frame[f"pstay{k}"] = fc["p_stay"][:, k]
            frame["h"] = fc["h"]
            frame["m"] = fc["m"]
            frame["logscore"] = fc["logscore"]
            frame["window"] = wi
            rows[spec.name].append(frame)
            saved["rows"][spec.name] = frame
            rec = {"window": wi, "spec": spec.name, "train_start": df.index[w.train_start], "test_start": df.index[w.train_end],
                   "test_end": df.index[w.test_end - 1], "loglik": fit.loglik, "aic": fit.aic, "bic": fit.bic,
                   "n_params": fit.params.n_params, "converged": fit.converged}
            for k in range(spec.K):
                rec[f"mu{k}"] = fit.params.mu[k]
                rec[f"sigma{k}"] = float(np.sqrt(fit.params.regime_variance[k]))
                if fit.params.nu is not None:
                    rec[f"nu{k}"] = fit.params.nu[k]
            if spec.K == 2:
                b1, b2 = fit.params.persistence_form()
                for j in range(len(b1)):
                    rec[f"beta1_{j}"] = b1[j]
                    rec[f"beta2_{j}"] = b2[j]
            if spec.name != specs[0].name:
                df_lr = fit.params.n_params - base_fit.params.n_params
                rec["lr_stat"], rec["lr_p"] = lr_test(base_fit.loglik, fit.loglik, df_lr)
            win_rows.append(rec)
            saved["win_rows"].append(rec)
        if ck:
            os.makedirs(checkpoint_dir, exist_ok=True)
            pickle.dump(saved, open(ck, "wb"))
        if verbose:
            msg = " | ".join(f"{k}: ll={v.loglik:.1f}" for k, v in fits.items())
            print(f"window {wi:3d} test {idx_test[0].date()}..{idx_test[-1].date()}  {msg}", flush=True)
    forecasts = {k: pd.concat(v) for k, v in rows.items()}
    return forecasts, pd.DataFrame(win_rows)
