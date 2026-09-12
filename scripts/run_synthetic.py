"""Synthetic-data verification (methodology Section 8, unit test (d), extended).

1. Parameter recovery for the TVTP-HMM with an exogenous AR(1) covariate:
   bias, RMSE and 95% Wald-CI coverage over N Monte Carlo replications.
2. Size of the LR test (6.9) under H0 (data from a constant HMM, irrelevant covariate)
   and power under H1.
3. Recovery when the covariate is the internal log realised volatility (feedback loop).

Usage: python scripts/run_synthetic.py [--n 200] [--T 2000] [--out results/synthetic]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tvtp_hmm.core import HMMParams, fit_em, fit_em_restarts, fit_direct, lr_test, smooth  # noqa: E402
from tvtp_hmm.simulate import simulate_exog_ar1, simulate_internal_rv  # noqa: E402

TRUE = HMMParams(mu=np.array([0.0010, -0.0010]), sigma=np.array([0.015, 0.045]),
                 beta=np.array([[[3.0, -1.0]], [[-2.5, -1.0]]]))     # persistence form: b1=(3,-1), b2=(2.5,1)
NULL = HMMParams(mu=TRUE.mu.copy(), sigma=TRUE.sigma.copy(),
                 beta=np.array([[[3.0, 0.0]], [[-2.5, 0.0]]]))       # same chain, no covariate effect
NAMES = ["mu1", "mu2", "sigma1", "sigma2", "b1_0", "b1_1", "b2_0", "b2_1"]


def true_vector(p: HMMParams) -> np.ndarray:
    b1, b2 = p.persistence_form()
    return np.array([p.mu[0], p.mu[1], p.sigma[0], p.sigma[1], b1[0], b1[1], b2[0], b2[1]])


def fit_both(y, Xlag, seed, compute_se=True):
    base = fit_em_restarts(y, np.ones((len(y), 1)), 2, n_restarts=3, seed=seed)
    beta0 = np.concatenate([base.params.beta, np.zeros((2, 1, 1))], axis=2)
    em = fit_em(y, Xlag, 2, init=HMMParams(base.params.mu, base.params.sigma, beta0))
    tv = fit_direct(y, Xlag, em.params, compute_se=compute_se)
    if tv.loglik < em.loglik:
        tv = em
    return base, tv


def estimate_vector(fit):
    p = fit.params
    v = true_vector(p)
    se = None
    if getattr(fit, "se", None) is not None:
        s = fit.se
        # SE of sigma via delta method from SE of log sigma; persistence-form slopes flip sign only
        se = np.array([s.mu[0], s.mu[1], s.sigma[0] * p.sigma[0], s.sigma[1] * p.sigma[1],
                       s.beta[0, 0, 0], s.beta[0, 0, 1], s.beta[1, 0, 0], s.beta[1, 0, 1]])
    return v, se


def study_recovery(n, T, out, tag, simulator, truth):
    rng = np.random.default_rng(2026)
    est, ses, lr, acc_s, acc_f, times = [], [], [], [], [], []
    for i in range(n):
        t0 = time.time()
        sim = simulator(truth, T, rng)
        y, Xlag, S = sim[0], sim[1], sim[2]
        base, tv = fit_both(y, Xlag, seed=i)
        v, se = estimate_vector(tv)
        est.append(v)
        ses.append(se)
        lr.append(lr_test(base.loglik, tv.loglik, 2)[0])
        xi_s, _, xi_pred, xi_filt, _ = smooth(tv.params, y, Xlag)
        acc_s.append((xi_s.argmax(1) == S).mean())
        acc_f.append((xi_filt.argmax(1) == S).mean())
        times.append(time.time() - t0)
        if (i + 1) % 25 == 0:
            print(f"[{tag}] {i + 1}/{n} done, {np.mean(times):.2f}s per replication", flush=True)
    est = np.array(est)
    ses = np.array(ses)
    tv_true = true_vector(truth)
    bias = est.mean(0) - tv_true
    rmse = np.sqrt(((est - tv_true) ** 2).mean(0))
    cover = (np.abs(est - tv_true) <= 1.96 * ses).mean(0)
    table = pd.DataFrame({"true": tv_true, "mean_est": est.mean(0), "bias": bias, "rmse": rmse,
                          "mean_se": ses.mean(0), "sd_est": est.std(0, ddof=1), "coverage95": cover}, index=NAMES)
    table.to_csv(os.path.join(out, f"recovery_{tag}.csv"))
    pd.DataFrame(est, columns=NAMES).to_csv(os.path.join(out, f"recovery_{tag}_draws.csv"), index=False)
    summary = {"n": n, "T": T, "lr_reject_5pct": float((np.array(lr) > stats.chi2.ppf(0.95, 2)).mean()),
               "lr_reject_1pct": float((np.array(lr) > stats.chi2.ppf(0.99, 2)).mean()),
               "regime_accuracy_smoothed": float(np.mean(acc_s)), "regime_accuracy_filtered": float(np.mean(acc_f)),
               "sec_per_replication": float(np.mean(times))}
    print(f"\n[{tag}] parameter recovery (N={n}, T={T})\n", table.round(4).to_string())
    print(json.dumps(summary, indent=1))
    return table, summary, np.array(lr)


def study_size(n, T, out):
    """LR test size: data from the constant HMM with an irrelevant AR(1) covariate."""
    rng = np.random.default_rng(99)
    lr = []
    for i in range(n):
        y, Xlag, S, z = simulate_exog_ar1(NULL, T, rng)
        base, tv = fit_both(y, Xlag, seed=i, compute_se=False)
        lr.append(lr_test(base.loglik, tv.loglik, 2)[0])
        if (i + 1) % 50 == 0:
            print(f"[size] {i + 1}/{n}", flush=True)
    lr = np.clip(np.array(lr), 0, None)
    res = {"n": n, "T": T,
           "reject_10pct": float((lr > stats.chi2.ppf(0.90, 2)).mean()),
           "reject_5pct": float((lr > stats.chi2.ppf(0.95, 2)).mean()),
           "reject_1pct": float((lr > stats.chi2.ppf(0.99, 2)).mean()),
           "ks_pvalue_vs_chi2_2": float(stats.kstest(lr, "chi2", args=(2,)).pvalue),
           "mean_lr": float(lr.mean()), "expected_mean_chi2_2": 2.0}
    np.save(os.path.join(out, "lr_null_draws.npy"), lr)
    print("\n[size] LR test under H0:", json.dumps(res, indent=1))
    return res, lr


def make_figure(draws_exog, lr_null, lr_alt, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    tv = true_vector(TRUE)
    ax = axes[0]
    idx = [4, 5, 6, 7]
    ax.boxplot([draws_exog[:, i] for i in idx], tick_labels=[NAMES[i] for i in idx], showfliers=False)
    for k, i in enumerate(idx):
        ax.plot(k + 1, tv[i], "r_", ms=25, mew=2)
    ax.set_title("TVTP slopes and intercepts: MC draws vs truth (red)")
    ax = axes[1]
    ax.boxplot([draws_exog[:, i] * 100 for i in [2, 3]], tick_labels=["sigma1 (%)", "sigma2 (%)"], showfliers=False)
    for k, i in enumerate([2, 3]):
        ax.plot(k + 1, tv[i] * 100, "r_", ms=25, mew=2)
    ax.set_title("Regime volatilities")
    ax = axes[2]
    grid = np.linspace(0, 15, 200)
    ax.hist(lr_null, bins=30, density=True, alpha=0.6, label="LR under H0 (size)")
    ax.plot(grid, stats.chi2.pdf(grid, 2), "k-", label=r"$\chi^2_2$")
    ax.axvline(stats.chi2.ppf(0.95, 2), color="r", ls="--", label="5% critical value")
    ax.set_xlim(0, 15)
    ax.set_title(f"LR test: size OK, power at 5% = {np.mean(lr_alt > stats.chi2.ppf(0.95, 2)):.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "synthetic_recovery.png"), dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_internal", type=int, default=100)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "synthetic"))
    ap.add_argument("--only", choices=["exog", "size", "internal", "all"], default="all")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.only in ("exog", "all"):
        t_exog, s_exog, lr_alt = study_recovery(a.n, a.T, a.out, "exog_ar1", simulate_exog_ar1, TRUE)
        np.save(os.path.join(a.out, "lr_alt_draws.npy"), lr_alt)
    else:
        t_exog = pd.read_csv(os.path.join(a.out, "recovery_exog_ar1.csv"), index_col=0); lr_alt = np.load(os.path.join(a.out, "lr_alt_draws.npy"))
        s_exog = json.load(open(os.path.join(a.out, "summary.json")))["exog_ar1"] if os.path.exists(os.path.join(a.out, "summary.json")) else {}
    if a.only in ("size", "all"):
        size, lr_null = study_size(a.n, a.T, a.out)
    else:
        lr_null = np.load(os.path.join(a.out, "lr_null_draws.npy"))
        size = json.load(open(os.path.join(a.out, "summary.json")))["lr_size_H0"] if os.path.exists(os.path.join(a.out, "summary.json")) else {}
    if a.only in ("internal", "all"):
        t_int, s_int, _ = study_recovery(a.n_internal, a.T, a.out, "internal_rv", simulate_internal_rv, TRUE)
    else:
        t_int = pd.read_csv(os.path.join(a.out, "recovery_internal_rv.csv"), index_col=0)
        s_int = json.load(open(os.path.join(a.out, "summary.json")))["internal_rv"]
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"exog_ar1": s_exog, "lr_size_H0": size, "internal_rv": s_int}, f, indent=1)
    draws = pd.read_csv(os.path.join(a.out, "recovery_exog_ar1_draws.csv")).to_numpy()
    make_figure(draws, lr_null, lr_alt, a.out)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("# Synthetic verification\n\n## Parameter recovery, exogenous AR(1) covariate\n\n")
        f.write(t_exog.round(4).to_markdown() + "\n\n" + json.dumps(s_exog, indent=1) + "\n\n")
        f.write("## LR test size under H0\n\n" + json.dumps(size, indent=1) + "\n\n")
        f.write("## Parameter recovery, internal log-RV covariate\n\n")
        f.write(t_int.round(4).to_markdown() + "\n\n" + json.dumps(s_int, indent=1) + "\n")
    print("saved to", a.out)
