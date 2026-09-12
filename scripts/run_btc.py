"""End-to-end pipeline on real BTC data (roadmap steps 3 to 7).

    python scripts/run_btc.py [--first_test 2018-01-01] [--train_len 1460] [--test_len 91] ...

Outputs go to results/btc/: full-sample fits with standard errors and LR tests, regime
dating around known events, transition response curves, walk-forward out-of-sample
forecasts, Diebold-Mariano tests, strategy backtests with costs, PSR / deflated Sharpe,
stationary-bootstrap confidence intervals, cost sensitivity, figures, and summary.md.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tvtp_hmm.core import HMMParams, fit_em_restarts, fit_direct, fit_em, smooth, lr_test, run_filter  # noqa: E402
from tvtp_hmm.data import load_btc, build_design, design_matrix, Standardizer  # noqa: E402
from tvtp_hmm.walkforward import ModelSpec, walk_forward  # noqa: E402
from tvtp_hmm import strategy as st  # noqa: E402
from tvtp_hmm import stats_tests as tst  # noqa: E402

EVENTS = {"COVID crash": "2020-03-12", "Terra/LUNA": "2022-05-09", "FTX collapse": "2022-11-08"}


def full_sample_fit(df, names, K, n_restarts, tag, warm=None, student_t=False):
    std = Standardizer().fit(df[[n + "_lag" for n in names]].to_numpy(float)) if names else None
    X = design_matrix(df, names, std)
    y = df["y"].to_numpy(float)
    best = fit_em_restarts(y, X, K, n_restarts=n_restarts, seed=1, student_t=student_t)
    if warm is not None and X.shape[1] > warm.d:
        beta = np.zeros((K, K - 1, X.shape[1])); beta[:, :, : warm.d] = warm.beta
        r = fit_em(y, X, K, init=HMMParams(warm.mu, warm.sigma, beta, None if warm.nu is None else warm.nu.copy()), student_t=student_t)
        if r.loglik > best.loglik:
            best = r
    fit = fit_direct(y, X, best.params, compute_se=True)
    if fit.loglik < best.loglik:
        fit = best
        fit.se = None
    fit.X, fit.std, fit.names, fit.tag = X, std, names, tag
    return fit


def fit_table(fits, base_name):
    rows = []
    for tag, f in fits.items():
        p, se = f.params, getattr(f, "se", None)
        rec = {"model": tag, "K": p.K, "n_params": p.n_params, "loglik": f.loglik, "aic": f.aic, "bic": f.bic}
        sd = np.sqrt(p.regime_variance)
        for k in range(p.K):
            rec[f"mu{k+1}"] = p.mu[k]; rec[f"sigma{k+1}_ann%"] = sd[k] * np.sqrt(365) * 100
            if se is not None:
                rec[f"se_mu{k+1}"] = se.mu[k]; rec[f"se_sigma{k+1}_ann%"] = se.sigma[k] * sd[k] * np.sqrt(365) * 100
            if p.nu is not None:
                rec[f"nu{k+1}"] = p.nu[k]
                if se is not None:
                    rec[f"se_nu{k+1}"] = se.nu[k] * (p.nu[k] - 2.0)   # delta method from log(nu - 2)
        if p.K == 2:
            b1, b2 = p.persistence_form()
            for j in range(p.d):
                lab = "0" if j == 0 else f.names[j - 1]
                rec[f"beta1_{lab}"] = b1[j]; rec[f"beta2_{lab}"] = b2[j]
                if se is not None:
                    rec[f"se_beta1_{lab}"] = se.beta[0, 0, j]; rec[f"se_beta2_{lab}"] = se.beta[1, 0, j]
            x0 = np.zeros(p.d); x0[0] = 1
            P0 = p.trans_at(x0)
            rec["p11_at_mean_z"] = P0[0, 0]; rec["p22_at_mean_z"] = P0[1, 1]
            rec["dur1_at_mean_z"] = 1 / (1 - P0[0, 0]); rec["dur2_at_mean_z"] = 1 / (1 - P0[1, 1])
        if tag != base_name and fits[base_name].params.K == p.K:
            b = fits[base_name]
            rec["lr_stat"], rec["lr_p"] = lr_test(b.loglik, f.loglik, p.n_params - b.params.n_params)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("model")


def regime_dating(df, fit, out):
    y = df["y"].to_numpy(float)
    xi_s, pair, xi_pred, xi_filt, ll = smooth(fit.params, y, fit.X)
    dating = pd.DataFrame({"p_turbulent_smoothed": xi_s[:, -1], "p_turbulent_filtered": xi_filt[:, -1]}, index=df.index)
    rows = []
    for name, date in EVENTS.items():
        d = pd.Timestamp(date)
        win = dating.loc[d - pd.Timedelta(days=10): d + pd.Timedelta(days=10)]
        rows.append({"event": name, "date": date,
                     "p_turb_smoothed_pre10d": float(dating.loc[:d - pd.Timedelta(days=10)].iloc[-1, 0]),
                     "p_turb_smoothed_event": float(dating.loc[d, "p_turbulent_smoothed"]) if d in dating.index else np.nan,
                     "p_turb_filtered_event": float(dating.loc[d, "p_turbulent_filtered"]) if d in dating.index else np.nan,
                     "p_turb_smoothed_max_pm10d": float(win.iloc[:, 0].max()),
                     "first_day_filtered_gt_0.5": str(win.index[(win.iloc[:, 1] > 0.5).to_numpy().argmax()].date()) if (win.iloc[:, 1] > 0.5).any() else "none"})
    return dating, pd.DataFrame(rows)


def response_curve(fit, cov_index, z_grid_std, cov_block_index):
    """p_ii(z) with delta-method 95% bands, z on the standardised scale."""
    p = fit.params
    b1, b2 = p.persistence_form()
    out = {}
    for i, b in enumerate([b1, b2]):
        x = np.zeros((len(z_grid_std), p.d)); x[:, 0] = 1; x[:, cov_index] = z_grid_std
        lin = x @ b
        prob = 1 / (1 + np.exp(-lin))
        band = None
        if getattr(fit, "cov", None) is not None:
            sl = slice(2 * p.K + i * p.d, 2 * p.K + (i + 1) * p.d)  # beta block for row i in pack() layout
            C = fit.cov[sl, sl]
            v = np.einsum("td,de,te->t", x, C, x)
            se = np.sqrt(np.clip(v, 0, None))
            band = (1 / (1 + np.exp(-(lin - 1.96 * se))), 1 / (1 + np.exp(-(lin + 1.96 * se))))
        out[i + 1] = (prob, band)
    return out


def strategies_from_forecasts(forecasts, df, cfg, K=2, rule="prop"):
    idx = forecasts["baseline"].index
    ret = df.loc[idx, "ret"].to_numpy(float)
    out = {}
    out["Buy&Hold"] = st.backtest(np.ones(len(idx)), ret, cfg["cost"])
    fb = forecasts["baseline"]
    xi_b = fb[[f"xi{k}" for k in range(K)]].to_numpy()
    out["VolTarget (no regime)"] = st.backtest(st.weights(xi_b, fb["h"].to_numpy(), cfg["target_vol"], cfg["w_max"], regime_conditioned=False), ret, cfg["cost"])
    for name, fc in forecasts.items():
        xi = fc[[f"xi{k}" for k in range(K)]].to_numpy()
        if rule == "prop":
            w = st.weights(xi, fc["h"].to_numpy(), cfg["target_vol"], cfg["w_max"])
        else:
            w = st.weights_kelly(fc["m"].to_numpy(), fc["h"].to_numpy(), cfg["gamma"], cfg["w_max"])
        out[name] = st.backtest(w, ret, cfg["cost"])
    return out, ret


def main(a):
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    df = load_btc(start=a.start)
    df, names = build_design(df, ["logrv", "exflow"], logrv={"window": a.rv_window}, exflow={"window": 30})
    print(f"data: {df.index[0].date()} .. {df.index[-1].date()}, T={len(df)}")
    summary = {"emission": a.emission, "data": {"source": "Coin Metrics Community Data (github.com/coinmetrics/data, csv/btc.csv, CC BY-NC 4.0)",
                        "start": str(df.index[0].date()), "end": str(df.index[-1].date()), "T": int(len(df)),
                        "covariates": {"logrv": f"log of {a.rv_window}-day realised volatility of daily log returns",
                                       "exflow": "(exchange inflow - outflow) / 30-day mean of total exchange flow, native units"}}}

    # ---------------- 1. full-sample fits ----------------
    fits = {}
    if a.skip_fullsample:
        print("\n== skipping full-sample fits ==")
    else:
        run_fullsample(df, a, out, summary, fits)
    specs, forecasts, windows = None, None, None
    # ---------------- 2. walk-forward ----------------
    print("\n== walk-forward ==")
    specs = [ModelSpec("baseline", [], a.K), ModelSpec("tvtp_logrv", [f"logrv{a.rv_window}"], a.K),
             ModelSpec("tvtp_exflow", ["exflow30"], a.K), ModelSpec("tvtp_both", [f"logrv{a.rv_window}", "exflow30"], a.K)]
    first_test = int(np.searchsorted(df.index.values, np.datetime64(a.first_test)))
    cached = [os.path.join(out, f"forecasts_{s.name}.csv") for s in specs] + [os.path.join(out, "walkforward_windows.csv")]
    if a.reuse and all(os.path.exists(c) for c in cached):
        print("reusing cached walk-forward forecasts from", out)
        forecasts = {s.name: pd.read_csv(os.path.join(out, f"forecasts_{s.name}.csv"), index_col=0, parse_dates=True) for s in specs}
        windows = pd.read_csv(os.path.join(out, "walkforward_windows.csv"), parse_dates=["train_start", "test_start", "test_end"])
    else:
        forecasts, windows = walk_forward(df, specs, a.train_len, a.test_len, first_test, a.n_restarts, polish=not a.no_polish,
                                          checkpoint_dir=os.path.join(out, "checkpoints"), student_t=(a.emission == "student_t"))
        windows.to_csv(os.path.join(out, "walkforward_windows.csv"), index=False)
        for k, v in forecasts.items():
            v.to_csv(os.path.join(out, f"forecasts_{k}.csv"))
    wf_lr = windows[windows.spec != "baseline"].groupby("spec").agg(n_windows=("lr_p", "size"), frac_lr_p_lt_05=("lr_p", lambda s: float((s < 0.05).mean())),
                                                                  median_lr=("lr_stat", "median"))
    base_bic = windows[windows.spec == "baseline"].set_index("window")["bic"]
    wf_lr["frac_bic_better"] = [float((windows[windows.spec == spec].set_index("window")["bic"] < base_bic).mean()) for spec in wf_lr.index]
    if {"beta1_1", "beta2_1"} <= set(windows.columns):   # persistence-form slopes are recorded for K = 2 only
        slopes = windows[windows.spec == "tvtp_logrv"][["beta1_1", "beta2_1"]].describe().loc[["mean", "50%", "min", "max"]]
    else:
        slopes = pd.DataFrame({"beta1_1": [np.nan] * 4, "beta2_1": [np.nan] * 4}, index=["mean", "50%", "min", "max"])
    print("\nper-window LR / BIC:\n", wf_lr.to_string(), "\n\nper-window TVTP-logrv slopes:\n", slopes.to_string())
    summary["walk_forward"] = {"n_windows": int(windows.window.nunique()), "train_len": a.train_len, "test_len": a.test_len,
                               "oos_start": str(forecasts["baseline"].index[0].date()), "oos_end": str(forecasts["baseline"].index[-1].date()),
                               "lr_bic": json.loads(wf_lr.to_json(orient="index")), "logrv_slopes": json.loads(slopes.to_json())}

    # ---------------- 3. forecast evaluation ----------------
    print("\n== out-of-sample forecast evaluation (Diebold-Mariano vs baseline) ==")
    idx = forecasts["baseline"].index
    y_oos = df.loc[idx, "y"].to_numpy(float)
    proxy = y_oos ** 2
    hb = forecasts["baseline"]["h"].to_numpy()
    # reference forecasts that do not use the HMM at all
    rv_prev = np.exp(df.loc[idx, f"logrv{a.rv_window}_lag"].to_numpy(float)) ** 2
    lam = 0.94
    ew = pd.Series(df["y"] ** 2).ewm(alpha=1 - lam, adjust=False).mean().shift(1).loc[idx].to_numpy()
    rows = []
    cand = {"baseline": hb, **{k: v["h"].to_numpy() for k, v in forecasts.items() if k != "baseline"},
            f"rolling RV{a.rv_window} (ref)": rv_prev, "EWMA(0.94) (ref)": ew}
    for name, h in cand.items():
        rec = {"model": name, "mean_qlike": tst.loss_qlike(h, proxy).mean(), "mean_mse_x1e6": tst.loss_mse(h, proxy).mean() * 1e6}
        if name in forecasts:
            rec["mean_logscore"] = forecasts[name]["logscore"].mean()
        if name != "baseline":
            dq = tst.dm_test(tst.loss_qlike(h, proxy), tst.loss_qlike(hb, proxy))
            dm = tst.dm_test(tst.loss_mse(h, proxy), tst.loss_mse(hb, proxy))
            rec.update({"dm_qlike": dq["dm_stat"], "p_qlike": dq["p_value"], "dm_mse": dm["dm_stat"], "p_mse": dm["p_value"]})
            if name in forecasts:
                dl = tst.dm_test(-forecasts[name]["logscore"].to_numpy(), -forecasts["baseline"]["logscore"].to_numpy())
                rec.update({"dm_logscore": dl["dm_stat"], "p_logscore": dl["p_value"]})
        rows.append(rec)
    fe = pd.DataFrame(rows).set_index("model")
    fe.to_csv(os.path.join(out, "forecast_evaluation.csv"))
    ht = forecasts["tvtp_logrv"]["h"].to_numpy()
    ref_rows = []
    for name, h in [(f"rolling RV{a.rv_window}", rv_prev), ("EWMA(0.94)", ew), ("baseline HMM", hb)]:
        dq = tst.dm_test(tst.loss_qlike(ht, proxy), tst.loss_qlike(h, proxy)); dm = tst.dm_test(tst.loss_mse(ht, proxy), tst.loss_mse(h, proxy))
        ref_rows.append({"tvtp_logrv vs": name, "dm_qlike": dq["dm_stat"], "p_qlike": dq["p_value"], "dm_mse": dm["dm_stat"], "p_mse": dm["p_value"]})
    fe_ref = pd.DataFrame(ref_rows).set_index("tvtp_logrv vs")
    fe_ref.to_csv(os.path.join(out, "forecast_evaluation_vs_reference.csv"))
    print("\nTVTP-logrv against reference forecasts (negative favours TVTP):\n", fe_ref.round(4).to_string())
    summary["forecast_evaluation_vs_reference"] = json.loads(fe_ref.to_json(orient="index"))
    print(fe.round(4).to_string())
    summary["forecast_evaluation"] = json.loads(fe.to_json(orient="index"))

    # ---------------- 4. strategies ----------------
    print("\n== strategy backtests (primary config) ==")
    cfg = {"target_vol": a.target_vol, "w_max": a.w_max, "cost": a.cost_bps / 1e4}
    strats, ret = strategies_from_forecasts(forecasts, df, cfg, a.K)
    mt = pd.DataFrame({k: st.metrics(v["r"].to_numpy(), v["turnover"].to_numpy()) for k, v in strats.items()}).T
    mt.to_csv(os.path.join(out, "strategy_metrics.csv"))
    print(mt.round(3).to_string())
    summary["strategy_config"] = cfg
    summary["strategy_metrics"] = json.loads(mt.to_json(orient="index"))
    pd.DataFrame({k: v["r"].to_numpy() for k, v in strats.items()}, index=idx).to_csv(os.path.join(out, "strategy_returns.csv"))

    # trials for the deflated Sharpe ratio: every configuration that could have been selected
    # (both allocation rules, every target / cap / risk-aversion value, every model)
    trials = {}
    for wm in [1.0, 1.5]:
        for tv in [0.15, 0.20, 0.30]:
            for name, fc in forecasts.items():
                xi = fc[[f"xi{k}" for k in range(a.K)]].to_numpy()
                trials[f"prop|{name}|tv={tv}|wmax={wm}"] = tst.sharpe(st.backtest(st.weights(xi, fc["h"].to_numpy(), tv, wm), ret, cfg["cost"])["r"].to_numpy())
            trials[f"voltarget|tv={tv}|wmax={wm}"] = tst.sharpe(st.backtest(st.weights(xi_b_dummy(forecasts, a.K), forecasts["baseline"]["h"].to_numpy(), tv, wm, regime_conditioned=False), ret, cfg["cost"])["r"].to_numpy())
        for g in [1.0, 2.0, 4.0]:
            for name, fc in forecasts.items():
                trials[f"kelly|{name}|gamma={g}|wmax={wm}"] = tst.sharpe(st.backtest(st.weights_kelly(fc["m"].to_numpy(), fc["h"].to_numpy(), g, wm), ret, cfg["cost"])["r"].to_numpy())
    pd.Series(trials).sort_values(ascending=False).to_csv(os.path.join(out, "trial_sharpes.csv"))

    # robustness rule: capped Kelly on the mixture forecasts
    cfg_k = {"gamma": a.gamma, "w_max": a.w_max, "cost": a.cost_bps / 1e4, "target_vol": a.target_vol}
    strats_k, _ = strategies_from_forecasts(forecasts, df, cfg_k, a.K, rule="kelly")
    mt_k = pd.DataFrame({k: st.metrics(v["r"].to_numpy(), v["turnover"].to_numpy()) for k, v in strats_k.items()}).T
    mt_k.to_csv(os.path.join(out, "strategy_metrics_kelly.csv"))
    print(f"\n== robustness rule: capped Kelly, gamma={a.gamma} ==\n", mt_k.round(3).to_string())
    summary["strategy_metrics_kelly"] = json.loads(mt_k.to_json(orient="index"))
    pd.DataFrame({k: v["r"].to_numpy() for k, v in strats_k.items()}, index=idx).to_csv(os.path.join(out, "strategy_returns_kelly.csv"))
    rk_base = strats_k["baseline"]["r"].to_numpy()
    summary["significance_kelly"] = {name: {"bootstrap_vs_baseline": tst.bootstrap_sharpe_diff(strats_k[name]["r"].to_numpy(), rk_base, B=a.B, mean_block=20, seed=7),
                                            "psr_vs_0": tst.probabilistic_sharpe_ratio(strats_k[name]["r"].to_numpy(), 0.0),
                                            "dsr": tst.deflated_sharpe_ratio(strats_k[name]["r"].to_numpy(), np.array(list(trials.values())))}
                                     for name in ["tvtp_logrv", "tvtp_exflow", "tvtp_both"]}
    sig = {}
    r_base = strats["baseline"]["r"].to_numpy()
    for name in ["tvtp_logrv", "tvtp_exflow", "tvtp_both"]:
        r = strats[name]["r"].to_numpy()
        sig[name] = {"psr_vs_0": tst.probabilistic_sharpe_ratio(r, 0.0),
                     "psr_vs_baseline_sr": tst.probabilistic_sharpe_ratio(r, tst.sharpe(r_base)),
                     "dsr": tst.deflated_sharpe_ratio(r, np.array(list(trials.values()))),
                     "bootstrap_vs_baseline": tst.bootstrap_sharpe_diff(r, r_base, B=a.B, mean_block=20, seed=7),
                     "bootstrap_vs_buyhold": tst.bootstrap_sharpe_diff(r, strats["Buy&Hold"]["r"].to_numpy(), B=a.B, mean_block=20, seed=7)}
    sig["baseline"] = {"psr_vs_0": tst.probabilistic_sharpe_ratio(r_base, 0.0),
                       "dsr": tst.deflated_sharpe_ratio(r_base, np.array(list(trials.values()))),
                       "bootstrap_vs_buyhold": tst.bootstrap_sharpe_diff(r_base, strats["Buy&Hold"]["r"].to_numpy(), B=a.B, mean_block=20, seed=7)}
    summary["significance"] = sig
    print("\n== significance (primary config) ==")
    for k, v in sig.items():
        print(k, json.dumps({kk: {m: (round(x, 4) if isinstance(x, float) else x) for m, x in vv.items()} for kk, vv in v.items()}, indent=None)[:900])

    # cost sensitivity
    cs = []
    for bps in [0, 10, 25, 50]:
        c2 = dict(cfg, cost=bps / 1e4)
        s2, _ = strategies_from_forecasts(forecasts, df, c2, a.K)
        rec = {"cost_bps": bps}
        for k, v in s2.items():
            rec[f"sharpe_{k}"] = tst.sharpe(v["r"].to_numpy())
        rec["sr_diff_tvtp_logrv_minus_baseline"] = rec["sharpe_tvtp_logrv"] - rec["sharpe_baseline"]
        cs.append(rec)
    cs = pd.DataFrame(cs).set_index("cost_bps")
    cs.to_csv(os.path.join(out, "cost_sensitivity.csv"))
    print("\ncost sensitivity (Sharpe):\n", cs.round(3).to_string())
    summary["cost_sensitivity"] = json.loads(cs.to_json(orient="index"))

    # ---------------- 5. figures ----------------
    if not a.skip_fullsample:
        make_figures(df, fits, summary['_dating_b'], summary['_dating_t'], forecasts, strats, idx, out, a)
    summary.pop('_dating_b', None); summary.pop('_dating_t', None)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    write_summary_md(summary, summary.get('_tab'), summary.get('_ev'), wf_lr, fe, mt, cs, sig, out, a)
    summary.pop('_tab', None); summary.pop('_ev', None)
    print("\nsaved to", out)


def run_fullsample(df, a, out, summary, fits):
    print("\n== full-sample fits ==")
    st = a.emission == "student_t"
    fits["HMM K=2"] = full_sample_fit(df, [], 2, a.n_restarts, "HMM K=2", student_t=st)
    warm = fits["HMM K=2"].params
    fits["TVTP logrv"] = full_sample_fit(df, [f"logrv{a.rv_window}"], 2, a.n_restarts, "TVTP logrv", warm, student_t=st)
    fits["TVTP exflow"] = full_sample_fit(df, ["exflow30"], 2, a.n_restarts, "TVTP exflow", warm, student_t=st)
    fits["TVTP both"] = full_sample_fit(df, [f"logrv{a.rv_window}", "exflow30"], 2, a.n_restarts, "TVTP both", warm, student_t=st)
    if not st:   # three-regime fits are reported for the Gaussian specification only
        fits["HMM K=3"] = full_sample_fit(df, [], 3, a.n_restarts, "HMM K=3")
        fits["TVTP logrv K=3"] = full_sample_fit(df, [f"logrv{a.rv_window}"], 3, a.n_restarts, "TVTP logrv K=3", fits["HMM K=3"].params)
    tab = fit_table(fits, "HMM K=2")
    if not st:
        b3, t3 = fits["HMM K=3"], fits["TVTP logrv K=3"]
        tab.loc["TVTP logrv K=3", "lr_stat"], tab.loc["TVTP logrv K=3", "lr_p"] = lr_test(b3.loglik, t3.loglik, t3.params.n_params - b3.params.n_params)
    tab.to_csv(os.path.join(out, "fullsample_fits.csv"))
    print(tab[[c for c in tab.columns if not c.startswith("se_")]].round(4).T.to_string())
    summary["full_sample"] = json.loads(tab.to_json(orient="index"))

    dating_b, ev_b = regime_dating(df, fits["HMM K=2"], out)
    dating_t, ev_t = regime_dating(df, fits["TVTP logrv"], out)
    # daily detail around each event (returns and real-time filtered probabilities), backing the narrative in the paper
    detail = []
    for name, date in EVENTS.items():
        dd = pd.Timestamp(date)
        win = df.loc[dd - pd.Timedelta(days=14): dd + pd.Timedelta(days=7), ["close", "y"]].copy()
        win["p_turb_filtered_hmm"] = dating_b.loc[win.index, "p_turbulent_filtered"].to_numpy()
        win["p_turb_filtered_tvtp"] = dating_t.loc[win.index, "p_turbulent_filtered"].to_numpy()
        win["event"] = name
        detail.append(win)
    pd.concat(detail).to_csv(os.path.join(out, "event_daily_detail.csv"))
    ev = pd.concat([ev_b.assign(model="HMM K=2"), ev_t.assign(model="TVTP logrv")])
    ev.to_csv(os.path.join(out, "event_dating.csv"), index=False)
    print("\nregime dating around events:\n", ev.to_string())
    summary["event_dating"] = ev.to_dict(orient="records")
    summary["_dating_b"], summary["_dating_t"], summary["_tab"], summary["_ev"] = dating_b, dating_t, tab, ev
    # transition-response summary for the log-RV TVTP model: standardisation constants, the volatility at which
    # p11(z) = p22(z), and persistence / expected duration on a grid of standardised covariate values
    ft = fits["TVTP logrv"]
    if ft.params.K == 2 and ft.std is not None:
        b1, b2 = ft.params.persistence_form()
        z_cross = (b2[0] - b1[0]) / (b1[1] - b2[1]) if abs(b1[1] - b2[1]) > 1e-12 else np.nan
        to_vol = lambda z: float(np.exp(z * ft.std.sd[0] + ft.std.mean[0]) * np.sqrt(365) * 100)
        grid = {}
        for zz in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            p11 = 1 / (1 + np.exp(-(b1[0] + b1[1] * zz))); p22 = 1 / (1 + np.exp(-(b2[0] + b2[1] * zz)))
            grid[str(zz)] = {"rv_ann_pct": to_vol(zz), "p11": float(p11), "p22": float(p22), "dur1": float(1 / (1 - p11)), "dur2": float(1 / (1 - p22))}
        summary["transition_response"] = {"z_mean_logrv": float(ft.std.mean[0]), "z_sd_logrv": float(ft.std.sd[0]),
                                          "crossover_z_std": float(z_cross), "crossover_rv_ann_pct": to_vol(z_cross) if np.isfinite(z_cross) else None,
                                          "grid": grid}


def xi_b_dummy(forecasts, K=2):
    return forecasts["baseline"][[f"xi{k}" for k in range(K)]].to_numpy()


def make_figures(df, fits, dating_b, dating_t, forecasts, strats, idx, out, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # regime dating: full sample (smoothed) and a real-time zoom on 2022 (filtered)
    y = df["y"].to_numpy(float)
    fb, ft = fits["HMM K=2"], fits["TVTP logrv"]
    filt_b = run_filter(fb.params, y, fb.X)[1][:, -1]
    filt_t = run_filter(ft.params, y, ft.X)[1][:, -1]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1.6, 1, 1.4]})
    ax = axes[0]
    ax.plot(df.index, df["close"], color="k", lw=0.8)
    ax.set_yscale("log"); ax.set_ylabel("BTC/USD (log)"); ax.set_title("Full sample")
    for name, d in EVENTS.items():
        ax.axvline(pd.Timestamp(d), color="tab:blue", ls=":", lw=1)
        ax.text(pd.Timestamp(d), df["close"].min() * 1.5, name, rotation=90, va="bottom", fontsize=8, color="tab:blue")
    ax = axes[1]
    ax.plot(dating_b.index, dating_b["p_turbulent_smoothed"].rolling(30, center=True).mean(), lw=0.9, label="HMM (constant P), 30-day mean")
    ax.plot(dating_t.index, dating_t["p_turbulent_smoothed"].rolling(30, center=True).mean(), lw=0.9, label="TVTP-HMM (log RV), 30-day mean")
    ax.set_ylabel("smoothed P(turbulent)"); ax.legend(fontsize=8); ax.set_xlim(df.index[0], df.index[-1])
    ax = axes[2]
    lo, hi = pd.Timestamp("2022-04-01"), pd.Timestamp("2023-01-15")
    m = (df.index >= lo) & (df.index <= hi)
    ax2 = ax.twinx()
    ax2.plot(df.index[m], df["close"][m], color="k", lw=0.8); ax2.set_ylabel("BTC/USD")
    ax.plot(df.index[m], filt_b[m], lw=1, label="HMM filtered P(turbulent), full-sample parameters")
    ax.plot(df.index[m], filt_t[m], lw=1, label="TVTP filtered P(turbulent), full-sample parameters")
    for name, d in EVENTS.items():
        if lo <= pd.Timestamp(d) <= hi:
            ax.axvline(pd.Timestamp(d), color="tab:blue", ls=":", lw=1)
            ax.text(pd.Timestamp(d), 0.02, name, rotation=90, va="bottom", fontsize=8, color="tab:blue")
    ax.set_ylim(0, 1.02); ax.set_ylabel("filtered P(turbulent)"); ax.set_title("Zoom: Terra/LUNA and FTX, filtered (causal) probabilities, full-sample parameters")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_regime_dating.png"), dpi=150); plt.close(fig)

    # transition response curves
    f = fits["TVTP logrv"]
    zs = np.linspace(-2.5, 2.5, 101)
    curves = response_curve(f, 1, zs, 0)
    z_raw = zs * f.std.sd[0] + f.std.mean[0]
    ann_vol = np.exp(z_raw) * np.sqrt(365) * 100
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, lab in [(1, "p11 (stay calm)"), (2, "p22 (stay turbulent)")]:
        prob, band = curves[i]
        ax.plot(ann_vol, prob, label=lab)
        if band is not None:
            ax.fill_between(ann_vol, band[0], band[1], alpha=0.2)
    x0 = np.zeros(f.params.d); x0[0] = 1
    P = fits["HMM K=2"].params.trans_at(np.ones(1))
    ax.axhline(P[0, 0], color="C0", ls="--", lw=0.8, label="constant HMM p11")
    ax.axhline(P[1, 1], color="C1", ls="--", lw=0.8, label="constant HMM p22")
    ax.set_xscale("log"); ax.set_xlabel(f"{a.rv_window}-day realised volatility, annualised %"); ax.set_ylabel("persistence probability")
    ax.set_title("TVTP persistence vs realised volatility (95% delta-method bands)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_transition_response.png"), dpi=150); plt.close(fig)

    # equity curves and weights
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    for k, v in strats.items():
        ax.plot(idx, np.cumprod(1 + v["r"].to_numpy()), lw=1, label=k)
    ax.set_yscale("log"); ax.set_ylabel("wealth (log)"); ax.legend(fontsize=8)
    ax.set_title(f"Out-of-sample {idx[0].date()}..{idx[-1].date()}, target vol {a.target_vol:.0%}, w_max {a.w_max}, cost {a.cost_bps} bps")
    ax = axes[1]
    ax.plot(idx, strats["baseline"]["w"], lw=0.7, label="HMM weight")
    ax.plot(idx, strats["tvtp_logrv"]["w"], lw=0.7, label="TVTP-logrv weight")
    ax.set_ylabel("position"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_equity.png"), dpi=150); plt.close(fig)


def write_summary_md(summary, tab, ev, wf_lr, fe, mt, cs, sig, out, a):
    L = []
    L.append("# Real-data results (generated by scripts/run_btc.py)\n")
    d = summary["data"]
    L.append(f"Data: {d['source']}. Sample {d['start']} to {d['end']}, T = {d['T']} daily observations. "
             f"Covariates: logrv = {d['covariates']['logrv']}; exflow = {d['covariates']['exflow']}.\n")
    if tab is not None:
        L.append("## Full-sample fits\n")
        show = [c for c in tab.columns if not c.startswith("se_")]
        L.append(tab[show].round(4).T.to_markdown() + "\n")
        L.append("Standard errors (numerical Hessian):\n")
        L.append(tab[[c for c in tab.columns if c.startswith("se_")]].round(4).T.to_markdown() + "\n")
        L.append("## Regime dating around known events (smoothed and filtered P(turbulent))\n")
        L.append(ev.round(3).to_markdown(index=False) + "\n")
    w = summary["walk_forward"]
    L.append(f"## Walk-forward\n\n{w['n_windows']} windows, rolling training length {w['train_len']} days, refit every {w['test_len']} days, "
             f"out-of-sample {w['oos_start']} to {w['oos_end']}.\n")
    L.append(wf_lr.round(3).to_markdown() + "\n")
    L.append("## Out-of-sample forecast evaluation (proxy: squared daily return; DM negative favours the row model over the constant HMM)\n")
    L.append(fe.round(4).to_markdown() + "\n")
    L.append(f"## Strategy backtest, primary configuration (target vol {a.target_vol}, w_max {a.w_max}, cost {a.cost_bps} bps, long-only)\n")
    L.append(mt.round(3).to_markdown() + "\n")
    L.append("## Significance of strategy performance\n")
    for k, v in sig.items():
        L.append(f"### {k}\n")
        for kk, vv in v.items():
            L.append(f"- {kk}: " + ", ".join(f"{m}={round(x, 4) if isinstance(x, float) else x}" for m, x in vv.items()))
        L.append("")
    L.append("### TVTP-logrv against reference forecasts (negative favours TVTP)\n")
    L.append(pd.DataFrame(summary["forecast_evaluation_vs_reference"]).T.round(4).to_markdown() + "\n")
    L.append(f"## Robustness rule: capped Kelly on mixture forecasts (gamma {a.gamma}, w_max {a.w_max}, cost {a.cost_bps} bps)\n")
    L.append(pd.DataFrame(summary["strategy_metrics_kelly"]).T.round(3).to_markdown() + "\n")
    for k, v in summary["significance_kelly"].items():
        L.append(f"- {k}: " + "; ".join(f"{kk}: " + ", ".join(f"{m}={round(x, 4) if isinstance(x, float) else x}" for m, x in vv.items()) for kk, vv in v.items()))
    L.append("")
    L.append("## Cost sensitivity, primary rule (annualised Sharpe)\n")
    L.append(cs.round(3).to_markdown() + "\n")
    L.append("Figures: fig_regime_dating.png, fig_transition_response.png, fig_equity.png\n")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--first_test", default="2018-01-01")
    ap.add_argument("--train_len", type=int, default=1460)
    ap.add_argument("--test_len", type=int, default=91)
    ap.add_argument("--rv_window", type=int, default=10)
    ap.add_argument("--K", type=int, default=2, help="regimes in the walk-forward comparison")
    ap.add_argument("--emission", choices=["gaussian", "student_t"], default="gaussian")
    ap.add_argument("--gamma", type=float, default=2.0, help="risk aversion of the capped-Kelly robustness rule")
    ap.add_argument("--n_restarts", type=int, default=5)
    ap.add_argument("--target_vol", type=float, default=0.20)
    ap.add_argument("--w_max", type=float, default=1.0)
    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--reuse", action="store_true", help="reuse cached walk-forward forecasts in --out if present")
    ap.add_argument("--no_polish", action="store_true", help="skip the L-BFGS polish in walk-forward windows (EM only)")
    ap.add_argument("--skip_fullsample", action="store_true", help="skip full-sample fits and figures (walk-forward, tests, strategies only)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "btc"))
    main(ap.parse_args())
