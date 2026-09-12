"""Build paper/tables.tex from results/. Every number in the paper comes from here.

    python scripts/make_paper_tables.py [--btc results/btc] [--k3 results/btc_k3] [--syn results/synthetic]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from tvtp_hmm import stats_tests as tst  # noqa: E402


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}"


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main(a):
    btc = os.path.join(ROOT, a.btc)
    S = json.load(open(os.path.join(btc, "summary.json")))
    fs = pd.read_csv(os.path.join(btc, "fullsample_fits.csv"), index_col=0)
    fe = pd.read_csv(os.path.join(btc, "forecast_evaluation.csv"), index_col=0)
    feref = pd.read_csv(os.path.join(btc, "forecast_evaluation_vs_reference.csv"), index_col=0)
    mt = pd.read_csv(os.path.join(btc, "strategy_metrics.csv"), index_col=0)
    mk = pd.read_csv(os.path.join(btc, "strategy_metrics_kelly.csv"), index_col=0)
    cs = pd.read_csv(os.path.join(btc, "cost_sensitivity.csv"), index_col=0)
    ev = pd.read_csv(os.path.join(btc, "event_dating.csv"))
    rec = pd.read_csv(os.path.join(ROOT, a.syn, "recovery_exog_ar1.csv"), index_col=0)
    rec_int = pd.read_csv(os.path.join(ROOT, a.syn, "recovery_internal_rv.csv"), index_col=0)
    syn = json.load(open(os.path.join(ROOT, a.syn, "summary.json")))
    k3 = None
    if a.k3 and os.path.exists(os.path.join(ROOT, a.k3, "summary.json")):
        k3 = json.load(open(os.path.join(ROOT, a.k3, "summary.json")))
        fe3 = pd.read_csv(os.path.join(ROOT, a.k3, "forecast_evaluation.csv"), index_col=0)
        mt3 = pd.read_csv(os.path.join(ROOT, a.k3, "strategy_metrics.csv"), index_col=0)
        mk3 = pd.read_csv(os.path.join(ROOT, a.k3, "strategy_metrics_kelly.csv"), index_col=0)

    tt = None
    if a.t and os.path.exists(os.path.join(ROOT, a.t, "summary.json")):
        tt = json.load(open(os.path.join(ROOT, a.t, "summary.json")))
        fst = pd.read_csv(os.path.join(ROOT, a.t, "fullsample_fits.csv"), index_col=0)
        fet = pd.read_csv(os.path.join(ROOT, a.t, "forecast_evaluation.csv"), index_col=0)
        mtt = pd.read_csv(os.path.join(ROOT, a.t, "strategy_metrics.csv"), index_col=0)
        mkt = pd.read_csv(os.path.join(ROOT, a.t, "strategy_metrics_kelly.csv"), index_col=0)
        wint = pd.read_csv(os.path.join(ROOT, a.t, "walkforward_windows.csv"))
    L = []
    d = S["data"]
    w = S["walk_forward"]
    cfg = S["strategy_config"]
    # ---------------- macros ----------------
    macros = {
        "DataStart": d["start"], "DataEnd": d["end"], "DataT": str(d["T"]),
        "OOSStart": w["oos_start"], "OOSEnd": w["oos_end"], "NWindows": str(w["n_windows"]),
        "TrainLen": str(w["train_len"]), "TestLen": str(w["test_len"]),
        "NOOS": str(int(mt.loc["baseline", "n_days"])),
        "TargetVol": f"{cfg['target_vol']*100:.0f}", "WMax": f"{cfg['w_max']:.0f}", "CostBps": f"{cfg['cost']*1e4:.0f}",
        "LRlogrv": fmt(fs.loc["TVTP logrv", "lr_stat"], 1), "LRexflow": fmt(fs.loc["TVTP exflow", "lr_stat"], 1),
        "LRexflowP": fmt(fs.loc["TVTP exflow", "lr_p"], 3), "LRboth": fmt(fs.loc["TVTP both", "lr_stat"], 1),
        "LRkthree": fmt(fs.loc["TVTP logrv K=3", "lr_stat"], 1),
        "SigCalm": fmt(fs.loc["HMM K=2", "sigma1_ann%"], 0), "SigTurb": fmt(fs.loc["HMM K=2", "sigma2_ann%"], 0),
        "SigCalmT": fmt(fs.loc["TVTP logrv", "sigma1_ann%"], 0), "SigTurbT": fmt(fs.loc["TVTP logrv", "sigma2_ann%"], 0),
        "BetaOneRV": fmt(fs.loc["TVTP logrv", "beta1_logrv10"], 3), "BetaOneRVse": fmt(fs.loc["TVTP logrv", "se_beta1_logrv10"], 3),
        "BetaTwoRV": fmt(fs.loc["TVTP logrv", "beta2_logrv10"], 3), "BetaTwoRVse": fmt(fs.loc["TVTP logrv", "se_beta2_logrv10"], 3),
        "PstayCalm": fmt(fs.loc["HMM K=2", "p11_at_mean_z"], 3), "PstayTurb": fmt(fs.loc["HMM K=2", "p22_at_mean_z"], 3),
        "DurCalm": fmt(fs.loc["HMM K=2", "dur1_at_mean_z"], 1), "DurTurb": fmt(fs.loc["HMM K=2", "dur2_at_mean_z"], 1),
        "BICtwo": fmt(fs.loc["HMM K=2", "bic"], 0), "BICthree": fmt(fs.loc["HMM K=3", "bic"], 0), "BICtvtp": fmt(fs.loc["TVTP logrv", "bic"], 0),
        "WFfracLR": f"{w['lr_bic']['tvtp_logrv']['frac_lr_p_lt_05']*100:.0f}", "WFfracBIC": f"{w['lr_bic']['tvtp_logrv']['frac_bic_better']*100:.0f}",
        "WFmedLR": fmt(w["lr_bic"]["tvtp_logrv"]["median_lr"], 1),
        "WFfracLRex": f"{w['lr_bic']['tvtp_exflow']['frac_lr_p_lt_05']*100:.0f}",
        "SlopeOneMin": fmt(w["logrv_slopes"]["beta1_1"]["min"], 2), "SlopeOneMax": fmt(w["logrv_slopes"]["beta1_1"]["max"], 2),
        "SlopeTwoMin": fmt(w["logrv_slopes"]["beta2_1"]["min"], 2), "SlopeTwoMax": fmt(w["logrv_slopes"]["beta2_1"]["max"], 2),
        "DMqlike": fmt(fe.loc["tvtp_logrv", "dm_qlike"], 2), "DMqlikeP": pval(fe.loc["tvtp_logrv", "p_qlike"]),
        "DMmse": fmt(fe.loc["tvtp_logrv", "dm_mse"], 2), "DMmseP": pval(fe.loc["tvtp_logrv", "p_mse"]),
        "DMls": fmt(fe.loc["tvtp_logrv", "dm_logscore"], 2), "DMlsP": pval(fe.loc["tvtp_logrv", "p_logscore"]),
        "DMewmaQ": fmt(feref.loc["EWMA(0.94)", "dm_qlike"], 2), "DMewmaQP": pval(feref.loc["EWMA(0.94)", "p_qlike"]),
        "DMrvQ": fmt(feref.loc["rolling RV10", "dm_qlike"], 2), "DMrvQP": pval(feref.loc["rolling RV10", "p_qlike"]),
        "SRbh": fmt(mt.loc["Buy&Hold", "sharpe"]), "SRvt": fmt(mt.loc["VolTarget (no regime)", "sharpe"]),
        "SRhmm": fmt(mt.loc["baseline", "sharpe"]), "SRtvtp": fmt(mt.loc["tvtp_logrv", "sharpe"]),
        "TOhmm": fmt(mt.loc["baseline", "ann_turnover"], 1), "TOtvtp": fmt(mt.loc["tvtp_logrv", "ann_turnover"], 1),
        "MDDbh": f"{mt.loc['Buy&Hold', 'max_drawdown']*100:.0f}", "MDDtvtp": f"{mt.loc['tvtp_logrv', 'max_drawdown']*100:.0f}",
        "VolTvtp": f"{mt.loc['tvtp_logrv', 'ann_vol']*100:.0f}",
        "SRhmmK": fmt(mk.loc["baseline", "sharpe"]), "SRtvtpK": fmt(mk.loc["tvtp_logrv", "sharpe"]),
        "MDDtvtpK": f"{mk.loc['tvtp_logrv', 'max_drawdown']*100:.0f}", "VolTvtpK": f"{mk.loc['tvtp_logrv', 'ann_vol']*100:.0f}",
        "WealthBH": fmt(mt.loc["Buy&Hold", "final_wealth"], 1), "WealthTvtpK": fmt(mk.loc["tvtp_logrv", "final_wealth"], 1),
    }
    sg = S["significance"]["tvtp_logrv"]
    sk = S["significance_kelly"]["tvtp_logrv"]
    macros.update({
        "BootDiff": fmt(sg["bootstrap_vs_baseline"]["sr_diff"]), "BootLo": fmt(sg["bootstrap_vs_baseline"]["ci_low"]),
        "BootHi": fmt(sg["bootstrap_vs_baseline"]["ci_high"]), "BootP": fmt(sg["bootstrap_vs_baseline"]["p_two_sided"], 3),
        "PSR": fmt(sg["psr_vs_0"]["psr"]), "DSR": fmt(sg["dsr"]["dsr"]), "NTrials": str(sg["dsr"]["n_trials"]), "SRzero": fmt(sg["dsr"]["sr0_ann"]),
        "BootDiffK": fmt(sk["bootstrap_vs_baseline"]["sr_diff"]), "BootLoK": fmt(sk["bootstrap_vs_baseline"]["ci_low"]),
        "BootHiK": fmt(sk["bootstrap_vs_baseline"]["ci_high"]), "BootPK": fmt(sk["bootstrap_vs_baseline"]["p_two_sided"], 3),
        "PSRK": fmt(sk["psr_vs_0"]["psr"]), "DSRK": fmt(sk["dsr"]["dsr"]),
        "GapZero": fmt(cs.loc[0, "sr_diff_tvtp_logrv_minus_baseline"]), "GapFifty": fmt(cs.loc[50, "sr_diff_tvtp_logrv_minus_baseline"]),
        "SynN": str(syn["exog_ar1"]["n"]), "SynT": str(syn["exog_ar1"]["T"]),
        "SynCovMin": fmt(rec["coverage95"].min(), 3), "SynCovMax": fmt(rec["coverage95"].max(), 3),
        "SynSizeTen": fmt(syn["lr_size_H0"]["reject_10pct"], 3), "SynSizeFive": fmt(syn["lr_size_H0"]["reject_5pct"], 3), "SynSizeOne": fmt(syn["lr_size_H0"]["reject_1pct"], 3),
        "SynKS": fmt(syn["lr_size_H0"]["ks_pvalue_vs_chi2_2"], 2), "SynPower": fmt(syn["exog_ar1"]["lr_reject_5pct"], 2),
        "SynAccS": f"{syn['exog_ar1']['regime_accuracy_smoothed']*100:.0f}", "SynAccF": f"{syn['exog_ar1']['regime_accuracy_filtered']*100:.0f}",
        "SynIntN": str(syn["internal_rv"]["n"]), "SynIntCovMin": fmt(rec_int["coverage95"].min(), 2), "SynIntCovMax": fmt(rec_int["coverage95"].max(), 2),
    })
    tr = S.get("transition_response", {})
    g = tr.get("grid", {})
    macros.update({"LRlogrvHalf": fmt(fs.loc["TVTP logrv", "lr_stat"] / 2, 0),
                   "CrossVol": fmt(tr.get("crossover_rv_ann_pct", np.nan), 0),
                   "LowVol": fmt(g.get("-2.0", {}).get("rv_ann_pct", np.nan), 0), "LowPcalm": fmt(g.get("-2.0", {}).get("p11", np.nan), 2),
                   "LowDurCalm": fmt(g.get("-2.0", {}).get("dur1", np.nan), 0), "LowDurTurb": fmt(g.get("-2.0", {}).get("dur2", np.nan), 1),
                   "HighVol": fmt(g.get("2.0", {}).get("rv_ann_pct", np.nan), 0), "HighPturb": fmt(g.get("2.0", {}).get("p22", np.nan), 2),
                   "HighDurTurb": fmt(g.get("2.0", {}).get("dur2", np.nan), 0), "HighDurCalm": fmt(g.get("2.0", {}).get("dur1", np.nan), 1)})
    ftx = ev[(ev.event == "FTX collapse") & (ev.model == "TVTP logrv")].iloc[0]
    covid = ev[(ev.event == "COVID crash") & (ev.model == "TVTP logrv")].iloc[0]
    macros.update({"FTXfirst": ftx["first_day_filtered_gt_0.5"], "FTXpre": fmt(ftx["p_turb_smoothed_pre10d"], 2),
                   "COVIDfirst": covid["first_day_filtered_gt_0.5"], "COVIDpre": fmt(covid["p_turb_smoothed_pre10d"], 2)})
    if k3 is not None:
        w3 = k3["walk_forward"]["lr_bic"]["tvtp_logrv"]
        s3 = k3["significance"]["tvtp_logrv"]["bootstrap_vs_baseline"]
        s3k = k3["significance_kelly"]["tvtp_logrv"]["bootstrap_vs_baseline"]
        macros.update({"KthreeWFfracLR": f"{w3['frac_lr_p_lt_05']*100:.0f}", "KthreeWFfracBIC": f"{w3['frac_bic_better']*100:.0f}",
                       "KthreeDMqlike": fmt(fe3.loc["tvtp_logrv", "dm_qlike"], 2), "KthreeDMqlikeP": pval(fe3.loc["tvtp_logrv", "p_qlike"]),
                       "KthreeDMls": fmt(fe3.loc["tvtp_logrv", "dm_logscore"], 2), "KthreeDMlsP": pval(fe3.loc["tvtp_logrv", "p_logscore"]),
                       "KthreeSRhmm": fmt(mt3.loc["baseline", "sharpe"]), "KthreeSRtvtp": fmt(mt3.loc["tvtp_logrv", "sharpe"]),
                       "KthreeSRhmmK": fmt(mk3.loc["baseline", "sharpe"]), "KthreeSRtvtpK": fmt(mk3.loc["tvtp_logrv", "sharpe"]),
                       "KthreeBootLo": fmt(s3["ci_low"]), "KthreeBootHi": fmt(s3["ci_high"]), "KthreeBootP": fmt(s3["p_two_sided"], 3),
                       "KthreeBootLoK": fmt(s3k["ci_low"]), "KthreeBootHiK": fmt(s3k["ci_high"]), "KthreeBootPK": fmt(s3k["p_two_sided"], 3),
                       "KthreeDMmse": fmt(fe3.loc["tvtp_logrv", "dm_mse"], 2), "KthreeDMmseP": pval(fe3.loc["tvtp_logrv", "p_mse"])})
    if tt is not None:
        wt = tt["walk_forward"]["lr_bic"]["tvtp_logrv"]
        st_ = tt["significance"]["tvtp_logrv"]["bootstrap_vs_baseline"]
        stk = tt["significance_kelly"]["tvtp_logrv"]["bootstrap_vs_baseline"]
        sl = wint[wint.spec == "tvtp_logrv"]
        macros.update({
            "TLLhmm": fmt(fst.loc["HMM K=2", "loglik"], 1), "TLLtvtp": fmt(fst.loc["TVTP logrv", "loglik"], 1),
            "TBIChmm": fmt(fst.loc["HMM K=2", "bic"], 0), "TBICtvtp": fmt(fst.loc["TVTP logrv", "bic"], 0),
            "TLR": fmt(fst.loc["TVTP logrv", "lr_stat"], 1), "TLRex": fmt(fst.loc["TVTP exflow", "lr_stat"], 1), "TLRexP": fmt(fst.loc["TVTP exflow", "lr_p"], 3),
            "TNuCalm": fmt(fst.loc["TVTP logrv", "nu1"], 1), "TNuTurb": fmt(fst.loc["TVTP logrv", "nu2"], 1),
            "TNuCalmH": fmt(fst.loc["HMM K=2", "nu1"], 1), "TNuTurbH": fmt(fst.loc["HMM K=2", "nu2"], 1),
            "TSigCalm": fmt(fst.loc["TVTP logrv", "sigma1_ann%"], 0), "TSigTurb": fmt(fst.loc["TVTP logrv", "sigma2_ann%"], 0),
            "TBetaOne": fmt(fst.loc["TVTP logrv", "beta1_logrv10"], 2), "TBetaOneSE": fmt(fst.loc["TVTP logrv", "se_beta1_logrv10"], 2),
            "TBetaTwo": fmt(fst.loc["TVTP logrv", "beta2_logrv10"], 2), "TBetaTwoSE": fmt(fst.loc["TVTP logrv", "se_beta2_logrv10"], 2),
            "TWFfracLR": f"{wt['frac_lr_p_lt_05']*100:.0f}", "TWFfracBIC": f"{wt['frac_bic_better']*100:.0f}", "TWFmedLR": fmt(wt["median_lr"], 1),
            "TSlopeOneMin": fmt(sl["beta1_1"].min(), 2), "TSlopeOneMax": fmt(sl["beta1_1"].max(), 2), "TSlopeOneMed": fmt(sl["beta1_1"].median(), 2),
            "TSlopeTwoMin": fmt(sl["beta2_1"].min(), 2), "TSlopeTwoMax": fmt(sl["beta2_1"].max(), 2), "TSlopeTwoMed": fmt(sl["beta2_1"].median(), 2),
            "TLShmm": fmt(fet.loc["baseline", "mean_logscore"], 4), "TLStvtp": fmt(fet.loc["tvtp_logrv", "mean_logscore"], 4),
            "TDMls": fmt(fet.loc["tvtp_logrv", "dm_logscore"], 2), "TDMlsP": pval(fet.loc["tvtp_logrv", "p_logscore"]),
            "TDMqlike": fmt(fet.loc["tvtp_logrv", "dm_qlike"], 2), "TDMqlikeP": pval(fet.loc["tvtp_logrv", "p_qlike"]),
            "TDMmse": fmt(fet.loc["tvtp_logrv", "dm_mse"], 2), "TDMmseP": pval(fet.loc["tvtp_logrv", "p_mse"]),
            "TSRhmm": fmt(mtt.loc["baseline", "sharpe"]), "TSRtvtp": fmt(mtt.loc["tvtp_logrv", "sharpe"]),
            "TSRhmmK": fmt(mkt.loc["baseline", "sharpe"]), "TSRtvtpK": fmt(mkt.loc["tvtp_logrv", "sharpe"]),
            "TBootLo": fmt(st_["ci_low"]), "TBootHi": fmt(st_["ci_high"]), "TBootP": fmt(st_["p_two_sided"], 3),
            "TBootLoK": fmt(stk["ci_low"]), "TBootHiK": fmt(stk["ci_high"]), "TBootPK": fmt(stk["p_two_sided"], 3),
            "GLStvtp": fmt(fe.loc["tvtp_logrv", "mean_logscore"], 4), "GLShmm": fmt(fe.loc["baseline", "mean_logscore"], 4),
        })
        macros["StudentTText"] = (
            "With Student-$t$ emissions (regime-specific degrees of freedom $\\nu_j$, estimated by an ECM step inside EM and polished by quasi-Newton; "
            "the regime variance is $\\sigma_j^2\\nu_j/(\\nu_j-2)$ and enters (\\ref{eq:fc}) in place of $\\sigma_j^2$) the picture changes more sharply. "
            "On the full sample the constant $t$-HMM reaches a log-likelihood of \\TLLhmm\\ (BIC \\TBIChmm), better than any Gaussian model including the three-regime ones, with "
            "$\\hat\\nu=(\\TNuCalmH, \\TNuTurbH)$: the calm regime is very heavy-tailed. Adding realised volatility to the transitions is still significant, LR $=\\TLR$ on two degrees of freedom, "
            "with slopes of the hypothesised sign and larger magnitude ($\\beta_{1,\\mathrm{RV}}=\\TBetaOne$ (\\TBetaOneSE), $\\beta_{2,\\mathrm{RV}}=\\TBetaTwo$ (\\TBetaTwoSE)), and the exchange-flow effect becomes marginally significant (LR $=\\TLRex$, $p=\\TLRexP$). "
            "Out of sample, however, the effect does not survive. In rolling four-year windows the LR test rejects in only \\TWFfracLR\\% of windows (median LR \\TWFmedLR), BIC prefers the TVTP model in \\TWFfracBIC\\%, and the slopes are unstable "
            "($\\beta_{1,\\mathrm{RV}}$ ranges from \\TSlopeOneMin\\ to \\TSlopeOneMax\\ across windows, median \\TSlopeOneMed). The $t$-HMM's own predictive log score (\\TLShmm) already exceeds that of the Gaussian TVTP model (\\GLStvtp), "
            "and the $t$-TVTP model adds nothing to it (\\TLStvtp, DM $t=\\TDMls$, $p=\\TDMlsP$); its variance forecasts are no better under QLIKE ($t=\\TDMqlike$, $p=\\TDMqlikeP$) and worse under MSE ($t=\\TDMmse$, $p=\\TDMmseP$). "
            "Sharpe ratios are \\TSRtvtp\\ against \\TSRhmm\\ under the proportional rule ([\\TBootLo, \\TBootHi], $p=\\TBootP$) and \\TSRtvtpK\\ against \\TSRhmmK\\ under Kelly ([\\TBootLoK, \\TBootHiK], $p=\\TBootPK$). "
            "Full tables are in the repository (\\texttt{results/btc\\_t/}).")
    else:
        macros["StudentTText"] = "[Student-$t$ results pending: run \\texttt{scripts/run\\_btc.py --emission student\\_t --out results/btc\\_t}.]"
    # cross-run checks that no single run can compute: DM tests between runs on the predictive log score, and the deflated
    # Sharpe ratio with every configuration backtested in the three runs pooled. Saved to results/btc/cross_run_checks.csv.
    if k3 is not None and tt is not None:
        fg = pd.read_csv(os.path.join(btc, "forecasts_tvtp_logrv.csv"), index_col=0, parse_dates=True)
        ftb = pd.read_csv(os.path.join(ROOT, a.t, "forecasts_baseline.csv"), index_col=0, parse_dates=True)
        f3b = pd.read_csv(os.path.join(ROOT, a.k3, "forecasts_baseline.csv"), index_col=0, parse_dates=True)
        assert fg.index.equals(ftb.index) and fg.index.equals(f3b.index)
        d_t = tst.dm_test(-ftb["logscore"].to_numpy(), -fg["logscore"].to_numpy())    # negative favours the t-HMM
        d_3 = tst.dm_test(-f3b["logscore"].to_numpy(), -fg["logscore"].to_numpy())    # negative favours the K=3 HMM
        pooled = np.concatenate([pd.read_csv(os.path.join(ROOT, p, "trial_sharpes.csv"), index_col=0).iloc[:, 0].to_numpy() for p in (a.btc, a.k3, a.t)])
        rp = pd.read_csv(os.path.join(btc, "strategy_returns.csv"), index_col=0)["tvtp_logrv"].to_numpy()
        rk = pd.read_csv(os.path.join(btc, "strategy_returns_kelly.csv"), index_col=0)["tvtp_logrv"].to_numpy()
        dsr_p = tst.deflated_sharpe_ratio(rp, pooled); dsr_k = tst.deflated_sharpe_ratio(rk, pooled)
        cross = pd.DataFrame([{"check": "DM log score: t-HMM (btc_t baseline) vs Gaussian TVTP log-RV (btc)", "stat": d_t["dm_stat"], "p_value": d_t["p_value"]},
                              {"check": "DM log score: K=3 constant HMM (btc_k3 baseline) vs K=2 Gaussian TVTP log-RV (btc)", "stat": d_3["dm_stat"], "p_value": d_3["p_value"]},
                              {"check": f"DSR TVTP log-RV proportional, pooled N={len(pooled)}", "stat": dsr_p["dsr"], "p_value": dsr_p["sr0_ann"]},
                              {"check": f"DSR TVTP log-RV Kelly, pooled N={len(pooled)}", "stat": dsr_k["dsr"], "p_value": dsr_k["sr0_ann"]}]).set_index("check")
        cross.to_csv(os.path.join(btc, "cross_run_checks.csv"))
        macros.update({"XDMtls": fmt(d_t["dm_stat"], 2), "XDMtlsP": fmt(d_t["p_value"], 3), "XDMkls": fmt(d_3["dm_stat"], 2), "XDMklsP": fmt(d_3["p_value"], 2),
                       "NTrialsAll": str(len(pooled)), "DSRpooled": fmt(dsr_p["dsr"]), "DSRKpooled": fmt(dsr_k["dsr"])})
    if k3 is not None:
        macros["KthreeText"] = (
            "Repeating the walk-forward exercise with $K=3$ (calm, intermediate and turbulent regimes, six transition slopes, EM without the quasi-Newton polish) "
            "gives the following. The realised-volatility TVTP model rejects the constant three-regime HMM in \\KthreeWFfracLR\\% of windows and has the lower BIC in \\KthreeWFfracBIC\\%. "
            "Against the three-regime baseline the Diebold--Mariano statistics are $t=\\KthreeDMqlike$ for QLIKE ($p=\\KthreeDMqlikeP$), $t=\\KthreeDMmse$ for MSE ($p=\\KthreeDMmseP$) and $t=\\KthreeDMls$ for the log score ($p\\KthreeDMlsP$). "
            "Under the proportional rule with $a=(1,\\tfrac12,0)$ in (\\ref{eq:rule}) the Sharpe ratios are \\KthreeSRtvtp\\ (TVTP) and \\KthreeSRhmm\\ (constant HMM), bootstrap interval for the difference [\\KthreeBootLo, \\KthreeBootHi] ($p=\\KthreeBootP$); "
            "under the Kelly rule \\KthreeSRtvtpK\\ against \\KthreeSRhmmK\\ ([\\KthreeBootLoK, \\KthreeBootHiK], $p=\\KthreeBootPK$). Full tables are in the repository (\\texttt{results/btc\\_k3/}).")
    else:
        macros["KthreeText"] = "[Three-regime walk-forward results pending: run \\texttt{scripts/run\\_btc.py --K 3 --out results/btc\\_k3} and regenerate tables.]"
    for k, v in macros.items():
        L.append(f"\\newcommand{{\\{k}}}{{{v}}}")

    # ---------------- Table: synthetic recovery ----------------
    L.append("\n% ---- Table: Monte Carlo recovery ----")
    L.append("\\begin{table}[t]\\centering\\small")
    L.append(f"\\caption{{Monte Carlo parameter recovery, TVTP-HMM with an exogenous AR(1) covariate ({macros['SynN']} replications, $T={macros['SynT']}$). "
             "Standard errors from the numerical Hessian, eq.~(\\ref{eq:se}). Bottom rows: same design with the internal log-RV covariate "
             f"({macros['SynIntN']} replications), slopes only.}}\\label{{tab:mc}}")
    L.append("\\begin{tabular}{lrrrrrr}\\toprule Parameter & True & Mean & Bias & RMSE & Mean SE & Coverage 95\\% \\\\ \\midrule")
    names = {"mu1": "$\\mu_1$", "mu2": "$\\mu_2$", "sigma1": "$\\sigma_1$", "sigma2": "$\\sigma_2$",
             "b1_0": "$\\beta_{1,0}$", "b1_1": "$\\beta_{1,1}$", "b2_0": "$\\beta_{2,0}$", "b2_1": "$\\beta_{2,1}$"}
    for i, r in rec.iterrows():
        nd = 4 if i.startswith(("mu", "sigma")) else 3
        L.append(f"{names[i]} & {r['true']:.{nd}f} & {r['mean_est']:.{nd}f} & {r['bias']:.{nd}f} & {r['rmse']:.{nd}f} & {r['mean_se']:.{nd}f} & {r['coverage95']:.3f} \\\\")
    L.append("\\midrule")
    for i in ["b1_1", "b2_1"]:
        r = rec_int.loc[i]
        L.append(f"{names[i]} (internal RV) & {r['true']:.3f} & {r['mean_est']:.3f} & {r['bias']:.3f} & {r['rmse']:.3f} & {r['mean_se']:.3f} & {r['coverage95']:.3f} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")

    # ---------------- Table: full-sample fits ----------------
    L.append("\n% ---- Table: full-sample fits ----")
    cols = ["HMM K=2", "TVTP logrv", "TVTP exflow", "TVTP both", "HMM K=3", "TVTP logrv K=3"]
    heads = ["HMM", "TVTP RV", "TVTP flow", "TVTP both", "HMM $K{=}3$", "TVTP RV $K{=}3$"]
    L.append("\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4.5pt}")   # seven numeric columns: \\small overflows the text block
    L.append("\\caption{Full-sample estimates, " + d["start"] + " to " + d["end"] + f" ($T={d['T']}$). Regime volatilities annualised. "
             "Slopes are the persistence-form coefficients of eq.~(\\ref{eq:tvtp2}) on the standardised covariate, standard errors in parentheses. "
             "LR tests each TVTP model against the constant HMM with the same $K$.}\\label{tab:fits}")
    L.append("\\begin{tabular}{l" + "r" * len(cols) + "}\\toprule & " + " & ".join(heads) + " \\\\ \\midrule")
    def row(label, key, nd=1, se=None, scale=1.0):
        cells = []
        for c in cols:
            v = fs.loc[c, key] if key in fs.columns else np.nan
            if pd.isna(v):
                cells.append("--")
            else:
                s = f"{v*scale:.{nd}f}"
                if se and se in fs.columns and not pd.isna(fs.loc[c, se]):
                    s += f" ({fs.loc[c, se]*scale:.{nd}f})"
                cells.append(s)
        L.append(label + " & " + " & ".join(cells) + " \\\\")
    row("Log-likelihood", "loglik", 1)
    row("Parameters", "n_params", 0)
    row("BIC", "bic", 1)
    row("$\\mu_1$ (\\%/day)", "mu1", 3, "se_mu1", 100)
    row("$\\mu_2$ (\\%/day)", "mu2", 3, "se_mu2", 100)
    row("$\\mu_3$ (\\%/day)", "mu3", 3, "se_mu3", 100)
    row("$\\sigma_1$ (\\% ann.)", "sigma1_ann%", 1, "se_sigma1_ann%")
    row("$\\sigma_2$ (\\% ann.)", "sigma2_ann%", 1, "se_sigma2_ann%")
    row("$\\sigma_3$ (\\% ann.)", "sigma3_ann%", 1, "se_sigma3_ann%")
    row("$\\beta_{1,\\mathrm{RV}}$", "beta1_logrv10", 3, "se_beta1_logrv10")
    row("$\\beta_{2,\\mathrm{RV}}$", "beta2_logrv10", 3, "se_beta2_logrv10")
    row("$\\beta_{1,\\mathrm{flow}}$", "beta1_exflow30", 3, "se_beta1_exflow30")
    row("$\\beta_{2,\\mathrm{flow}}$", "beta2_exflow30", 3, "se_beta2_exflow30")
    row("$p_{11}$ at $\\bar z$", "p11_at_mean_z", 3)
    row("$p_{22}$ at $\\bar z$", "p22_at_mean_z", 3)
    row("LR statistic", "lr_stat", 1)
    cells = []
    for c in cols:
        p = fs.loc[c, "lr_p"] if "lr_p" in fs.columns else np.nan
        cells.append("--" if pd.isna(p) else pval(p))
    L.append("LR $p$-value & " + " & ".join(cells) + " \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")

    # ---------------- Table: OOS forecast evaluation ----------------
    L.append("\n% ---- Table: forecast evaluation ----")
    L.append("\\begin{table}[t]\\centering\\small")
    L.append(f"\\caption{{Out-of-sample one-day-ahead variance and density forecasts, {w['oos_start']} to {w['oos_end']} ({macros['NOOS']} days, "
             f"{w['n_windows']} refits). Proxy for realised variance: squared daily return. DM statistics test each row against the constant HMM "
             "(negative favours the row); HLN small-sample correction, Newey--West bandwidth $\\lfloor T^{1/3}\\rfloor$. "
             "Reference forecasts use no regime model. $^{***}$, $^{**}$, $^{*}$: 1, 5, 10\\%.}\\label{tab:fc}")
    L.append("\\begin{tabular}{lrrrrrr}\\toprule Model & QLIKE & MSE ($\\times10^{6}$) & Log score & DM QLIKE & DM MSE & DM log score \\\\ \\midrule")
    labels = {"baseline": "Constant HMM", "tvtp_logrv": "TVTP, log RV", "tvtp_exflow": "TVTP, exchange flow", "tvtp_both": "TVTP, both",
              "rolling RV10 (ref)": "Rolling RV$_{10}$ (ref.)", "EWMA(0.94) (ref)": "EWMA(0.94) (ref.)"}
    for i, r in fe.iterrows():
        ls = "--" if pd.isna(r.get("mean_logscore", np.nan)) else f"{r['mean_logscore']:.4f}"
        dq = "--" if pd.isna(r.get("dm_qlike", np.nan)) else f"{r['dm_qlike']:.2f}$^{{{stars(r['p_qlike'])}}}$"
        dm = "--" if pd.isna(r.get("dm_mse", np.nan)) else f"{r['dm_mse']:.2f}$^{{{stars(r['p_mse'])}}}$"
        dl = "--" if pd.isna(r.get("dm_logscore", np.nan)) else f"{r['dm_logscore']:.2f}$^{{{stars(r['p_logscore'])}}}$"
        L.append(f"{labels[i]} & {r['mean_qlike']:.4f} & {r['mean_mse_x1e6']:.2f} & {ls} & {dq} & {dm} & {dl} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")

    # ---------------- Table: strategies ----------------
    L.append("\n% ---- Table: strategies ----")
    L.append("\\begin{table}[t]\\centering\\small")
    L.append(f"\\caption{{Out-of-sample strategy performance, {w['oos_start']} to {w['oos_end']}, proportional cost {macros['CostBps']}~bps per unit turnover, long-only. "
             f"Panel A: primary rule, eq.~(\\ref{{eq:rule}}), target volatility {macros['TargetVol']}\\%, cap {macros['WMax']}. "
             "Panel B: capped-Kelly rule, eq.~(\\ref{eq:kelly}), $\\gamma=2$. Sharpe and Sortino annualised (365 days).}\\label{tab:strat}")
    L.append("\\begin{tabular}{lrrrrrr}\\toprule Strategy & Ann. return & Ann. vol & Sharpe & Sortino & Max DD & Turnover/yr \\\\ \\midrule")
    slabels = {"Buy&Hold": "Buy and hold", "VolTarget (no regime)": "Volatility target, no regime", "baseline": "Constant HMM",
               "tvtp_logrv": "TVTP, log RV", "tvtp_exflow": "TVTP, exchange flow", "tvtp_both": "TVTP, both"}
    for panel, m in [("Panel A: proportional rule", mt), ("Panel B: capped Kelly", mk)]:
        L.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{panel}}}}} \\\\")
        for i, r in m.iterrows():
            if panel.startswith("Panel B") and i in ("Buy&Hold", "VolTarget (no regime)"):
                continue
            L.append(f"\\quad {slabels[i]} & {r['ann_return']*100:.1f}\\% & {r['ann_vol']*100:.1f}\\% & {r['sharpe']:.2f} & {r['sortino']:.2f} & {r['max_drawdown']*100:.0f}\\% & {r['ann_turnover']:.1f} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")

    # ---------------- Table: significance ----------------
    L.append("\n% ---- Table: significance ----")
    L.append("\\begin{table}[t]\\centering\\small")
    L.append("\\caption{Inference on strategy performance. PSR: probability that the true Sharpe exceeds zero. DSR: deflated Sharpe ratio, "
             f"benchmark $\\mathrm{{SR}}_0$ = expected maximum Sharpe among the $N={macros['NTrials']}$ configurations tried (both rules, all targets, caps, "
             "risk aversions and models). Bootstrap: stationary block bootstrap (2000 draws, mean block 20 days) of the annualised Sharpe difference "
             "against the constant-HMM strategy under the same rule.}\\label{tab:sig}")
    L.append("\\begin{tabular}{llrrrrr}\\toprule Rule & Model & Sharpe & PSR & DSR & $\\Delta$SR vs HMM [95\\% CI] & $p$ \\\\ \\midrule")
    for rule, key, m in [("Proportional", "significance", mt), ("Capped Kelly", "significance_kelly", mk)]:
        for name in ["tvtp_logrv", "tvtp_exflow", "tvtp_both"]:
            v = S[key][name]
            b = v["bootstrap_vs_baseline"]
            L.append(f"{rule} & {slabels[name]} & {m.loc[name, 'sharpe']:.2f} & {v['psr_vs_0']['psr']:.2f} & {v['dsr']['dsr']:.2f} & "
                     f"{b['sr_diff']:.2f} [{b['ci_low']:.2f}, {b['ci_high']:.2f}] & {b['p_two_sided']:.3f} \\\\")
        vb = S["significance"]["baseline"] if key == "significance" else None
        if vb:
            L.append(f"{rule} & Constant HMM & {m.loc['baseline', 'sharpe']:.2f} & {vb['psr_vs_0']['psr']:.2f} & {vb['dsr']['dsr']:.2f} & -- & -- \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")

    # ---------------- Table: cost sensitivity ----------------
    L.append("\n% ---- Table: cost sensitivity ----")
    L.append("\\begin{table}[t]\\centering\\small")
    L.append("\\caption{Annualised Sharpe ratio of the proportional rule as a function of the transaction cost.}\\label{tab:cost}")
    L.append("\\begin{tabular}{rrrrrr}\\toprule Cost (bps) & Buy and hold & Vol target & Constant HMM & TVTP, log RV & Difference \\\\ \\midrule")
    for c, r in cs.iterrows():
        L.append(f"{c} & {r['sharpe_Buy&Hold']:.2f} & {r['sharpe_VolTarget (no regime)']:.2f} & {r['sharpe_baseline']:.2f} & {r['sharpe_tvtp_logrv']:.2f} & {r['sr_diff_tvtp_logrv_minus_baseline']:+.2f} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")

    pdir = os.path.join(ROOT, "paper", "generated")
    os.makedirs(pdir, exist_ok=True)
    text = "\n".join(L) + "\n"
    # macros go to macros.tex; each "% ---- Table: name ----" block goes to tab_<name>.tex
    parts = text.split("\n% ---- Table: ")
    with open(os.path.join(pdir, "macros.tex"), "w") as f:
        f.write(parts[0])
    for block in parts[1:]:
        name, body = block.split(" ----\n", 1)
        fname = {"Monte Carlo recovery": "tab_mc", "full-sample fits": "tab_fits", "forecast evaluation": "tab_fc",
                 "strategies": "tab_strat", "significance": "tab_sig", "cost sensitivity": "tab_cost"}[name]
        with open(os.path.join(pdir, fname + ".tex"), "w") as f:
            f.write(body)
    print("wrote", pdir, "with", len(macros), "macros", "(K=3 included)" if k3 else "(no K=3 run found)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc", default="results/btc")
    ap.add_argument("--k3", default="results/btc_k3")
    ap.add_argument("--syn", default="results/synthetic")
    ap.add_argument("--t", default="results/btc_t")
    main(ap.parse_args())
