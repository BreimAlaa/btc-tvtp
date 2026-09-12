"""Unit tests mapped to Section 8 of docs/methodology_derivation.md.

(a) TVTP with zero slopes reproduces the baseline likelihood to machine precision
(b) filtered / predicted probabilities and transition rows sum to one
(c) consistency identities of the pairwise smoothed probabilities, eq. (4.3)
(d) parameter recovery on simulated data (short version; full study in scripts/run_synthetic.py)
(e) analytic score (6.6) matches a finite-difference gradient of Q_P
plus: EM monotonicity, relabelling invariance, smoother equals Rabiner backward pass,
Viterbi agrees with a brute-force search on a tiny sample, and cross-checks of the
baseline against hmmlearn and of the TVTP likelihood against statsmodels.
"""
import numpy as np
import pytest

from tvtp_hmm.core import (HMMParams, build_trans, gaussian_log_emission, hamilton_filter, kim_smoother, viterbi,
                           loglik, smooth, fit_em, fit_em_restarts, fit_direct, _q_row, lr_test, ergodic_distribution)
from tvtp_hmm.simulate import simulate_exog_ar1, simulate_internal_rv

TRUE = HMMParams(mu=np.array([0.001, -0.001]), sigma=np.array([0.015, 0.045]),
                 beta=np.array([[[3.0, -1.0]], [[-2.5, -1.0]]]))   # p11 = L(3 - z), p22 = L(2.5 + z)


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(7)
    y, Xlag, S, z = simulate_exog_ar1(TRUE, 1500, rng)
    return y, Xlag, S


def test_a_zero_slope_nesting(sim):
    y, Xlag, _ = sim
    base = fit_em_restarts(y, np.ones((len(y), 1)), 2, n_restarts=2)
    beta = np.concatenate([base.params.beta, np.zeros((2, 1, 1))], axis=2)
    nested = HMMParams(base.params.mu, base.params.sigma, beta)
    assert np.isclose(loglik(nested, y, Xlag), base.loglik, rtol=0, atol=1e-9)


def test_b_probabilities_sum_to_one(sim):
    y, Xlag, _ = sim
    log_eta = gaussian_log_emission(y, TRUE.mu, TRUE.sigma)
    trans = build_trans(TRUE.beta, Xlag)
    xi_pred, xi_filt, ll = hamilton_filter(log_eta, trans, TRUE.initial_distribution(Xlag))
    assert np.allclose(xi_pred.sum(1), 1) and np.allclose(xi_filt.sum(1), 1) and np.allclose(trans.sum(2), 1)
    assert np.all(xi_filt >= 0) and np.isfinite(ll).all()


def test_c_pairwise_identities(sim):
    y, Xlag, _ = sim
    xi_s, pair, xi_pred, xi_filt, ll = smooth(TRUE, y, Xlag)
    assert np.allclose(pair[1:].sum(axis=1), xi_s[1:], atol=1e-10)      # sum over origin i gives xi_{t|T}(j)
    assert np.allclose(pair[1:].sum(axis=2), xi_s[:-1], atol=1e-10)     # sum over destination j gives xi_{t-1|T}(i)
    assert np.allclose(xi_s.sum(1), 1)


def test_smoother_matches_rabiner_backward(sim):
    """Kim smoother (4.2) equals the scaled forward-backward posterior."""
    y, Xlag, _ = sim
    log_eta = gaussian_log_emission(y, TRUE.mu, TRUE.sigma)
    trans = build_trans(TRUE.beta, Xlag)
    pi = TRUE.initial_distribution(Xlag)
    xi_pred, xi_filt, ll = hamilton_filter(log_eta, trans, pi)
    xi_s, pair = kim_smoother(xi_pred, xi_filt, trans)
    T, K = log_eta.shape
    eta = np.exp(log_eta - log_eta.max(1, keepdims=True))
    beta_hat = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        b = trans[t + 1] @ (eta[t + 1] * beta_hat[t + 1])
        beta_hat[t] = b / b.sum()
    post = xi_filt * beta_hat
    post /= post.sum(1, keepdims=True)
    assert np.allclose(post, xi_s, atol=1e-8)


def test_viterbi_bruteforce():
    rng = np.random.default_rng(3)
    p = HMMParams(np.array([0.0, 0.0]), np.array([0.01, 0.04]), np.array([[[2.0, 0.5]], [[-2.0, -0.5]]]))
    y, Xlag, S, _ = simulate_exog_ar1(p, 9, rng)
    log_eta = gaussian_log_emission(y, p.mu, p.sigma)
    trans = build_trans(p.beta, Xlag)
    pi = p.initial_distribution(Xlag)
    path = viterbi(log_eta, trans, pi)
    import itertools
    best, best_path = -np.inf, None
    for cand in itertools.product(range(2), repeat=9):
        lp = np.log(pi[cand[0]]) + log_eta[0, cand[0]]
        for t in range(1, 9):
            lp += np.log(trans[t, cand[t - 1], cand[t]]) + log_eta[t, cand[t]]
        if lp > best:
            best, best_path = lp, cand
    assert tuple(path) == best_path


def test_e_analytic_score_vs_finite_difference(sim):
    y, Xlag, _ = sim
    xi_s, pair, *_ = smooth(TRUE, y, Xlag)
    X = Xlag[1:]
    for i in range(2):
        W = pair[1:, i, :]
        b = np.array([[1.0, 0.3]])
        q, g, H = _q_row(b, X, W)
        eps = 1e-6
        g_fd = np.empty_like(g)
        for k in range(len(g)):
            e = np.zeros(len(g)); e[k] = eps
            g_fd[k] = (_q_row((b.ravel() + e).reshape(1, 2), X, W)[0] - _q_row((b.ravel() - e).reshape(1, 2), X, W)[0]) / (2 * eps)
        assert np.allclose(g, g_fd, rtol=1e-5, atol=1e-6)
        # Hessian negative definite (concavity, Section 6.5)
        assert np.all(np.linalg.eigvalsh(H) < 0)


def test_em_monotone_and_relabel(sim):
    y, Xlag, _ = sim
    r = fit_em(y, Xlag, 2)
    # EM is monotone in L conditional on the initial distribution; tying pi to the ergodic
    # distribution of the updated P (Section 5.3) can move L by the first-period term only,
    # which is O(1e-6) here. Anything larger would signal a broken M-step.
    d = np.diff(r.history)
    assert np.all(d >= -1e-4) and (d < 0).sum() <= 1
    assert r.history[-1] - r.history[0] > 1.0
    assert r.params.sigma[0] < r.params.sigma[1]
    # relabelling is a likelihood invariance
    swapped = r.params.relabel(np.array([1, 0]))
    assert np.isclose(loglik(swapped, y, Xlag), loglik(r.params, y, Xlag), atol=1e-8)


def test_d_parameter_recovery_short(sim):
    y, Xlag, S = sim
    base = fit_em_restarts(y, np.ones((len(y), 1)), 2, n_restarts=3)
    beta0 = np.concatenate([base.params.beta, np.zeros((2, 1, 1))], axis=2)
    r = fit_direct(y, Xlag, fit_em(y, Xlag, 2, init=HMMParams(base.params.mu, base.params.sigma, beta0)).params, compute_se=True)
    b1, b2 = r.params.persistence_form()
    assert abs(b1[1] - (-1.0)) < 0.5 and abs(b2[1] - 1.0) < 0.5
    assert np.allclose(r.params.sigma, TRUE.sigma, rtol=0.15)
    # slope standard errors are finite and the LR test rejects H0 on data generated under H1
    assert np.isfinite(r.se.beta).all()
    stat, p = lr_test(base.loglik, r.loglik, 2)
    assert p < 0.01
    # regime dating accuracy of the smoothed path
    xi_s, *_ = smooth(r.params, y, Xlag)
    assert (xi_s.argmax(1) == S).mean() > 0.85


def test_internal_rv_simulation_runs():
    rng = np.random.default_rng(11)
    y, Xlag, S, z, ref = simulate_internal_rv(TRUE, 800, rng)
    assert y.shape == (800,) and Xlag.shape == (800, 2) and np.isfinite(Xlag).all()
    assert np.isfinite(loglik(TRUE, y, Xlag))


def test_ergodic_distribution():
    P = np.array([[0.9, 0.1], [0.3, 0.7]])
    pi = ergodic_distribution(P)
    assert np.allclose(pi, [0.75, 0.25])


def test_crosscheck_hmmlearn_baseline(sim):
    pytest.importorskip("hmmlearn")
    from hmmlearn.hmm import GaussianHMM
    y, _, _ = sim
    ours = fit_em_restarts(y, np.ones((len(y), 1)), 2, n_restarts=3, tol=1e-8)
    # evaluate hmmlearn's likelihood at OUR parameters (same model, same initial distribution)
    m = GaussianHMM(n_components=2, covariance_type="diag", init_params="", params="")
    P = build_trans(ours.params.beta, np.ones((1, 1)))[0]
    m.startprob_ = ergodic_distribution(P)
    m.transmat_ = P
    m.means_ = ours.params.mu[:, None]
    m.covars_ = (ours.params.sigma ** 2)[:, None]
    assert np.isclose(m.score(y[:, None]), ours.loglik, atol=1e-6)


def test_crosscheck_statsmodels_tvtp(sim):
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    y, Xlag, _ = sim
    base = fit_em_restarts(y, np.ones((len(y), 1)), 2, n_restarts=3)
    beta0 = np.concatenate([base.params.beta, np.zeros((2, 1, 1))], axis=2)
    ours = fit_direct(y, Xlag, fit_em(y, Xlag, 2, init=HMMParams(base.params.mu, base.params.sigma, beta0)).params)
    mod = MarkovRegression(y, k_regimes=2, exog_tvtp=Xlag, switching_variance=True)
    res = mod.fit(search_reps=10, disp=False, maxiter=500)
    # statsmodels initialises the chain differently and its optimiser can stop early;
    # both maximisers must agree to within a small margin on the optimum
    assert ours.loglik >= res.llf - 0.5
    assert abs(ours.loglik - res.llf) < 5.0


# ----------------------------------------------------------------------------
# Student-t emissions
# ----------------------------------------------------------------------------
def test_student_t_density_and_gaussian_limit():
    from scipy import stats
    from tvtp_hmm.core import student_t_log_emission
    y = np.linspace(-0.1, 0.1, 41)
    mu, sigma, nu = np.array([0.001, -0.002]), np.array([0.01, 0.03]), np.array([4.0, 8.0])
    ours = student_t_log_emission(y, mu, sigma, nu)
    ref = np.column_stack([stats.t.logpdf(y, nu[j], loc=mu[j], scale=sigma[j]) for j in range(2)])
    assert np.allclose(ours, ref, atol=1e-10)
    big = student_t_log_emission(y, mu, sigma, np.array([1e6, 1e6]))
    assert np.allclose(big, gaussian_log_emission(y, mu, sigma), atol=1e-2)


def test_student_t_em_recovery():
    rng = np.random.default_rng(5)
    true = HMMParams(np.array([0.001, -0.001]), np.array([0.012, 0.035]), np.array([[[3.0, -1.0]], [[-2.5, -1.0]]]), np.array([4.0, 6.0]))
    y, Xlag, S, _ = simulate_exog_ar1(true, 3000, rng)
    base = fit_em_restarts(y, np.ones((len(y), 1)), 2, n_restarts=2, student_t=True)
    beta0 = np.concatenate([base.params.beta, np.zeros((2, 1, 1))], axis=2)
    em = fit_em(y, Xlag, 2, init=HMMParams(base.params.mu, base.params.sigma, beta0, base.params.nu), student_t=True)
    r = fit_direct(y, Xlag, em.params)
    assert r.params.nu is not None and np.all(r.params.nu > 2)
    assert np.allclose(r.params.sigma, true.sigma, rtol=0.2)
    assert np.all(np.abs(r.params.nu - true.nu) < 3.0)
    b1, b2 = r.params.persistence_form()
    assert abs(b1[1] + 1.0) < 0.5 and abs(b2[1] - 1.0) < 0.5
    # EM history increases and the t model beats a Gaussian fit on t data
    d = np.diff(em.history)
    assert np.all(d >= -1e-4)
    g = fit_direct(y, Xlag, fit_em(y, Xlag, 2).params)
    assert r.loglik > g.loglik + 5
    assert r.params.n_params == g.params.n_params + 2
    # forecast variance uses sigma^2 nu/(nu-2)
    assert np.allclose(r.params.regime_variance, r.params.sigma ** 2 * r.params.nu / (r.params.nu - 2))


# ----------------------------------------------------------------------------
# Edge cases in the evaluation and walk-forward layers
# ----------------------------------------------------------------------------
def test_stats_edge_cases():
    from tvtp_hmm import stats_tests as tst
    r = np.random.default_rng(0).normal(0.001, 0.02, 500)
    same = tst.dm_test(r ** 2, r ** 2)
    assert same["dm_stat"] == 0.0 and same["p_value"] == 1.0
    with pytest.raises(ValueError):
        tst.dm_test(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    assert np.isnan(tst.sharpe(np.zeros(10)))
    psr = tst.probabilistic_sharpe_ratio(r)
    assert 0 <= psr["psr"] <= 1
    dsr1 = tst.deflated_sharpe_ratio(r, np.array([0.5]))
    assert np.isnan(dsr1["dsr"]) and dsr1["n_trials"] == 1
    dsr = tst.deflated_sharpe_ratio(r, np.array([0.1, 0.5, 0.9, 1.2]))
    assert 0 <= dsr["dsr"] <= 1 and dsr["sr0_ann"] > 0
    bs = tst.bootstrap_sharpe_diff(r, r * 0.9, B=50, mean_block=5)
    assert bs["ci_low"] <= bs["sr_diff"] <= bs["ci_high"]


def test_strategy_edge_cases():
    from tvtp_hmm import strategy as st
    xi = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    h = np.array([0.0, 1e-4, 4e-4])          # zero variance must not produce inf weights
    w = st.weights(xi, h, 0.2, 1.0)
    assert np.all(np.isfinite(w)) and np.all(w >= 0) and np.all(w <= 1.0)
    assert w[1] == 0.0                        # fully turbulent regime -> flat
    wk = st.weights_kelly(np.array([-0.01, 0.01, 0.0]), h, 2.0, 1.5)
    assert wk[0] == 0.0 and 0 <= wk[1] <= 1.5 and wk[2] == 0.0
    bt = st.backtest(np.array([1.0, 1.0, 0.0]), np.array([0.01, -0.02, 0.03]), cost=0.001)
    assert np.isclose(bt["turnover"].sum(), 2.0)
    assert np.isclose(bt["r"].iloc[0], 0.01 - 0.001)
    m = st.metrics(np.full(30, 0.001))          # no negative days: sortino is nan, not an exception
    assert np.isnan(m["sortino"]) and m["max_drawdown"] == 0.0


def test_walk_forward_rejects_bad_config():
    import pandas as pd
    from tvtp_hmm.walkforward import ModelSpec, walk_forward
    df = pd.DataFrame({"y": np.random.default_rng(1).normal(0, 0.02, 200)}, index=pd.date_range("2020-01-01", periods=200))
    with pytest.raises(ValueError):
        walk_forward(df, [ModelSpec("baseline", [], 2)], train_len=300, test_len=50, verbose=False)
    with pytest.raises(ValueError):
        walk_forward(df, [ModelSpec("tvtp", ["z"], 2)], train_len=100, test_len=50, verbose=False)


def test_fit_rejects_short_or_nonfinite_input():
    with pytest.raises(ValueError, match="at least three"):
        fit_em(np.array([0.01, -0.02]), np.ones((2, 1)), 2, max_iter=5)
    with pytest.raises(ValueError, match="finite"):
        fit_em(np.array([0.01, np.nan, 0.02]), np.ones((3, 1)), 2, max_iter=5)
    with pytest.raises(ValueError, match="finite"):
        fit_em(np.array([0.01, 0.02, 0.03]), np.array([[1.0], [np.nan], [1.0]]), 2, max_iter=5)


def test_short_sample_and_k1_guards():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 0.02, 6)
    r = fit_em(y, np.ones((6, 1)), 2, max_iter=5)          # 3 <= T < 10 used to crash inside init_params
    assert np.isfinite(r.loglik)
    with pytest.raises(ValueError, match="K=1"):
        fit_em(y, np.column_stack([np.ones(6), rng.normal(size=6)]), 1, max_iter=5)


def test_walk_forward_windows_validation():
    from tvtp_hmm.data import walk_forward_windows
    with pytest.raises(ValueError):
        list(walk_forward_windows(100, 10, 0))              # used to be an infinite generator
    with pytest.raises(ValueError):
        list(walk_forward_windows(100, 10, -5))
    assert [(w.train_start, w.train_end, w.test_end) for w in walk_forward_windows(25, 10, 7)] == [(0, 10, 17), (7, 17, 24), (14, 24, 25)]


def test_strategy_returns_frame_alignment():
    """backtest() returns a RangeIndex frame; building a dated frame from its columns must use the values, not the Series."""
    import pandas as pd
    from tvtp_hmm import strategy as st
    idx = pd.date_range("2020-01-01", periods=3)
    bt = st.backtest(np.array([1.0, 0.5, 0.0]), np.array([0.01, -0.02, 0.03]), cost=0.0)
    bad = pd.DataFrame({"r": bt["r"]}, index=idx)
    good = pd.DataFrame({"r": bt["r"].to_numpy()}, index=idx)
    assert bad["r"].isna().all() and np.allclose(good["r"], [0.01, -0.01, 0.0])


def test_forecast_segment_is_causal_in_the_test_covariates():
    """Out-of-sample forecasts must not depend on covariate or return values dated after them, even for short segments
    (the initial distribution used to be evaluated at the mean regressor of train+test rows)."""
    from tvtp_hmm.walkforward import forecast_segment
    rng = np.random.default_rng(4)
    y, Xlag, _, _ = simulate_exog_ar1(TRUE, 40, rng)
    a = forecast_segment(TRUE, y, Xlag, 5)          # a 5-day training prefix keeps the initial condition visible
    y2, X2 = y.copy(), Xlag.copy()
    y2[20:] = rng.normal(0, 0.3, 20); X2[20:, 1] = rng.normal(0, 5, 20)
    b = forecast_segment(TRUE, y2, X2, 5)
    assert np.array_equal(a["xi_pred"][:15], b["xi_pred"][:15]) and np.array_equal(a["h"][:15], b["h"][:15])
    assert np.array_equal(a["logscore"][:15], b["logscore"][:15])
