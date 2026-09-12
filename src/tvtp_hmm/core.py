"""
Core recursions and estimators for the (TVTP-)HMM.

Notation follows docs/methodology_derivation.md:

    y_t            log return, t = 0..T-1
    Xlag[t]        regressor vector x_{t-1} that drives the transition INTO period t
                   (Xlag[0] is unused; constant HMM: Xlag = ones((T,1)))
    trans[t,i,j]   p_{ij,t} = Pr(S_t=j | S_{t-1}=i, x_{t-1})   (trans[0] unused)
    xi_pred[t]     xi_{t|t-1}
    xi_filt[t]     xi_{t|t}
    xi_smooth[t]   xi_{t|T}
    pair[t,i,j]    Pr(S_{t-1}=i, S_t=j | F_T), eq. (4.3), t >= 1

Transition parameterisation (eq. 6.3): multinomial logit per origin row with
the last regime as reference category,

    p_{ij,t} = exp(beta[i,j] . x_{t-1}) / sum_l exp(beta[i,l] . x_{t-1}),  beta[i,K-1] = 0

so beta has shape (K, K-1, d). The constant-transition HMM is the special case
d = 1 (intercept only). For K = 2 the "persistence" form of eq. (6.2) is
    p_11 = Lambda(beta[0,0] . x),   p_22 = Lambda(-beta[1,0] . x).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:  # numba accelerates the sequential loops; everything works without it
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def deco(f):
            return f
        return deco


LOG2PI = np.log(2.0 * np.pi)
_TINY = 1e-300


# ----------------------------------------------------------------------------
# Emissions
# ----------------------------------------------------------------------------
def gaussian_log_emission(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """log f_j(y_t) for Gaussian regimes, eq. (1.2). Returns (T, K)."""
    y = np.asarray(y, float)[:, None]
    return -0.5 * LOG2PI - np.log(sigma)[None, :] - 0.5 * ((y - mu[None, :]) / sigma[None, :]) ** 2


def student_t_log_emission(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """log f_j(y_t) for Student-t regimes with location mu_j, scale sigma_j and nu_j degrees of freedom.
    Variance of regime j is sigma_j^2 nu_j / (nu_j - 2); nu_j > 2 is enforced by the parameterisation."""
    from scipy.special import gammaln
    y = np.asarray(y, float)[:, None]
    d2 = ((y - mu[None, :]) / sigma[None, :]) ** 2
    nu = nu[None, :]
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(sigma)[None, :]
            - 0.5 * (nu + 1) * np.log1p(d2 / nu))


def log_emission(y: np.ndarray, params: "HMMParams") -> np.ndarray:
    if params.nu is None:
        return gaussian_log_emission(y, params.mu, params.sigma)
    return student_t_log_emission(y, params.mu, params.sigma, params.nu)


# ----------------------------------------------------------------------------
# Transition matrices
# ----------------------------------------------------------------------------
def build_trans(beta: np.ndarray, Xlag: np.ndarray) -> np.ndarray:
    """P_t for every t from beta (K, K-1, d) and Xlag (T, d). Returns (T, K, K)."""
    K = beta.shape[0]
    T = Xlag.shape[0]
    logits = np.zeros((T, K, K))
    # logits[t, i, j] = beta[i, j] . x_{t-1} for j < K-1 ; last column is 0
    logits[:, :, : K - 1] = np.einsum("td,ijd->tij", Xlag, beta)
    logits -= logits.max(axis=2, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=2, keepdims=True)
    return P


def ergodic_distribution(P: np.ndarray) -> np.ndarray:
    """Stationary distribution pi with pi' P = pi', eq. (2.3) for general K."""
    K = P.shape[0]
    A = np.vstack([P.T - np.eye(K), np.ones((1, K))])
    b = np.zeros(K + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, 1e-12, None)
    return pi / pi.sum()


def beta_from_probs(P: np.ndarray) -> np.ndarray:
    """Inverse of build_trans for a constant matrix: beta (K, K-1, 1)."""
    K = P.shape[0]
    Pc = np.clip(P, 1e-10, 1.0)
    beta = np.log(Pc[:, : K - 1] / Pc[:, K - 1][:, None])
    return beta[:, :, None]


# ----------------------------------------------------------------------------
# Recursions (numba-compiled)
# ----------------------------------------------------------------------------
@njit(cache=True)
def hamilton_filter(log_eta, trans, pi):
    """Eqs. (3.1)-(3.5) with P_t (eq. 6.4). Returns xi_pred, xi_filt, loglik_t."""
    T, K = log_eta.shape
    xi_pred = np.empty((T, K))
    xi_filt = np.empty((T, K))
    ll = np.empty(T)
    prev = pi.copy()
    num = np.empty(K)
    for t in range(T):
        # prediction step
        if t == 0:
            for j in range(K):
                xi_pred[t, j] = pi[j]
        else:
            for j in range(K):
                acc = 0.0
                for i in range(K):
                    acc += trans[t, i, j] * prev[i]
                xi_pred[t, j] = acc
        # update step, scaled so exp() never under/overflows
        m = log_eta[t, 0]
        for j in range(1, K):
            if log_eta[t, j] > m:
                m = log_eta[t, j]
        s = 0.0
        for j in range(K):
            num[j] = xi_pred[t, j] * np.exp(log_eta[t, j] - m)
            s += num[j]
        if s < _TINY:
            s = _TINY
        ll[t] = np.log(s) + m
        for j in range(K):
            xi_filt[t, j] = num[j] / s
            prev[j] = xi_filt[t, j]
    return xi_pred, xi_filt, ll


@njit(cache=True)
def kim_smoother(xi_pred, xi_filt, trans):
    """Eq. (4.2) backward recursion and pairwise probabilities eq. (4.3)."""
    T, K = xi_filt.shape
    xi_s = np.empty((T, K))
    pair = np.zeros((T, K, K))
    for j in range(K):
        xi_s[T - 1, j] = xi_filt[T - 1, j]
    for t in range(T - 2, -1, -1):
        for i in range(K):
            acc = 0.0
            for j in range(K):
                denom = xi_pred[t + 1, j]
                if denom < _TINY:
                    denom = _TINY
                r = xi_s[t + 1, j] / denom
                acc += trans[t + 1, i, j] * r
                pair[t + 1, i, j] = xi_s[t + 1, j] * trans[t + 1, i, j] * xi_filt[t, i] / denom
            xi_s[t, i] = xi_filt[t, i] * acc
    return xi_s, pair


@njit(cache=True)
def viterbi(log_eta, trans, pi):
    """Eq. (5.6): most probable regime path in log space."""
    T, K = log_eta.shape
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=np.int64)
    for j in range(K):
        delta[0, j] = np.log(max(pi[j], _TINY)) + log_eta[0, j]
    for t in range(1, T):
        for j in range(K):
            best = -1e300
            arg = 0
            for i in range(K):
                v = delta[t - 1, i] + np.log(max(trans[t, i, j], _TINY))
                if v > best:
                    best = v
                    arg = i
            delta[t, j] = best + log_eta[t, j]
            psi[t, j] = arg
    path = np.empty(T, dtype=np.int64)
    best = -1e300
    for j in range(K):
        if delta[T - 1, j] > best:
            best = delta[T - 1, j]
            path[T - 1] = j
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


# ----------------------------------------------------------------------------
# Parameter container
# ----------------------------------------------------------------------------
@dataclass
class HMMParams:
    mu: np.ndarray          # (K,)
    sigma: np.ndarray       # (K,)  Gaussian sd, or Student-t scale when nu is set
    beta: np.ndarray        # (K, K-1, d)
    nu: Optional[np.ndarray] = None   # (K,) Student-t degrees of freedom; None = Gaussian emissions

    @property
    def K(self) -> int:
        return self.mu.shape[0]

    @property
    def d(self) -> int:
        return self.beta.shape[2]

    @property
    def n_params(self) -> int:
        return 2 * self.K + self.K * (self.K - 1) * self.d + (self.K if self.nu is not None else 0)

    @property
    def regime_variance(self) -> np.ndarray:
        """Var(y_t | S_t = j): sigma^2 for Gaussian, sigma^2 nu/(nu-2) for Student-t."""
        if self.nu is None:
            return self.sigma ** 2
        return self.sigma ** 2 * self.nu / (self.nu - 2.0)

    def copy(self) -> "HMMParams":
        return HMMParams(self.mu.copy(), self.sigma.copy(), self.beta.copy(), None if self.nu is None else self.nu.copy())

    def trans_at(self, x: np.ndarray) -> np.ndarray:
        """Transition matrix evaluated at one regressor vector x (d,)."""
        return build_trans(self.beta, np.asarray(x, float)[None, :])[0]

    def initial_distribution(self, Xlag: np.ndarray) -> np.ndarray:
        """Ergodic distribution of P evaluated at the mean regressor (Section 6.3)."""
        xbar = Xlag[1:].mean(axis=0) if Xlag.shape[0] > 1 else Xlag[0]
        return ergodic_distribution(self.trans_at(xbar))

    def relabel(self, order: np.ndarray) -> "HMMParams":
        """Permute regimes so that new regime k is old regime order[k] (Section 5.5)."""
        K = self.K
        full = np.concatenate([self.beta, np.zeros((K, 1, self.d))], axis=1)  # (K, K, d)
        full = full[order][:, order]                                          # permute rows and cols
        full = full - full[:, K - 1:K, :]                                     # re-normalise reference
        return HMMParams(self.mu[order].copy(), self.sigma[order].copy(), full[:, : K - 1, :].copy(),
                         None if self.nu is None else self.nu[order].copy())

    def sorted_by_sigma(self) -> "HMMParams":
        return self.relabel(np.argsort(np.sqrt(self.regime_variance)))

    # packing for direct numerical ML -------------------------------------
    def pack(self) -> np.ndarray:
        parts = [self.mu, np.log(self.sigma), self.beta.ravel()]
        if self.nu is not None:
            parts.append(np.log(self.nu - 2.0))       # nu = 2 + exp(.) keeps the variance finite
        return np.concatenate(parts)

    @staticmethod
    def unpack(vec: np.ndarray, K: int, d: int, student_t: bool = False) -> "HMMParams":
        mu = vec[:K]
        sigma = np.exp(vec[K: 2 * K])
        nb = K * (K - 1) * d
        beta = vec[2 * K: 2 * K + nb].reshape(K, K - 1, d)
        nu = 2.0 + np.exp(vec[2 * K + nb: 2 * K + nb + K]) if student_t else None
        return HMMParams(mu.copy(), sigma.copy(), beta.copy(), nu)

    def persistence_form(self):
        """For K=2: (beta_1, beta_2) of eq. (6.2), p_ii = Lambda(beta_i . x)."""
        assert self.K == 2
        return self.beta[0, 0].copy(), -self.beta[1, 0].copy()


# ----------------------------------------------------------------------------
# Likelihood evaluation
# ----------------------------------------------------------------------------
def run_filter(params: HMMParams, y: np.ndarray, Xlag: np.ndarray, pi: Optional[np.ndarray] = None):
    log_eta = log_emission(y, params)
    trans = build_trans(params.beta, Xlag)
    if pi is None:
        pi = params.initial_distribution(Xlag)
    xi_pred, xi_filt, ll = hamilton_filter(log_eta, trans, np.asarray(pi, float))
    return xi_pred, xi_filt, ll, trans, log_eta


def loglik(params: HMMParams, y: np.ndarray, Xlag: np.ndarray) -> float:
    return float(run_filter(params, y, Xlag)[2].sum())


def smooth(params: HMMParams, y: np.ndarray, Xlag: np.ndarray):
    xi_pred, xi_filt, ll, trans, log_eta = run_filter(params, y, Xlag)
    xi_s, pair = kim_smoother(xi_pred, xi_filt, trans)
    return xi_s, pair, xi_pred, xi_filt, ll


def decode(params: HMMParams, y: np.ndarray, Xlag: np.ndarray) -> np.ndarray:
    log_eta = log_emission(y, params)
    trans = build_trans(params.beta, Xlag)
    return viterbi(log_eta, trans, params.initial_distribution(Xlag))


# ----------------------------------------------------------------------------
# M-step for the transition block, eqs. (6.5)-(6.8)
# ----------------------------------------------------------------------------
def _q_row(b_i: np.ndarray, X: np.ndarray, W: np.ndarray):
    """Q_P for one origin row and its gradient/Hessian.

    b_i : (K-1, d) logits coefficients (reference = last regime)
    X   : (n, d) regressors x_{t-1}
    W   : (n, K) weights pair[t, i, :]
    """
    n, K = W.shape
    logits = np.zeros((n, K))
    logits[:, : K - 1] = X @ b_i.T
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    logp = logits - lse[:, None]
    q = float((W * logp).sum())
    p = np.exp(logp)
    w_tot = W.sum(axis=1)                                          # gamma_{t-1}(i)
    resid = W[:, : K - 1] - w_tot[:, None] * p[:, : K - 1]        # eq. (6.6) bracket
    grad = (resid[:, :, None] * X[:, None, :]).sum(axis=0)         # (K-1, d)
    # Hessian eq. (6.7): block (m,n) = -sum_t w_t p_m (1[m=n] - p_n) x x'
    pm = p[:, : K - 1]
    cov = -(w_tot[:, None, None] * (np.einsum("tm,mn->tmn", pm, np.eye(K - 1)) - np.einsum("tm,tn->tmn", pm, pm)))
    H = np.einsum("tmn,ta,tb->manb", cov, X, X).reshape((K - 1) * X.shape[1], (K - 1) * X.shape[1])
    return q, grad.ravel(), H


def mstep_transition(beta: np.ndarray, Xlag: np.ndarray, pair: np.ndarray, max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Maximise Q_P row by row. Closed form (5.2) when d = 1, Newton (6.6)-(6.8) otherwise."""
    K, _, d = beta.shape
    X = Xlag[1:]
    W_all = pair[1:]                                     # (T-1, K, K)
    new = beta.copy()
    if d == 1:
        P = W_all.sum(axis=0)
        P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-300, None)
        return beta_from_probs(P)
    for i in range(K):
        W = W_all[:, i, :]
        b = new[i].reshape(-1).copy()
        q, g, H = _q_row(b.reshape(K - 1, d), X, W)
        for _ in range(max_iter):
            Hr = H - 1e-8 * np.eye(H.shape[0])
            step = -np.linalg.solve(Hr, g)
            # backtracking line search: Q is concave so a full step nearly always works
            alpha = 1.0
            while alpha > 1e-6:
                b_try = b + alpha * step
                q_try, g_try, H_try = _q_row(b_try.reshape(K - 1, d), X, W)
                if q_try >= q - 1e-12:
                    break
                alpha *= 0.5
            b, q, g, H = b_try, q_try, g_try, H_try
            if np.abs(g).max() < tol:
                break
        new[i] = b.reshape(K - 1, d)
    return new


# ----------------------------------------------------------------------------
# EM / GEM (Sections 5.3 and 6.4-6.5)
# ----------------------------------------------------------------------------
@dataclass
class FitResult:
    params: HMMParams
    loglik: float
    n_iter: int
    converged: bool
    history: list = field(default_factory=list)
    T: int = 0

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.params.n_params

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.params.n_params * np.log(self.T)


def init_params(y: np.ndarray, K: int, d: int, rng: Optional[np.random.Generator] = None, jitter: float = 0.0,
                student_t: bool = False, nu0: float = 5.0) -> HMMParams:
    """Section 8: initialise regimes by rolling-volatility quantiles, p_ii = 0.95, slopes 0."""
    y = np.asarray(y, float)
    win = min(10, y.shape[0])
    rv = np.sqrt(np.convolve(y ** 2, np.ones(win) / win, mode="same"))
    qs = np.quantile(rv, np.linspace(0, 1, K + 1))
    mu = np.empty(K)
    sigma = np.empty(K)
    for k in range(K):
        mask = (rv >= qs[k]) & (rv <= qs[k + 1])
        if mask.sum() < 5:
            mask = np.ones_like(mask)
        mu[k] = y[mask].mean()
        sigma[k] = max(y[mask].std(), 1e-4)
    if rng is not None and jitter > 0:
        mu = mu + rng.normal(0, jitter * y.std(), K)
        sigma = sigma * np.exp(rng.normal(0, jitter, K))
    P = np.full((K, K), 0.05 / max(K - 1, 1))
    np.fill_diagonal(P, 0.95)
    beta = np.zeros((K, K - 1, d))
    beta[:, :, 0] = beta_from_probs(P)[:, :, 0]
    nu = np.full(K, nu0) if student_t else None
    if student_t:  # sigma is a scale: shrink the sd initialisation so the implied variance matches
        sigma = sigma * np.sqrt((nu0 - 2.0) / nu0)
    return HMMParams(mu, sigma, beta, nu).sorted_by_sigma()


def _mstep_student_t(y: np.ndarray, w: np.ndarray, params: HMMParams, n_inner: int = 3, var_floor: float = 1e-8):
    """ECM update of (mu_j, sigma_j, nu_j) for weighted Student-t regimes.

    Uses the scale-mixture representation: given nu, the weights u_t = (nu+1)/(nu + d_t^2) turn the
    location/scale update into weighted least squares; nu is then updated by a 1-D bounded search on
    the weighted t log-likelihood. Each pass increases Q, so the outer algorithm is a generalised EM."""
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln
    mu, sigma, nu = params.mu.copy(), params.sigma.copy(), params.nu.copy()
    K = mu.shape[0]
    for j in range(K):
        wj = w[:, j]
        if wj.sum() < 1e-8:
            continue
        for _ in range(n_inner):
            d2 = ((y - mu[j]) / sigma[j]) ** 2
            u = (nu[j] + 1.0) / (nu[j] + d2)
            mu[j] = (wj * u * y).sum() / (wj * u).sum()
            sigma[j] = np.sqrt(max((wj * u * (y - mu[j]) ** 2).sum() / wj.sum(), var_floor))
            d2 = ((y - mu[j]) / sigma[j]) ** 2

            def negq(log_nu_m2, d2=d2, wj=wj, sig=sigma[j]):
                v = 2.0 + np.exp(log_nu_m2)
                ll = gammaln((v + 1) / 2) - gammaln(v / 2) - 0.5 * np.log(v * np.pi) - np.log(sig) - 0.5 * (v + 1) * np.log1p(d2 / v)
                return -(wj * ll).sum()
            res = minimize_scalar(negq, bounds=(np.log(0.05), np.log(200.0)), method="bounded")
            nu[j] = 2.0 + np.exp(res.x)
    return mu, sigma, nu


def fit_em(y: np.ndarray, Xlag: np.ndarray, K: int, init: Optional[HMMParams] = None,
           max_iter: int = 500, tol: float = 1e-6, var_floor: float = 1e-8,
           rng: Optional[np.random.Generator] = None, jitter: float = 0.0, student_t: bool = False) -> FitResult:
    """Baum-Welch for the constant HMM (d=1) and generalised EM for the TVTP model (d>1).
    With student_t=True the emissions are Student-t and the emission M-step is the ECM update above."""
    y = np.asarray(y, float)
    T = y.shape[0]
    d = Xlag.shape[1]
    if y.ndim != 1 or Xlag.ndim != 2 or Xlag.shape[0] != T:
        raise ValueError("y must be one-dimensional and Xlag must have one row per observation")
    if T < 3:
        raise ValueError("fit_em needs at least three observations")
    if K < 1 or K > T:
        raise ValueError("K must be between one and the number of observations")
    if K == 1 and d > 1:
        raise ValueError("K=1 has no transition parameters, so covariates cannot enter; pass Xlag = ones((T, 1))")
    if not np.isfinite(y).all() or not np.isfinite(Xlag).all():
        raise ValueError("y and Xlag must contain only finite values")
    params = init.copy() if init is not None else init_params(y, K, d, rng, jitter, student_t=student_t)
    if params.d != d:
        raise ValueError("init has wrong covariate dimension")
    prev_ll = -np.inf
    history = []
    converged = False
    for it in range(1, max_iter + 1):
        xi_s, pair, xi_pred, xi_filt, ll = smooth(params, y, Xlag)
        cur_ll = float(ll.sum())
        history.append(cur_ll)
        if cur_ll - prev_ll < tol and it > 1:
            converged = True
            break
        prev_ll = cur_ll
        # M-step, emissions eq. (5.3) (Gaussian) or the ECM update (Student-t)
        w = xi_s
        if params.nu is None:
            wsum = np.clip(w.sum(axis=0), 1e-12, None)
            mu = (w * y[:, None]).sum(axis=0) / wsum
            sig2 = (w * (y[:, None] - mu[None, :]) ** 2).sum(axis=0) / wsum
            sigma = np.sqrt(np.maximum(sig2, var_floor))
            nu = None
        else:
            mu, sigma, nu = _mstep_student_t(y, w, params, var_floor=var_floor)
        # M-step, transitions eqs. (5.2) / (6.5)-(6.8)
        beta = mstep_transition(params.beta, Xlag, pair)
        params = HMMParams(mu, sigma, beta, nu)
    params = params.sorted_by_sigma()
    final_ll = loglik(params, y, Xlag)
    return FitResult(params, final_ll, len(history), converged, history, T)


def fit_em_restarts(y, Xlag, K, n_restarts=10, seed=0, **kw) -> FitResult:
    rng = np.random.default_rng(seed)
    best = fit_em(y, Xlag, K, **kw)
    for _ in range(n_restarts - 1):
        try:
            r = fit_em(y, Xlag, K, rng=rng, jitter=0.5, **kw)
        except (np.linalg.LinAlgError, FloatingPointError):
            continue
        if np.isfinite(r.loglik) and r.loglik > best.loglik:
            best = r
    return best


# ----------------------------------------------------------------------------
# Direct numerical ML and standard errors (Sections 5.4, 6.5 step 4)
# ----------------------------------------------------------------------------
def fit_direct(y: np.ndarray, Xlag: np.ndarray, init: HMMParams, compute_se: bool = False, **opt_kw):
    from scipy.optimize import minimize
    K, d = init.K, init.d
    st = init.nu is not None
    y = np.asarray(y, float)

    def nll(v):
        p = HMMParams.unpack(v, K, d, st)
        val = -loglik(p, y, Xlag)
        return val if np.isfinite(val) else 1e300

    res = minimize(nll, init.pack(), method="L-BFGS-B", options={"maxiter": 2000, **opt_kw})
    params = HMMParams.unpack(res.x, K, d, st).sorted_by_sigma()
    out = FitResult(params, -float(res.fun), int(res.nit), bool(res.success), [], y.shape[0])
    if compute_se:
        H = numerical_hessian(nll, params.pack())
        try:
            cov = np.linalg.inv(H)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            se = np.full(H.shape[0], np.nan)
        # same layout as pack(); the sigma slot holds the SE of log(sigma), so SE(sigma) = se.sigma * sigma;
        # the nu slot (if any) holds the SE of log(nu - 2)
        nb = K * (K - 1) * d
        out.se = HMMParams(se[:K].copy(), se[K:2 * K].copy(), se[2 * K:2 * K + nb].reshape(K, K - 1, d).copy(),
                           se[2 * K + nb:2 * K + nb + K].copy() if st else None)
        out.cov = cov
    return out


def numerical_hessian(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian, eq. (5.5)."""
    n = x.shape[0]
    H = np.empty((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / eps ** 2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps ** 2)
    return H


# ----------------------------------------------------------------------------
# Forecast outputs, Section 7
# ----------------------------------------------------------------------------
def moment_forecasts(params: HMMParams, xi_pred: np.ndarray):
    """m_{t|t-1} and h_{t|t-1} from eq. (7.2) for every t (regime variance sigma^2 nu/(nu-2) for Student-t)."""
    m = xi_pred @ params.mu
    h = xi_pred @ (params.regime_variance + params.mu ** 2) - m ** 2
    return m, h


def lr_test(ll_restricted: float, ll_full: float, df: int):
    from scipy.stats import chi2
    stat = 2.0 * (ll_full - ll_restricted)
    return stat, float(chi2.sf(max(stat, 0.0), df))
