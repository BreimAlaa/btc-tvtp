# Methodology: Hamilton Filter, HMM Estimation, and the TVTP Extension

Working derivation for Section 3 of the FDP, in one consistent notation. Everything is derived from three stated assumptions so the paper can present the TVTP model as a one-line modification of the baseline rather than a separate model. Section 9 maps which equations belong in the 15-page body, which in an appendix, and which live only in the repository.

---

## 1. Notation and conventions

**Data.** Trading days $t = 1, \dots, T$. Close price $C_t$. Daily log return

$$
y_t = \ln C_t - \ln C_{t-1}. \tag{1.1}
$$

Covariate $z_t$ (scalar; one of realised volatility, exchange net-flow, funding rate), observed by the close of day $t$. Covariate regressor vector $x_t = (1, z_t)'$, dimension $d = 2$ (extends to $d > 2$ for two covariates).

**Information set.** $\mathcal{F}_t = \{y_1, \dots, y_t, z_0, \dots, z_t\}$: everything observable at the close of day $t$.

**Latent regime.** $S_t \in \{1, \dots, K\}$. Regime 1 is the low-volatility regime, ordered by $\sigma_1 < \sigma_2 < \cdots < \sigma_K$ (this ordering is the identification constraint, Section 5.5).

**Regime probability vectors.** $\boldsymbol{\xi}_{t|s}$ is the $K \times 1$ vector with $j$-th entry

$$
\xi_{t|s}(j) = \Pr(S_t = j \mid \mathcal{F}_s).
$$

$s = t-1$: predicted; $s = t$: filtered; $s = T$: smoothed.

**Emissions.** $f_j(y_t) = f(y_t \mid S_t = j; \theta)$. Baseline emission is Gaussian,

$$
f_j(y) = \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp\!\left\{-\frac{(y - \mu_j)^2}{2\sigma_j^2}\right\}, \tag{1.2}
$$

collected in the $K \times 1$ vector $\boldsymbol{\eta}_t = (f_1(y_t), \dots, f_K(y_t))'$.

**Transition matrix.** $\mathbf{P} = [p_{ij}]$, $K \times K$, rows index the origin regime, columns the destination:

$$
p_{ij} = \Pr(S_t = j \mid S_{t-1} = i), \qquad \sum_{j=1}^K p_{ij} = 1. \tag{1.3}
$$

In the TVTP model this becomes $\mathbf{P}_t = [p_{ij,t}]$, the matrix governing the move from $S_{t-1}$ to $S_t$.

**Operators.** $\odot$ elementwise product, $\oslash$ elementwise division, $\mathbf{1}$ the $K \times 1$ vector of ones, $\Lambda(u) = (1 + e^{-u})^{-1}$ the logistic function.

**Parameters.** $\theta$ collects all free parameters; the exact contents are listed per model in Sections 2 and 6.

---

## 2. Baseline model: constant-transition Gaussian HMM

The model is fully specified by three assumptions.

**(A1) First-order Markov regimes.**
$$
\Pr(S_t = j \mid S_{t-1} = i, S_{t-2}, \dots, \mathcal{F}_{t-1}) = p_{ij}. \tag{2.1}
$$

**(A2) Conditional independence of emissions.**
$$
f(y_t \mid S_t = j, S_{1:t-1}, \mathcal{F}_{t-1}) = f_j(y_t). \tag{2.2}
$$

**(A3) Initial distribution.** $\Pr(S_1 = j) = \pi_j$. For a stationary ergodic chain this is the unique $\boldsymbol{\pi}$ solving $\boldsymbol{\pi}' \mathbf{P} = \boldsymbol{\pi}'$, $\boldsymbol{\pi}'\mathbf{1} = 1$. For $K = 2$:

$$
\pi_1 = \frac{1 - p_{22}}{2 - p_{11} - p_{22}}, \qquad \pi_2 = \frac{1 - p_{11}}{2 - p_{11} - p_{22}}. \tag{2.3}
$$

Under (A1)–(A3) the complete-data likelihood factorizes as

$$
f(y_{1:T}, S_{1:T}; \theta) = \pi_{S_1} f_{S_1}(y_1) \prod_{t=2}^{T} p_{S_{t-1} S_t} f_{S_t}(y_t). \tag{2.4}
$$

**Parameter vector.** $\theta = (\mu_1, \dots, \mu_K, \sigma_1^2, \dots, \sigma_K^2, \{p_{ij}\}_{j \neq i})$, giving $2K + K(K-1)$ free parameters ($\boldsymbol{\pi}$ tied to $\mathbf{P}$ via (2.3)). For $K = 2$: six parameters.

**Why this reproduces the stylized facts (Cont 2001).** The unconditional distribution of $y_t$ is a mixture of Gaussians with regime-specific variances, so it has excess kurtosis even with Gaussian components; persistence in $S_t$ ($p_{ii}$ close to 1) produces volatility clustering. Expected duration of regime $i$ is

$$
\mathbb{E}[D_i] = \frac{1}{1 - p_{ii}}. \tag{2.5}
$$

---

## 3. The Hamilton filter

Goal: compute $\boldsymbol{\xi}_{t|t}$ and the likelihood recursively, using only $\mathcal{F}_t$ at each step. Two steps per period.

### 3.1 Prediction step

By the law of total probability and (A1),

$$
\xi_{t|t-1}(j) = \sum_{i=1}^K \Pr(S_t = j \mid S_{t-1} = i, \mathcal{F}_{t-1}) \Pr(S_{t-1} = i \mid \mathcal{F}_{t-1}) = \sum_{i=1}^K p_{ij}\, \xi_{t-1|t-1}(i),
$$

that is,

$$
\boldsymbol{\xi}_{t|t-1} = \mathbf{P}' \boldsymbol{\xi}_{t-1|t-1}. \tag{3.1}
$$

### 3.2 Update step

Joint density of the new observation and the regime, given the past, using (A2):

$$
f(y_t, S_t = j \mid \mathcal{F}_{t-1}) = f_j(y_t)\, \xi_{t|t-1}(j). \tag{3.2}
$$

Marginalising over $j$ gives the one-step-ahead predictive density of the return, which is the period-$t$ likelihood contribution:

$$
\ell_t \equiv f(y_t \mid \mathcal{F}_{t-1}) = \sum_{j=1}^K f_j(y_t)\, \xi_{t|t-1}(j) = \mathbf{1}'(\boldsymbol{\xi}_{t|t-1} \odot \boldsymbol{\eta}_t). \tag{3.3}
$$

Bayes' rule then gives the filtered probability:

$$
\xi_{t|t}(j) = \Pr(S_t = j \mid y_t, \mathcal{F}_{t-1}) = \frac{f_j(y_t)\, \xi_{t|t-1}(j)}{\ell_t}, \qquad
\boldsymbol{\xi}_{t|t} = \frac{\boldsymbol{\xi}_{t|t-1} \odot \boldsymbol{\eta}_t}{\mathbf{1}'(\boldsymbol{\xi}_{t|t-1} \odot \boldsymbol{\eta}_t)}. \tag{3.4}
$$

### 3.3 Initialisation and recursion

Set $\boldsymbol{\xi}_{1|0} = \boldsymbol{\pi}$ from (2.3), then iterate (3.1) and (3.4) for $t = 1, \dots, T$. Each step costs $O(K^2)$; the full pass is $O(TK^2)$.

### 3.4 Log-likelihood

By the prediction-error decomposition,

$$
\mathcal{L}(\theta) = \ln f(y_{1:T}; \theta) = \sum_{t=1}^T \ln \ell_t. \tag{3.5}
$$

This is the objective for direct maximum likelihood (Section 5.4) and the quantity compared across models by AIC, BIC and the likelihood-ratio test (Section 6.6).

### 3.5 Relation to the forward algorithm (Rabiner 1989)

Rabiner's forward variable is the unnormalized joint $\alpha_t(j) = f(y_{1:t}, S_t = j)$, with recursion $\alpha_t(j) = f_j(y_t) \sum_i \alpha_{t-1}(i) p_{ij}$. The Hamilton filter is exactly the scaled forward algorithm:

$$
\xi_{t|t}(j) = \frac{\alpha_t(j)}{\sum_k \alpha_t(k)}, \qquad \ell_t = \frac{\sum_k \alpha_t(k)}{\sum_k \alpha_{t-1}(k)}. \tag{3.6}
$$

The normalisation in (3.4) is Rabiner's scaling constant $c_t = 1/\ell_t$, and it is what prevents underflow over long samples. This is the bridge between the econometric (Hamilton) and signal-processing (Rabiner) presentations; the paper only needs the one sentence.

---

## 4. Smoothing (Kim 1994)

Smoothed probabilities $\boldsymbol{\xi}_{t|T}$ use the whole sample. They are needed inside the EM algorithm and for in-sample regime dating (for example, locating the FTX collapse), but never for trading decisions.

### 4.1 Derivation

Decompose over $S_{t+1}$:

$$
\Pr(S_t = i \mid \mathcal{F}_T) = \sum_{j=1}^K \Pr(S_{t+1} = j \mid \mathcal{F}_T)\, \Pr(S_t = i \mid S_{t+1} = j, \mathcal{F}_T).
$$

Under (A1)–(A2), once $S_{t+1}$ is known, the future observations $y_{t+1:T}$ carry no further information about $S_t$, so

$$
\Pr(S_t = i \mid S_{t+1} = j, \mathcal{F}_T) = \Pr(S_t = i \mid S_{t+1} = j, \mathcal{F}_t) = \frac{p_{ij}\, \xi_{t|t}(i)}{\xi_{t+1|t}(j)}, \tag{4.1}
$$

where the last equality is Bayes' rule with (3.1) in the denominator. Combining,

$$
\xi_{t|T}(i) = \xi_{t|t}(i) \sum_{j=1}^K p_{ij}\, \frac{\xi_{t+1|T}(j)}{\xi_{t+1|t}(j)}, \qquad
\boldsymbol{\xi}_{t|T} = \boldsymbol{\xi}_{t|t} \odot \left\{ \mathbf{P} \left( \boldsymbol{\xi}_{t+1|T} \oslash \boldsymbol{\xi}_{t+1|t} \right) \right\}. \tag{4.2}
$$

Run backward from $\boldsymbol{\xi}_{T|T}$ (the last filtered vector) to $t = 1$, reusing the stored $\boldsymbol{\xi}_{t|t}$ and $\boldsymbol{\xi}_{t+1|t}$ from the forward pass.

### 4.2 Smoothed transition probabilities

The EM algorithm also needs the pairwise smoothed probabilities. From (4.1),

$$
\xi_t(i, j) \equiv \Pr(S_{t-1} = i, S_t = j \mid \mathcal{F}_T) = \xi_{t|T}(j)\, \frac{p_{ij}\, \xi_{t-1|t-1}(i)}{\xi_{t|t-1}(j)}, \qquad t = 2, \dots, T. \tag{4.3}
$$

Consistency checks: $\sum_i \xi_t(i,j) = \xi_{t|T}(j)$ and $\sum_j \xi_t(i,j) = \xi_{t-1|T}(i)$. Both are useful unit tests in the repo.

Equation (4.3) is the same quantity Rabiner writes as $\alpha_{t-1}(i) p_{ij} f_j(y_t) \beta_t(j) / f(y_{1:T})$; the form above avoids storing a separate backward variable.

---

## 5. Estimation of the baseline: EM (Baum-Welch) and Viterbi

### 5.1 EM objective

Write $\gamma_t(j) = \xi_{t|T}(j)$ for the smoothed marginal. Taking the conditional expectation of $\ln$(2.4) given $\mathcal{F}_T$ under current parameters $\theta^{(m)}$:

$$
Q(\theta \mid \theta^{(m)}) = \sum_{j} \gamma_1(j) \ln \pi_j
\;+\; \sum_{t=2}^T \sum_{i,j} \xi_t(i,j) \ln p_{ij}
\;+\; \sum_{t=1}^T \sum_{j} \gamma_t(j) \ln f_j(y_t). \tag{5.1}
$$

The three terms involve disjoint parameter blocks, so the M-step maximises each separately.

### 5.2 E-step

Run the filter (3.1)–(3.4), the smoother (4.2), and compute (4.3), all at $\theta^{(m)}$.

### 5.3 M-step (closed form for Gaussian emissions)

Transition probabilities (maximise the second term subject to rows summing to one):

$$
\hat{p}_{ij} = \frac{\sum_{t=2}^T \xi_t(i,j)}{\sum_{t=2}^T \gamma_{t-1}(i)}. \tag{5.2}
$$

Emission parameters (weighted MLE of a Gaussian):

$$
\hat{\mu}_j = \frac{\sum_{t=1}^T \gamma_t(j)\, y_t}{\sum_{t=1}^T \gamma_t(j)}, \qquad
\hat{\sigma}_j^2 = \frac{\sum_{t=1}^T \gamma_t(j)\, (y_t - \hat{\mu}_j)^2}{\sum_{t=1}^T \gamma_t(j)}. \tag{5.3}
$$

Initial distribution: if treated as free, $\hat{\pi}_j = \gamma_1(j)$. With $T \approx 2000$ daily observations the choice is immaterial; the cleaner option is to tie $\boldsymbol{\pi}$ to the ergodic distribution (2.3) of the current $\hat{\mathbf{P}}$ and drop the first term of (5.1). Strictly, EM is then monotone in $\mathcal{L}$ conditional on $\boldsymbol{\pi}$; updating $\boldsymbol{\pi}$ with $\hat{\mathbf{P}}$ can move $\mathcal{L}$ by the first-period term only, which is $O(10^{-6})$ in practice (the unit test allows for it).

Iterate E and M steps until $\mathcal{L}(\theta^{(m+1)}) - \mathcal{L}(\theta^{(m)}) < 10^{-6}$. EM is monotone in $\mathcal{L}$ (Dempster, Laird & Rubin 1977) but converges to a local optimum; use several random restarts (five in the empirical pipeline, more if the likelihood surface looks multimodal) and keep the best likelihood.

### 5.4 Direct numerical ML (equivalent alternative)

Maximise (3.5) directly with a quasi-Newton optimiser (L-BFGS-B) on an unconstrained reparameterisation:

$$
\sigma_j = \exp(\tilde{\sigma}_j), \qquad p_{ij} = \frac{\exp(\tilde{p}_{ij})}{\sum_l \exp(\tilde{p}_{il})} \;\; (\tilde{p}_{iK} \equiv 0). \tag{5.4}
$$

EM and direct ML converge to the same MLE. The recommended pipeline is EM to convergence (robust from poor starting values), then one L-BFGS polish on (3.5) to obtain the numerical Hessian for standard errors:

$$
\widehat{\mathrm{Var}}(\hat{\theta}) = \left[ -\nabla^2_\theta \mathcal{L}(\hat{\theta}) \right]^{-1}. \tag{5.5}
$$

### 5.5 Identification and label switching

The likelihood is invariant to permuting regime labels. After each estimation (and in every walk-forward window) relabel so that $\hat{\sigma}_1 < \hat{\sigma}_2 < \cdots < \hat{\sigma}_K$. This makes "regime 1 = calm" a stable meaning across windows, which the trading rule and the covariate-sign hypotheses (Section 6.2) rely on.

### 5.6 Choosing $K$

Testing $K$ versus $K+1$ regimes by likelihood ratio is nonstandard: the transition parameters of the extra regime are unidentified under the null, so the LR statistic is not $\chi^2$ (Hansen 1992). Select $K$ on the baseline by BIC, $\text{BIC} = -2\mathcal{L} + k \ln T$, comparing $K = 2$ and $K = 3$, then hold $K$ fixed for the TVTP model so the comparison isolates the effect of time variation alone. On the BTC sample BIC favours $K = 3$ for the constant HMM; the paper keeps $K = 2$ as the main specification (interpretable calm/turbulent split, four fewer transition parameters) and reports the $K = 3$ TVTP fit as a robustness check.

### 5.7 Viterbi decoding

Most probable regime path $\hat{S}_{1:T} = \arg\max_{S_{1:T}} f(S_{1:T} \mid y_{1:T})$, in log space:

$$
\delta_1(j) = \ln \pi_j + \ln f_j(y_1), \qquad
\delta_t(j) = \max_{i} \left[ \delta_{t-1}(i) + \ln p_{ij} \right] + \ln f_j(y_t), \qquad
\psi_t(j) = \arg\max_i \left[ \delta_{t-1}(i) + \ln p_{ij} \right]. \tag{5.6}
$$

Backtrack: $\hat{S}_T = \arg\max_j \delta_T(j)$, $\hat{S}_t = \psi_{t+1}(\hat{S}_{t+1})$. Like smoothing, this is a full-sample tool used for descriptive regime dating only.

---

## 6. The TVTP extension

### 6.1 Modified assumption

Replace (A1) by

**(A1′) Covariate-driven Markov transitions.**
$$
\Pr(S_t = j \mid S_{t-1} = i, S_{t-2}, \dots, \mathcal{F}_{t-1}) = p_{ij}(z_{t-1}; \boldsymbol{\beta}) \equiv p_{ij,t}, \tag{6.1}
$$

together with

**(A4) Predetermined covariate.** $z_{t-1} \in \mathcal{F}_{t-1}$.

(A2) and (A3) are unchanged. The transition from $t-1$ to $t$ is driven by the covariate observed at the close of $t-1$. This lag is what makes the model usable for forecasting and trading without look-ahead: at the close of day $t$ the matrix $\mathbf{P}_{t+1} = \mathbf{P}(z_t)$ governing tomorrow's transition is already known.

Two cases for the covariate:

- **Internal covariate** (realised volatility, a function of $y_{1:t-1}$). Then (6.1) is simply a richer specification of the return process: the regime sequence is no longer Markov on its own, but $(S_t, \mathcal{F}_t)$ is, and (2.4) with $p_{ij} \to p_{ij,t}$ is a fully specified joint density for $y_{1:T}$.
- **External covariate** (funding rate, exchange net-flow). The likelihood is conditional on the covariate path, which is treated as exogenous: $z_t$ is assumed not to be driven by the contemporaneous regime $S_t$. This is the setting of Filardo (1994) and Diebold, Lee & Weinbach (1994); endogenous-switching relaxations exist (Kim, Piger & Startz 2008) but are out of scope.

### 6.2 Parameterisation of $\mathbf{P}_t$

**Two regimes (primary specification).** Each row has one free probability, modelled as a logistic function of $x_{t-1} = (1, z_{t-1})'$:

$$
p_{11,t} = \Lambda(\boldsymbol{\beta}_1' x_{t-1}) = \frac{1}{1 + \exp\{-(\beta_{1,0} + \beta_{1,1} z_{t-1})\}}, \qquad p_{12,t} = 1 - p_{11,t},
$$
$$
p_{22,t} = \Lambda(\boldsymbol{\beta}_2' x_{t-1}) = \frac{1}{1 + \exp\{-(\beta_{2,0} + \beta_{2,1} z_{t-1})\}}, \qquad p_{21,t} = 1 - p_{22,t}. \tag{6.2}
$$

$\beta_{i,0}$ is the log-odds of staying in regime $i$ when $z = 0$ (after standardising $z$, this is the persistence at the covariate's training-sample mean); $\beta_{i,1}$ is the marginal effect of the covariate on the log-odds of persistence. Filardo (1994) used the probit link $\Phi(\cdot)$; the logistic gives the same qualitative behaviour with a cleaner score (Section 6.5) and is the standard choice in the recent literature.

Testable sign hypotheses for $z$ = realised volatility, with regime 1 calm: $\beta_{1,1} < 0$ (rising realised volatility makes the calm regime less persistent) and $\beta_{2,1} > 0$ (rising realised volatility makes the turbulent regime more persistent). Analogous predictions can be written for funding-rate extremes and exchange inflows.

**General $K$.** Multinomial logit per row with the last column as reference category:

$$
p_{ij,t} = \frac{\exp(\boldsymbol{\beta}_{ij}' x_{t-1})}{\sum_{l=1}^K \exp(\boldsymbol{\beta}_{il}' x_{t-1})}, \qquad \boldsymbol{\beta}_{iK} \equiv \mathbf{0}. \tag{6.3}
$$

For $K = 2$ this reduces to (6.2) with $\boldsymbol{\beta}_1 = \boldsymbol{\beta}_{11}$ and $\boldsymbol{\beta}_2 = -\boldsymbol{\beta}_{21}$.

**Parameter vector.** $\theta = (\mu_1, \dots, \mu_K, \sigma_1^2, \dots, \sigma_K^2, \boldsymbol{\beta})$ with $\boldsymbol{\beta}$ of dimension $K(K-1)d$. For $K = 2$, $d = 2$: eight parameters versus six in the baseline.

**Time-varying expected duration.** From (2.5), $\mathbb{E}[D_i \mid z] = 1/(1 - p_{ii}(z))$; plotting this against $z$ is the most direct way to show what the covariate does.

### 6.3 Modified Hamilton filter

Repeating the prediction-step derivation with (A1′) and (A4):

$$
\xi_{t|t-1}(j) = \sum_{i} \Pr(S_t = j \mid S_{t-1} = i, \mathcal{F}_{t-1})\, \xi_{t-1|t-1}(i) = \sum_i p_{ij}(z_{t-1})\, \xi_{t-1|t-1}(i),
$$

that is,

$$
\boxed{\;\boldsymbol{\xi}_{t|t-1} = \mathbf{P}_t' \boldsymbol{\xi}_{t-1|t-1}, \qquad \mathbf{P}_t = \mathbf{P}(z_{t-1}; \boldsymbol{\beta}).\;} \tag{6.4}
$$

The update step (3.4), likelihood contribution (3.3), and log-likelihood (3.5) are unchanged. The smoother (4.2), pairwise probabilities (4.3), and Viterbi (5.6) carry over with $p_{ij} \to p_{ij,t}$ ($\mathbf{P} \to \mathbf{P}_{t+1}$ in (4.2), since it maps $S_t$ to $S_{t+1}$). The conditional-independence argument behind (4.1) still holds: given $S_{t+1}$ and $\mathcal{F}_t$, the future $y_{t+1:T}$ depends on $S_t$ through nothing, because $z_{t+1}, z_{t+2}, \dots$ are functions of observables and $y_{t+1}$ depends only on $S_{t+1}$.

**Initialisation.** There is no stationary distribution when $\mathbf{P}_t$ varies. Set $\boldsymbol{\xi}_{1|0}$ to the ergodic distribution (2.3) of $\mathbf{P}(\bar{z})$ evaluated at the training-sample mean covariate. The influence of the initial condition decays geometrically and is negligible after a few dozen observations.

The entire extension, from the filter's point of view, is the replacement $\mathbf{P}' \to \mathbf{P}_t'$ in one line. This is the sentence to put in the paper.

### 6.4 Estimation: what changes in EM

The E-step is unchanged (filter, smoother, (4.3) with time-varying probabilities). In the M-step, (5.3) is unchanged. The transition block of (5.1) becomes

$$
Q_{\mathbf{P}}(\boldsymbol{\beta}) = \sum_{t=2}^T \sum_{i,j} \xi_t(i,j)\, \ln p_{ij}(z_{t-1}; \boldsymbol{\beta}), \tag{6.5}
$$

which no longer has a closed-form maximiser. It separates by row $i$, and each row is a weighted multinomial logistic regression: observations $x_{t-1}$, "response" weights $\xi_t(i,j)$, total weight $\gamma_{t-1}(i) = \sum_j \xi_t(i,j)$.

### 6.5 Score and Hessian for the transition parameters

For the general form (6.3), using $\partial \ln p_{ij,t} / \partial \boldsymbol{\beta}_{im} = x_{t-1}(\mathbb{1}[j = m] - p_{im,t})$:

$$
\frac{\partial Q_{\mathbf{P}}}{\partial \boldsymbol{\beta}_{im}} = \sum_{t=2}^T x_{t-1} \left[ \xi_t(i,m) - \gamma_{t-1}(i)\, p_{im,t} \right], \tag{6.6}
$$

$$
\frac{\partial^2 Q_{\mathbf{P}}}{\partial \boldsymbol{\beta}_{im} \partial \boldsymbol{\beta}_{in}'} = -\sum_{t=2}^T \gamma_{t-1}(i)\, p_{im,t} \left( \mathbb{1}[m = n] - p_{in,t} \right) x_{t-1} x_{t-1}'. \tag{6.7}
$$

The Hessian is negative semidefinite (it is minus a weighted sum of multinomial covariance matrices), so $Q_{\mathbf{P}}$ is concave in each $\boldsymbol{\beta}_i$ and Newton-Raphson converges from any start.

For the two-regime form (6.2), row $i$ has the single vector $\boldsymbol{\beta}_i$ and

$$
\frac{\partial Q_{\mathbf{P}}}{\partial \boldsymbol{\beta}_i} = \sum_{t=2}^T x_{t-1} \left[ \xi_t(i,i) - \gamma_{t-1}(i)\, \Lambda(\boldsymbol{\beta}_i' x_{t-1}) \right], \qquad
\frac{\partial^2 Q_{\mathbf{P}}}{\partial \boldsymbol{\beta}_i \partial \boldsymbol{\beta}_i'} = -\sum_{t=2}^T \gamma_{t-1}(i)\, \Lambda_t (1 - \Lambda_t)\, x_{t-1} x_{t-1}', \tag{6.8}
$$

with $\Lambda_t = \Lambda(\boldsymbol{\beta}_i' x_{t-1})$. Setting the score to zero and reading (6.6) at a constant covariate recovers (5.2), which confirms the nesting.

**Recommended estimation pipeline.**

1. Estimate the baseline by EM (Section 5.3), five restarts as in the empirical pipeline (more if the surface looks multimodal), relabel by $\sigma$.
2. Warm-start the TVTP model at the baseline solution: $\hat{\mu}, \hat{\sigma}$ as estimated, $\beta_{i,0} = \operatorname{logit}(\hat{p}_{ii})$, $\beta_{i,1} = 0$. At this point the TVTP likelihood equals the baseline likelihood exactly, which is a useful correctness check.
3. Run generalized EM: E-step as before; M-step (5.3) for emissions and Newton iterations on (6.5) using (6.6)–(6.8) for $\boldsymbol{\beta}$ (a handful of iterations suffice; the problem is two-dimensional per row).
4. Polish with L-BFGS on (3.5) (now a function of $\boldsymbol{\beta}$ through (6.4)); obtain the numerical Hessian for standard errors of $\hat{\boldsymbol{\beta}}$.

Steps 3 and 4 give the same MLE; either alone is acceptable. Reporting $\hat{\beta}_{i,1}$ with its standard error gives a $t$-test on each covariate slope, complementing the joint LR test below.

### 6.6 Nesting and the likelihood-ratio test

Setting $\beta_{i,1} = 0$ for all $i$ in (6.2) or the slope blocks to zero in (6.3) recovers the constant-transition HMM exactly. The null hypothesis $H_0 : \beta_{i,1} = 0 \;\forall i$ is a regular hypothesis provided the data contain $K$ distinct regimes, so that the constant HMM is itself identified: all parameters are then identified under $H_0$ and lie in the interior of the parameter space. With fewer genuine regimes than $K$ the transition parameters are unidentified and the test is oversized (in a Monte Carlo with one true regime and $K=2$ it rejected 37\% of the time at the 5\% level). Hence the standard result applies:

$$
\mathrm{LR} = 2\left[ \mathcal{L}_{\text{TVTP}}(\hat{\theta}) - \mathcal{L}_{\text{HMM}}(\hat{\theta}_0) \right] \;\xrightarrow{d}\; \chi^2_{K(K-1)(d-1)} \quad \text{under } H_0, \tag{6.9}
$$

with $K(K-1)(d-1) = 2$ degrees of freedom for $K = 2$ and one covariate. This contrasts with the regime-count test of Section 5.6, which is nonstandard. AIC and BIC are reported alongside.

The LR test is an in-sample statement about fit. The project's claim rests on the out-of-sample tests of Section 7 and step 7 of the roadmap; the LR result is the necessary first hurdle, not the conclusion.

### 6.7 Covariate preprocessing

- Standardise $z$ using the mean and standard deviation of the current training window only; store them and apply to the test segment. Refit per walk-forward window.
- For realised volatility use the log transform before standardising: $z_t = \ln \mathrm{RV}_t$, where $\mathrm{RV}_t$ is a range-based daily estimator from OHLC (Parkinson or Garman-Klass) or a rolling sum of squared returns, or the sum of squared intraday returns if hourly data are used. The log keeps the logistic argument in a range where $\Lambda$ is not saturated.
- For a second covariate, extend $x_{t-1} = (1, z_{1,t-1}, z_{2,t-1})'$; $d = 3$ and the LR degrees of freedom become $K(K-1) \cdot 2$.

---

## 7. Outputs used downstream (forecast evaluation and allocation)

All forecasting quantities are computed at the close of day $t$ from $\mathcal{F}_t$ and refer to day $t+1$.

**One-step regime forecast.**
$$
\boldsymbol{\xi}_{t+1|t} = \mathbf{P}_{t+1}' \boldsymbol{\xi}_{t|t}, \qquad \mathbf{P}_{t+1} = \mathbf{P}(z_t). \tag{7.1}
$$

For the baseline, $\mathbf{P}_{t+1} = \mathbf{P}$. Both models produce a forecast on every date from identical information, so pairwise loss comparisons (Diebold-Mariano) are well posed.

**One-step conditional mean and variance** (moments of a $K$-component Gaussian mixture):

$$
m_{t+1|t} = \sum_{j} \xi_{t+1|t}(j)\, \mu_j, \qquad
h_{t+1|t} = \sum_{j} \xi_{t+1|t}(j)\left( \sigma_j^2 + \mu_j^2 \right) - m_{t+1|t}^2. \tag{7.2}
$$

$h_{t+1|t}$ is the volatility forecast that (i) enters the Diebold-Mariano comparison against a realised-variance proxy (MSE or QLIKE loss; QLIKE is robust to proxy noise, Patton 2011) and (ii) scales the position in the allocation rule.

**One-step predictive density** (for density-forecast scoring, optional):
$$
f(y_{t+1} \mid \mathcal{F}_t) = \sum_j \xi_{t+1|t}(j)\, f_j(y_{t+1}). \tag{7.3}
$$

**Multi-step forecasts.** For the TVTP model, $h$-step forecasts with $h > 1$ require future covariate values $z_{t+1}, \dots$ that are unknown at $t$. The project uses one-step forecasts only, which is all a daily-rebalanced rule needs; this is stated once as a scope decision.

**Timing table (no look-ahead).** At the close of day $t$:

| Known at close of $t$ | Computed at close of $t$ | Used for |
|---|---|---|
| $y_t$, $z_t$, $\hat{\theta}$ from training window | $\boldsymbol{\xi}_{t \mid t}$ via (3.4) | filtering |
| | $\mathbf{P}_{t+1} = \mathbf{P}(z_t)$ | tomorrow's transition |
| | $\boldsymbol{\xi}_{t+1 \mid t}$, $h_{t+1 \mid t}$ via (7.1)–(7.2) | forecast and position |
| | position $w_{t+1}$ held over $(t, t+1]$ | earns $w_{t+1}\, y_{t+1}$ minus costs |

**Walk-forward protocol.** For each window $w$: estimate $\hat{\theta}^{(w)}$ on the training segment; run the filter forward over the test segment with $\hat{\theta}^{(w)}$ held fixed, carrying $\boldsymbol{\xi}_{t|t}$ over from the end of the training segment rather than re-initialising. The filter is causal by construction, so filtered and predicted probabilities on the test segment are legitimate out-of-sample quantities. Smoothed probabilities and Viterbi paths are never used on the test segment.

---

## 8. Numerical implementation notes (repo)

- **Filter in log space when needed.** For the log-likelihood use $\ln \ell_t = \operatorname{logsumexp}_j\{\ln \xi_{t|t-1}(j) + \ln f_j(y_t)\}$; normalise $\boldsymbol{\xi}_{t|t}$ by subtracting the max before exponentiating. In double precision with daily data the plain form (3.3)–(3.4) is usually fine because normalisation happens every step, but the log form costs nothing.
- **Variance floor.** Guard $\hat{\sigma}_j^2 \ge 10^{-8}$ in (5.3) to prevent a regime collapsing onto a single observation.
- **Initial values for EM.** Split returns by rolling-volatility quantiles (or k-means on $|y_t|$) to initialise $\mu_j, \sigma_j$; initialise $p_{ii} = 0.95$. Random restarts perturb these. For Student-$t$ emissions (implemented as a robustness option) the emission M-step is an ECM update: scale-mixture weights $u_t = (\nu+1)/(\nu + d_t^2)$ for location and scale, then a bounded one-dimensional search for $\nu$.
- **Convergence.** Stop EM on $\Delta \mathcal{L} < 10^{-6}$ or 500 iterations. Assert monotonicity of $\mathcal{L}$ across iterations as a unit test.
- **Unit tests worth writing.** (a) TVTP with slopes zero reproduces baseline likelihood to machine precision. (b) Filtered probabilities sum to one. (c) Consistency identities under (4.3). (d) Simulated data from a known TVTP model recovers parameters (parameter-recovery test; report bias and RMSE over, say, 200 simulations). (e) Analytic score (6.6) matches finite-difference gradient of $Q_{\mathbf{P}}$.
- **Standard errors.** Numerical Hessian of (3.5) at $\hat{\theta}$ in the unconstrained parameterisation (5.4), then delta method for reported quantities. A parametric bootstrap (simulate from $\hat{\theta}$, refit) is a robustness alternative.
- **Complexity.** One filter pass is $O(TK^2)$; one EM iteration is $O(TK^2)$ plus the Newton inner loop, negligible. Walk-forward with monthly refits over five years of daily data is seconds of compute.
- **Reference implementations.** `hmmlearn` (constant-transition Gaussian HMM, useful for cross-checking the baseline) and `statsmodels.tsa.regime_switching.MarkovRegression` (supports TVTP through `exog_tvtp`; a natural cross-check for the TVTP estimates). The project's own implementation should be the one in the paper so the derivation and the code match line for line.

---

## 9. What goes where (15-page budget)

**Paper body, Methodology section (target 3 to 3.5 pages).**
- Notation (Section 1, compressed to one paragraph plus (1.1)–(1.3)).
- Baseline model: (2.1)–(2.4), one sentence on stylized facts, (2.5).
- Hamilton filter: (3.1), (3.3), (3.4), (3.5), one sentence on (3.6) linking to Rabiner.
- Estimation: one paragraph on EM with (5.2)–(5.3), one sentence on identification (Section 5.5), one sentence on BIC for $K$ (Section 5.6).
- TVTP: (A1′), (A4), (6.2) with sign hypotheses, boxed (6.4) with the "one-line modification" statement, (6.5)–(6.6) and the pipeline summary, nesting and LR test (6.9).
- Forecast outputs: (7.1)–(7.2) and the timing table or an equivalent two-sentence description.

**Appendix (counts toward the 15 pages, so keep to one page or omit).**
- Kim smoother derivation (4.1)–(4.3), Viterbi (5.6), Hessian (6.7)–(6.8), general-$K$ multinomial form (6.3), reparameterisation (5.4).

**Repository only.**
- Section 8 in full, as `docs/implementation_notes.md`, plus the unit-test list mapped to actual test files.

---

## 10. References used in this section

Already in the verified list: Hamilton (1989); Rabiner (1989); Filardo (1994); Cont (2001); Diebold & Mariano (1995).

Candidate additions, cited above, to verify before adding to the reference list:

- Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *Journal of Econometrics*, 60(1–2), 1–22. [smoother (4.2)]
- Diebold, F. X., Lee, J.-H., & Weinbach, G. C. (1994). Regime switching with time-varying transition probabilities. In C. Hargreaves (Ed.), *Nonstationary Time Series Analysis and Cointegration*. Oxford University Press. [TVTP, parallel precedent to Filardo]
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society B*, 39(1), 1–38. [EM monotonicity]
- Hansen, B. E. (1992). The likelihood ratio test under nonstandard conditions: Testing the Markov switching model of GNP. *Journal of Applied Econometrics*, 7(S1), S61–S82. [why $K$ is chosen by BIC, not LR]
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press, Chapter 22. [textbook form of the filter and smoother]
- Kim, C.-J., Piger, J., & Startz, R. (2008). Estimation of Markov regime-switching regression models with endogenous switching. *Journal of Econometrics*, 143(2), 263–273. [one-line caveat on exogeneity, optional]
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256. [QLIKE loss for the DM test]
