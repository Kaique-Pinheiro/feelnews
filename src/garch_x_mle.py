"""
Implementacao manual de GARCH(1,1) com regressor exogeno na equacao de
variancia (GARCH-X), via maxima verossimilhanca com scipy.optimize.

Necessario porque a biblioteca `arch` NAO suporta regressores exogenos
na equacao de variancia: o parametro `x` de arch_model() so e aplicado
a equacao da MEDIA (e apenas com mean='LS'), sendo silenciosamente
ignorado com mean='Constant'. Confirmado lendo o docstring/source de
arch.univariate.mean.arch_model.

Equacao de variancia:
    sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2 + gamma*x_t

(x_t aqui e o regressor JA alinhado pelo chamador - ou seja, quem chama
esta funcao ja deve passar x_t = sentimento defasado em 1 dia. Este
modulo nao aplica nenhum shift adicional.)

Inovacoes padronizadas t de Student, com nu (graus de liberdade)
estimado junto. Erros-padrao via Hessiana numerica da log-verossimilhanca
negativa (aproximacao classica de MLE).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

FLOOR_VAR = 1e-6
PENALTY_WEIGHT = 1e6
STATIONARITY_LIMIT = 0.999


@dataclass
class GarchXResult:
    param_names: list
    params: np.ndarray
    se: np.ndarray
    pvalues: np.ndarray
    loglik: float
    aic: float
    bic: float
    n_obs: int
    sigma2: np.ndarray
    converged: bool

    def as_dict(self) -> dict:
        return dict(zip(self.param_names, self.params))

    def se_dict(self) -> dict:
        return dict(zip(self.param_names, self.se))

    def pvalue_dict(self) -> dict:
        return dict(zip(self.param_names, self.pvalues))


def _unpack(params: np.ndarray, has_x: bool):
    mu, omega, alpha, beta = params[0], params[1], params[2], params[3]
    if has_x:
        gamma, nu = params[4], params[5]
    else:
        gamma, nu = 0.0, params[4]
    return mu, omega, alpha, beta, gamma, nu


def compute_sigma2_series(params: np.ndarray, returns: np.ndarray, x: np.ndarray | None) -> np.ndarray:
    has_x = x is not None
    mu, omega, alpha, beta, gamma, _ = _unpack(params, has_x)
    eps = returns - mu
    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = max(np.var(returns), FLOOR_VAR)
    for t in range(1, n):
        exog = gamma * x[t] if has_x else 0.0
        sigma2[t] = max(omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1] + exog, FLOOR_VAR)
    return sigma2


def _neg_loglik(params: np.ndarray, returns: np.ndarray, x: np.ndarray | None) -> float:
    has_x = x is not None
    mu, omega, alpha, beta, gamma, nu = _unpack(params, has_x)
    if nu <= 2.01 or omega <= 0 or alpha < 0 or beta < 0:
        return 1e10

    sigma2 = compute_sigma2_series(params, returns, x)
    eps = returns - mu
    z = eps / np.sqrt(sigma2)

    ll = (
        special.gammaln((nu + 1) / 2)
        - special.gammaln(nu / 2)
        - 0.5 * np.log(np.pi * (nu - 2))
        - 0.5 * np.log(sigma2)
        - ((nu + 1) / 2) * np.log(1 + z**2 / (nu - 2))
    )

    penalty = 0.0
    if alpha + beta >= STATIONARITY_LIMIT:
        penalty = PENALTY_WEIGHT * (alpha + beta - STATIONARITY_LIMIT + 1e-3)

    return -np.sum(ll) + penalty


def _numerical_hessian(f, x0: np.ndarray, step: float = 1e-4) -> np.ndarray:
    n = len(x0)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x1, x2, x3, x4 = (x0.copy() for _ in range(4))
            x1[i] += step; x1[j] += step
            x2[i] += step; x2[j] -= step
            x3[i] -= step; x3[j] += step
            x4[i] -= step; x4[j] -= step
            H[i, j] = (f(x1) - f(x2) - f(x3) + f(x4)) / (4 * step * step)
            H[j, i] = H[i, j]
    return H


def fit_garch11_x(returns: np.ndarray, x: np.ndarray | None = None, warm_start: np.ndarray | None = None) -> GarchXResult:
    """Ajusta GARCH(1,1)-t por MLE. Se x is None, e o baseline sem regressor.

    returns deve estar na MESMA escala usada no Passo 2 (log_return * 100)
    para estabilidade numerica do otimizador.

    warm_start: valores iniciais [mu, omega, alpha, beta, gamma, nu] (gamma
    omitido se has_x=False). Usar os parametros ja convergidos do baseline
    (com gamma=0) para inicializar o GARCH-X evita que o otimizador caia
    num minimo local pior que o do modelo aninhado - por construcao, o
    log-likelihood do GARCH-X no otimo nunca pode ser PIOR que o do
    baseline (gamma=0 e um caso particular), entao um resultado pior
    indica falha de convergencia, nao um resultado real.
    """
    has_x = x is not None
    returns = np.asarray(returns, dtype=float)
    if has_x:
        x = np.asarray(x, dtype=float)

    var0 = np.var(returns)
    if has_x:
        x0 = warm_start if warm_start is not None else np.array([returns.mean(), var0 * 0.10, 0.05, 0.80, 0.0, 8.0])
        bounds = [(-5, 5), (1e-6, 50), (0, 0.5), (0, 0.999), (-50, 50), (2.05, 30)]
        names = ["mu", "omega", "alpha[1]", "beta[1]", "gamma_sentiment", "nu"]
    else:
        x0 = warm_start if warm_start is not None else np.array([returns.mean(), var0 * 0.10, 0.05, 0.80, 8.0])
        bounds = [(-5, 5), (1e-6, 50), (0, 0.5), (0, 0.999), (2.05, 30)]
        names = ["mu", "omega", "alpha[1]", "beta[1]", "nu"]

    result = optimize.minimize(
        _neg_loglik,
        x0,
        args=(returns, x),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    # multi-start: tenta tambem a partir de um ponto generico; fica com o melhor
    if has_x:
        generic_x0 = np.array([returns.mean(), var0 * 0.10, 0.05, 0.80, 0.0, 8.0])
        result_generic = optimize.minimize(
            _neg_loglik, generic_x0, args=(returns, x), method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 2000, "ftol": 1e-10},
        )
        if result_generic.fun < result.fun:
            result = result_generic

    # polimento com Nelder-Mead (sem gradiente): L-BFGS-B pode terminar em
    # ABNORMAL_TERMINATION_IN_LNSRCH quando a superficie de verossimilhanca
    # e mal-condicionada (parametros em escalas muito diferentes, ex: gamma
    # vs. nu). Nelder-Mead nao depende de gradiente numerico e refina o
    # ponto sem risco de piorar (comparamos o valor da funcao objetivo).
    result_nm = optimize.minimize(
        _neg_loglik, result.x, args=(returns, x), method="Nelder-Mead",
        bounds=bounds, options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )
    if result_nm.fun < result.fun:
        result = result_nm

    loglik = -result.fun
    k = len(x0)
    n = len(returns)
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik

    try:
        H = _numerical_hessian(lambda p: _neg_loglik(p, returns, x), result.x)
        cov = np.linalg.inv(H)
        se = np.sqrt(np.abs(np.diag(cov)))
        pvalues = 2 * (1 - stats.norm.cdf(np.abs(result.x / se)))
    except Exception:
        se = np.full(k, np.nan)
        pvalues = np.full(k, np.nan)

    sigma2 = compute_sigma2_series(result.x, returns, x)

    return GarchXResult(
        param_names=names,
        params=result.x,
        se=se,
        pvalues=pvalues,
        loglik=loglik,
        aic=aic,
        bic=bic,
        n_obs=n,
        sigma2=sigma2,
        converged=result.success,
    )
