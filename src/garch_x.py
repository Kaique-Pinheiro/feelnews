"""
Passo 4 - GARCH-X: GARCH(1,1) com sentimento defasado como regressor
exogeno na equacao de variancia.

sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2 + gamma * X_{t-1}

Usa a MLE manual em src/garch_x_mle.py (a biblioteca `arch` nao suporta
regressor exogeno na equacao de variancia - ver docstring daquele modulo).

Ajustado apenas na janela em que ha sentimento real, e comparado a um
GARCH(1,1) baseline reajustado NA MESMA JANELA e com a MESMA rotina de
MLE (elimina diferencas de otimizador/software como fator de confusao -
a unica diferenca entre os dois fits e a presenca do regressor gamma).
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from garch_x_mle import fit_garch11_x

warnings.filterwarnings("ignore")

DATA_PROCESSED_DIR = "data/processed"
OUTPUTS_DIR = "outputs"
SCALE = 100


def load_aligned_data() -> pd.DataFrame:
    returns = pd.read_csv(f"{DATA_PROCESSED_DIR}/bvsp_log_returns.csv", index_col=0, parse_dates=True)["log_return"]
    sentiment = pd.read_csv(f"{DATA_PROCESSED_DIR}/daily_sentiment.csv", index_col=0, parse_dates=True)

    df = pd.DataFrame({"log_return": returns}).join(sentiment[["sentiment_score"]], how="inner")

    # ---- CORTE ANTI LOOK-AHEAD BIAS ----
    # sentimento do dia t so pode explicar a variancia do dia t+1.
    # sentiment_lag1 na linha t contem o sentimento do dia t-1: nunca
    # usamos informacao contemporanea ou futura para "prever" o presente.
    df["sentiment_lag1"] = df["sentiment_score"].shift(1)
    df = df.dropna(subset=["sentiment_lag1"])
    # ------------------------------------

    return df


def print_result(label: str, res, extra_param: str | None = None) -> None:
    params = res.as_dict()
    se = res.se_dict()
    pvalues = res.pvalue_dict()
    persistence = params["alpha[1]"] + params["beta[1]"]

    print(f"\n--- {label} ---")
    print(f"  n_obs: {res.n_obs} | convergiu: {res.converged}")
    for name in res.param_names:
        p = pvalues[name]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"  {name:18s} = {params[name]:9.4f}  (se={se[name]:.4f}, p={p:.4f}) {sig}")
    print(f"  persistencia (alpha+beta): {persistence:.4f}")
    print(f"  log-likelihood: {res.loglik:.2f} | AIC: {res.aic:.2f} | BIC: {res.bic:.2f}")

    if extra_param and extra_param in params:
        gamma = params[extra_param]
        p_gamma = pvalues[extra_param]
        sign_txt = "negativo -> sentimento negativo ontem eleva a variancia hoje (consistente com a hipotese)" \
            if gamma < 0 else "positivo -> direcao contra-intuitiva em relacao a hipotese"
        sig_txt = "estatisticamente significante (p<0.05)" if p_gamma < 0.05 else "NAO significante (p>=0.05)"
        print(f"  gamma_sentiment: {sig_txt}, sinal {sign_txt}")


def plot_comparison(df: pd.DataFrame, sigma2_baseline, sigma2_x, out_path: str) -> None:
    vol_baseline = (sigma2_baseline**0.5 / SCALE) * (252**0.5)
    vol_x = (sigma2_x**0.5 / SCALE) * (252**0.5)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, vol_baseline, label="GARCH(1,1) baseline (mesma janela)", color="#2a78d6", linewidth=1.4)
    ax.plot(df.index, vol_x, label="GARCH-X (sentimento defasado)", color="#e34948", linewidth=1.4, linestyle="--")
    ax.set_title("Volatilidade condicional anualizada - baseline vs. GARCH-X")
    ax.set_xlabel("Data")
    ax.set_ylabel("Volatilidade anualizada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    df = load_aligned_data()
    print(f"Janela alinhada (com sentimento defasado disponivel): {len(df)} pregoes")
    print(f"  {df.index.min().date()} a {df.index.max().date()}")

    r = (df["log_return"] * SCALE).values
    x = df["sentiment_lag1"].values

    res_baseline = fit_garch11_x(r, x=None)
    print_result("GARCH(1,1) baseline (mesma janela, mesma MLE)", res_baseline)

    # warm start: parametros convergidos do baseline + gamma=0 (ver docstring
    # de fit_garch11_x - garante que o GARCH-X nao converge para um ponto
    # pior que o modelo aninhado, o que seria impossivel na otima global)
    p_base = res_baseline.as_dict()
    warm_start = np.array([p_base["mu"], p_base["omega"], p_base["alpha[1]"], p_base["beta[1]"], 0.0, p_base["nu"]])

    res_x = fit_garch11_x(r, x=x, warm_start=warm_start)
    print_result("GARCH-X (sentimento defasado)", res_x, extra_param="gamma_sentiment")

    if res_x.loglik < res_baseline.loglik - 1e-6:
        print("  [AVISO] GARCH-X convergiu pior que o baseline aninhado - problema de otimizacao, nao resultado real.")

    aic_improved = res_x.aic < res_baseline.aic
    bic_improved = res_x.bic < res_baseline.bic
    lr_stat = 2 * (res_x.loglik - res_baseline.loglik)  # teste da razao de verossimilhanca, 1 g.l. extra
    from scipy.stats import chi2
    lr_pvalue = 1 - chi2.cdf(lr_stat, df=1)

    print("\n--- Comparacao baseline vs. GARCH-X ---")
    print(f"  AIC:  baseline={res_baseline.aic:.2f}  GARCH-X={res_x.aic:.2f}  (menor é melhor) -> melhora: {aic_improved}")
    print(f"  BIC:  baseline={res_baseline.bic:.2f}  GARCH-X={res_x.bic:.2f}  (menor é melhor) -> melhora: {bic_improved}")
    print(f"  Teste da razao de verossimilhanca (1 g.l.): LR={lr_stat:.3f}, p-value={lr_pvalue:.4f}")
    print(f"  Sentimento adiciona poder explicativo significante? {'SIM' if lr_pvalue < 0.05 else 'NAO'}")

    for name, res in [("baseline", res_baseline), ("garchx", res_x)]:
        pd.Series(res.as_dict()).to_csv(f"{DATA_PROCESSED_DIR}/garch_{name}_samewindow_params.csv")

    cond_vol = pd.DataFrame(
        {
            "cond_vol_baseline": (res_baseline.sigma2**0.5) / SCALE,
            "cond_vol_garchx": (res_x.sigma2**0.5) / SCALE,
        },
        index=df.index,
    )
    cond_vol.to_csv(f"{DATA_PROCESSED_DIR}/garch_comparison_cond_vol.csv")

    plot_comparison(df, res_baseline.sigma2, res_x.sigma2, f"{OUTPUTS_DIR}/03_garch_vs_garchx_cond_vol.png")

    with open(f"{DATA_PROCESSED_DIR}/garch_x_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Janela alinhada: {len(df)} pregoes ({df.index.min().date()} a {df.index.max().date()})\n\n")
        for label, res in [("BASELINE", res_baseline), ("GARCH-X", res_x)]:
            f.write(f"{label}\n")
            for name in res.param_names:
                f.write(f"  {name} = {res.as_dict()[name]:.4f} (se={res.se_dict()[name]:.4f}, p={res.pvalue_dict()[name]:.4f})\n")
            f.write(f"  loglik={res.loglik:.2f} aic={res.aic:.2f} bic={res.bic:.2f}\n\n")
        f.write(f"LR stat={lr_stat:.3f} p-value={lr_pvalue:.4f}\n")

    print(f"\nParametros salvos em {DATA_PROCESSED_DIR}/garch_garchx_samewindow_params.csv")
    print(f"Grafico salvo em {OUTPUTS_DIR}/03_garch_vs_garchx_cond_vol.png")


if __name__ == "__main__":
    main()
