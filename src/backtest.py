"""
Passo 5 - Backtest comparativo out-of-sample: GARCH(1,1) baseline vs.
GARCH-X (sentimento defasado), avaliados contra volatilidade realizada
(janela movel de 5 dias) no periodo de teste.

Split cronologico 80/20. Parametros estimados uma unica vez no treino
(nao ha reestimacao diaria/janela expansivel - simplificacao consciente
por restricao de tempo, documentada no README). As previsoes de 1 passo
a frente no teste usam sempre apenas retornos e sentimento ja realizados
ate o dia anterior (nunca informacao futura).
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from garch_x import load_aligned_data
from garch_x_mle import compute_sigma2_series, fit_garch11_x

DATA_PROCESSED_DIR = "data/processed"
OUTPUTS_DIR = "outputs"
SCALE = 100
TRAIN_FRACTION = 0.8
RV_WINDOW = 5


def realized_volatility(scaled_returns: np.ndarray, window: int = RV_WINDOW) -> np.ndarray:
    """RV_t = media movel de r^2 nos ultimos `window` dias (incl. t). Proxy da variancia 'verdadeira'."""
    r2 = scaled_returns**2
    rv = pd.Series(r2).rolling(window, min_periods=window).mean().values
    return rv


def qlike(rv: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    ratio = rv / forecast
    return ratio - np.log(ratio) - 1


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 5) -> tuple[float, float]:
    """DM test: H0 = mesma acuracia preditiva. d_t = loss_a - loss_b.
    Erro-padrao via Newey-West (HAC) para corrigir autocorrelacao em d_t."""
    d = loss_a - loss_b
    n = len(d)
    d_mean = d.mean()

    gamma0 = np.sum((d - d_mean) ** 2) / n
    var_d = gamma0
    for k in range(1, lag + 1):
        cov_k = np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / n
        weight = 1 - k / (lag + 1)
        var_d += 2 * weight * cov_k
    se_d = np.sqrt(var_d / n)

    dm_stat = d_mean / se_d
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value


def main():
    df = load_aligned_data()
    n = len(df)
    split = int(n * TRAIN_FRACTION)
    print(f"Amostra alinhada: {n} pregoes | treino: {split} | teste: {n - split}")
    print(f"  treino: {df.index[0].date()} a {df.index[split - 1].date()}")
    print(f"  teste:  {df.index[split].date()} a {df.index[-1].date()}")

    r_full = (df["log_return"] * SCALE).values
    x_full = df["sentiment_lag1"].values
    r_train = r_full[:split]
    x_train = x_full[:split]

    print("\nAjustando parametros no periodo de treino...")
    res_base = fit_garch11_x(r_train, x=None)
    p_base = res_base.as_dict()
    warm_start = np.array([p_base["mu"], p_base["omega"], p_base["alpha[1]"], p_base["beta[1]"], 0.0, p_base["nu"]])
    res_x = fit_garch11_x(r_train, x=x_train, warm_start=warm_start)
    print(f"  baseline: alpha={p_base['alpha[1]']:.4f} beta={p_base['beta[1]']:.4f} loglik={res_base.loglik:.2f}")
    p_x = res_x.as_dict()
    print(f"  GARCH-X:  alpha={p_x['alpha[1]']:.4f} beta={p_x['beta[1]']:.4f} gamma={p_x['gamma_sentiment']:.4f} loglik={res_x.loglik:.2f}")

    # ---- previsoes 1 passo a frente na amostra de teste, parametros fixos ----
    # A recursao usa retornos/sentimento REAIS ate t-1 para prever t - nunca
    # informacao de t ou posterior. Rodar a recursao sobre a serie completa
    # (treino+teste) com os parametros fixos do treino e depois fatiar o
    # teste e equivalente a uma previsao recursiva de 1 passo, pois cada
    # sigma2_t so depende de eps_{t-1}, sigma2_{t-1} e x_{t-1}.
    sigma2_baseline_full = compute_sigma2_series(res_base.params, r_full, None)
    sigma2_x_full = compute_sigma2_series(res_x.params, r_full, x_full)

    sigma2_baseline_test = sigma2_baseline_full[split:]
    sigma2_x_test = sigma2_x_full[split:]

    rv_full = realized_volatility(r_full)
    rv_test = rv_full[split:]

    valid = ~np.isnan(rv_test)  # primeiros RV_WINDOW-1 dias do teste podem faltar RV
    rv_test = rv_test[valid]
    sigma2_baseline_test = sigma2_baseline_test[valid]
    sigma2_x_test = sigma2_x_test[valid]
    test_dates = df.index[split:][valid]
    n_test_valid = len(rv_test)
    print(f"\nDias de teste com RV valida: {n_test_valid}")

    metrics = {}
    for name, forecast in [("baseline", sigma2_baseline_test), ("garchx", sigma2_x_test)]:
        mse = np.mean((rv_test - forecast) ** 2)
        mae = np.mean(np.abs(rv_test - forecast))
        ql = qlike(rv_test, forecast)
        metrics[name] = {"MSE": mse, "MAE": mae, "QLIKE": ql.mean()}

    print("\n--- Metricas out-of-sample (variancia na escala retornos*100) ---")
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df.to_string())

    ql_base = qlike(rv_test, sigma2_baseline_test)
    ql_x = qlike(rv_test, sigma2_x_test)
    dm_stat, dm_pvalue = diebold_mariano(ql_base, ql_x)

    print(f"\n--- Teste de Diebold-Mariano (QLIKE baseline - QLIKE GARCH-X) ---")
    print(f"  DM statistic: {dm_stat:.4f}  |  p-value: {dm_pvalue:.4f}")
    better = "GARCH-X" if metrics["garchx"]["QLIKE"] < metrics["baseline"]["QLIKE"] else "baseline"
    print(f"  Menor QLIKE (melhor previsao out-of-sample): {better}")
    print(f"  Diferenca estatisticamente significante? {'SIM' if dm_pvalue < 0.05 else 'NAO'} (p={dm_pvalue:.4f})")

    if dm_pvalue >= 0.05:
        direction = "favorecem levemente o GARCH-X" if better == "GARCH-X" else "favorecem o baseline"
        print(f"\n  CONCLUSAO HONESTA: as metricas pontuais out-of-sample {direction}, mas a diferenca")
        print("  NAO e estatisticamente significante (DM test, p>=0.05) - consistente com o Passo 4")
        print("  (LR test in-sample tambem nao significante). Com N=50 no teste, nao ha evidencia")
        print("  robusta de que o sentimento (LLM, janela de 1 ano) melhore a previsao de volatilidade")
        print("  do Ibovespa alem do GARCH(1,1) baseline - o resultado e inconclusivo, nao negativo.")

    metrics_df.to_csv(f"{DATA_PROCESSED_DIR}/backtest_metrics.csv")
    with open(f"{DATA_PROCESSED_DIR}/backtest_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Treino: {df.index[0].date()} a {df.index[split - 1].date()} ({split} obs)\n")
        f.write(f"Teste:  {df.index[split].date()} a {df.index[-1].date()} ({n_test_valid} obs validas)\n\n")
        f.write(metrics_df.to_string() + "\n\n")
        f.write(f"Diebold-Mariano stat={dm_stat:.4f} p-value={dm_pvalue:.4f}\n")
        f.write(f"Melhor modelo (menor QLIKE): {better}\n")

    pd.DataFrame(
        {
            "realized_vol": np.sqrt(rv_test) / SCALE * np.sqrt(252),
            "forecast_baseline": np.sqrt(sigma2_baseline_test) / SCALE * np.sqrt(252),
            "forecast_garchx": np.sqrt(sigma2_x_test) / SCALE * np.sqrt(252),
        },
        index=test_dates,
    ).to_csv(f"{DATA_PROCESSED_DIR}/backtest_forecasts.csv")

    print(f"\nMetricas salvas em {DATA_PROCESSED_DIR}/backtest_metrics.csv")
    print(f"Previsoes de teste salvas em {DATA_PROCESSED_DIR}/backtest_forecasts.csv")


if __name__ == "__main__":
    main()
