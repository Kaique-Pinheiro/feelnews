"""
Fase 2 - Camada de decisao: vol-targeting sobre o Ibovespa usando as
previsoes de variancia do GARCH baseline e do GARCH-X (mesmo split
treino/teste e mesmos parametros do Passo 5 - nao ha reestimacao aqui).

peso(t) = vol_alvo_anualizada / vol_prevista(t), capado em [0, 1.5]

vol_prevista(t) e a previsao de 1 passo a frente feita com informacao ate
t-1 (a mesma recursao usada no backtest - sigma2_t so depende de dados
ate t-1), entao o peso de t nunca usa informacao de t. Capital nao
alocado (1 - peso) rende CDI (taxa constante aproximada, documentada).

Parametros de decisao (vol_alvo, cap, CDI) sao escolhas a priori,
documentadas - nao foram otimizados nos dados (evita overfitting).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from garch_x import load_aligned_data
from garch_x_mle import compute_sigma2_series, fit_garch11_x

DATA_PROCESSED_DIR = "data/processed"
OUTPUTS_DIR = "outputs"
SCALE = 100
TRAIN_FRACTION = 0.8

VOL_TARGET_ANNUAL = 0.15   # 15% a.a. - escolha a priori, entre a vol historica do Ibovespa (~17-18%) e um alvo conservador
WEIGHT_CAP = 1.5           # alavancagem maxima
WEIGHT_FLOOR = 0.0         # sem posicao vendida
CDI_ANNUAL = 0.10          # aproximacao constante (~Selic/CDI recente), nao a serie historica real - simplificacao documentada
CDI_DAILY = (1 + CDI_ANNUAL) ** (1 / 252) - 1

BLUE = "#2a78d6"   # GARCH(1,1) baseline
RED = "#e34948"    # GARCH-X
INK = "#0b0b0b"    # buy-and-hold (benchmark passivo)
GRID = "#e1e0d9"


def forecast_vol_annual(sigma2_scaled: np.ndarray) -> np.ndarray:
    """sigma2 esta na escala (log_return*100)^2; converte para vol anualizada real."""
    return np.sqrt(sigma2_scaled) / SCALE * np.sqrt(252)


def compute_weights(vol_forecast_annual: np.ndarray) -> np.ndarray:
    raw = VOL_TARGET_ANNUAL / vol_forecast_annual
    return np.clip(raw, WEIGHT_FLOOR, WEIGHT_CAP)


def strategy_returns(weights: np.ndarray, r_ibov_simple: np.ndarray) -> np.ndarray:
    """Retorno simples da estrategia: peso no Ibovespa, resto rende CDI."""
    return weights * r_ibov_simple + (1 - weights) * CDI_DAILY


def compute_metrics(name: str, port_r: np.ndarray, weights: np.ndarray, n_days: int) -> dict:
    wealth = np.cumprod(1 + port_r)
    ret_acum = wealth[-1] - 1
    ret_annual = (1 + ret_acum) ** (252 / n_days) - 1
    vol_annual = np.std(np.log(1 + port_r)) * np.sqrt(252)
    sharpe = (ret_annual - CDI_ANNUAL) / vol_annual if vol_annual > 0 else np.nan
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1
    max_dd = drawdown.min()
    turnover = np.mean(np.abs(np.diff(weights))) if len(weights) > 1 else 0.0

    return {
        "estrategia": name,
        "retorno_acumulado": ret_acum,
        "retorno_anualizado": ret_annual,
        "vol_anualizada": vol_annual,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "turnover_medio": turnover,
        "wealth": wealth,
    }


def main():
    df = load_aligned_data()
    n = len(df)
    split = int(n * TRAIN_FRACTION)
    print(f"Reaproveitando split do Passo 5: treino={split}, teste={n - split} pregoes")
    print(f"Parametros de decisao: vol_alvo={VOL_TARGET_ANNUAL:.0%} a.a. | cap={WEIGHT_CAP}x | CDI~{CDI_ANNUAL:.0%} a.a. (aprox. constante)")

    r_full = (df["log_return"] * SCALE).values
    x_full = df["sentiment_lag1"].values
    r_train = r_full[:split]
    x_train = x_full[:split]

    res_base = fit_garch11_x(r_train, x=None)
    p_base = res_base.as_dict()
    warm_start = np.array([p_base["mu"], p_base["omega"], p_base["alpha[1]"], p_base["beta[1]"], 0.0, p_base["nu"]])
    res_x = fit_garch11_x(r_train, x=x_train, warm_start=warm_start)

    sigma2_baseline_full = compute_sigma2_series(res_base.params, r_full, None)
    sigma2_x_full = compute_sigma2_series(res_x.params, r_full, x_full)

    sigma2_baseline_test = sigma2_baseline_full[split:]
    sigma2_x_test = sigma2_x_full[split:]

    # retorno simples real do Ibovespa no periodo de teste (nao escalado)
    log_ret_test = df["log_return"].values[split:]
    r_ibov_simple_test = np.exp(log_ret_test) - 1
    test_dates = df.index[split:]
    n_test = len(test_dates)

    vol_fc_baseline = forecast_vol_annual(sigma2_baseline_test)
    vol_fc_x = forecast_vol_annual(sigma2_x_test)

    w_baseline = compute_weights(vol_fc_baseline)
    w_x = compute_weights(vol_fc_x)
    w_bh = np.ones(n_test)  # buy-and-hold: sempre 100% alocado

    port_r_baseline = strategy_returns(w_baseline, r_ibov_simple_test)
    port_r_x = strategy_returns(w_x, r_ibov_simple_test)
    port_r_bh = r_ibov_simple_test  # buy-and-hold puro, sem CDI (sempre investido)

    m_bh = compute_metrics("Buy-and-hold Ibovespa", port_r_bh, w_bh, n_test)
    m_base = compute_metrics("Vol-targeting (GARCH baseline)", port_r_baseline, w_baseline, n_test)
    m_x = compute_metrics("Vol-targeting (GARCH-X)", port_r_x, w_x, n_test)

    print("\n--- Metricas de estrategia (periodo de teste, out-of-sample, 50 pregoes) ---")
    metrics_table = pd.DataFrame([
        {k: v for k, v in m.items() if k != "wealth"} for m in [m_bh, m_base, m_x]
    ]).set_index("estrategia")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(metrics_table.to_string())

    metrics_table.to_csv(f"{DATA_PROCESSED_DIR}/strategy_metrics.csv")

    weights_df = pd.DataFrame(
        {"peso_baseline": w_baseline, "peso_garchx": w_x},
        index=test_dates,
    )
    weights_df.to_csv(f"{DATA_PROCESSED_DIR}/strategy_weights.csv")

    wealth_df = pd.DataFrame(
        {
            "buy_and_hold": m_bh["wealth"],
            "vol_target_baseline": m_base["wealth"],
            "vol_target_garchx": m_x["wealth"],
        },
        index=test_dates,
    )
    wealth_df.to_csv(f"{DATA_PROCESSED_DIR}/strategy_wealth.csv")

    # ---- visualizacao: patrimonio (topo) + exposicao (base), eixos separados ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, height_ratios=[1.3, 1])

    ax1.plot(test_dates, m_bh["wealth"], label="Buy-and-hold Ibovespa", color=INK, linewidth=1.4)
    ax1.plot(test_dates, m_base["wealth"], label="Vol-targeting (GARCH baseline)", color=BLUE, linewidth=1.4, linestyle="--")
    ax1.plot(test_dates, m_x["wealth"], label="Vol-targeting (GARCH-X)", color=RED, linewidth=1.4, linestyle="--")
    ax1.axhline(1.0, color=GRID, linewidth=0.8)
    ax1.set_ylabel("Patrimonio acumulado\n(base=1.0)")
    ax1.set_title("Backtest de estrategia out-of-sample: buy-and-hold vs. vol-targeting")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", color=GRID, linewidth=0.6)

    ax2.plot(test_dates, w_baseline, label="Exposicao - GARCH baseline", color=BLUE, linewidth=1.3)
    ax2.plot(test_dates, w_x, label="Exposicao - GARCH-X", color=RED, linewidth=1.3)
    ax2.axhline(1.0, color=INK, linewidth=0.8, linestyle=":", label="Buy-and-hold (100%)")
    ax2.set_ylabel("Exposicao ao Ibovespa\n(peso da carteira)")
    ax2.set_xlabel("Data")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(axis="y", color=GRID, linewidth=0.6)

    fig.tight_layout()
    out = f"{OUTPUTS_DIR}/07_strategy_backtest.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"\nGrafico salvo em {out}")
    print(f"Metricas salvas em {DATA_PROCESSED_DIR}/strategy_metrics.csv")

    return metrics_table


if __name__ == "__main__":
    main()
