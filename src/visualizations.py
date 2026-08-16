"""
Passo 6 - Visualizacoes finais para o relatorio (max 5 paginas, visual).

Paleta: azul #2a78d6 (baseline / serie 1) e vermelho #e34948 (GARCH-X /
serie 8) - par adjacente validado para contraste de daltonismo. Sentimento
e volatilidade NUNCA no mesmo eixo (dual-axis e o erro #1 de grafico) -
uso subplots empilhados compartilhando o eixo x.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from garch_x import load_aligned_data

DATA_PROCESSED_DIR = "data/processed"
OUTPUTS_DIR = "outputs"

BLUE = "#2a78d6"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def plot_sentiment_vs_volatility():
    df = load_aligned_data()
    rv_daily_var = (df["log_return"] * 100) ** 2
    rv_annual_vol = np.sqrt(rv_daily_var.rolling(5, min_periods=5).mean()) / 100 * np.sqrt(252)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[1, 1.2])

    colors = np.where(df["sentiment_score"] >= 0, BLUE, RED)
    ax1.bar(df.index, df["sentiment_score"], color=colors, width=1.2)
    ax1.axhline(0, color=MUTED, linewidth=0.8)
    ax1.set_ylabel("Score de sentimento\n(-1 a +1)")
    ax1.set_title("Sentimento diario (manchetes financeiras) vs. volatilidade realizada - Ibovespa")
    ax1.set_ylim(-1, 1)
    ax1.grid(axis="y", color=GRID, linewidth=0.6)

    ax2.plot(df.index, rv_annual_vol, color=INK, linewidth=1.2)
    ax2.set_ylabel("Vol. realizada anualizada\n(janela movel 5 dias)")
    ax2.set_xlabel("Data")
    ax2.grid(axis="y", color=GRID, linewidth=0.6)

    fig.tight_layout()
    out = f"{OUTPUTS_DIR}/04_sentiment_vs_volatility.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Salvo: {out}")


def plot_backtest_comparison():
    df = pd.read_csv(f"{DATA_PROCESSED_DIR}/backtest_forecasts.csv", index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(df.index, df["realized_vol"], label="Volatilidade realizada (RV, janela 5d)", color=INK, linewidth=1.6)
    ax.plot(df.index, df["forecast_baseline"], label="Previsao GARCH(1,1) baseline", color=BLUE, linewidth=1.3, linestyle="--")
    ax.plot(df.index, df["forecast_garchx"], label="Previsao GARCH-X (sentimento)", color=RED, linewidth=1.3, linestyle="--")
    ax.set_title("Backtest out-of-sample: previsao de volatilidade vs. realizada (periodo de teste)")
    ax.set_xlabel("Data")
    ax.set_ylabel("Volatilidade anualizada")
    ax.legend()
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    out = f"{OUTPUTS_DIR}/05_backtest_oos_comparison.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Salvo: {out}")


def plot_metrics_table():
    base_params = pd.read_csv(f"{DATA_PROCESSED_DIR}/garch_baseline_samewindow_params.csv", index_col=0).iloc[:, 0]
    x_params = pd.read_csv(f"{DATA_PROCESSED_DIR}/garch_garchx_samewindow_params.csv", index_col=0).iloc[:, 0]
    backtest = pd.read_csv(f"{DATA_PROCESSED_DIR}/backtest_metrics.csv", index_col=0)

    with open(f"{DATA_PROCESSED_DIR}/backtest_summary.txt", encoding="utf-8") as f:
        summary_text = f.read()
    dm_line = [l for l in summary_text.splitlines() if l.startswith("Diebold-Mariano")][0]

    persist_base = base_params["alpha[1]"] + base_params["beta[1]"]
    persist_x = x_params["alpha[1]"] + x_params["beta[1]"]

    rows = [
        ["alpha (reacao a choques)", f"{base_params['alpha[1]']:.4f}", f"{x_params['alpha[1]']:.4f}"],
        ["beta (persistencia)", f"{base_params['beta[1]']:.4f}", f"{x_params['beta[1]']:.4f}"],
        ["alpha + beta", f"{persist_base:.4f}", f"{persist_x:.4f}"],
        ["gamma (sentimento defasado)", "-", f"{x_params['gamma_sentiment']:.4f}"],
        ["MSE out-of-sample", f"{backtest.loc['baseline', 'MSE']:.4f}", f"{backtest.loc['garchx', 'MSE']:.4f}"],
        ["MAE out-of-sample", f"{backtest.loc['baseline', 'MAE']:.4f}", f"{backtest.loc['garchx', 'MAE']:.4f}"],
        ["QLIKE out-of-sample", f"{backtest.loc['baseline', 'QLIKE']:.4f}", f"{backtest.loc['garchx', 'QLIKE']:.4f}"],
    ]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metrica", "GARCH(1,1) baseline", "GARCH-X (sentimento)"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(INK)
        else:
            cell.set_facecolor("#fcfcfb")

    ax.set_title("Comparativo GARCH baseline vs. GARCH-X (mesma janela, 250 pregoes)\n" + dm_line, fontsize=10, pad=20)
    fig.tight_layout()
    out = f"{OUTPUTS_DIR}/06_metrics_table.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {out}")


def main():
    plot_sentiment_vs_volatility()
    plot_backtest_comparison()
    plot_metrics_table()


if __name__ == "__main__":
    main()
