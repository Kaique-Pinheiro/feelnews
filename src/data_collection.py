"""
Passo 1 - Coleta de dados do Ibovespa e diagnostico exploratorio.

Baixa historico do ^BVSP via yfinance, calcula log-retornos e roda
diagnosticos que justificam o uso de GARCH: estacionariedade (ADF) e
evidencia de clustering de volatilidade (Ljung-Box em retornos^2).
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

TICKER = "^BVSP"
YEARS = 5
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
OUTPUTS_DIR = "outputs"


def download_prices(ticker: str, years: int) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance retornou vazio para {ticker}. Checar conexao/ticker.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].dropna()
    log_ret = np.log(close / close.shift(1)).dropna()
    log_ret.name = "log_return"
    return log_ret


def descriptive_stats(log_ret: pd.Series) -> pd.Series:
    return pd.Series(
        {
            "n_obs": log_ret.shape[0],
            "mean": log_ret.mean(),
            "std": log_ret.std(),
            "annualized_vol": log_ret.std() * np.sqrt(252),
            "skew": log_ret.skew(),
            "kurtosis": log_ret.kurtosis(),  # excess kurtosis (0 = normal)
            "min": log_ret.min(),
            "max": log_ret.max(),
        }
    )


def adf_test(series: pd.Series) -> dict:
    stat, pvalue, used_lag, nobs, crit_values, _ = adfuller(series, autolag="AIC")
    return {
        "adf_statistic": stat,
        "p_value": pvalue,
        "used_lag": used_lag,
        "n_obs": nobs,
        "critical_values": crit_values,
        "stationary_5pct": pvalue < 0.05,
    }


def volatility_clustering_test(log_ret: pd.Series, lags: int = 10) -> pd.DataFrame:
    squared = log_ret**2
    result = acorr_ljungbox(squared, lags=[lags], return_df=True)
    return result


def plot_returns(log_ret: pd.Series, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(log_ret.index, log_ret.values, linewidth=0.6, color="#1f4e79")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Ibovespa (^BVSP) - Log-retornos diarios")
    ax.set_xlabel("Data")
    ax.set_ylabel("Log-retorno")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    print(f"Baixando {TICKER} ({YEARS} anos)...")
    prices = download_prices(TICKER, YEARS)
    prices.to_csv(f"{DATA_RAW_DIR}/bvsp_prices.csv")
    print(f"  {prices.shape[0]} pregoes baixados: {prices.index.min().date()} a {prices.index.max().date()}")

    log_ret = compute_log_returns(prices)
    log_ret.to_csv(f"{DATA_PROCESSED_DIR}/bvsp_log_returns.csv")

    print("\n--- Estatisticas descritivas (log-retornos) ---")
    stats = descriptive_stats(log_ret)
    print(stats.to_string())

    print("\n--- Teste ADF (estacionariedade) ---")
    adf = adf_test(log_ret)
    print(f"  ADF statistic: {adf['adf_statistic']:.4f}")
    print(f"  p-value:       {adf['p_value']:.6f}")
    print(f"  Estacionario a 5%? {'SIM' if adf['stationary_5pct'] else 'NAO'}")

    print("\n--- Teste Ljung-Box em retornos^2 (clustering de volatilidade) ---")
    lb = volatility_clustering_test(log_ret)
    print(lb.to_string())
    lb_pvalue = lb["lb_pvalue"].iloc[0]
    print(f"  Evidencia de efeito ARCH (clustering)? {'SIM' if lb_pvalue < 0.05 else 'NAO'} (p={lb_pvalue:.2e})")

    plot_returns(log_ret, f"{OUTPUTS_DIR}/01_log_returns_series.png")
    print(f"\nGrafico salvo em {OUTPUTS_DIR}/01_log_returns_series.png")

    summary_path = f"{DATA_PROCESSED_DIR}/diagnostics_step1.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("DIAGNOSTICO EXPLORATORIO - IBOVESPA LOG-RETORNOS\n")
        f.write("=" * 50 + "\n\n")
        f.write(stats.to_string() + "\n\n")
        f.write(f"ADF statistic: {adf['adf_statistic']:.4f}\n")
        f.write(f"ADF p-value:   {adf['p_value']:.6f}\n")
        f.write(f"Estacionario a 5%: {adf['stationary_5pct']}\n\n")
        f.write(lb.to_string() + "\n")
        f.write(f"\nEfeito ARCH (clustering) presente: {lb_pvalue < 0.05}\n")
    print(f"Resumo salvo em {summary_path}")


if __name__ == "__main__":
    main()
