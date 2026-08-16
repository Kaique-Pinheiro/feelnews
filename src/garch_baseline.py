"""
Passo 2 - GARCH(1,1) baseline sobre log-retornos do Ibovespa.

Ajusta GARCH(1,1) com distribuicao t de Student, interpreta os
parametros (omega, alpha, beta) e extrai a serie de volatilidade
condicional em amostra.
"""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from arch import arch_model

warnings.filterwarnings("ignore")

DATA_PROCESSED_DIR = "data/processed"
OUTPUTS_DIR = "outputs"
SCALE = 100  # escala dos retornos para estabilidade numerica do otimizador


def load_returns() -> pd.Series:
    s = pd.read_csv(f"{DATA_PROCESSED_DIR}/bvsp_log_returns.csv", index_col=0, parse_dates=True)
    return s["log_return"]


def fit_garch11(returns: pd.Series):
    scaled = returns * SCALE
    model = arch_model(scaled, mean="Constant", vol="Garch", p=1, q=1, dist="t")
    result = model.fit(disp="off")
    return result


def interpret_params(result) -> dict:
    p = result.params
    omega, alpha, beta = p["omega"], p["alpha[1]"], p["beta[1]"]
    persistence = alpha + beta
    unconditional_var_scaled = omega / (1 - persistence)
    # devolve para a escala original dos retornos (variancia escala com SCALE^2)
    unconditional_vol_annualized = ((unconditional_var_scaled / SCALE**2) ** 0.5) * (252**0.5)
    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "unconditional_annual_vol": unconditional_vol_annualized,
        "nu_student_t": p.get("nu", None),
    }


def plot_conditional_vol(result, returns: pd.Series, out_path: str) -> None:
    cond_vol_annualized = (result.conditional_volatility / SCALE) * (252**0.5)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(returns.index, cond_vol_annualized, color="#c0392b", linewidth=0.9)
    ax.set_title("GARCH(1,1) - Volatilidade condicional anualizada (in-sample)")
    ax.set_xlabel("Data")
    ax.set_ylabel("Volatilidade anualizada")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    returns = load_returns()
    print(f"Ajustando GARCH(1,1)-t em {len(returns)} observacoes...")
    result = fit_garch11(returns)

    print("\n--- Sumario do modelo ---")
    print(result.summary())

    interp = interpret_params(result)
    print("\n--- Interpretacao dos parametros ---")
    print(f"  omega (piso de variancia, escala x100):  {interp['omega']:.6f}")
    print(f"  alpha (reacao a choques):                {interp['alpha']:.4f}")
    print(f"  beta  (persistencia de variancia):       {interp['beta']:.4f}")
    print(f"  alpha + beta (persistencia total):       {interp['persistence']:.4f}")
    print(f"  vol incondicional anualizada implicada:  {interp['unconditional_annual_vol']:.2%}")
    if interp["nu_student_t"]:
        print(f"  graus de liberdade (t-Student):          {interp['nu_student_t']:.2f}")

    if interp["persistence"] >= 0.999:
        print("  AVISO: persistencia >= 0.999 -> processo quase-integrado (IGARCH-like), choques nao decaem.")
    elif interp["persistence"] >= 0.90:
        print("  Persistencia alta (0.90-0.999): choques de volatilidade se dissipam lentamente (semanas).")
    else:
        print("  Persistencia moderada/baixa: volatilidade reverte rapido a media.")

    result.params.to_csv(f"{DATA_PROCESSED_DIR}/garch_baseline_params.csv")
    cond_vol = (result.conditional_volatility / SCALE)
    cond_vol.name = "garch_cond_vol_daily"
    cond_vol.to_csv(f"{DATA_PROCESSED_DIR}/garch_baseline_cond_vol.csv")

    plot_conditional_vol(result, returns, f"{OUTPUTS_DIR}/02_garch_baseline_cond_vol.png")
    print(f"\nParametros salvos em {DATA_PROCESSED_DIR}/garch_baseline_params.csv")
    print(f"Vol condicional salva em {DATA_PROCESSED_DIR}/garch_baseline_cond_vol.csv")
    print(f"Grafico salvo em {OUTPUTS_DIR}/02_garch_baseline_cond_vol.png")


if __name__ == "__main__":
    main()
