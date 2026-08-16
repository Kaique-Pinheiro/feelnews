"""
Passo 3b - Score de sentimento das manchetes via LLM.

Consome exclusivamente a interface generica src.sentiment_provider.get_sentiment
(nunca chama Gemini/Anthropic diretamente) - o provider e trocavel via env var
LLM_PROVIDER sem alterar este arquivo.

Para cada dia, agrega os itens classificados (um por manchete) em:
  - sentiment_score: media do sentimento das manchetes do dia
  - sentiment_std:   dispersao (dias de consenso vs. dia com sinais mistos)
  - intensity_mean:  media da intensidade/urgencia do tom
  - risk_category_dominante: categoria de risco mais frequente do dia

NAO faz nenhum alinhamento temporal com retornos aqui - isso e feito em
src/garch_x.py, onde o corte de look-ahead bias (defasar sentimento em
1 dia) e aplicado explicitamente.
"""

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from sentiment_provider import get_sentiment  # unica dependencia de LLM permitida aqui

DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"


def build_prompt_text(headlines: list[str]) -> str:
    return "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))


def score_day(headlines: list[str]) -> list[dict]:
    if not headlines:
        return []
    texto = build_prompt_text(headlines)
    result = get_sentiment(texto)
    items = result["items"]
    if len(items) != len(headlines):
        raise ValueError(f"Esperado {len(headlines)} itens, recebido {len(items)}")
    return items


def main():
    provider = os.getenv("LLM_PROVIDER", "gemini")
    print(f"Provider de sentimento: {provider}")

    news = pd.read_csv(f"{DATA_RAW_DIR}/news_headlines.csv", encoding="utf-8")
    news["headlines_list"] = news["headlines"].fillna("").apply(
        lambda s: [h.strip() for h in s.split(" | ") if h.strip()]
    )

    total_headlines = news["n_headlines"].sum()
    print(f"Pontuando sentimento de {total_headlines} manchetes em {len(news)} dias...")
    t0 = time.time()

    daily_rows = []
    detail_rows = []
    for i, row in news.iterrows():
        headlines = row["headlines_list"]
        items = score_day(headlines)

        for h, it in zip(headlines, items):
            detail_rows.append(
                {
                    "date": row["date"],
                    "headline": h,
                    "sentimento": it["sentimento"],
                    "intensidade": it["intensidade"],
                    "categoria_risco": it["categoria_risco"],
                }
            )

        if items:
            sent_series = pd.Series([it["sentimento"] for it in items])
            int_series = pd.Series([it["intensidade"] for it in items])
            cat_series = pd.Series([it["categoria_risco"] for it in items])
            daily_rows.append(
                {
                    "date": row["date"],
                    "sentiment_score": sent_series.mean(),
                    "sentiment_std": sent_series.std() if len(sent_series) > 1 else 0.0,
                    "intensity_mean": int_series.mean(),
                    "risk_category_dominante": cat_series.mode().iloc[0],
                    "n_headlines": len(headlines),
                }
            )
        else:
            daily_rows.append(
                {
                    "date": row["date"],
                    "sentiment_score": 0.0,
                    "sentiment_std": 0.0,
                    "intensity_mean": 0.0,
                    "risk_category_dominante": "sem_noticia",
                    "n_headlines": 0,
                }
            )

        if (i + 1) % 15 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (len(news) - i - 1)
            print(f"  {i + 1}/{len(news)} dias | {elapsed:.0f}s decorridos | ~{remaining:.0f}s restantes")
            if remaining > 600:
                print("  [AVISO] ritmo indica que vai passar de 10min. Avaliar cortar janela se necessario.")

    daily_df = pd.DataFrame(daily_rows)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.set_index("date")
    daily_df.to_csv(f"{DATA_PROCESSED_DIR}/daily_sentiment.csv", encoding="utf-8")

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(f"{DATA_PROCESSED_DIR}/sentiment_headlines_detail.csv", index=False, encoding="utf-8")

    print(f"\nConcluido em {time.time() - t0:.0f}s")
    print(f"  Score medio diario: {daily_df['sentiment_score'].mean():.3f} | desvio: {daily_df['sentiment_score'].std():.3f}")
    print(f"  Min: {daily_df['sentiment_score'].min():.3f} | Max: {daily_df['sentiment_score'].max():.3f}")
    print("  Distribuicao categoria_risco dominante por dia:")
    print(daily_df["risk_category_dominante"].value_counts().to_string())
    print(f"  Salvo em {DATA_PROCESSED_DIR}/daily_sentiment.csv")
    print(f"  Detalhe por manchete salvo em {DATA_PROCESSED_DIR}/sentiment_headlines_detail.csv")


if __name__ == "__main__":
    main()
