"""
Passo 3a - Coleta de manchetes financeiras sobre o mercado brasileiro.

Fonte: Google News RSS com filtro de data (after:/before:), consultado
dia a dia. Nao exige chave de API. Cobertura limitada a uma janela
recente (WINDOW_DAYS) porque nao ha acesso gratuito a arquivo historico
completo de noticias - essa e uma limitacao documentada no README.
"""

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pandas as pd

DATA_RAW_DIR = "data/raw"
WINDOW_DAYS = 365
MAX_HEADLINES_PER_DAY = 10
QUERY = 'ibovespa OR "bolsa de valores" OR "mercado financeiro" Brasil'
BASE_URL = "https://news.google.com/rss/search"
HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_DELAY = 0.4  # segundos entre requests, para nao sobrecarregar/ser bloqueado


def fetch_day_headlines(day: datetime, retries: int = 2) -> list[str]:
    day_str = day.strftime("%Y-%m-%d")
    next_day_str = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    q = f"{QUERY} after:{day_str} before:{next_day_str}"
    params = {"q": q, "hl": "pt-BR", "gl": "BR", "ceid": "BR:pt-419"}
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            data = urllib.request.urlopen(req, timeout=10).read()
            root = ET.fromstring(data)
            items = root.findall(".//item")
            titles = [it.findtext("title", "").strip() for it in items]
            titles = [t for t in titles if t]
            return titles[:MAX_HEADLINES_PER_DAY]
        except Exception as e:
            if attempt == retries:
                print(f"  [AVISO] falha em {day_str} apos {retries + 1} tentativas: {e}")
                return []
            time.sleep(0.5)
    return []


def main():
    end_date = datetime(2026, 8, 14)  # ultimo pregao disponivel nos dados de preco
    start_date = end_date - timedelta(days=WINDOW_DAYS)
    days = pd.date_range(start_date, end_date, freq="D")

    print(f"Coletando manchetes de {start_date.date()} a {end_date.date()} ({len(days)} dias)...")
    t0 = time.time()

    rows = []
    for i, day in enumerate(days):
        titles = fetch_day_headlines(day)
        rows.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "n_headlines": len(titles),
                "headlines": " | ".join(titles),
            }
        )
        time.sleep(REQUEST_DELAY)

        if (i + 1) % 15 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (len(days) - i - 1)
            print(f"  {i + 1}/{len(days)} dias | {elapsed:.0f}s decorridos | ~{remaining:.0f}s restantes")
            if remaining > 240:
                print("  [AVISO] coleta esta lenta. Considerar reduzir WINDOW_DAYS se necessario.")

    df = pd.DataFrame(rows)
    out_path = f"{DATA_RAW_DIR}/news_headlines.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    total_headlines = df["n_headlines"].sum()
    days_with_news = (df["n_headlines"] > 0).sum()
    print(f"\nConcluido em {time.time() - t0:.0f}s")
    print(f"  Dias com pelo menos 1 manchete: {days_with_news}/{len(days)}")
    print(f"  Total de manchetes coletadas: {total_headlines}")
    print(f"  Salvo em {out_path}")


if __name__ == "__main__":
    main()
