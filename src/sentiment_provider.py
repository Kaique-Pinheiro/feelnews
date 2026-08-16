"""
Camada de abstracao de provider de LLM para extracao de sentimento.

O resto do pipeline (sentiment_scoring.py) so conhece a funcao publica
get_sentiment(texto) -> dict. O provider concreto (Gemini ou Anthropic)
e escolhido em runtime pela variavel de ambiente LLM_PROVIDER, e pode
ser trocado sem alterar nenhum outro modulo.

IMPORTANTE - vies de look-ahead: este modulo apenas classifica texto
recebido; ele nao sabe nada sobre datas de pregao. O corte temporal
(sentimento do dia t so pode explicar volatilidade de t+1, nunca t)
e responsabilidade de quem consome esta serie (ver src/garch_x.py,
onde a serie de sentimento e explicitamente defasada em 1 dia antes
de entrar na equacao de variancia).

Contrato de saida esperado do LLM (JSON):
{
  "items": [
    {"sentimento": float [-1,1], "intensidade": float [0,1], "categoria_risco": str},
    ...
  ]
}
"""

import hashlib
import json
import os
import re
import time
from collections import deque

from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = "data/processed/sentiment_cache.json"
VALID_CATEGORIES = {"macro", "politico", "setorial", "corporativo"}

SYSTEM_PROMPT = """Voce e um analista quantitativo especializado em mercado financeiro brasileiro.
Voce recebe uma lista numerada de manchetes de noticias. Para CADA manchete, classifique:

- sentimento: float continuo entre -1.0 (extremamente negativo/pessimista para o mercado,
  ex: crise, colapso, forte queda, risco elevado) e +1.0 (extremamente positivo/otimista,
  ex: forte alta, recorde, confianca elevada). 0.0 = neutro/sem viés direcional.
- intensidade: float entre 0.0 (tom ameno, rotineiro) e 1.0 (tom urgente, chocante, extremo)
- categoria_risco: exatamente uma destas strings: "macro", "politico", "setorial", "corporativo"
  (macro = juros/inflacao/cambio/PIB; politico = eleicoes/governo/legislacao;
   setorial = tendencia de um setor inteiro; corporativo = evento de empresa especifica)

Julgue cada manchete de forma independente. Responda APENAS com um JSON no formato:
{"items": [{"sentimento": <float>, "intensidade": <float>, "categoria_risco": "<str>"}, ...]}
na MESMA ORDEM das manchetes recebidas, sem nenhum texto antes ou depois do JSON."""


class RateLimiter:
    """Janela deslizante simples: no maximo max_calls a cada period_seconds."""

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self.timestamps: deque = deque()

    def wait(self):
        now = time.time()
        while self.timestamps and now - self.timestamps[0] > self.period:
            self.timestamps.popleft()
        if len(self.timestamps) >= self.max_calls:
            sleep_for = self.period - (now - self.timestamps[0]) + 0.05
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.timestamps.append(time.time())


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _cache_key(provider: str, model: str, texto: str) -> str:
    raw = f"{provider}::{model}::{SYSTEM_PROMPT}::{texto}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_json_dict(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Nenhum JSON encontrado na resposta: {text[:300]}")
    data = json.loads(match.group(0))
    if "items" not in data or not isinstance(data["items"], list):
        raise ValueError(f"JSON sem chave 'items' valida: {text[:300]}")
    for item in data["items"]:
        item["sentimento"] = max(-1.0, min(1.0, float(item.get("sentimento", 0.0))))
        item["intensidade"] = max(0.0, min(1.0, float(item.get("intensidade", 0.0))))
        cat = str(item.get("categoria_risco", "macro")).lower().strip()
        item["categoria_risco"] = cat if cat in VALID_CATEGORIES else "macro"
    return data


GEMINI_MODEL = "gemini-3.5-flash-lite"  # gemini-2.5-flash-lite foi descontinuado para novas contas
_gemini_limiter = RateLimiter(max_calls=14, period_seconds=60)  # margem sob o limite de 15/min do free tier
_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY nao encontrada no .env")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _gemini_get_sentiment(texto: str, retries: int = 3) -> dict:
    from google.genai import types
    from google.genai.errors import ClientError

    client = _get_gemini_client()
    for attempt in range(retries + 1):
        _gemini_limiter.wait()
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=texto,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            return _parse_json_dict(resp.text)
        except ClientError as e:
            if e.code == 429:
                backoff = 2 ** attempt * 5
                print(f"    [rate limit Gemini] aguardando {backoff}s...")
                time.sleep(backoff)
            elif attempt == retries:
                raise
            else:
                time.sleep(2 ** attempt)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Falha ao obter resposta do Gemini apos retries")


_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY nao encontrada no .env")
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def _anthropic_get_sentiment(texto: str, retries: int = 3) -> dict:
    client = _get_anthropic_client()
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1500,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": texto}],
            )
            return _parse_json_dict(resp.content[0].text)
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Falha ao obter resposta da Anthropic apos retries")


def get_sentiment(texto: str) -> dict:
    """Interface publica e unica que o resto do pipeline deve usar.

    Recebe um bloco de texto (uma ou mais manchetes numeradas) e retorna
    {"items": [{"sentimento": float, "intensidade": float, "categoria_risco": str}, ...]}
    na mesma ordem dos itens presentes no texto de entrada.

    Provider escolhido via env var LLM_PROVIDER ('gemini' | 'anthropic').
    Resultados sao cacheados em disco por hash(provider+model+prompt+texto),
    entao rodar o script de novo nao reprocessa o que ja foi classificado.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = GEMINI_MODEL if provider == "gemini" else "claude-haiku-4-5-20251001"

    cache = _load_cache()
    key = _cache_key(provider, model, texto)
    if key in cache:
        return cache[key]

    if provider == "gemini":
        result = _gemini_get_sentiment(texto)
    elif provider == "anthropic":
        result = _anthropic_get_sentiment(texto)
    else:
        raise ValueError(f"LLM_PROVIDER desconhecido: {provider!r}. Use 'gemini' ou 'anthropic'.")

    cache[key] = result
    _save_cache(cache)
    return result
