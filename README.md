# FeelNews

Modelo de previsão de volatilidade do Ibovespa que testa se sentimento extraído de manchetes financeiras por LLM adiciona poder preditivo a um GARCH(1,1) tradicional.

## Hipótese central

A volatilidade do mercado não reage apenas ao histórico de retornos, mas também ao tom do noticiário financeiro. Picos de sentimento negativo poderiam anteceder aumentos de volatilidade ainda não capturados pelos preços, funcionando como proxy de risco percebido. Testamos isso comparando um GARCH(1,1) baseline contra um GARCH-X que inclui sentimento defasado como regressor na equação de variância.

**Resultado, direto ao ponto: a hipótese não se confirma nesta amostra.** O sentimento não adiciona poder explicativo estatisticamente significante, nem in-sample nem out-of-sample. Ver [Resultados](#resultados) e [Limitações](#limitações) para o porquê e o que isso significa (e não significa).

## Dados

- **Ativo**: Ibovespa (`^BVSP`), via `yfinance`
- **Preços**: 5 anos de histórico diário (2021-08-16 a 2026-08-14), 1247 pregões, para o diagnóstico exploratório e o GARCH baseline "de referência"
- **Notícias**: manchetes financeiras sobre o mercado brasileiro, coletadas via Google News RSS (filtro de data dia a dia, sem chave de API), janela de 365 dias corridos (2025-08-15 a 2026-08-14) — 3.450 manchetes em 366 dias, cobertura de 100%
- **Sentimento**: cada manchete classificada por LLM (Gemini `gemini-3.5-flash-lite`, `temperature=0`) em três dimensões: `sentimento` (-1 a +1), `intensidade` (0 a 1) e `categoria_risco` (macro/político/setorial/corporativo)
- **Janela alinhada para GARCH-X**: interseção de retornos e sentimento defasado = 250 pregões (2025-08-15 a 2026-08-14)

## Metodologia

### 1. Diagnóstico exploratório (`src/data_collection.py`)

Log-retornos ($r_t = \ln(P_t/P_{t-1})$) sobre os 5 anos de preços. Confirmação empírica dos pré-requisitos para GARCH:

| Teste | Resultado | Conclusão |
|---|---|---|
| ADF (estacionariedade) | stat=-35.43, p≈0 | Estacionário — apto para GARCH |
| Ljung-Box em r² (clustering de volatilidade) | p=3.9e-06 | Efeito ARCH presente — justifica GARCH sobre vol. constante |

Vol. anualizada da amostra: 17.66% · curtose excedente: 1.03 (caudas pesadas, moderado) · assimetria: -0.09

### 2. GARCH(1,1) baseline "de referência" (`src/garch_baseline.py`)

Ajustado com a biblioteca `arch` sobre os 5 anos completos, distribuição t de Student (justificada pela curtose excedente).

| Parâmetro | Valor | Leitura |
|---|---|---|
| α | 0.0243 | Reação a choques — moderada-baixa |
| β | 0.9705 | Persistência — muito alta (a variância "lembra" de choques por semanas) |
| α + β | 0.9949 | Quase-integrado, mas abaixo do limiar IGARCH (0.999) |
| ν (Student-t) | 10.79 (p<0.001) | Confirma caudas pesadas — normal seria mal especificada |

### 3. Coleta e scoring de sentimento (`src/news_collection.py`, `src/sentiment_scoring.py`, `src/sentiment_provider.py`)

Cada manchete é julgada individualmente pelo LLM (não o dia inteiro de uma vez — evita que o modelo faça sua própria agregação, tarefa não pedida e difícil de auditar). O score diário é a **média** dos scores das manchetes do dia. A camada `sentiment_provider.py` abstrai o provider de LLM (`LLM_PROVIDER=gemini|anthropic`) atrás de uma única função `get_sentiment(texto) -> dict` — o resto do pipeline nunca chama a API diretamente.

Distribuição da categoria de risco dominante por dia (366 dias): macro 321 · corporativo 25 · setorial 11 · político 9 — o noticiário coletado é dominado por eventos macroeconômicos, coerente com a cobertura típica de "Ibovespa" no Google News.

### 4. GARCH-X (`src/garch_x_mle.py`, `src/garch_x.py`)

$$\sigma_t^2 = \omega + \alpha \cdot \varepsilon_{t-1}^2 + \beta \cdot \sigma_{t-1}^2 + \gamma \cdot X_{t-1}$$

**Nota técnica importante**: a biblioteca `arch` não suporta regressor exógeno na equação de variância — o parâmetro `x` de `arch_model()` só é aplicado à equação da média, e é silenciosamente ignorado com `mean='Constant'` (confirmado no código-fonte da lib). Implementamos a MLE manualmente (`scipy.optimize`, inovações t de Student, erros-padrão via Hessiana numérica), com warm-start a partir dos parâmetros do baseline e polimento Nelder-Mead — necessário porque o GARCH-X, sendo um superconjunto do baseline (γ=0 é um caso particular), nunca pode convergir para uma verossimilhança pior; quando isso acontecia, era sinal de falha de otimização, não resultado real.

**Corte anti look-ahead bias**: `X_{t-1}` é sempre o sentimento do dia anterior. O sentimento do dia $t$ nunca entra na equação que explica a variância do dia $t$ — ver comentário explícito em `src/garch_x.py::load_aligned_data()`.

Baseline e GARCH-X foram ambos reajustados na mesma janela de 250 pregões com a mesma rotina de MLE (elimina diferenças de otimizador/software como fator de confusão — a única diferença entre os dois fits é a presença de γ).

### 5. Backtest out-of-sample (`src/backtest.py`)

Split cronológico 80/20 (200 treino / 50 teste). Parâmetros estimados **uma única vez** no treino (não há reestimação diária/janela expansível — simplificação consciente por restrição de tempo, ver [Limitações](#limitações)). Previsões recursivas de 1 passo à frente no teste, usando sempre apenas retornos e sentimento já realizados até o dia anterior. Benchmark: volatilidade realizada em janela móvel de 5 dias (proxy padrão na ausência de dados intradiários). Métrica principal: **QLIKE** (Patton, 2011) — robusta a ruído no proxy de vol. realizada, ao contrário de MSE/MAE. Significância: teste de Diebold-Mariano com erro-padrão Newey-West (HAC).

## Resultados

### In-sample (250 pregões, mesma janela)

| | GARCH(1,1) baseline | GARCH-X |
|---|---|---|
| α | 0.0373 | 0.0368 |
| β | 0.9595 | 0.9622 |
| γ (sentimento) | — | 0.0494 (p=0.203, **não significante**) |
| Log-likelihood | -366.06 | -365.30 |
| AIC | **742.12** | 742.61 |
| BIC | **759.72** | 763.74 |

Teste da razão de verossimilhança (γ=0 vs. γ livre, 1 g.l.): **LR=1.51, p=0.22** — a inclusão do sentimento não melhora o ajuste de forma estatisticamente significante. AIC e BIC preferem o modelo mais simples.

### Out-of-sample (50 pregões de teste)

| Métrica | GARCH(1,1) baseline | GARCH-X |
|---|---|---|
| MSE | 0.4701 | **0.4200** |
| MAE | 0.6042 | **0.5422** |
| QLIKE | 0.2337 | **0.2089** |

Diebold-Mariano (QLIKE): **stat=0.886, p=0.376** — as métricas pontuais favorecem levemente o GARCH-X, mas a diferença **não é estatisticamente significante**.

### Conclusão honesta

O sentimento extraído por LLM, nesta amostra (~1 ano de manchetes, dominadas por notícias macro genéricas do Ibovespa), **não demonstra poder explicativo incremental estatisticamente significante** sobre o GARCH(1,1) — nem in-sample (teste LR) nem out-of-sample (teste DM). O resultado out-of-sample é numericamente favorável ao GARCH-X, mas com N=50 no teste essa diferença é indistinguível de ruído amostral. O resultado correto é **"inconclusivo com leve tendência favorável"**, não "sentimento não funciona" — a amostra é pequena demais para uma afirmação mais forte em qualquer direção.

Ver `outputs/06_metrics_table.png` para a tabela completa e `outputs/04_sentiment_vs_volatility.png` / `outputs/05_backtest_oos_comparison.png` para a inspeção visual.

## Limitações

- **Janela de sentimento pequena para padrões GARCH.** GARCH tipicamente pede 500+ observações para inferência estável; usamos 250. Expandimos de 91 para 365 dias corridos durante o desenvolvimento justamente porque N=63 produzia parâmetros degenerados (α=β=0) — ver histórico de commits. Mesmo com N=250, alguns parâmetros do GARCH-X têm erros-padrão elevados.
- **Sem acesso a arquivo histórico de notícias gratuito.** Google News RSS com filtro de data cobre bem ~1 ano; antes disso a densidade de manchetes cai abruptamente (testamos: 2024-06 retornou 1 manchete/dia vs. ~10/dia em 2025). Isso impede alinhar sentimento com os 5 anos completos de preços usados no GARCH baseline "de referência" — por isso o comparativo justo (Passo 4/5) usa uma subamostra de 250 pregões, não os 5 anos inteiros.
- **Cobertura de notícia dominada por macro.** 321 de 366 dias têm "macro" como categoria de risco dominante — o sinal de sentimento captura muito mais "clima geral do mercado" do que eventos idiossincráticos setoriais/corporativos/políticos, que teoricamente teriam sinal mais forte e mais defasado em relação ao preço.
- **Parâmetros fixos no backtest** (sem reestimação expansível dia a dia) — simplificação por tempo. Um backtest mais rigoroso reestimaria o modelo a cada novo dia de teste.
- **Agregação diária, não intradiária.** O score de sentimento agrega todas as manchetes do dia numa média simples; não há ponderação por horário de publicação em relação ao fechamento do pregão.
- **MLE manual, não a biblioteca `arch`.** Implementação própria testada e validada (warm-start + multi-start + polimento Nelder-Mead para evitar mínimos locais), mas não passou pelo mesmo nível de escrutínio de anos de uso em produção que a biblioteca `arch` teve para o caso sem regressor exógeno.
- **`gemini-2.5-flash-lite`, o modelo originalmente planejado, foi descontinuado para novas contas** durante o desenvolvimento; usamos `gemini-3.5-flash-lite` (ver commits).

## Estrutura do projeto

```
FeelNews/
├── src/
│   ├── data_collection.py      # Passo 1: preços, log-retornos, ADF, Ljung-Box
│   ├── garch_baseline.py       # Passo 2: GARCH(1,1) via biblioteca arch (5 anos)
│   ├── news_collection.py      # Passo 3a: manchetes via Google News RSS
│   ├── sentiment_provider.py   # Passo 3b: abstração de provider LLM (Gemini/Anthropic)
│   ├── sentiment_scoring.py    # Passo 3b: agregação diária de sentimento
│   ├── garch_x_mle.py          # Passo 4: MLE manual do GARCH-X
│   ├── garch_x.py              # Passo 4: fit + comparação baseline vs. GARCH-X
│   ├── backtest.py             # Passo 5: split treino/teste, MSE/MAE/QLIKE, Diebold-Mariano
│   └── visualizations.py       # Passo 6: gráficos finais
├── data/
│   ├── raw/                    # preços e manchetes brutos (não versionado)
│   └── processed/              # séries derivadas, parâmetros, métricas
├── outputs/                    # gráficos e tabela em alta resolução (PNG, 220 dpi)
├── notebooks/
├── requirements.txt
└── .env                        # ANTHROPIC_API_KEY / GEMINI_API_KEY (não versionado)
```

## Como rodar

```bash
pip install -r requirements.txt
```

Crie um `.env` na raiz com:
```
GEMINI_API_KEY=sua-chave
LLM_PROVIDER=gemini
```

Execute em ordem (cada passo lê os artefatos salvos pelo anterior em `data/processed/`):

```bash
python src/data_collection.py
python src/garch_baseline.py
python src/news_collection.py
python src/sentiment_scoring.py
python src/garch_x.py
python src/backtest.py
python src/visualizations.py
```

## Tecnologias

Python 3.14 · `yfinance` · `arch` · `statsmodels` · `scipy` · `pandas`/`numpy` · `matplotlib` · `google-genai` (Gemini) / `anthropic` (Claude) para extração de sentimento.
