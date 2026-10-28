# Simulador do Brasileirão

Simulador de Monte Carlo para o Campeonato Brasileiro Série A. Ajusta um modelo de Poisson para
gols (estilo Dixon-Coles) com resultados passados, simula o restante da temporada e mostra as
chances em um painel interativo (Streamlit). O foco é o Corinthians, com cenários "e se".

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # e preencha FOOTBALL_DATA_TOKEN
```

Um token gratuito está em https://www.football-data.org/client/register.

## Uso

```bash
brasileirao fetch          # atualiza a temporada atual no cache local (data/raw)
brasileirao simulate       # mostra as chances de posição final no terminal
brasileirao backtest       # avalia o modelo em rodadas passadas
brasileirao verify         # confere nossa tabela com a tabela oficial das temporadas passadas
brasileirao report --out site   # gera o relatório HTML estático
streamlit run src/brasileirao/dashboard/app.py
pytest
```

## Configuração

Todas as opções são variáveis de ambiente (ou `.env`). Veja `.env.example`. Os tamanhos das zonas
(`LIBERTADORES_SPOTS`, `SULA_SPOTS`, `RELEGATION_SPOTS`) devem ser conferidos com as regras da
temporada.

## Relatório estático (GitHub Pages)

`brasileirao report --out site` gera uma página com as chances, o mapa de calor das posições e os
cenários do Corinthians. O workflow `.github/workflows/pages.yml` gera e publica a página todos os
dias. Ele precisa do segredo `FOOTBALL_DATA_TOKEN` no repositório.

## Testes e integração contínua

`.github/workflows/ci.yml` roda `pytest` e o build do Docker a cada push e pull request.

## Contêiner

```bash
docker build -t brasileirao .
docker run --rm -p 8501:8501 --env-file .env -v "$PWD/data:/data" brasileirao
```

O contêiner lê a configuração do ambiente e guarda o cache em `/data`, então a imagem não tem
estado.

## Estrutura

- `src/brasileirao/sources/`: clientes de dados (football-data.org)
- `src/brasileirao/standings.py`: tabela e critérios de desempate
- `src/brasileirao/poisson.py`: ajuste do modelo e probabilidades de placar
- `src/brasileirao/simulate.py`: simulador vetorizado da temporada
- `src/brasileirao/service.py`: lógica compartilhada pelo CLI, pelo painel e pelo relatório
- `src/brasileirao/dashboard/app.py`: aplicativo Streamlit
- `src/brasileirao/report.py`: relatório HTML estático
- `src/brasileirao/backtest.py`: avaliação do modelo
