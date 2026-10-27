# Brasileirão simulator

Monte Carlo simulator for the Campeonato Brasileiro Série A. It fits a Poisson goals model
(Dixon-Coles style) on past results, simulates the rest of the season, and shows the odds in an
interactive Streamlit dashboard. The focus is Corinthians, with what-if scenarios.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then add your FOOTBALL_DATA_TOKEN
```

Get a free token at https://www.football-data.org/client/register.

## Usage

```bash
brasileirao fetch           # refresh the current season into data/raw
brasileirao simulate        # print final-position odds in the terminal
streamlit run src/brasileirao/dashboard/app.py
pytest
```

## Static report (GitHub Pages)

`brasileirao report --out site` writes a static HTML page with the odds, the position heatmap, and
Corinthians' scenarios. The workflow in `.github/workflows/pages.yml` builds it every day and
publishes it to GitHub Pages. It needs a repository secret named `FOOTBALL_DATA_TOKEN`.

## Configuration

All settings are environment variables (or `.env`). See `.env.example`. Zone sizes
(`LIBERTADORES_SPOTS`, `SULA_SPOTS`, `RELEGATION_SPOTS`) should be checked against the current
season's rules.

## Containers

```bash
docker build -t brasileirao .
docker run --rm -p 8501:8501 --env-file .env -v "$PWD/data:/data" brasileirao
```

The container reads its configuration from the environment and keeps the data cache in `/data`,
so the image holds no state.

## Layout

- `src/brasileirao/sources/` data clients (football-data.org)
- `src/brasileirao/standings.py` table and tiebreakers
- `src/brasileirao/poisson.py` model fit and scoreline probabilities
- `src/brasileirao/simulate.py` vectorized season simulator
- `src/brasileirao/service.py` orchestration shared by CLI and dashboard
- `src/brasileirao/dashboard/app.py` Streamlit app
