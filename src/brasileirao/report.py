import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px

from brasileirao.config import Settings
from brasileirao.formatting import pt_int, pt_num, pt_pct
from brasileirao.service import (
    SCENARIO_PRESETS,
    Context,
    build_context,
    find_focus_team,
    focus_constraints,
    focus_matches,
    preset_choices,
    simulate,
    zones_from,
)
from brasileirao.simulate import position_probabilities, summarize

PERCENT_COLUMNS = ["title", "libertadores", "sula", "relegation"]

COLUMN_LABELS = {
    "team": "Time",
    "scenario": "Cenário",
    "exp_points": "Pontos esperados",
    "exp_position": "Posição esperada",
    "title": "Título",
    "libertadores": "Libertadores",
    "sula": "Sul-Americana",
    "relegation": "Rebaixamento",
}

CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; color: #1a1a1a; }
h1 { margin-bottom: 0.2rem; }
.meta, .note { color: #555; font-size: 0.9rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92rem; }
th, td { padding: 0.35rem 0.6rem; border-bottom: 1px solid #ddd; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f4f4f4; }
tr.focus td { background: #fff4d6; font-weight: 600; }
"""


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in PERCENT_COLUMNS:
        if col in display:
            display[col] = display[col].map(pt_pct)
    for col in ["exp_points", "exp_position"]:
        display[col] = display[col].map(pt_num)
    return display.rename(columns=COLUMN_LABELS)


def _mark_focus(html: str, focus_name: str) -> str:
    pattern = rf"<tr>(\s*<td>{re.escape(escape(focus_name))}</td>)"
    return re.sub(pattern, r'<tr class="focus">\1', html, count=1)


def build_report(settings: Settings, out_dir: Path) -> None:
    ctx: Context = build_context(settings)
    zones = zones_from(settings)
    generated = datetime.now(timezone.utc)

    focus = find_focus_team(ctx.table, settings.focus_team)
    if focus.empty:
        raise ValueError(f"Time em foco '{settings.focus_team}' não encontrado na temporada {settings.season}")
    focus_id = int(focus["team_id"].iloc[0])
    focus_name = focus["team"].iloc[0]
    base_summary = summarize(simulate(ctx, settings), ctx.table, zones)
    odds_html = _mark_focus(
        _format_table(base_summary).drop(columns=["tla"]).to_html(index=False, escape=True, border=0),
        focus_name,
    )

    scenario_rows = []
    for label in SCENARIO_PRESETS:
        choices = preset_choices(ctx, focus_id, label)
        summary = summarize(
            simulate(ctx, settings, focus_constraints(ctx, focus_id, choices)), ctx.table, zones
        )
        row = summary[summary["team"] == focus_name].iloc[0].to_dict()
        row["scenario"] = label
        scenario_rows.append(row)
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_html = _format_table(scenario_df).to_html(index=False, escape=True, border=0)

    result = simulate(ctx, settings)
    probs = position_probabilities(result) * 100
    heatmap = px.imshow(
        probs,
        x=[str(k) for k in range(1, len(result.team_ids) + 1)],
        y=ctx.table["team"].tolist(),
        color_continuous_scale="Blues",
        labels={"x": "Posição final", "y": "Time", "color": "%"},
        aspect="auto",
    )
    heatmap.update_layout(height=700, margin=dict(l=10, r=10, t=30, b=10))
    heatmap_html = heatmap.to_html(full_html=False, include_plotlyjs="cdn")

    page = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulador do Brasileirão {settings.season}</title>
<style>{CSS}</style>
</head>
<body>
<h1>Campeonato Brasileiro Série A {settings.season}: simulador da temporada</h1>
<p class="meta">Atualizado em {generated:%d/%m/%Y %H:%M} UTC · {pt_int(settings.n_simulations)} simulações · {len(ctx.open_matches)} jogos restantes · modelo ajustado com {settings.history_seasons + 1} temporadas</p>
<p class="note">Estimativas de um modelo de Poisson para gols. Use para comparar cenários. Não são previsões.</p>

<h2>Chances de cada time</h2>
{odds_html}

<h2>Probabilidade de cada posição final</h2>
{heatmap_html}

<h2>{escape(focus_name)}: cenários</h2>
<p class="note">Cada cenário fixa os jogos restantes de {escape(focus_name)}. Mandante e visitante podem ter resultados diferentes. Em "Modelo", nenhum jogo é fixado.</p>
{scenario_html}

<p class="note">Fonte dos dados: football-data.org.</p>
</body>
</html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
