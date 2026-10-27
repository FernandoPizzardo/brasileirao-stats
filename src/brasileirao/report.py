import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px

from brasileirao.config import Settings
from brasileirao.service import Context, build_context, simulate, zones_from
from brasileirao.simulate import position_probabilities, summarize

PERCENT_COLUMNS = ["title", "libertadores", "sula", "relegation"]

SCENARIOS = [
    ("Model", None),
    ("Wins every remaining match", "Win"),
    ("Draws every remaining match", "Draw"),
    ("Loses every remaining match", "Loss"),
]

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


def focus_constraints(ctx: Context, focus_id: int, choice: str | None) -> dict[int, set[str]]:
    if choice is None:
        return {}
    constraints: dict[int, set[str]] = {}
    for m in ctx.open_matches:
        if focus_id not in (m.home_id, m.away_id):
            continue
        is_home = m.home_id == focus_id
        if choice == "Draw":
            allowed = {"D"}
        elif (choice == "Win") == is_home:
            allowed = {"H"}
        else:
            allowed = {"A"}
        constraints[m.id] = allowed
    return constraints


def _odds_table(summary: pd.DataFrame, focus_name: str) -> str:
    display = summary.copy()
    for col in PERCENT_COLUMNS:
        display[col] = display[col].map(lambda v: f"{v:.1%}")
    display["exp_points"] = display["exp_points"].map(lambda v: f"{v:.1f}")
    display["exp_position"] = display["exp_position"].map(lambda v: f"{v:.1f}")
    display = display.rename(
        columns={
            "team": "Team",
            "tla": "TLA",
            "exp_points": "Exp. points",
            "exp_position": "Exp. position",
            "title": "Title",
            "libertadores": "Libertadores",
            "sula": "Sul-Americana",
            "relegation": "Relegation",
        }
    ).drop(columns=["TLA"])
    return display.to_html(index=False, escape=True, border=0)


def _mark_focus(html: str, focus_name: str) -> str:
    pattern = rf"<tr>(\s*<td>{re.escape(escape(focus_name))}</td>)"
    return re.sub(pattern, r'<tr class="focus">\1', html, count=1)


def _scenario_table(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    for col in PERCENT_COLUMNS:
        df[col] = df[col].map(lambda v: f"{v:.1%}")
    df["exp_points"] = df["exp_points"].map(lambda v: f"{v:.1f}")
    df["exp_position"] = df["exp_position"].map(lambda v: f"{v:.1f}")
    df = df.rename(
        columns={
            "scenario": "Scenario",
            "exp_points": "Exp. points",
            "exp_position": "Exp. position",
            "title": "Title",
            "libertadores": "Libertadores",
            "sula": "Sul-Americana",
            "relegation": "Relegation",
        }
    )
    return df.to_html(index=False, escape=True, border=0)


def build_report(settings: Settings, out_dir: Path) -> None:
    ctx = build_context(settings)
    zones = zones_from(settings)
    generated = datetime.now(timezone.utc)

    base_summary = summarize(simulate(ctx, settings), ctx.table, zones)
    focus = ctx.table[ctx.table["team"].str.contains(settings.focus_team, case=False, regex=False)]
    if focus.empty:
        raise ValueError(f"Focus team '{settings.focus_team}' not found in season {settings.season}")
    focus_id = int(focus["team_id"].iloc[0])
    focus_name = focus["team"].iloc[0]

    scenario_rows = []
    for label, choice in SCENARIOS:
        summary = summarize(
            simulate(ctx, settings, focus_constraints(ctx, focus_id, choice)), ctx.table, zones
        )
        row = summary[summary["team"] == focus_name].iloc[0].to_dict()
        row["scenario"] = label
        scenario_rows.append(row)

    result = simulate(ctx, settings)
    probs = position_probabilities(result) * 100
    heatmap = px.imshow(
        probs,
        x=[str(k) for k in range(1, len(result.team_ids) + 1)],
        y=ctx.table["team"].tolist(),
        color_continuous_scale="Blues",
        labels={"x": "Final position", "y": "Team", "color": "%"},
        aspect="auto",
    )
    heatmap.update_layout(height=700, margin=dict(l=10, r=10, t=30, b=10))
    heatmap_html = heatmap.to_html(full_html=False, include_plotlyjs="cdn")

    odds_html = _mark_focus(_odds_table(base_summary, focus_name), focus_name)
    scenario_html = _scenario_table(scenario_rows)

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brasileirão {settings.season} simulator</title>
<style>{CSS}</style>
</head>
<body>
<h1>Campeonato Brasileiro Série A {settings.season}: season simulator</h1>
<p class="meta">Updated {generated:%Y-%m-%d %H:%M} UTC · {settings.n_simulations:,} simulations · {len(ctx.open_matches)} matches left · model fitted on {settings.history_seasons + 1} seasons</p>
<p class="note">Estimates from a Poisson goals model. Use them to compare scenarios. They are not predictions.</p>

<h2>Odds for each team</h2>
{odds_html}

<h2>Final position probabilities</h2>
{heatmap_html}

<h2>{escape(focus_name)}: what-if scenarios</h2>
<p class="note">Each scenario forces every remaining match of {escape(focus_name)} to the stated result.</p>
{scenario_html}

<p class="note">Source: football-data.org.</p>
</body>
</html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
