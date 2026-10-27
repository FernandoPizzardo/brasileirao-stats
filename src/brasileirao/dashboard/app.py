import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from brasileirao.config import get_settings
from brasileirao.formatting import pt_int, pt_num, pt_pct, pt_signed
from brasileirao.poisson import outcome_probs, scoreline_pmf
from brasileirao.service import (
    Context,
    build_context,
    find_focus_team,
    focus_constraints,
    focus_matches,
    refresh_cache,
    simulate,
    zones_from,
)
from brasileirao.simulate import SimulationResult, position_probabilities, summarize

settings = get_settings()
zones = zones_from(settings)

OPTIONS = ["Model", "Win", "Draw", "Loss"]
OPTION_LABELS = {"Model": "Modelo", "Win": "Vitória", "Draw": "Empate", "Loss": "Derrota"}

st.set_page_config(page_title="Simulador do Brasileirão", layout="wide")


@st.cache_resource(show_spinner="Carregando partidas e ajustando o modelo...")
def get_context() -> Context:
    return build_context(settings)


@st.cache_data(show_spinner="Simulando a temporada...")
def cached_simulation(_ctx: Context, choices_key: tuple) -> SimulationResult:
    constraints = {match_id: set(outcomes) for match_id, outcomes in choices_key}
    return simulate(_ctx, settings, constraints or None)


def percent_table(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.copy()
    for col in ["title", "libertadores", "sula", "relegation"]:
        display[col] = display[col] * 100
    return display


PERCENT_FORMAT = "%.1f%%"

with st.sidebar:
    st.header("Dados")
    if st.button("Atualizar pela API"):
        with st.spinner("Buscando dados em football-data.org..."):
            refresh_cache(settings)
        get_context.clear()
        st.rerun()

ctx = get_context()
focus = find_focus_team(ctx.table, settings.focus_team)
focus_name = focus["team"].iloc[0] if not focus.empty else None

st.title(f"Campeonato Brasileiro Série A {settings.season} — simulador")
st.caption(
    f"{pt_int(settings.n_simulations)} simulações · modelo ajustado em {ctx.as_of:%d/%m/%Y %H:%M} UTC · "
    f"{len(ctx.open_matches)} jogos restantes"
)

tab_odds, tab_whatif, tab_match, tab_model = st.tabs(
    [
        "Classificação e chances",
        f"Cenários de {focus_name}" if focus_name else "Cenários",
        "Previsão de partida",
        "Modelo",
    ]
)

with tab_odds:
    result = cached_simulation(ctx, ())
    summary = summarize(result, ctx.table, zones)
    st.dataframe(
        percent_table(summary).rename(
            columns={
                "team": "Time",
                "tla": "Sigla",
                "exp_points": "Pontos esperados",
                "exp_position": "Posição esperada",
                "title": "Título",
                "libertadores": "Libertadores",
                "sula": "Sul-Americana",
                "relegation": "Rebaixamento",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Pontos esperados": st.column_config.NumberColumn(format="%.1f"),
            "Posição esperada": st.column_config.NumberColumn(format="%.1f"),
            "Título": st.column_config.NumberColumn(format=PERCENT_FORMAT),
            "Libertadores": st.column_config.NumberColumn(format=PERCENT_FORMAT),
            "Sul-Americana": st.column_config.NumberColumn(format=PERCENT_FORMAT),
            "Rebaixamento": st.column_config.NumberColumn(format=PERCENT_FORMAT),
        },
    )

    probs = position_probabilities(result) * 100
    fig = px.imshow(
        probs,
        x=[str(k) for k in range(1, len(ctx.table) + 1)],
        y=ctx.table["team"].tolist(),
        color_continuous_scale="Blues",
        labels={"x": "Posição final", "y": "Time", "color": "Probabilidade (%)"},
        aspect="auto",
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")

with tab_whatif:
    if focus.empty:
        st.warning(f"Time em foco '{settings.focus_team}' não encontrado na temporada {settings.season}.")
    else:
        focus_id = int(focus["team_id"].iloc[0])
        st.subheader(f"Defina os resultados dos jogos restantes de {focus_name}")
        st.caption("'Modelo' mantém o resultado simulado. As outras opções fixam o resultado em todas as simulações.")

        choices: dict[int, str] = {}
        for m in focus_matches(ctx, focus_id):
            is_home = m.home_id == focus_id
            opponent = m.away_name if is_home else m.home_name
            label = f"Rodada {m.matchday} · {'em casa contra' if is_home else 'fora contra'} {opponent}"
            choice = st.selectbox(label, OPTIONS, format_func=OPTION_LABELS.get, key=f"what_if_{m.id}")
            if choice != "Model":
                choices[m.id] = choice

        constraints = focus_constraints(ctx, focus_id, choices)
        key = tuple(sorted((match_id, "".join(sorted(allowed))) for match_id, allowed in constraints.items()))
        baseline = summarize(cached_simulation(ctx, ()), ctx.table, zones)
        scenario = summarize(cached_simulation(ctx, key), ctx.table, zones)
        b = baseline[baseline["team"] == focus_name].iloc[0]
        s = scenario[scenario["team"] == focus_name].iloc[0]

        cols = st.columns(4)
        cols[0].metric("Pontos esperados", pt_num(s.exp_points), pt_signed(s.exp_points - b.exp_points))
        cols[1].metric("Título", pt_pct(s.title), f"{pt_signed((s.title - b.title) * 100)} p.p.")
        cols[2].metric("Libertadores", pt_pct(s.libertadores), f"{pt_signed((s.libertadores - b.libertadores) * 100)} p.p.")
        cols[3].metric(
            "Rebaixamento",
            pt_pct(s.relegation),
            f"{pt_signed((s.relegation - b.relegation) * 100)} p.p.",
            delta_color="inverse",
        )

        st.markdown("**Distribuição da posição final**")
        idx = cached_simulation(ctx, key).team_ids.index(focus_id)
        dist = position_probabilities(cached_simulation(ctx, key))[idx] * 100
        fig = px.bar(
            x=list(range(1, len(dist) + 1)),
            y=dist,
            labels={"x": "Posição final", "y": "Probabilidade (%)"},
        )
        st.plotly_chart(fig, width="stretch")

with tab_match:
    names = dict(zip(ctx.table["team_id"], ctx.table["team"]))
    team_ids = ctx.table["team_id"].tolist()
    left, right = st.columns(2)
    home = left.selectbox("Mandante", team_ids, format_func=names.get, index=0)
    away = right.selectbox("Visitante", team_ids, format_func=names.get, index=1)
    if home == away:
        st.warning("Escolha dois times diferentes.")
    else:
        lam_h, lam_a = ctx.model.expected_goals(home, away)
        pmf = scoreline_pmf(lam_h, lam_a, ctx.model.rho)
        p_home, p_draw, p_away = outcome_probs(pmf)
        cols = st.columns(5)
        cols[0].metric(f"Vitória de {names[home]}", pt_pct(p_home))
        cols[1].metric("Empate", pt_pct(p_draw))
        cols[2].metric(f"Vitória de {names[away]}", pt_pct(p_away))
        cols[3].metric("Gols esperados (mandante)", pt_num(lam_h, 2))
        cols[4].metric("Gols esperados (visitante)", pt_num(lam_a, 2))
        fig = px.imshow(
            pmf[:6, :6] * 100,
            x=[str(g) for g in range(6)],
            y=[str(g) for g in range(6)],
            labels={"x": f"Gols de {names[away]}", "y": f"Gols de {names[home]}", "color": "%"},
            color_continuous_scale="Blues",
            text_auto=".1f",
        )
        st.plotly_chart(fig, width="stretch")

with tab_model:
    st.metric("Vantagem de jogar em casa (multiplicador de gols)", pt_num(np.exp(ctx.model.home_adv), 2))
    st.caption(f"Correlação em placares baixos (rho): {pt_num(ctx.model.rho, 3)}")
    strengths = pd.DataFrame(
        {
            "team_id": ctx.model.team_ids,
            "attack": ctx.model.attack,
            "defence": ctx.model.defence,
        }
    )
    strengths = strengths[strengths["team_id"].isin(names)]
    strengths.insert(0, "team", strengths["team_id"].map(names))
    strengths = (
        strengths.drop(columns="team_id")
        .rename(columns={"attack": "Ataque", "defence": "Defesa (gols sofridos)", "team": "Time"})
        .sort_values("Ataque", ascending=False, ignore_index=True)
    )
    st.dataframe(strengths, hide_index=True, width="stretch")
