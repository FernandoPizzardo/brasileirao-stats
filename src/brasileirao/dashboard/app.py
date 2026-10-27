import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from brasileirao.config import get_settings
from brasileirao.poisson import outcome_probs, scoreline_pmf
from brasileirao.service import Context, build_context, refresh_cache, simulate, zones_from
from brasileirao.simulate import SimulationResult, position_probabilities, summarize

settings = get_settings()
zones = zones_from(settings)
CHOICES = ["Model", "Win", "Draw", "Loss"]

st.set_page_config(page_title="Brasileirão simulator", layout="wide")


@st.cache_resource(show_spinner="Loading matches and fitting model...")
def get_context() -> Context:
    return build_context(settings)


@st.cache_data(show_spinner="Simulating season...")
def cached_simulation(_ctx: Context, constraints_key: tuple) -> SimulationResult:
    constraints = {match_id: set(outcomes) for match_id, outcomes in constraints_key}
    return simulate(_ctx, settings, constraints or None)


def pct(value: float) -> str:
    return f"{value:.1%}"


def percent_table(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.copy()
    for col in ["title", "libertadores", "sula", "relegation"]:
        display[col] = display[col] * 100
    return display


PERCENT_FORMAT = "%.1f%%"


with st.sidebar:
    st.header("Data")
    if st.button("Refresh from API"):
        with st.spinner("Fetching from football-data.org..."):
            refresh_cache(settings)
        get_context.clear()
        st.rerun()

ctx = get_context()
st.title(f"Campeonato Brasileiro Série A {settings.season} — simulator")
st.caption(
    f"{settings.n_simulations:,} simulations · model fitted {ctx.as_of:%Y-%m-%d %H:%M} UTC · "
    f"{len(ctx.open_matches)} matches left"
)

tab_odds, tab_whatif, tab_match, tab_model = st.tabs(
    ["Odds", "Corinthians what-if", "Match predictor", "Model"]
)

with tab_odds:
    result = cached_simulation(ctx, ())
    summary = summarize(result, ctx.table, zones)
    st.dataframe(
        percent_table(summary),
        hide_index=True,
        width="stretch",
        column_config={
            "exp_points": st.column_config.NumberColumn("Expected points", format="%.1f"),
            "exp_position": st.column_config.NumberColumn("Expected position", format="%.1f"),
            "title": st.column_config.NumberColumn("Title", format=PERCENT_FORMAT),
            "libertadores": st.column_config.NumberColumn("Libertadores", format=PERCENT_FORMAT),
            "sula": st.column_config.NumberColumn("Sul-Americana", format=PERCENT_FORMAT),
            "relegation": st.column_config.NumberColumn("Relegation", format=PERCENT_FORMAT),
        },
    )

    probs = position_probabilities(result) * 100
    teams = ctx.table["team"].tolist()
    fig = px.imshow(
        probs,
        x=[str(k) for k in range(1, len(teams) + 1)],
        y=teams,
        color_continuous_scale="Blues",
        labels={"x": "Final position", "y": "Team", "color": "Probability (%)"},
        aspect="auto",
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")

with tab_whatif:
    focus = ctx.table[ctx.table["team"].str.contains(settings.focus_team, case=False, regex=False)]
    if focus.empty:
        st.warning(f"Focus team '{settings.focus_team}' not found in season {settings.season}.")
    else:
        focus_id = int(focus["team_id"].iloc[0])
        focus_name = focus["team"].iloc[0]
        st.subheader(f"Set outcomes for {focus_name}'s remaining matches")
        st.caption("'Model' keeps the simulated result. Other choices force the outcome in every simulation.")

        remaining = [m for m in ctx.open_matches if focus_id in (m.home_id, m.away_id)]
        constraints: dict[int, set[str]] = {}
        for m in remaining:
            is_home = m.home_id == focus_id
            label = f"Round {m.matchday} · {'vs' if is_home else 'at'} {m.away_name if is_home else m.home_name}"
            choice = st.selectbox(label, CHOICES, key=f"what_if_{m.id}")
            if choice == "Model":
                continue
            if choice == "Draw":
                allowed = {"D"}
            elif (choice == "Win") == is_home:
                allowed = {"H"}
            else:
                allowed = {"A"}
            constraints[m.id] = allowed

        key = tuple(sorted((match_id, "".join(sorted(allowed))) for match_id, allowed in constraints.items()))
        baseline = summarize(cached_simulation(ctx, ()), ctx.table, zones)
        scenario = summarize(cached_simulation(ctx, key), ctx.table, zones)
        b = baseline[baseline["team"] == focus_name].iloc[0]
        s = scenario[scenario["team"] == focus_name].iloc[0]

        cols = st.columns(4)
        cols[0].metric("Expected points", f"{s.exp_points:.1f}", f"{s.exp_points - b.exp_points:+.1f}")
        cols[1].metric("Title", pct(s.title), f"{(s.title - b.title) * 100:+.1f} pp")
        cols[2].metric("Libertadores", pct(s.libertadores), f"{(s.libertadores - b.libertadores) * 100:+.1f} pp")
        cols[3].metric(
            "Relegation",
            pct(s.relegation),
            f"{(s.relegation - b.relegation) * 100:+.1f} pp",
            delta_color="inverse",
        )

        st.markdown("**Final position distribution**")
        result = cached_simulation(ctx, key)
        idx = result.team_ids.index(focus_id)
        dist = position_probabilities(result)[idx] * 100
        fig = px.bar(
            x=list(range(1, len(dist) + 1)),
            y=dist,
            labels={"x": "Final position", "y": "Probability (%)"},
        )
        st.plotly_chart(fig, width="stretch")

with tab_match:
    names = dict(zip(ctx.table["team_id"], ctx.table["team"]))
    team_ids = ctx.table["team_id"].tolist()
    left, right = st.columns(2)
    home = left.selectbox("Home", team_ids, format_func=names.get, index=0)
    away = right.selectbox("Away", team_ids, format_func=names.get, index=1)
    if home == away:
        st.warning("Pick two different teams.")
    else:
        lam_h, lam_a = ctx.model.expected_goals(home, away)
        pmf = scoreline_pmf(lam_h, lam_a, ctx.model.rho)
        p_home, p_draw, p_away = outcome_probs(pmf)
        cols = st.columns(5)
        cols[0].metric(f"{names[home]} win", pct(p_home))
        cols[1].metric("Draw", pct(p_draw))
        cols[2].metric(f"{names[away]} win", pct(p_away))
        cols[3].metric("Expected goals (home)", f"{lam_h:.2f}")
        cols[4].metric("Expected goals (away)", f"{lam_a:.2f}")
        fig = px.imshow(
            pmf[:6, :6] * 100,
            x=[str(g) for g in range(6)],
            y=[str(g) for g in range(6)],
            labels={"x": f"{names[away]} goals", "y": f"{names[home]} goals", "color": "%"},
            color_continuous_scale="Blues",
            text_auto=".1f",
        )
        st.plotly_chart(fig, width="stretch")

with tab_model:
    st.metric("Home advantage (goal multiplier)", f"{np.exp(ctx.model.home_adv):.2f}")
    st.caption(f"Low-score correlation rho: {ctx.model.rho:.3f}")
    strengths = pd.DataFrame(
        {
            "team_id": ctx.model.team_ids,
            "attack": ctx.model.attack,
            "defence (goals allowed)": ctx.model.defence,
        }
    )
    strengths = strengths[strengths["team_id"].isin(names)]
    strengths.insert(0, "team", strengths["team_id"].map(names))
    strengths = strengths.drop(columns="team_id").sort_values("attack", ascending=False, ignore_index=True)
    st.dataframe(strengths, hide_index=True, width="stretch")
