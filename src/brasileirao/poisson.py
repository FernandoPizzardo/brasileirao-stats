from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from brasileirao.models import Match

RHO_BOUND = 0.15


@dataclass(frozen=True, eq=False)
class Model:
    team_ids: list[int]
    attack: np.ndarray
    defence: np.ndarray
    mu: float
    home_adv: float
    rho: float

    def expected_goals(self, home_id: int, away_id: int) -> tuple[float, float]:
        h = self.team_ids.index(home_id)
        a = self.team_ids.index(away_id)
        lam_home = np.exp(self.mu + self.home_adv + self.attack[h] + self.defence[a])
        lam_away = np.exp(self.mu + self.attack[a] + self.defence[h])
        return float(lam_home), float(lam_away)


def _tau(x: np.ndarray, y: np.ndarray, lam_home: np.ndarray, lam_away: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lam_home)
    tau = np.where((x == 0) & (y == 0), 1 - lam_home * lam_away * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lam_home * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + lam_away * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return np.maximum(tau, 1e-10)


def _neg_log_likelihood(theta, home, away, hg, ag, weights, n, ridge) -> float:
    attack = theta[:n]
    defence = theta[n : 2 * n]
    mu, home_adv, rho = theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]
    lam_home = np.exp(mu + home_adv + attack[home] + defence[away])
    lam_away = np.exp(mu + attack[away] + defence[home])
    tau = _tau(hg, ag, lam_home, lam_away, rho)
    ll = (
        np.log(tau)
        + hg * np.log(lam_home) - lam_home - gammaln(hg + 1)
        + ag * np.log(lam_away) - lam_away - gammaln(ag + 1)
    )
    return -np.sum(weights * ll) + ridge * (np.sum(attack**2) + np.sum(defence**2))


def fit(matches: list[Match], as_of: datetime, half_life_days: float = 365.0, ridge: float = 0.1) -> Model:
    """Dixon-Coles style fit: time-decayed weights, ridge on team strengths."""
    played = [m for m in matches if m.is_played and m.utc_date <= as_of]
    if not played:
        raise ValueError("No finished matches to fit the model")

    team_ids = sorted({m.home_id for m in matches} | {m.away_id for m in matches})
    index = {team_id: i for i, team_id in enumerate(team_ids)}
    n = len(team_ids)

    home = np.array([index[m.home_id] for m in played])
    away = np.array([index[m.away_id] for m in played])
    hg = np.array([m.home_goals for m in played])
    ag = np.array([m.away_goals for m in played])
    age_days = np.array([(as_of - m.utc_date).total_seconds() / 86400 for m in played])
    weights = 0.5 ** (age_days / half_life_days)

    x0 = np.zeros(2 * n + 3)
    x0[2 * n] = np.log(1.3)
    bounds = [(None, None)] * (2 * n + 2) + [(-RHO_BOUND, RHO_BOUND)]
    res = minimize(
        _neg_log_likelihood,
        x0,
        args=(home, away, hg, ag, weights, n, ridge),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not res.success:
        raise RuntimeError(f"Model fit did not converge: {res.message}")

    theta = res.x
    return Model(
        team_ids=team_ids,
        attack=theta[:n].copy(),
        defence=theta[n : 2 * n].copy(),
        mu=float(theta[2 * n]),
        home_adv=float(theta[2 * n + 1]),
        rho=float(theta[2 * n + 2]),
    )


def scoreline_pmf(lam_home: float, lam_away: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Joint probability matrix: rows are home goals, columns are away goals."""
    goals = np.arange(max_goals + 1)
    pmf = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
    pmf[0, 0] *= 1 - lam_home * lam_away * rho
    pmf[0, 1] *= 1 + lam_home * rho
    pmf[1, 0] *= 1 + lam_away * rho
    pmf[1, 1] *= 1 - rho
    pmf = np.clip(pmf, 0, None)
    return pmf / pmf.sum()


def outcome_probs(pmf: np.ndarray) -> tuple[float, float, float]:
    home_win = float(np.tril(pmf, -1).sum())
    draw = float(np.trace(pmf))
    away_win = float(np.triu(pmf, 1).sum())
    return home_win, draw, away_win
