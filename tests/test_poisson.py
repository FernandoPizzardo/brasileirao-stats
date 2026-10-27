from datetime import datetime, timedelta, timezone

import numpy as np

from brasileirao.models import Match
from brasileirao.poisson import fit, outcome_probs, scoreline_pmf


def synthetic_matches(cycles=40, seed=0):
    rng = np.random.default_rng(seed)
    attack = np.array([0.4, 0.1, -0.1, -0.4])
    defence = np.zeros(4)
    mu, home_adv = 0.25, 0.2
    as_of = datetime(2026, 10, 1, tzinfo=timezone.utc)
    matches, mid = [], 0
    for cycle in range(cycles):
        for h in range(4):
            for a in range(4):
                if h == a:
                    continue
                lam_h = np.exp(mu + home_adv + attack[h] + defence[a])
                lam_a = np.exp(mu + attack[a] + defence[h])
                mid += 1
                matches.append(
                    Match(
                        id=mid,
                        season=2026,
                        matchday=cycle,
                        utc_date=as_of - timedelta(days=cycle * 7 + h),
                        status="FINISHED",
                        home_id=h + 1,
                        home_name=f"T{h + 1}",
                        away_id=a + 1,
                        away_name=f"T{a + 1}",
                        home_goals=int(rng.poisson(lam_h)),
                        away_goals=int(rng.poisson(lam_a)),
                    )
                )
    return matches, as_of


def test_fit_recovers_strength_ordering():
    matches, as_of = synthetic_matches()
    model = fit(matches, as_of, half_life_days=3650, ridge=0.01)
    assert model.attack[0] > model.attack[1] > model.attack[2] > model.attack[3]
    assert 0.0 < model.home_adv < 0.4


def test_scoreline_probabilities_are_consistent():
    pmf = scoreline_pmf(1.4, 1.1, rho=-0.1)
    assert abs(pmf.sum() - 1) < 1e-9
    home, draw, away = outcome_probs(pmf)
    assert abs(home + draw + away - 1) < 1e-9
    assert home > away
