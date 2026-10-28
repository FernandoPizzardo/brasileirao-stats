from dataclasses import dataclass
from datetime import datetime, timezone

from brasileirao.models import Match
from brasileirao.report import _mark_focus
from brasileirao.service import SCENARIO_PRESETS, focus_constraints, preset_choices


def open_match(id, home, away):
    return Match(
        id=id,
        season=2026,
        matchday=id,
        utc_date=datetime(2026, 12, 1, tzinfo=timezone.utc),
        status="TIMED",
        home_id=home,
        home_name=f"T{home}",
        away_id=away,
        away_name=f"T{away}",
    )


@dataclass
class FakeContext:
    open_matches: list
    as_of: object = None
    model: object = None
    table: object = None
    fixtures: list = None


FOCUS = 1
CTX = FakeContext(open_matches=[open_match(10, FOCUS, 2), open_match(11, 3, FOCUS), open_match(12, 4, 5)])


def test_home_win_away_loss_preset():
    choices = preset_choices(CTX, FOCUS, "Vence em casa e perde fora")
    assert choices == {10: "Win", 11: "Loss"}
    # Home win is H. Away loss means the home side wins, so also H.
    assert focus_constraints(CTX, FOCUS, choices) == {10: {"H"}, 11: {"H"}}


def test_away_win_home_loss_preset():
    choices = preset_choices(CTX, FOCUS, "Perde em casa e vence fora")
    # Home loss means the away side wins, and an away win for the focus team is also A.
    assert focus_constraints(CTX, FOCUS, choices) == {10: {"A"}, 11: {"A"}}


def test_draw_preset_constrains_both_venues():
    choices = preset_choices(CTX, FOCUS, "Empata todos os jogos restantes")
    assert focus_constraints(CTX, FOCUS, choices) == {10: {"D"}, 11: {"D"}}


def test_model_preset_fixes_nothing():
    assert set(SCENARIO_PRESETS) >= {"Modelo"}
    choices = preset_choices(CTX, FOCUS, "Modelo")
    assert focus_constraints(CTX, FOCUS, choices) == {}


def test_focus_row_is_highlighted():
    html = "<table>\n<tr>\n<td>Outro</td>\n</tr>\n<tr>\n<td>SC Corinthians</td>\n</tr>\n</table>"
    marked = _mark_focus(html, "SC Corinthians")
    assert marked.count('<tr class="focus">') == 1
    assert '<tr class="focus">\n<td>SC Corinthians</td>' in marked
