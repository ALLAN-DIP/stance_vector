from typing import Any, Dict

from diplomacy import Game
import pytest

from stance_vector import StanceExtraction


class StanceTester(StanceExtraction):
    """Need to make `get_stance` non-abstract for testing."""

    def get_stance(self, log: Any, messages: Any) -> Dict[str, Dict[str, float]]:
        return super().get_stance(log, messages)


def test_extract_terr():
    game = Game()
    my_id = "FRANCE"
    stance = StanceTester(my_id, game)
    game.process()  # Method does not work until first phase ends
    territories = stance.extract_terr()
    assert territories == {
        "AUSTRIA": ["BUD", "TRI", "VIE"],
        "ENGLAND": ["EDI", "LON", "LVP"],
        "FRANCE": ["BRE", "MAR", "PAR"],
        "GERMANY": ["BER", "KIE", "MUN"],
        "ITALY": ["NAP", "ROM", "VEN"],
        "RUSSIA": ["MOS", "SEV", "STP", "WAR"],
        "TURKEY": ["ANK", "CON", "SMY"],
    }


def test_get_prev_m_phase():
    game = Game()
    my_id = "FRANCE"
    stance = StanceTester(my_id, game)
    prev_phase = stance.get_prev_m_phase()
    assert prev_phase is None


def test_get_stance():
    game = Game()
    my_id = "FRANCE"
    stance = StanceTester(my_id, game)
    with pytest.raises(NotImplementedError):
        stance.get_stance(None, None)
