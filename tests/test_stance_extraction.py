from typing import Any, Dict
from unittest.mock import ANY

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
    start_state = {
        "timestamp": ANY,
        "zobrist_hash": "1919110489198082658",
        "note": "",
        "name": "S1901M",
        "units": {
            "AUSTRIA": ["A BUD", "A VIE", "F TRI"],
            "ENGLAND": ["F EDI", "F LON", "A LVP"],
            "FRANCE": ["F BRE", "A MAR", "A PAR"],
            "GERMANY": ["F KIE", "A BER", "A MUN"],
            "ITALY": ["F NAP", "A ROM", "A VEN"],
            "RUSSIA": ["A WAR", "A MOS", "F SEV", "F STP/SC"],
            "TURKEY": ["F ANK", "A CON", "A SMY"],
        },
        "retreats": {
            "AUSTRIA": {},
            "ENGLAND": {},
            "FRANCE": {},
            "GERMANY": {},
            "ITALY": {},
            "RUSSIA": {},
            "TURKEY": {},
        },
        "centers": {
            "AUSTRIA": ["BUD", "TRI", "VIE"],
            "ENGLAND": ["EDI", "LON", "LVP"],
            "FRANCE": ["BRE", "MAR", "PAR"],
            "GERMANY": ["BER", "KIE", "MUN"],
            "ITALY": ["NAP", "ROM", "VEN"],
            "RUSSIA": ["MOS", "SEV", "STP", "WAR"],
            "TURKEY": ["ANK", "CON", "SMY"],
        },
        "homes": {
            "AUSTRIA": ["BUD", "TRI", "VIE"],
            "ENGLAND": ["EDI", "LON", "LVP"],
            "FRANCE": ["BRE", "MAR", "PAR"],
            "GERMANY": ["BER", "KIE", "MUN"],
            "ITALY": ["NAP", "ROM", "VEN"],
            "RUSSIA": ["MOS", "SEV", "STP", "WAR"],
            "TURKEY": ["ANK", "CON", "SMY"],
        },
        "influence": {
            "AUSTRIA": ["BUD", "VIE", "TRI"],
            "ENGLAND": ["EDI", "LON", "LVP"],
            "FRANCE": ["BRE", "MAR", "PAR"],
            "GERMANY": ["KIE", "BER", "MUN"],
            "ITALY": ["NAP", "ROM", "VEN"],
            "RUSSIA": ["WAR", "MOS", "SEV", "STP"],
            "TURKEY": ["ANK", "CON", "SMY"],
        },
        "civil_disorder": {
            "AUSTRIA": 0,
            "ENGLAND": 0,
            "FRANCE": 0,
            "GERMANY": 0,
            "ITALY": 0,
            "RUSSIA": 0,
            "TURKEY": 0,
        },
        "builds": {
            "AUSTRIA": {"count": 0, "homes": []},
            "ENGLAND": {"count": 0, "homes": []},
            "FRANCE": {"count": 0, "homes": []},
            "GERMANY": {"count": 0, "homes": []},
            "ITALY": {"count": 0, "homes": []},
            "RUSSIA": {"count": 0, "homes": []},
            "TURKEY": {"count": 0, "homes": []},
        },
    }
    game = Game()
    my_id = "FRANCE"
    stance = StanceTester(my_id, game)
    prev_phase = stance.get_prev_m_phase()
    assert prev_phase.to_dict() == {
        "name": "S1901M",
        "state": start_state,
        "orders": {
            "AUSTRIA": None,
            "ENGLAND": None,
            "FRANCE": None,
            "GERMANY": None,
            "ITALY": None,
            "RUSSIA": None,
            "TURKEY": None,
        },
        "results": {},
        "messages": [],
    }
    game.process()
    prev_phase = stance.get_prev_m_phase()
    assert prev_phase.to_dict() == {
        "name": "S1901M",
        "state": start_state,
        "orders": {
            "AUSTRIA": [],
            "ENGLAND": [],
            "FRANCE": [],
            "GERMANY": [],
            "ITALY": [],
            "RUSSIA": [],
            "TURKEY": [],
        },
        "results": {
            "A BUD": [],
            "A VIE": [],
            "F TRI": [],
            "F EDI": [],
            "F LON": [],
            "A LVP": [],
            "F BRE": [],
            "A MAR": [],
            "A PAR": [],
            "F KIE": [],
            "A BER": [],
            "A MUN": [],
            "F NAP": [],
            "A ROM": [],
            "A VEN": [],
            "A WAR": [],
            "A MOS": [],
            "F SEV": [],
            "F STP/SC": [],
            "F ANK": [],
            "A CON": [],
            "A SMY": [],
        },
        "messages": [],
    }


def test_get_stance():
    game = Game()
    my_id = "FRANCE"
    stance = StanceTester(my_id, game)
    with pytest.raises(NotImplementedError):
        stance.get_stance(None, None)
