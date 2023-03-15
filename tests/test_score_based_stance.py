from diplomacy import Game

from stance_vector import ScoreBasedStance


def test_game():
    game = Game()
    my_id = "FRANCE"
    score_stance = ScoreBasedStance(my_id, game)
    scores = score_stance.extract_scores()
    assert scores == {
        "AUSTRIA": 3,
        "ENGLAND": 3,
        "FRANCE": 3,
        "GERMANY": 3,
        "ITALY": 3,
        "RUSSIA": 4,
        "TURKEY": 3,
    }
    stances = score_stance.get_stance()
    assert stances["FRANCE"] == {
        "AUSTRIA": 0,
        "ENGLAND": 0,
        "FRANCE": 0,
        "GERMANY": 0,
        "ITALY": 0,
        "RUSSIA": -1,
        "TURKEY": 0,
    }

    # S1901M
    assert game.get_current_phase() == "S1901M"
    game.set_orders("FRANCE", ["A MAR H", "A PAR H", "F BRE - PIC"])
    game.set_orders("ENGLAND", ["A LVP - WAL", "F EDI - NTH", "F LON - ENG"])
    game.set_orders("GERMANY", ["A BER - MUN", "A MUN - BUR", "F KIE - HOL"])
    game.process()
    # The scores are unchanged even though Germany captures Holland
    # because the scores are the number of centers each power has,
    # and centers aren't transferred (at least in this game engine)
    # until the adjustments phase.
    scores = score_stance.extract_scores()
    assert scores == {
        "AUSTRIA": 3,
        "ENGLAND": 3,
        "FRANCE": 3,
        "GERMANY": 3,
        "ITALY": 3,
        "RUSSIA": 4,
        "TURKEY": 3,
    }
    stances = score_stance.get_stance()
    assert stances["FRANCE"] == {
        "AUSTRIA": 0,
        "ENGLAND": 0,
        "FRANCE": 0,
        "GERMANY": 0,
        "ITALY": 0,
        "RUSSIA": -1,
        "TURKEY": 0,
    }

    # F1901M
    assert game.get_current_phase() == "F1901M"
    game.set_orders("FRANCE", ["A MAR - BUR", "A PAR - BRE", "F PIC H"])
    game.set_orders("ENGLAND", ["A WAL - BEL VIA", "F ENG C A WAL - BEL", "F NTH - HEL"])
    game.set_orders("GERMANY", ["A BUR - MAR", "A MUN - RUH", "F HOL H"])
    game.process()
    scores = score_stance.extract_scores()
    assert scores == {
        "AUSTRIA": 3,
        "ENGLAND": 4,
        "FRANCE": 3,
        "GERMANY": 4,
        "ITALY": 3,
        "RUSSIA": 4,
        "TURKEY": 3,
    }
    stances = score_stance.get_stance()
    assert stances["FRANCE"] == {
        "AUSTRIA": 0,
        "ENGLAND": -1,
        "FRANCE": 0,
        "GERMANY": -1,
        "ITALY": 0,
        "RUSSIA": -1,
        "TURKEY": 0,
    }

    # W1901A
    assert game.get_current_phase() == "W1901A"
