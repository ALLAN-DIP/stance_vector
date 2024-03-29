from diplomacy import Game
from pytest import approx

from stance_vector import ActionBasedStance

RANDOM_SEED = 0


def test_order_parser() -> None:
    game = Game()
    my_id = "FRANCE"
    action_stance = ActionBasedStance(my_id, game, discount_factor=0.5)

    parsed_order = action_stance.order_parser("EAT HAM SAND")
    assert parsed_order == ("UNKNOWN", "UNKNOWN", "UNKNOWN")


def test_get_stance_non_verbose() -> None:
    game = Game()
    my_id = "FRANCE"
    action_stance = ActionBasedStance(my_id, game, discount_factor=0.5, random_seed=RANDOM_SEED)

    # S1901M
    assert game.get_current_phase() == "S1901M"
    game.set_orders("FRANCE", ["A MAR H", "A PAR H", "F BRE - PIC"])
    game.set_orders("ENGLAND", ["A LVP - WAL", "F EDI - NTH", "F LON - ENG"])
    game.set_orders("GERMANY", ["A BER - MUN", "A MUN - BUR", "F KIE - HOL"])
    game.process()
    stances = action_stance.get_stance(game, verbose=False)
    assert stances["FRANCE"] == {
        "AUSTRIA": -1,
        "ENGLAND": 0.05,
        "FRANCE": 0.05,
        "GERMANY": 0.05,
        "ITALY": 0.05,
        "RUSSIA": 0.05,
        "TURKEY": 0.05,
    }


def test_get_stance_long_game() -> None:
    game = Game()
    my_id = "FRANCE"
    action_stance = ActionBasedStance(
        my_id, game, discount_factor=0.5, year_threshold=1902, random_seed=RANDOM_SEED
    )

    # S1901M
    assert game.get_current_phase() == "S1901M"
    game.set_orders("FRANCE", ["A MAR H", "A PAR H", "F BRE - PIC"])
    game.set_orders("ENGLAND", ["A LVP - WAL", "F EDI - NTH", "F LON - ENG"])
    game.set_orders("GERMANY", ["A BER - MUN", "A MUN - BUR", "F KIE - HOL"])
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert stances["FRANCE"] == {
        "AUSTRIA": -1,
        "ENGLAND": 0.05,
        "FRANCE": 0.05,
        "GERMANY": 0.05,
        "ITALY": 0.05,
        "RUSSIA": 0.05,
        "TURKEY": 0.05,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from 0.1 to 0.05 by a factor 0.5.\nMy stance to AUSTRIA becomes -1.0 because I plan to betray AUSTRIA to break the peace.\nMy final stance score to AUSTRIA is -1.0.",
        "ENGLAND": "My stance to ENGLAND decays from 0.1 to 0.05 by a factor 0.5.\nMy final stance score to ENGLAND is 0.05.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from 0.1 to 0.05 by a factor 0.5.\nMy final stance score to GERMANY is 0.05.",
        "ITALY": "My stance to ITALY decays from 0.1 to 0.05 by a factor 0.5.\nMy final stance score to ITALY is 0.05.",
        "RUSSIA": "My stance to RUSSIA decays from 0.1 to 0.05 by a factor 0.5.\nMy final stance score to RUSSIA is 0.05.",
        "TURKEY": "My stance to TURKEY decays from 0.1 to 0.05 by a factor 0.5.\nMy final stance score to TURKEY is 0.05.",
    }

    # F1901M
    assert game.get_current_phase() == "F1901M"
    game.set_orders("FRANCE", ["A MAR - BUR", "A PAR - BRE", "F PIC H"])
    game.set_orders("ENGLAND", ["A WAL - BEL VIA", "F ENG C A WAL - BEL", "F NTH - HEL"])
    game.set_orders("GERMANY", ["A BUR - MAR", "A MUN - RUH", "F HOL H"])
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert approx(stances["FRANCE"]) == {
        "AUSTRIA": -0.5,
        "ENGLAND": 0.025,
        "FRANCE": 0.025,
        "GERMANY": -0.975,
        "ITALY": 0.025,
        "RUSSIA": 0.025,
        "TURKEY": 0.025,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from -1.0 to -0.5 by a factor 0.5.\nMy final stance score to AUSTRIA is -0.5.",
        "ENGLAND": "My stance to ENGLAND decays from 0.05 to 0.025 by a factor 0.5.\nMy final stance score to ENGLAND is 0.025.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from 0.05 to 0.025 by a factor 0.5.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict moves towards me.\nMy final stance score to GERMANY is -0.97.",
        "ITALY": "My stance to ITALY decays from 0.05 to 0.025 by a factor 0.5.\nMy final stance score to ITALY is 0.025.",
        "RUSSIA": "My stance to RUSSIA decays from 0.05 to 0.025 by a factor 0.5.\nMy final stance score to RUSSIA is 0.025.",
        "TURKEY": "My stance to TURKEY decays from 0.05 to 0.025 by a factor 0.5.\nMy final stance score to TURKEY is 0.025.",
    }

    # W1901A
    assert game.get_current_phase() == "W1901A"
    game.set_orders("ENGLAND", ["A LON B"])
    game.set_orders("GERMANY", ["A MUN B"])
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert approx(stances["FRANCE"]) == {
        "AUSTRIA": -0.25,
        "ENGLAND": 0.0125,
        "FRANCE": 0.0125,
        "GERMANY": -1.4875,
        "ITALY": 0.0125,
        "RUSSIA": 0.0125,
        "TURKEY": 0.0125,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from -0.5 to -0.25 by a factor 0.5.\nMy final stance score to AUSTRIA is -0.25.",
        "ENGLAND": "My stance to ENGLAND decays from 0.025 to 0.013 by a factor 0.5.\nMy final stance score to ENGLAND is 0.013.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from -0.97 to -0.49 by a factor 0.5.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict moves towards me.\nMy final stance score to GERMANY is -1.5.",
        "ITALY": "My stance to ITALY decays from 0.025 to 0.013 by a factor 0.5.\nMy final stance score to ITALY is 0.013.",
        "RUSSIA": "My stance to RUSSIA decays from 0.025 to 0.013 by a factor 0.5.\nMy final stance score to RUSSIA is 0.013.",
        "TURKEY": "My stance to TURKEY decays from 0.025 to 0.013 by a factor 0.5.\nMy final stance score to TURKEY is 0.013.",
    }

    # S1902M
    assert game.get_current_phase() == "S1902M"
    game.set_orders("FRANCE", ["A BRE H", "A MAR - GAS", "F PIC H"])
    game.set_orders("ENGLAND", ["A BEL S F PIC", "F ENG S A BRE", "F HEL - HOL"])
    game.set_orders("GERMANY", ["A BUR - PAR", "A RUH - BUR", "F HOL H"])
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert approx(stances["FRANCE"]) == {
        "AUSTRIA": -0.125,
        "ENGLAND": 3.00625,
        "FRANCE": 0.00625,
        "GERMANY": -1.74375,
        "ITALY": 0.00625,
        "RUSSIA": 0.00625,
        "TURKEY": 0.00625,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from -0.25 to -0.12 by a factor 0.5.\nMy final stance score to AUSTRIA is -0.12.",
        "ENGLAND": "My stance to ENGLAND decays from 0.013 to 0.0063 by a factor 0.5.\nMy stance to ENGLAND increases by 2.0 because of receiving their support.\nMy stance to ENGLAND increases by 1.0 because they could attack but didn't.\nMy final stance score to ENGLAND is 3.0.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from -1.5 to -0.74 by a factor 0.5.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict moves towards me.\nMy final stance score to GERMANY is -1.7.",
        "ITALY": "My stance to ITALY decays from 0.013 to 0.0063 by a factor 0.5.\nMy final stance score to ITALY is 0.0063.",
        "RUSSIA": "My stance to RUSSIA decays from 0.013 to 0.0063 by a factor 0.5.\nMy final stance score to RUSSIA is 0.0063.",
        "TURKEY": "My stance to TURKEY decays from 0.013 to 0.0063 by a factor 0.5.\nMy final stance score to TURKEY is 0.0063.",
    }

    # F1902M
    assert game.get_current_phase() == "F1902M"
    game.set_orders("FRANCE", ["A BRE - PAR", "A GAS - BUR", "F PIC - BEL"])
    game.set_orders("ENGLAND", ["A BEL - HOL", "F ENG S F PIC - BEL", "F HEL S A BEL - HOL"])
    game.set_orders("GERMANY", ["A BUR S A PAR - PIC", "A PAR - PIC", "F HOL H"])
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert approx(stances["FRANCE"]) == {
        "AUSTRIA": -0.0625,
        "ENGLAND": 3.503125,
        "FRANCE": 0.003125,
        "GERMANY": -1.871875,
        "ITALY": 0.003125,
        "RUSSIA": 0.003125,
        "TURKEY": 0.003125,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from -0.12 to -0.062 by a factor 0.5.\nMy final stance score to AUSTRIA is -0.062.",
        "ENGLAND": "My stance to ENGLAND decays from 3.0 to 1.5 by a factor 0.5.\nMy stance to ENGLAND increases by 1.0 because of receiving their support.\nMy stance to ENGLAND increases by 1.0 because they could attack but didn't.\nMy final stance score to ENGLAND is 3.5.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from -1.7 to -0.87 by a factor 0.5.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict moves towards me.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict support.\nMy stance to GERMANY increases by 1.0 because of receiving their support.\nMy final stance score to GERMANY is -1.9.",
        "ITALY": "My stance to ITALY decays from 0.0063 to 0.0031 by a factor 0.5.\nMy final stance score to ITALY is 0.0031.",
        "RUSSIA": "My stance to RUSSIA decays from 0.0063 to 0.0031 by a factor 0.5.\nMy final stance score to RUSSIA is 0.0031.",
        "TURKEY": "My stance to TURKEY decays from 0.0063 to 0.0031 by a factor 0.5.\nMy final stance score to TURKEY is 0.0031.",
    }

    # From this point, only need to test end-of-game flip, so no actions are taken

    # F1902R
    assert game.get_current_phase() == "F1902R"
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert approx(stances["FRANCE"]) == {
        "AUSTRIA": -0.03125,
        "ENGLAND": 3.7515625,
        "FRANCE": 0.0015625,
        "GERMANY": -1.9359375,
        "ITALY": 0.0015625,
        "RUSSIA": 0.0015625,
        "TURKEY": 0.0015625,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from -0.062 to -0.031 by a factor 0.5.\nMy final stance score to AUSTRIA is -0.031.",
        "ENGLAND": "My stance to ENGLAND decays from 3.5 to 1.8 by a factor 0.5.\nMy stance to ENGLAND increases by 1.0 because of receiving their support.\nMy stance to ENGLAND increases by 1.0 because they could attack but didn't.\nMy final stance score to ENGLAND is 3.8.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from -1.9 to -0.94 by a factor 0.5.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict moves towards me.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict support.\nMy stance to GERMANY increases by 1.0 because of receiving their support.\nMy final stance score to GERMANY is -1.9.",
        "ITALY": "My stance to ITALY decays from 0.0031 to 0.0016 by a factor 0.5.\nMy final stance score to ITALY is 0.0016.",
        "RUSSIA": "My stance to RUSSIA decays from 0.0031 to 0.0016 by a factor 0.5.\nMy final stance score to RUSSIA is 0.0016.",
        "TURKEY": "My stance to TURKEY decays from 0.0031 to 0.0016 by a factor 0.5.\nMy final stance score to TURKEY is 0.0016.",
    }

    # W1902A
    assert game.get_current_phase() == "W1902A"
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert approx(stances["FRANCE"]) == {
        "AUSTRIA": -0.015625,
        "ENGLAND": 3.87578125,
        "FRANCE": 0.00078125,
        "GERMANY": -1.96796876,
        "ITALY": 0.00078125,
        "RUSSIA": 0.00078125,
        "TURKEY": 0.00078125,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from -0.031 to -0.016 by a factor 0.5.\nMy final stance score to AUSTRIA is -0.016.",
        "ENGLAND": "My stance to ENGLAND decays from 3.8 to 1.9 by a factor 0.5.\nMy stance to ENGLAND increases by 1.0 because of receiving their support.\nMy stance to ENGLAND increases by 1.0 because they could attack but didn't.\nMy final stance score to ENGLAND is 3.9.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from -1.9 to -0.97 by a factor 0.5.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict moves towards me.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict support.\nMy stance to GERMANY increases by 1.0 because of receiving their support.\nMy final stance score to GERMANY is -2.0.",
        "ITALY": "My stance to ITALY decays from 0.0016 to 0.00078 by a factor 0.5.\nMy final stance score to ITALY is 0.00078.",
        "RUSSIA": "My stance to RUSSIA decays from 0.0016 to 0.00078 by a factor 0.5.\nMy final stance score to RUSSIA is 0.00078.",
        "TURKEY": "My stance to TURKEY decays from 0.0016 to 0.00078 by a factor 0.5.\nMy final stance score to TURKEY is 0.00078.",
    }

    # S1903M
    assert game.get_current_phase() == "S1903M"
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert approx(stances["FRANCE"]) == {
        "AUSTRIA": -0.0078125,
        "ENGLAND": -1,
        "FRANCE": -1,
        "GERMANY": -1,
        "ITALY": -1,
        "RUSSIA": -1,
        "TURKEY": -1,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from -0.016 to -0.0078 by a factor 0.5.\nMy final stance score to AUSTRIA is -0.0078.",
        "ENGLAND": "My stance to ENGLAND decays from 3.9 to 1.9 by a factor 0.5.\nMy stance to ENGLAND increases by 1.0 because they could attack but didn't.\nMy stance to ENGLAND becomes -1.0, because I plan to betray everyone after year 1902.\nMy final stance score to ENGLAND is -1.0.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from -2.0 to -0.98 by a factor 0.5.\nMy stance to GERMANY increases by 1.0 because they could attack but didn't.\nMy stance to GERMANY becomes -1.0, because I plan to betray everyone after year 1902.\nMy final stance score to GERMANY is -1.0.",
        "ITALY": "My stance to ITALY decays from 0.00078 to 0.00039 by a factor 0.5.\nMy stance to ITALY becomes -1.0, because I plan to betray everyone after year 1902.\nMy final stance score to ITALY is -1.0.",
        "RUSSIA": "My stance to RUSSIA decays from 0.00078 to 0.00039 by a factor 0.5.\nMy stance to RUSSIA becomes -1.0, because I plan to betray everyone after year 1902.\nMy final stance score to RUSSIA is -1.0.",
        "TURKEY": "My stance to TURKEY decays from 0.00078 to 0.00039 by a factor 0.5.\nMy stance to TURKEY becomes -1.0, because I plan to betray everyone after year 1902.\nMy final stance score to TURKEY is -1.0.",
    }


def test_update_stance() -> None:
    game = Game()
    my_id = "FRANCE"
    action_stance = ActionBasedStance(my_id, game, discount_factor=0.5)

    stances = action_stance.stance
    assert stances["FRANCE"] == {
        "AUSTRIA": 0.1,
        "ENGLAND": 0.1,
        "FRANCE": 0.1,
        "GERMANY": 0.1,
        "ITALY": 0.1,
        "RUSSIA": 0.1,
        "TURKEY": 0.1,
    }

    action_stance.update_stance("FRANCE", "ENGLAND", 0.5)

    stances = action_stance.stance
    assert stances["FRANCE"] == {
        "AUSTRIA": 0.1,
        "ENGLAND": 0.5,
        "FRANCE": 0.1,
        "GERMANY": 0.1,
        "ITALY": 0.1,
        "RUSSIA": 0.1,
        "TURKEY": 0.1,
    }
