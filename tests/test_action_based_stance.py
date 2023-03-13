from diplomacy import Game
import numpy as np

from stance_vector import ActionBasedStance


def test_get_stance():
    np.random.seed(0)

    game = Game()
    my_id = "FRANCE"
    action_stance = ActionBasedStance(my_id, game, discount_factor=0.5)

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
        "AUSTRIA": "My stance to AUSTRIA decays from 0.1 to 0.05 by a factor 0.5.\nMy stance to AUSTRIA becomes -1 because I plan to betray AUSTRIA to break the peace.\n My final stance score to AUSTRIA is -1.",
        "ENGLAND": "My stance to ENGLAND decays from 0.1 to 0.05 by a factor 0.5.\n My final stance score to ENGLAND is 0.05.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from 0.1 to 0.05 by a factor 0.5.\n My final stance score to GERMANY is 0.05.",
        "ITALY": "My stance to ITALY decays from 0.1 to 0.05 by a factor 0.5.\n My final stance score to ITALY is 0.05.",
        "RUSSIA": "My stance to RUSSIA decays from 0.1 to 0.05 by a factor 0.5.\n My final stance score to RUSSIA is 0.05.",
        "TURKEY": "My stance to TURKEY decays from 0.1 to 0.05 by a factor 0.5.\n My final stance score to TURKEY is 0.05.",
    }

    game.set_orders("FRANCE", ["A MAR - BUR", "A PAR - BRE", "F PIC H"])
    game.set_orders("ENGLAND", ["A WAL - BEL VIA", "F ENG C A WAL - BEL", "F NTH - HEL"])
    game.set_orders("GERMANY", ["A BUR - MAR", "A MUN - RUH", "F HOL H"])
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert stances["FRANCE"] == {
        "AUSTRIA": -0.5,
        "ENGLAND": 0.025,
        "FRANCE": 0.025,
        "GERMANY": -0.975,
        "ITALY": 0.025,
        "RUSSIA": 0.025,
        "TURKEY": 0.025,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from -1 to -0.5 by a factor 0.5.\n My final stance score to AUSTRIA is -0.5.",
        "ENGLAND": "My stance to ENGLAND decays from 0.05 to 0.025 by a factor 0.5.\n My final stance score to ENGLAND is 0.025.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from 0.05 to 0.025 by a factor 0.5.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict moves towards me.\n My final stance score to GERMANY is -0.975.",
        "ITALY": "My stance to ITALY decays from 0.05 to 0.025 by a factor 0.5.\n My final stance score to ITALY is 0.025.",
        "RUSSIA": "My stance to RUSSIA decays from 0.05 to 0.025 by a factor 0.5.\n My final stance score to RUSSIA is 0.025.",
        "TURKEY": "My stance to TURKEY decays from 0.05 to 0.025 by a factor 0.5.\n My final stance score to TURKEY is 0.025.",
    }

    game.set_orders("ENGLAND", ["A LON B"])
    game.set_orders("GERMANY", ["A MUN B"])
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert stances["FRANCE"] == {
        "AUSTRIA": -0.25,
        "ENGLAND": 0.0125,
        "FRANCE": 0.0125,
        "GERMANY": -1.4875,
        "ITALY": 0.0125,
        "RUSSIA": 0.0125,
        "TURKEY": 0.0125,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from -0.5 to -0.25 by a factor 0.5.\n My final stance score to AUSTRIA is -0.25.",
        "ENGLAND": "My stance to ENGLAND decays from 0.025 to 0.0125 by a factor 0.5.\n My final stance score to ENGLAND is 0.0125.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from -0.975 to -0.4875 by a factor 0.5.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict moves towards me.\n My final stance score to GERMANY is -1.4875.",
        "ITALY": "My stance to ITALY decays from 0.025 to 0.0125 by a factor 0.5.\n My final stance score to ITALY is 0.0125.",
        "RUSSIA": "My stance to RUSSIA decays from 0.025 to 0.0125 by a factor 0.5.\n My final stance score to RUSSIA is 0.0125.",
        "TURKEY": "My stance to TURKEY decays from 0.025 to 0.0125 by a factor 0.5.\n My final stance score to TURKEY is 0.0125.",
    }

    game.set_orders("FRANCE", ["A BRE H", "A MAR - GAS", "F PIC H"])
    game.set_orders("ENGLAND", ["A BEL S F PIC", "F ENG S A BRE", "F HEL - HOL"])
    game.set_orders("GERMANY", ["A BUR - PAR", "A RUH - BUR", "F HOL H"])
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert stances["FRANCE"] == {
        "AUSTRIA": -0.125,
        "ENGLAND": 3.00625,
        "FRANCE": 0.00625,
        "GERMANY": -1.74375,
        "ITALY": 0.00625,
        "RUSSIA": 0.00625,
        "TURKEY": 0.00625,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from -0.25 to -0.125 by a factor 0.5.\n My final stance score to AUSTRIA is -0.125.",
        "ENGLAND": "My stance to ENGLAND decays from 0.0125 to 0.00625 by a factor 0.5.\nMy stance to ENGLAND increases by 2.0 because of receiving their support.\nMy stance to ENGLAND increases by 1.0 because they could attack but didn't.\n My final stance score to ENGLAND is 3.00625.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from -1.4875 to -0.74375 by a factor 0.5.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict moves towards me.\n My final stance score to GERMANY is -1.74375.",
        "ITALY": "My stance to ITALY decays from 0.0125 to 0.00625 by a factor 0.5.\n My final stance score to ITALY is 0.00625.",
        "RUSSIA": "My stance to RUSSIA decays from 0.0125 to 0.00625 by a factor 0.5.\n My final stance score to RUSSIA is 0.00625.",
        "TURKEY": "My stance to TURKEY decays from 0.0125 to 0.00625 by a factor 0.5.\n My final stance score to TURKEY is 0.00625.",
    }

    game.set_orders("FRANCE", ["A BRE - PAR", "A GAS - BUR", "F PIC - BEL"])
    game.set_orders("ENGLAND", ["A BEL - HOL", "F ENG S F PIC - BEL", "F HEL S A BEL - HOL"])
    game.set_orders("GERMANY", ["A BUR S A PAR - PIC", "A PAR - PIC", "F HOL H"])
    game.process()
    stances, stance_log = action_stance.get_stance(game, verbose=True)
    assert stances["FRANCE"] == {
        "AUSTRIA": -0.0625,
        "ENGLAND": 3.503125,
        "FRANCE": 0.003125,
        "GERMANY": -1.8718750000000002,
        "ITALY": 0.003125,
        "RUSSIA": 0.003125,
        "TURKEY": 0.003125,
    }
    assert stance_log["FRANCE"] == {
        "AUSTRIA": "My stance to AUSTRIA decays from -0.125 to -0.0625 by a factor 0.5.\n My final stance score to AUSTRIA is -0.0625.",
        "ENGLAND": "My stance to ENGLAND decays from 3.00625 to 1.503125 by a factor 0.5.\nMy stance to ENGLAND increases by 1.0 because of receiving their support.\nMy stance to ENGLAND increases by 1.0 because they could attack but didn't.\n My final stance score to ENGLAND is 3.503125.",
        "FRANCE": "",
        "GERMANY": "My stance to GERMANY decays from -1.74375 to -0.871875 by a factor 0.5.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict moves towards me.\nMy stance to GERMANY decreases by 1.0 because of their hostile/conflict support.\nMy stance to GERMANY increases by 1.0 because of receiving their support.\n My final stance score to GERMANY is -1.8718750000000002.",
        "ITALY": "My stance to ITALY decays from 0.00625 to 0.003125 by a factor 0.5.\n My final stance score to ITALY is 0.003125.",
        "RUSSIA": "My stance to RUSSIA decays from 0.00625 to 0.003125 by a factor 0.5.\n My final stance score to RUSSIA is 0.003125.",
        "TURKEY": "My stance to TURKEY decays from 0.00625 to 0.003125 by a factor 0.5.\n My final stance score to TURKEY is 0.003125.",
    }
