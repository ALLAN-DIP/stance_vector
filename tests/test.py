__authors__ = "Runzhe Yang"
__email__ = ""

from diplomacy import Game
import numpy as np

from stance_vector import ActionBasedStance

np.random.seed(0)

game = Game()
my_id = 'FRANCE'
action_stance = ActionBasedStance(my_id, game, discount_factor=0.5)

game.set_orders('FRANCE', ['A MAR H', 'A PAR H', 'F BRE - PIC'])
game.set_orders('ENGLAND', ['A LVP - WAL', 'F EDI - NTH', 'F LON - ENG'])
game.set_orders('GERMANY', ['A BER - MUN', 'A MUN - BUR', 'F KIE - HOL'])
game.process()
stances, stance_log = action_stance.get_stance(game, verbose=True)
print(stances['FRANCE'])
print(stance_log['FRANCE'])

game.set_orders('FRANCE', ['A MAR - BUR', 'A PAR - BRE', 'F PIC H'])
game.set_orders('ENGLAND', ['A WAL - BEL VIA', 'F ENG C A WAL - BEL', 'F NTH - HEL'])
game.set_orders('GERMANY', ['A BUR - MAR', 'A MUN - RUH', 'F HOL H'])
game.process()
stances, stance_log = action_stance.get_stance(game, verbose=True)
print(stances['FRANCE'])
print(stance_log['FRANCE'])

game.set_orders('ENGLAND', ['A LON B'])
game.set_orders('GERMANY', ['A MUN B'])
game.process()
stances, stance_log = action_stance.get_stance(game, verbose=True)
print(stances['FRANCE'])
print(stance_log['FRANCE'])

game.set_orders('FRANCE', ['A BRE H', 'A MAR - GAS', 'F PIC H'])
game.set_orders('ENGLAND', ['A BEL S F PIC', 'F ENG S A BRE', 'F HEL - HOL'])
game.set_orders('GERMANY', ['A BUR - PAR', 'A RUH - BUR', 'F HOL H'])
game.process()
stances, stance_log = action_stance.get_stance(game, verbose=True)
print(stances['FRANCE'])
print(stance_log['FRANCE'])


# game.process()
game.set_orders('FRANCE', ['A BRE - PAR', 'A GAS - BUR', 'F PIC - BEL'])
game.set_orders('ENGLAND', ['A BEL - HOL', 'F ENG S F PIC - BEL', 'F HEL S A BEL - HOL'])
game.set_orders('GERMANY', ['A BUR S A PAR - PIC', 'A PAR - PIC', 'F HOL H'])
game.process()
stances, stance_log = action_stance.get_stance(game, verbose=True)
print(stances['FRANCE'])
print(stance_log['FRANCE'])

