__authors__ = "Runzhe Yang"
__email__ = ""

from diplomacy import Game
from stance_vector import ActionBasedStance

game = Game()
my_id = 'FRANCE'
action_stance = ActionBasedStance(my_id, game)

game.process()
game.set_orders('FRANCE', ['A MAR H', 'A PAR H', 'F BRE - PIC'])
game.set_orders('ENGLAND', ['A LVP - WAL', 'F EDI - NTH', 'F LON - ENG'])
game.set_orders('GERMANY', ['A BER - MUN', 'A MUN - BUR', 'F KIE - HOL'])
game.process()
print(action_stance.get_stance()['FRANCE'])

game.process()
game.set_orders('FRANCE', ['A MAR - BUR', 'A PAR - BRE', 'F PIC H'])
game.set_orders('ENGLAND', ['A WAL - BEL VIA', 'F ENG C A WAL - BEL', 'F NTH - HEL'])
game.set_orders('GERMANY', ['A BUR - MAR', 'A MUN - RUH', 'F HOL H'])
game.process()
print(action_stance.get_stance()['FRANCE'])

game.process()
game.set_orders('ENGLAND', ['A LON B'])
game.set_orders('GERMANY', ['A MUN B'])
game.process()
print(action_stance.get_stance()['FRANCE'])

game.process()
game.set_orders('FRANCE', ['A BRE H', 'A MAR - GAS', 'F PIC H'])
game.set_orders('ENGLAND', ['A BEL S F PIC', 'F ENG S A BRE', 'F HEL - HOL'])
game.set_orders('GERMANY', ['A BUR - PAR', 'A RUH - BUR', 'F HOL H'])
game.process()
print(action_stance.get_stance()['FRANCE'])
