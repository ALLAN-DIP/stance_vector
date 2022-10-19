__authors__ = "Runzhe Yang"
__email__ = ""

from diplomacy import Game
from stance_vector import ActionBasedStance

game = Game()
my_id = 'Germany'
action_stance = ActionBasedStance(my_id, game)
print(action_stance.get_stance())