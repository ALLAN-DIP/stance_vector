"""
    Stance vector modules

    Tested on the turn-level game logs in
    https://github.com/DenisPeskov/2020_acl_diplomacy/blob/master/utils/ExtraGameData.zip

"""

from stance_vector.action_based_stance import ActionBasedStance as ActionBasedStance
from stance_vector.score_based_stance import ScoreBasedStance as ScoreBasedStance
from stance_vector.stance_extraction import StanceExtraction as StanceExtraction
