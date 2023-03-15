from typing import Dict, Optional

from diplomacy import Game
import numpy as np

from .stance_extraction import StanceExtraction


class ScoreBasedStance(StanceExtraction):
    """
    A turn-level score-based subjective stance vector baseline
    "Whoever stronger than me must be evil, whoever weaker than me can be my ally."
    Stance on nation k =
        sign(my score - k's score)
    """

    scores: Optional[Dict[str, int]]
    stance: Optional[Dict[str, Dict[str, float]]]  # type: ignore[assignment]

    def __init__(self, my_identity: str, game: Game) -> None:
        super().__init__(my_identity, game)
        self.scores = None
        self.stance = None

    def extract_scores(self) -> Dict[str, int]:
        """
        Extract scores at the end of each round.
            game_rec: the turn-level JSON log of a game,
        Returns a dict of scores for all nations
        """
        scores = {n: 0 for n in self.nations}
        for n in self.nations:
            scores[n] = len(self.game.powers[n].centers) if self.game.powers[n].centers else 0
        return scores

    def get_stance(self) -> Dict[str, Dict[str, float]]:  # type: ignore[override]
        """
        Extract turn-level subjective stance of nation n on nation k.
            game_rec: the turn-level JSON log of a game,
            messages is not used
        Returns a bi-level dictionary of stance score stance[n][k]
        """
        # extract territory info
        self.scores = self.extract_scores()

        self.stance = {
            n: {
                k: np.sign(self.scores[n] - self.scores[k]) if self.scores[n] > 0 else 0
                for k in self.nations
            }
            for n in self.nations
        }

        return self.stance
