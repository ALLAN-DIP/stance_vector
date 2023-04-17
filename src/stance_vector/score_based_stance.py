from itertools import product
from typing import Dict

from diplomacy import Game

from .stance_extraction import StanceExtraction


class ScoreBasedStance(StanceExtraction):
    """A turn-level score-based subjective stance vector baseline.

    "Whoever stronger than me must be evil, whoever weaker than me can be my ally."

    Stance on nation k = sign(my score - k's score)
    """

    scores: Dict[str, int]

    def __init__(self, my_identity: str, game: Game) -> None:
        super().__init__(my_identity, game)
        self.scores = {n: 0 for n in self.nations}
        self.stance = {n: {k: 0 for k in self.nations} for n in self.nations}

    def extract_scores(self) -> Dict[str, int]:
        """Extract scores at the end of each round.

        A nation's score is the number of centers it controls.

        Returns a dict of scores for all nations
        """
        return {n: len(self.game.get_centers(n)) for n in self.nations}

    def get_stance(self) -> Dict[str, Dict[str, float]]:  # type: ignore[override]
        """Extract turn-level subjective stance of nation n on nation k.

        Returns a bi-level dictionary of stance score stance[n][k]
        """
        self.scores = self.extract_scores()

        for n, k in product(self.nations, repeat=2):
            if self.scores[n] > 0 and self.scores[n] > self.scores[k]:
                self.stance[n][k] = 1
            elif self.scores[n] > 0 and self.scores[n] < self.scores[k]:
                self.stance[n][k] = -1
            else:
                self.stance[n][k] = 0

        return self.stance
