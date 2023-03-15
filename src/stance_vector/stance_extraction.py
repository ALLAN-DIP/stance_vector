from abc import ABC, abstractmethod
from typing import Any, Dict, List

from diplomacy import Game, GamePhaseData


class StanceExtraction(ABC):
    """Abstract Base Class for stance vector extraction."""

    identity: str
    nations: List[str]
    current_round: int
    territories: Dict[str, List[str]]
    stance: Dict[str, Dict[str, float]]
    game: Game

    def __init__(self, my_identity: str, game: Game) -> None:
        self.identity = my_identity
        self.nations = sorted(game.get_map_power_names())
        self.current_round = 0
        self.territories = {n: [] for n in self.nations}
        self.stance = {n: {k: 0.1 for k in self.nations} for n in self.nations}
        self.game = game

    def extract_terr(self) -> Dict[str, List[str]]:
        """Extract current territories for each nation from the turn-level JSON log of a game."""

        def unit2loc(units: str) -> List[str]:
            return [unit[2:5] for unit in units]

        # Obtain orderable location from the previous state
        m_phase_data = self.get_prev_m_phase()
        terr = {}
        for nation in self.nations:
            locs = (
                unit2loc(m_phase_data.state["units"][nation])
                + unit2loc(m_phase_data.state["retreats"][nation])
                + m_phase_data.state["centers"][nation]
            )
            terr[nation] = sorted(set(locs))
        return terr

    def get_prev_m_phase(self) -> GamePhaseData:
        phase_hist = self.game.get_phase_history()
        prev_m_phase_name = None
        for phase_data in reversed(phase_hist):
            if phase_data.name.endswith("M"):
                prev_m_phase_name = phase_data.name
                break
        if prev_m_phase_name:
            prev_m_phase = self.game.get_phase_from_history(prev_m_phase_name, self.game.role)
        else:
            prev_m_phase = self.game.get_phase_data()
            # Remove private messages between other powers
            prev_m_phase.messages = self.game.filter_messages(prev_m_phase.messages, self.game.role)
        return prev_m_phase

    @abstractmethod
    def get_stance(self, log: Any, messages: Any) -> Dict[str, Dict[str, float]]:
        """
        Abstract method to extract stance of nation n on nation k,
        for all pairs of nations n, k at the current round, given
        the game history and messages
            log: the turn-level JSON log
                 or a list of turn-level logs of the game,
            messages: a dict of dialog lists with other nations in a given round
        Returns a bi-level dictionary stance[n][k]
        """
        raise NotImplementedError()
