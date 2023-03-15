__authors__ = "Runzhe Yang"
__email__ = ""

from abc import ABC, abstractmethod
from copy import deepcopy

from diplomacy import Game
from diplomacy.utils import strings
import numpy as np

"""
    Stance vector modules

    Tested on the turn-level game logs in
    https://github.com/DenisPeskov/2020_acl_diplomacy/blob/master/utils/ExtraGameData.zip

"""


class StanceExtraction(ABC):
    """
    Abstract Base Class for stance vector extraction
    """

    def __init__(self, my_identity: str, game: Game) -> None:
        self.identity = my_identity
        self.nations = list(game.get_map_power_names())
        self.current_round = 0
        self.territories = {n: [] for n in self.nations}
        self.stance = {n: {k: 0.1 for k in self.nations} for n in self.nations}
        self.game = game

    def extract_terr(self):
        """
        Extract current terrirories for each nation from
           game_rec: the turn-level JSON log of a game
        """

        def unit2loc(units):
            locs = []
            for u in units:
                locs.append(u[2:5])
            return locs

        # obtain orderable location from the previous state
        m_phase_data = self.get_prev_m_phase()
        terr = {n: unit2loc(m_phase_data.state["units"][n]) for n in self.nations}
        terr = {n: terr[n] + unit2loc(m_phase_data.state["retreats"][n]) for n in self.nations}
        terr = {
            n: list(np.unique(terr[n] + m_phase_data.state["centers"][n])) for n in self.nations
        }
        return terr

    def get_prev_m_phase(self):
        prev_m_phase_name = None
        phase_hist = self.game.get_phase_history()
        for i in range(len(phase_hist) - 1, -1, -1):
            if phase_hist[i].name[-1] == "M":
                prev_m_phase_name = phase_hist[i].name
                break
        if prev_m_phase_name:
            return self.game.get_phase_history(
                prev_m_phase_name, prev_m_phase_name, self.game.role
            )[0]
        else:
            return None

    @abstractmethod
    def get_stance(self, log, messages) -> dict:
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


class ActionBasedStance(StanceExtraction):
    """
    A turn-level action-based objective stance vector baseline
    "Whoever attacks me is my enemy, whoever supports me is my friend."
    Stance on nation k =  discount* Stance on nation k
        - alpha1 * count(k's hostile moves)
        - alpha2 * count(k's conflict moves)
        - beta1 * k's count(hostile supports/convoys)
        - beta2 * count(k's conflict supports/convoys)
        + gamma1 * count(k's friendly supports/convoys)
        + gamma2 * count(k's unrealized hostile moves)
    """

    def __init__(
        self,
        my_identity,
        game,
        invasion_coef=1.0,
        conflict_coef=0.5,
        invasive_support_coef=1.0,
        conflict_support_coef=0.5,
        friendly_coef=1.0,
        unrealized_coef=1.0,
        discount_factor=0.5,
        end_game_flip=True,
        year_threshold=1918,
        random_betrayal=True,
    ) -> None:
        super().__init__(my_identity, game)
        # hyperparametes weighting different actions
        self.alpha1 = invasion_coef
        self.alpha2 = conflict_coef
        self.discount = discount_factor
        self.beta1 = invasive_support_coef
        self.beta2 = conflict_support_coef
        self.gamma1 = friendly_coef
        self.gamma2 = unrealized_coef
        self.end_game_flip = end_game_flip
        self.year_threshold = year_threshold
        self.random_betrayal = random_betrayal

    def __game_deepcopy__(self, game):
        """Fast deep copy implementation, from Paquette's game engine https://github.com/diplomacy/diplomacy"""
        if game.__class__.__name__ != "Game":
            cls = list(game.__class__.__bases__)[0]
            result = cls.__new__(cls)
        else:
            cls = game.__class__
            result = cls.__new__(cls)
        # Deep copying
        for key in game._slots:
            if key in [
                "map",
                "renderer",
                "powers",
                "channel",
                "notification_callbacks",
                "data",
                "__weakref__",
            ]:
                continue
            setattr(result, key, deepcopy(getattr(game, key)))
        setattr(result, "map", game.map)
        setattr(result, "powers", {})
        for power in game.powers.values():
            result.powers[power.name] = deepcopy(power)
            setattr(result.powers[power.name], "game", result)
        result.role = strings.SERVER_TYPE
        self.game = result

    def order_parser(self, order: str):
        """
        Dipnet order syntax based on
        https://docs.google.com/document/d/16RODa6KDX7vNNooBdciI4NqSVN31lToto3MLTNcEHk0/edit

        The parser will return a tuple:
        (order_type, unit_location, source_location, *target_location)
        """
        order_comp = order.split()
        if len(order_comp) == 3:
            if order_comp[2] in ["H"]:
                return ("HOLD", order_comp[1], order_comp[1])
        elif len(order_comp) == 4:
            if order_comp[2] in ["-", "R"]:
                return ("MOVE", order_comp[1], order_comp[1], order_comp[3])
        elif len(order_comp) == 5:
            if order_comp[2] in ["-"]:
                return ("MOVE", order_comp[1], order_comp[1], order_comp[3])
            elif order_comp[2] in ["S"]:
                return ("SUPPORT", order_comp[1], order_comp[4])
        elif len(order_comp) == 7:
            if order_comp[2] in ["S"]:
                return ("SUPPORT", order_comp[1], order_comp[4], order_comp[6])
            elif order_comp[2] in ["C"]:
                return ("CONVOY", order_comp[1], order_comp[4], order_comp[6])
        return ("UNKNOWN", "UNKNOWN", "UNKNOWN")

    def extract_hostile_moves(self, nation: str):
        """
        Extract hostile moves toward a nation and evaluate
        the hostility scores it holds to other nations
            nation: standing point
            game_rec: the turn-level JSON log of a game
        Returns
            hostility: a dict of hostility move scores of the given nation
            hostile_moves: a list of hostile moves against the given nation
            conflict_moves: a list of conflict moves against the given nation
        """
        hostility = {n: 0 for n in self.nations}
        hostile_moves = []
        conflit_moves = []

        # extract my target cities

        m_phase_data = self.get_prev_m_phase()

        my_targets = []
        my_orders = m_phase_data.orders[nation]
        for order in my_orders:
            order = self.order_parser(order)
            if order[0] == "MOVE":
                target = order[-1]
                if target not in self.territories[nation]:
                    my_targets.append(target)

        # extract other's hostile MOVEs

        for opp in self.nations:
            if opp == nation:
                continue
            opp_orders = m_phase_data.orders[opp]
            if len(opp_orders) == 0:
                continue
            for order in opp_orders:
                order = self.order_parser(order)
                if order[0] == "MOVE":
                    target = order[-1]
                    unit = order[1]
                    # invasion or cut support/convoy
                    if target in self.territories[nation]:
                        hostility[opp] += self.alpha1
                        hostile_moves.append(unit + "-" + target)
                    # seize the same city
                    elif target in my_targets:
                        hostility[opp] += self.alpha2
                        conflit_moves.append(unit + "-" + target)

        return hostility, hostile_moves, conflit_moves

    def extract_hostile_supports(self, nation, hostile_mov, conflit_mov):
        """
        Extract hostile support toward a nation and evaluate
        the hostility scores it holds to other nations
            nation: standing point
            hostile_mov: a list of hostile moves against the given nation
            conflit_mov: a list of conflict moves against the given nation
            game_rec: the turn-level JSON log of a game
        Returns
            hostility: dict of hostility support scores of the given nation
            hostile_supports: list of hostile supports against the given nation
            conflit_supports: list of conflict supports against the given nation
        """
        hostility = {n: 0 for n in self.nations}
        hostile_supports = []
        conflit_supports = []
        m_phase_data = self.get_prev_m_phase()

        # extract other's hostile MOVEs

        for opp in self.nations:
            if opp == nation:
                continue
            opp_orders = m_phase_data.orders[opp]
            if len(opp_orders) == 0:
                continue
            for order in opp_orders:
                order = self.order_parser(order)
                if order[0] in ["SUPPORT", "CONVOY"]:
                    unit = order[1]
                    source = order[2]
                    # if not supporting a HOLD
                    if len(order) > 3:
                        target = order[3]
                        support = source + "-" + target
                        # support invasion or support a cut support/convoy
                        if support in hostile_mov:
                            hostility[opp] += self.beta1
                            hostile_supports.append(unit + ":" + source + "-" + target)
                        # support an attack to seize the same city
                        elif target in conflit_mov:
                            hostility[opp] += self.beta2
                            conflit_supports.append(unit + ":" + source + "-" + target)

        return hostility, hostile_supports, conflit_supports

    def extract_friendly_supports(self, nation):
        """
        Extract friendly support toward a nation and evaluate
        the friend scores it holds to other nations
            nation: standing point
            game_rec: the turn-level JSON log of a game
        Returns
            friendship: dict of friend scores of the given nation
            friendly_supports: list of friendly supports for the given nation
        """
        friendship = {n: 0 for n in self.nations}
        friendly_supports = []
        m_phase_data = self.get_prev_m_phase()
        # extract others' friendly SUPPORT

        for opp in self.nations:
            if opp == nation:
                continue
            opp_orders = m_phase_data.orders[opp]

            if len(opp_orders) == 0:
                continue
            for order in opp_orders:
                order = self.order_parser(order)
                unit = order[1]
                if order[0] in ["SUPPORT", "CONVOY"]:
                    source = order[2]
                    # any kind of support to me
                    if source in self.territories[nation]:
                        friendship[opp] += self.gamma1
                        if len(order) > 3:
                            target = order[3]
                            friendly_supports.append(unit + ":" + source + "-" + target)
                        else:
                            friendly_supports.append(unit + ":" + source)

        return friendship, friendly_supports

    def extract_unrealized_hostile_moves(self, nation: str):
        """
        Extract unrealized hostile moves toward a nation and evaluate
        the friendship scores it holds to other nations
            nation: standing point
        Returns
            friendship:
            unrealized_hostile_moves: a list of potential hostile moves against the given nation
        """
        friendship = {n: 0 for n in self.nations}
        unrealized_hostile_moves = []

        m_phase_data = self.get_prev_m_phase()

        # extract other's unrealized hostile MOVEs

        for opp in self.nations:
            if opp == nation:
                continue
            opp_orders = m_phase_data.orders[opp]
            opp_units = m_phase_data.state["units"][opp]
            adj_pairs = set()
            for opp_unit in opp_units:
                for loc in self.territories[nation]:
                    if opp_unit[0] == "A":
                        if self.game._abuts("A", opp_unit[2:5], "-", loc):
                            adj_pairs.add(opp_unit[2:5] + "-" + loc)

            if len(adj_pairs) > 0:
                friendship[opp] = self.gamma2

            if len(opp_orders) == 0:
                continue
            for order in opp_orders:
                order = self.order_parser(order)
                if order[0] == "MOVE":
                    target = order[-1]
                    unit = order[1]
                    # invasion or cut support/convoy
                    if target in self.territories[nation]:
                        hostile_order = unit + "-" + target
                        if hostile_order in adj_pairs:
                            adj_pairs.remove(hostile_order)
                            friendship[opp] = 0

        return friendship, adj_pairs

    def get_stance(self, game, message=None, verbose=False):
        """
        Extract turn-level objective stance of nation n on nation k.
            game_rec: the turn-level JSON log of a game,
            messages is not used
        Returns a bi-level dictionary of stance score stance[n][k]
        """
        # deepcopy NetworkGame to Game
        self.__game_deepcopy__(game)
        # extract territory info
        self.territories = self.extract_terr()

        # extract hostile moves
        hostility_to, hostile_mov_to, conflit_mov_to = {}, {}, {}
        for n in self.nations:
            hostility_to[n], hostile_mov_to[n], conflit_mov_to[n] = self.extract_hostile_moves(n)

        # extract hostile supports
        hostility_s_to, hostile_sup_to, conflit_sup_to = {}, {}, {}
        for n in self.nations:
            hostility_s_to[n], hostile_sup_to[n], conflit_sup_to[n] = self.extract_hostile_supports(
                n, hostile_mov_to[n], conflit_mov_to[n]
            )

        # extract friendly supports
        friendship_to, friendly_sup_to = {}, {}
        for n in self.nations:
            friendship_to[n], friendly_sup_to[n] = self.extract_friendly_supports(n)

        # extract unrealized hostile moves
        friendship_ur_to, unrealized_move_to = {}, {}
        for n in self.nations:
            friendship_ur_to[n], unrealized_move_to[n] = self.extract_unrealized_hostile_moves(n)

        self.stance_prev = self.stance

        self.stance = {
            n: {
                k: self.discount * self.stance[n][k]
                - hostility_to[n][k]
                - hostility_s_to[n][k]
                + friendship_to[n][k]
                + friendship_ur_to[n][k]
                for k in self.nations
            }
            for n in self.nations
        }

        # simple heuristic to make all other coutries enermy
        if self.end_game_flip:
            m_phase_data = self.get_prev_m_phase()
            if int(m_phase_data.name[1:5]) > self.year_threshold:
                for n in self.nations:
                    for k in self.nations:
                        if self.stance[n][k] > 0:
                            self.stance[n][k] = -1

        # randomly chose one enermy if stance are all postivie
        flipped = {n: {k: False for k in self.nations} for n in self.nations}
        if self.random_betrayal:
            for n in self.nations:
                all_possitive = True
                for k in self.nations:
                    if self.stance[n][k] < 0:
                        all_possitive = False
                if all_possitive:
                    flip_k = np.random.choice([k for k in self.nations if k != n])
                    self.stance[n][flip_k] = -1
                    flipped[n][flip_k] = True

        if not verbose:
            return self.stance
        else:
            log = {n: {k: "" for k in self.nations} for n in self.nations}
            for n in self.nations:
                for k in self.nations:
                    if k == n:
                        continue
                    total = (
                        -hostility_to[n][k]
                        - hostility_s_to[n][k]
                        + friendship_to[n][k]
                        + friendship_ur_to[n][k]
                    )
                    log[n][k] += "My stance to {} decays from {} to {} by a factor {}.".format(
                        k,
                        self.stance_prev[n][k],
                        self.discount * self.stance_prev[n][k],
                        self.discount,
                    )
                    if hostility_to[n][k] != 0:
                        log[n][
                            k
                        ] += "\nMy stance to {} decreases by {} because of their hostile/conflict moves towards me.".format(
                            k, hostility_to[n][k]
                        )
                    if hostility_s_to[n][k] != 0:
                        log[n][
                            k
                        ] += "\nMy stance to {} decreases by {} because of their hostile/conflict support.".format(
                            k, hostility_s_to[n][k]
                        )
                    if friendship_to[n][k] != 0:
                        log[n][
                            k
                        ] += "\nMy stance to {} increases by {} because of receiving their support.".format(
                            k, friendship_to[n][k]
                        )
                    if friendship_ur_to[n][k] > 0:
                        log[n][
                            k
                        ] += "\nMy stance to {} increases by {} because they could attack but didn't.".format(
                            k, friendship_ur_to[n][k]
                        )
                    if friendship_ur_to[n][k] < 0:
                        log[n][
                            k
                        ] += "\nMy stance to {} decreases by {} because of they could be a threat.".format(
                            k, friendship_ur_to[n][k]
                        )
                    if self.random_betrayal and flipped[n][k]:
                        log[n][
                            k
                        ] += "\nMy stance to {} becomes {} because I plan to betray {} to break the peace.".format(
                            k, self.stance[n][k], k
                        )
                    if self.end_game_flip and (int(m_phase_data.name[1:5]) > self.year_threshold):
                        log[n][
                            k
                        ] += "\nMy stance to {} becomes {}, because I plan to betray everyone after year {}.".format(
                            k, friendship_ur_to[n][k], m_phase_data.name[1:5]
                        )
                    log[n][k] += "\n My final stance score to {} is {}.".format(
                        k, self.stance[n][k]
                    )

            return self.stance, log

    def update_stance(self, my_id, opp_id, value):
        """
        Force update the stance value
        could be used when receiving ally proposal
        """
        self.stance[my_id][opp_id] = value


class ScoreBasedStance(StanceExtraction):
    """
    A turn-level score-based subjective stance vector baseline
    "Whoever stronger than me must be evil, whoever weaker than me can be my ally."
    Stance on nation k =
        sign(my score - k's score)
    """

    def __init__(self, my_identity: str, game: Game) -> None:
        super().__init__(my_identity, game)
        self.scores = None
        self.stance = None

    def extract_scores(self):
        """
        Extract scores at the end of each round.
            game_rec: the turn-level JSON log of a game,
        Returns a dict of scores for all nations
        """
        scores = {n: 0 for n in self.nations}
        for n in self.nations:
            scores[n] = len(self.game.powers[n].centers) if self.game.powers[n].centers else 0
        return scores

    def get_stance(self):
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
