import math
import random

from .card import Card, Suit, Rank
import numpy as np

# 1) Helper: Skat card-point values
CARD_POINTS = {
    Rank.ACE:   11,
    Rank.TEN:   10,
    Rank.KING:   4,
    Rank.QUEEN:  3,
    Rank.JACK:   2,
    Rank.NINE:   0,
    Rank.EIGHT:  0,
    Rank.SEVEN:  0,
}

# 2) Absolute Jack order (♣ > ♠ > ♥ > ♦)
JACK_ORDER = {
    Suit.CLUBS:    4,
    Suit.SPADES:   3,
    Suit.HEARTS:   2,
    Suit.DIAMONDS: 1,
}

class Trick:
    def __init__(self, game_type: str, trump: Suit, leader: str):
        self.game_type = game_type
        self.trump     = trump
        self.leader    = leader
        self.cards     = []
        self.suit_led  = None

    def play(self, player: str, card: Card):
        if not self.cards:
            self.suit_led = card.suit
        self.cards.append((player, card))

    def is_complete(self) -> bool:
        return len(self.cards) == 3

    def winner(self) -> str:
        best_player, best_card = self.cards[0]
        for player, card in self.cards[1:]:
            if _beats(card, best_card, self.suit_led, self.trump, self.game_type):
                best_player, best_card = player, card
        return best_player

    def points(self) -> int:
        return sum(CARD_POINTS[card.rank] for _, card in self.cards)

def _beats(c1: Card, c2: Card, led: Suit, trump: Suit, game_type: str) -> bool:
    if game_type == "null":
        order = [Rank.ACE, Rank.TEN, Rank.KING, Rank.QUEEN,
                 Rank.JACK, Rank.NINE, Rank.EIGHT, Rank.SEVEN]
        return order.index(c1.rank) < order.index(c2.rank)

    # Grand/Suit: Jacks outrank everything
    if c1.rank == Rank.JACK or c2.rank == Rank.JACK:
        if c1.rank == Rank.JACK and c2.rank == Rank.JACK:
            return JACK_ORDER[c1.suit] > JACK_ORDER[c2.suit]
        return c1.rank == Rank.JACK

    # Suit game: trump suit next
    if game_type == "suit":
        if c1.suit == trump and c2.suit != trump:
            return True
        if c2.suit == trump and c1.suit != trump:
            return False

    # Follow-suit
    if c1.suit == led and c2.suit != led:
        return True
    if c2.suit == led and c1.suit != led:
        return False

    order = [Rank.ACE, Rank.TEN, Rank.KING, Rank.QUEEN,
             Rank.NINE, Rank.EIGHT, Rank.SEVEN]
    return order.index(c1.rank) < order.index(c2.rank)


class TrickTakingState:
    def __init__(self,
                 hands: dict,
                 declarer: str,
                 game_type: str,
                 trump: Suit,
                 base_value: int = 1,
                 is_hand: bool = False,
                 is_ouvert: bool = False,
                 contra: bool = False,
                 re: bool = False):
        self.hands            = {p: list(cards) for p, cards in hands.items()}
        self.declarer         = declarer
        self.game_type        = game_type
        self.trump            = trump if game_type == "suit" else None
        self.base_value       = base_value
        self.is_hand          = is_hand
        self.is_ouvert        = is_ouvert
        self.contra           = contra
        self.re               = re
        self.order            = ["F", "M", "R"]
        self.current_player   = "F"
        self.current_trick    = Trick(game_type, self.trump, leader="F")
        self.completed_tricks = []

    def clone(self) -> "TrickTakingState":
        new = TrickTakingState(
            {p: list(cards) for p, cards in self.hands.items()},
            self.declarer,
            self.game_type,
            self.trump,
            base_value=self.base_value,
            is_hand=self.is_hand,
            is_ouvert=self.is_ouvert,
            contra=self.contra,
            re=self.re
        )
        new.order = list(self.order)
        new.current_player = self.current_player
        # clone current trick
        t = self.current_trick
        new.current_trick = Trick(t.game_type, t.trump, t.leader)
        new.current_trick.cards = list(t.cards)
        new.current_trick.suit_led = t.suit_led
        # clone completed tricks
        new.completed_tricks = []
        for old in self.completed_tricks:
            ct = Trick(old.game_type, old.trump, old.leader)
            ct.cards = list(old.cards)
            ct.suit_led = old.suit_led
            new.completed_tricks.append(ct)
        return new

    def get_legal_actions(self):
        hand = self.hands[self.current_player]
        led  = self.current_trick.suit_led

        if led is not None:
            follow = [c for c in hand if c.suit == led]
            if follow:
                return follow

            if self.game_type in ("suit", "grand"):
                trumps = [c for c in hand
                          if c.rank == Rank.JACK
                          or (self.game_type == "suit" and c.suit == self.trump)]
                if trumps:
                    return trumps

        return list(hand)

    def apply(self, card: Card):
        self.hands[self.current_player].remove(card)
        self.current_trick.play(self.current_player, card)
        idx = self.order.index(self.current_player)
        self.current_player = self.order[(idx + 1) % 3]

        if self.current_trick.is_complete():
            winner = self.current_trick.winner()
            self.completed_tricks.append(self.current_trick)
            self.current_trick  = Trick(self.game_type, self.trump, leader=winner)
            self.current_player = winner

        return self

    def is_terminal(self) -> bool:
        return all(len(h) == 0 for h in self.hands.values())

    def _sum_points(self, player: str) -> int:
        return sum(t.points() for t in self.completed_tricks
                   if t.winner() == player)

    def final_result(self) -> dict:
        total_pts = sum(t.points() for t in self.completed_tricks)
        dec_pts   = self._sum_points(self.declarer)
        opp_pts   = total_pts - dec_pts

        if self.game_type == "null":
            made = (opp_pts == 0)
        else:
            made = (dec_pts >= 61)

        mult = 1
        if self.is_hand:   mult += 1
        if self.is_ouvert: mult += 1
        if self.contra:    mult *= 2
        if self.re:        mult *= 2

        # Schneider / Schwarz
        if opp_pts <= 30:
            mult += 1
            if opp_pts == 0:
                mult += 1

        return {
            'declarer_pts':   dec_pts,
            'opp_pts':        opp_pts,
            'made':           made,
            'multiplier':     mult,
            'contract_value': self.base_value * mult
        }

    def to_feature_vector(self) -> np.ndarray:
        vec = np.zeros(255, dtype=np.float32)

        # 1) Hands: 3×32 = 96 dims
        for i, p in enumerate(["F","M","R"]):
            for c in self.hands[p]:
                vec[i*32 + c.index] = 1.0

        # 2) Current trick cards: next 96 dims
        offset = 96
        for j, (_, c) in enumerate(self.current_trick.cards):
            vec[offset + j*32 + c.index] = 1.0

        # 3) Game type: next 3 dims
        offset += 96
        type_map = {"null":0, "grand":1, "suit":2}
        vec[offset + type_map[self.game_type]] = 1.0

        # 4) Trump suit: next 4 dims
        offset += 3
        if self.game_type == "suit" and self.trump is not None:
            suit_map = {Suit.CLUBS:0, Suit.SPADES:1,
                        Suit.HEARTS:2, Suit.DIAMONDS:3}
            vec[offset + suit_map[self.trump]] = 1.0

        # 5) Current player to move: next 3 dims
        offset += 4
        player_map = {"F":0, "M":1, "R":2}
        vec[offset + player_map[self.current_player]] = 1.0

        return vec

def heuristic_play(state: TrickTakingState) -> Card:
    legal = state.get_legal_actions()

    if state.current_trick.cards:
        led = state.current_trick.suit_led
        _, current = state.current_trick.cards[0]
        for _, c in state.current_trick.cards[1:]:
            if _beats(c, current, led, state.trump, state.game_type):
                current = c

        winners = [c for c in legal
                   if _beats(c, current, led, state.trump, state.game_type)]
        if winners:
            return min(winners, key=lambda c: CARD_POINTS[c.rank])

    return min(legal, key=lambda c: CARD_POINTS[c.rank])


def simulate_rollout_tricks(state: TrickTakingState) -> float:
    st = state.clone()
    while not st.is_terminal():
        move = heuristic_play(st)
        st.apply(move)
    return st._sum_points(st.declarer) / 120.0

BIAS_WEIGHT = 0.5

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

class Node:

    def __init__(self, state, parent=None, action=None, prior: float = 0.0):
        self.state    = state
        self.parent   = parent
        self.action   = action
        self.prior    = prior
        self.children = []
        self.visits   = 0
        self.value    = 0.0

    def uct_score(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        bias    = BIAS_WEIGHT * (self.prior / (1 + self.visits))
        return exploit + explore + bias

    def best_child(self, c: float = 1.4):
        return max(self.children, key=lambda n: n.uct_score(c))


def mcts_trick_phase(root_state,
                     iterations: int = 2000,
                     policy=None
                    ) -> Card:

    root = Node(root_state)

    network_priors = {}
    if policy is not None:

        feat = root_state.to_feature_vector()
        logits = policy.predict(feat[None, :])[0]

        legal = root_state.get_legal_actions()

        idxs = [c.index for c in legal]
        legal_logits = logits[idxs]
        priors = _softmax(legal_logits)
        network_priors = dict(zip(legal, priors))

    max_point = max(CARD_POINTS.values())

    for _ in range(iterations * 2):
        node = root
        # 1) SELECTION
        while node.children and not node.state.is_terminal():
            node = node.best_child()

        # 2) EXPANSION
        if not node.state.is_terminal():
            legal_actions = node.state.get_legal_actions()
            for act in legal_actions:

                if node is root and policy is not None:
                    prior = network_priors.get(act, 0.0)
                else:

                    prior = CARD_POINTS[act.rank] / max_point
                child_state = node.state.clone().apply(act)
                node.children.append(
                    Node(child_state, parent=node, action=act, prior=prior)
                )

            node = random.choice(node.children)

        # 3) SIMULATION
        reward = simulate_rollout_tricks(node.state)

        # 4) BACKPROPAGATION
        while node:
            node.visits += 1
            node.value   += reward
            node = node.parent

    if root_state.current_player == root_state.declarer:
        best = max(root.children, key=lambda n: n.value / n.visits)
    else:
        best = min(root.children, key=lambda n: n.value / n.visits)

    return best.action
