import math
import copy
from src.game.state import evaluate_game_type, evaluate_suit_game

# ─── State Classes ───────────────────────────────────────────────────────────────

class GameTypeState:
    def __init__(self, hand):
        self.hand     = hand[:]
        self.selected = None

    def get_legal_actions(self):
        return ["suit", "grand", "null"] if self.selected is None else []

    def apply(self, action):
        if self.selected is None:
            self.selected = action
        return self

    def is_terminal(self):
        return self.selected is not None

    def reward(self):
        values = evaluate_game_type(self.hand)
        return values.get(self.selected, 0)


class TrumpSelectionState:
    def __init__(self, hand):
        self.hand     = hand[:]
        self.selected = None

    def get_legal_actions(self):
        return ["clubs", "spades", "hearts", "diamonds"] if self.selected is None else []

    def apply(self, action):
        if self.selected is None:
            self.selected = action
        return self

    def is_terminal(self):
        return self.selected is not None

    def reward(self):
        return evaluate_suit_game(self.hand, self.selected)

# ─── Rollout Policies ────────────────────────────────────────────────────────────

def rollout_game_type_policy(state: GameTypeState) -> str:
    best, best_val = None, -float("inf")
    for a in state.get_legal_actions():
        tmp = copy.deepcopy(state)
        tmp.apply(a)
        val = tmp.reward()
        if val > best_val:
            best, best_val = a, val
    return best

def rollout_trump_policy(state: TrumpSelectionState) -> str:
    best, best_val = None, -float("inf")
    for a in state.get_legal_actions():
        tmp = copy.deepcopy(state)
        tmp.apply(a)
        val = tmp.reward()
        if val > best_val:
            best, best_val = a, val
    return best

# ─── Generic MCTS Boilerplate ────────────────────────────────────────────────────

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state        = state
        self.parent       = parent
        self.action       = action
        self.children     = []
        self.visits       = 0
        self.total_reward = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def uct_score(self, c_puct=1.4):
        if self.visits == 0:
            return float("inf")
        q = self.total_reward / self.visits
        u = c_puct * math.sqrt(math.log(self.parent.visits) / self.visits)
        return q + u

    def best_child(self, c_puct=1.4):
        return max(self.children, key=lambda n: n.uct_score(c_puct))

    def expand_one(self):
        tried = {c.action for c in self.children}
        for a in self.state.get_legal_actions():
            if a not in tried:
                new_state = copy.deepcopy(self.state)
                new_state.apply(a)
                child = Node(new_state, parent=self, action=a)
                self.children.append(child)
                return child
        raise RuntimeError("No untried actions left")

    def update(self, reward):
        self.visits       += 1
        self.total_reward += reward

def mcts_search(initial_state, iterations=2000, c_puct=1.4):
    root = Node(initial_state)
    for _ in range(iterations * 2):
        node, sim_state = root, copy.deepcopy(initial_state)

        # Selection
        while node.children and not sim_state.is_terminal():
            node = node.best_child(c_puct)
            sim_state.apply(node.action)

        # Expansion
        if not sim_state.is_terminal():
            node = node.expand_one()
            sim_state = copy.deepcopy(node.state)

        # Simulation
        if isinstance(sim_state, GameTypeState):
            a      = rollout_game_type_policy(sim_state)
            sim_state.apply(a)
            reward = sim_state.reward()
        else:
            a      = rollout_trump_policy(sim_state)
            sim_state.apply(a)
            reward = sim_state.reward()

        # Backpropagation
        while node:
            node.update(reward)
            node = node.parent

    return max(root.children, key=lambda n: n.visits).action

# ─── Public API (Config1: ALL players use AI) ───────────────────────────────────

def select_game_type_entry(declarer, ai_hand):
    """Ignore 'declarer'; always use AI for config1."""
    return select_game_type_ai(ai_hand)

def select_game_type_ai(ai_hand):
    """First select game type; if suit, then select trump."""
    gtype = mcts_search(GameTypeState(ai_hand), iterations=2000)
    if gtype != "suit":
        return gtype, None
    trump = mcts_search(TrumpSelectionState(ai_hand), iterations=2000)
    return "suit", trump
