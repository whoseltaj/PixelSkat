import math
import random
import copy
from src.game.state import evaluate_game_type

class SkatPhaseState:
    def __init__(self, hand, skat_cards):
        self.hand = hand[:]
        self.skat_cards = skat_cards[:]
        self.picked_up = None

    def get_legal_actions(self):
        if self.picked_up is None:
            return ["pickup", "skip"]
        return []

    def apply(self, action):
        if self.picked_up is None:
            if action == "skip":
                self.picked_up = False
            elif action == "pickup":
                self.picked_up = True
                self.hand.extend(self.skat_cards)
                if not getattr(self, "interactive", False):

                    discard_indices = random.sample(range(len(self.hand)), 2)
                    for i in sorted(discard_indices, reverse=True):
                        self.hand.pop(i)
            else:
                raise ValueError("Invalid action for skat phase: " + str(action))
        return self

    def is_terminal(self):
        return self.picked_up is not None

def rollout_skat_policy(state: SkatPhaseState) -> str:
    base_vals   = evaluate_game_type(state.hand)
    pickup_hand = state.hand + state.skat_cards
    pickup_vals = evaluate_game_type(pickup_hand)

    best_base   = max(base_vals.values())
    best_pickup = max(pickup_vals.values())

    return "pickup" if (best_pickup - best_base) > 3 else "skip"
def simulate_rollout_skat(state: SkatPhaseState) -> float:
    st = copy.deepcopy(state)
    if not st.is_terminal():
        action = rollout_skat_policy(st)
        st.apply(action)
    return max(evaluate_game_type(st.hand).values())


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def uct_score(self, c_puct: float = 1.0) -> float:

        if self.visits == 0:
            return float("inf")
        q = self.total_reward / self.visits
        u = c_puct * math.sqrt(math.log(self.parent.visits) / self.visits)
        return q + u

    def best_child(self, c_param: float = 1.0):

        return max(self.children, key=lambda n: n.uct_score(c_param))

    def expand(self, action, new_state):
        child_node = Node(new_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward


def mcts_skat_phase(initial_state, iterations=2000, c_puct=1.0):

    root = Node(initial_state)

    for _ in range(iterations * 2):
        # 1) SELECTION:
        node = root
        sim_state = copy.deepcopy(initial_state)
        while node.children and not sim_state.is_terminal():
            # pick child by UCT
            node = node.best_child(c_param=c_puct)
            sim_state.apply(node.action)

        # 2) EXPANSION:
        if not sim_state.is_terminal():
            for action in sim_state.get_legal_actions():
                next_state = copy.deepcopy(sim_state)
                next_state.apply(action)
                node.children.append(Node(next_state, parent=node, action=action))

            node = random.choice(node.children)
            sim_state = copy.deepcopy(node.state)

        # 3) SIMULATION:
        reward = simulate_rollout_skat(sim_state)

        # 4) BACKPROPAGATION:
        back = node
        while back is not None:
            back.update(reward)
            back = back.parent


    best = max(root.children, key=lambda n: n.visits)
    return best.action


def interactive_declarer_phase(final_winner, hand, skat_cards, iterations=500, c_puct=1.0):

    print("=== Declarer Skat Phase (AI-driven) ===\n")
    print(f"Initial hand ({len(hand)} cards): {', '.join(str(card) for card in hand)}")
    print(f"Skat cards (face-down): {', '.join(str(card) for card in skat_cards)}\n")

    # Initialize state
    state = SkatPhaseState(hand, skat_cards)
    # Mark interactive so rollout doesn't auto-discard on apply('pickup')
    state.interactive = True

    # Run MCTS to choose "pickup" or "skip"
    action = mcts_skat_phase(state, iterations=iterations, c_puct=c_puct)
    print(f"{final_winner} chooses to '{action}'.\n")

    # Apply the chosen action
    state.apply(action)

    # If pickup, show new 12-card hand, then prompt discard simulation for AI
    if action == "pickup":
        print(f"After pickup, hand ({len(state.hand)} cards): {', '.join(str(card) for card in state.hand)}")
        # Since interactive=True, we must still simulate AI discard:
        # (mcts_skat_phase already did random discard in apply())
        print(f"After discarding 2 cards, hand ({len(state.hand)} cards): {', '.join(str(card) for card in state.hand)}\n")
    else:
        print("Hand game (no skat picked up).\n")

    return state.hand


