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





def interactive_declarer_phase(final_winner, hand, skat_cards):

    print("=== Declarer Skat Phase ===\n")
    print("There are 2 skat cards on the table (face-down).")

    state = SkatPhaseState(hand, skat_cards)

    state.interactive = True
    recommended_action = mcts_skat_phase(state, iterations=500)

    if final_winner == "M":
        print("Your current hand:")
        print(", ".join(str(card) for card in hand))
        print(f"\nMCTS recommends you to '{recommended_action}'.")
        choice = input(
            "Do you want to pick up the skat? Enter 'yes' for pickup or 'no' for Hand game: ").strip().lower()
        user_decision = "pickup" if choice.startswith("y") else "skip"
        print(f"You choose: {user_decision}.\n")
        if user_decision == "pickup":

            state.picked_up = True
            state.hand.extend(state.skat_cards)  # Hand becomes 12 cards.
            print("Your hand after picking up the skat (12 cards):")
            print(f"(Hand length: {len(state.hand)})")
            for i, card in enumerate(state.hand):
                print(f"{i}: {card}", end="    ")
            print("\n")

            while True:
                discard_input = input("Enter two indices (comma-separated) of the cards to discard: ").strip()
                try:
                    indices = [int(x.strip()) for x in discard_input.split(",")]
                    if len(indices) != 2:
                        print("Select exactly 2 cards. Try again.")
                        continue
                    if any(i < 0 or i >= len(state.hand) for i in indices):
                        print("One or both indices out of range. Try again.")
                        continue
                    for i in sorted(indices, reverse=True):
                        discarded_card = state.hand.pop(i)
                        print(f"Discarded {discarded_card}")
                    break
                except Exception:
                    print("Invalid input; please enter two numbers separated by a comma.")
        else:
            print("You choose to play a Hand game (skip picking up the skat).")
    else:
        # AI declarer path:
        print(f"{final_winner} (AI) uses MCTS to decide the skat phase.")
        recommended_action = mcts_skat_phase(state, iterations=500)
        print(f"MCTS recommends '{recommended_action}'.")
        state.apply(recommended_action)
        print(f"{final_winner} updates their hand automatically.\n")

    if final_winner == "M":
        print("Updated hand after skat phase:")
        print(", ".join(str(card) for card in state.hand))
    return state.hand

