import math
import copy
import numpy as np
import random

bid_values = [18, 20, 22, 23, 24, 27, 30, 33, 35, 36, 40, 44, 45, 46, 48, 50,
              54, 55, 59, 60, 63, 66, 70, 72, 77, 80, 81, 84, 88, 90, 96, 99]


def evaluate_hand(hand):

    card_points = {"A": 11, "10": 10, "K": 4, "Q": 3, "J": 2, "9": 0, "8": 0, "7": 0}
    return sum(card_points[card.rank.value] for card in hand)


def evaluate_suit_game(hand, trump_suit):

    base = evaluate_hand(hand)
    bonus = sum(5 for card in hand if card.suit.value.lower() == trump_suit.lower()) \
            + sum(3 for card in hand if card.rank.value == "J")
    return base + bonus


def evaluate_grand_game(hand):

    base = evaluate_hand(hand)
    bonus = sum(3 for card in hand if card.rank.value == "J")
    return base + bonus


def evaluate_null_game(hand):

    return -evaluate_hand(hand)


def evaluate_game_type(hand):
    suits = {}
    for card in hand:
        suits.setdefault(card.suit.value, 0)
        suits[card.suit.value] += 1
    trump_suit = max(suits, key=lambda s: suits[s]) if suits else None
    return {
        "suit": evaluate_suit_game(hand, trump_suit) if trump_suit else evaluate_hand(hand),
        "grand": evaluate_grand_game(hand),
        "null": evaluate_null_game(hand)
    }


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

    def best_child(self, c_param=1.0):
        best, best_score = None, -1e9
        for child in self.children:
            exploit = child.total_reward / child.visits
            explore = c_param * math.sqrt(math.log(self.visits) / (1 + child.visits))
            score = exploit + explore
            if score > best_score:
                best_score, best = score, child
        return best

    def expand(self, action, new_state):
        child_node = Node(new_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward



class BiddingState:
    def __init__(self, current_bid, player_turn, players_in, hand_F, hand_M, hand_R):
        self.current_bid = current_bid
        self.player_turn = player_turn
        self.players_in = players_in[:]
        self.hand_F = hand_F
        self.hand_M = hand_M
        self.hand_R = hand_R

    def get_legal_actions(self):


        idx = bid_values.index(self.current_bid)

        next_bids = bid_values[idx + 1:]


        if self.player_turn == "M":
            threshold = None
        elif self.player_turn == "F":
            threshold = 27
        elif self.player_turn == "R":
            threshold = 40
        else:
            threshold = None


        legal = []
        for bid in next_bids:
            if threshold is None or bid <= threshold:
                legal.append(bid)
                break


        legal.append("pass")
        return legal

    def get_untried_action(self):
        return random.choice(self.get_legal_actions())

    def apply(self, action):
        if action == "pass":
            if self.player_turn in self.players_in:
                self.players_in.remove(self.player_turn)
        else:
            self.current_bid = action
        if self.players_in:
            if self.player_turn in self.players_in:
                current_index = self.players_in.index(self.player_turn)
            else:
                current_index = 0
            self.player_turn = self.players_in[(current_index + 1) % len(self.players_in)]
        return self

    def is_terminal(self):
        return len(self.players_in) <= 1




def rollout_policy(state):

    actions = state.get_legal_actions()
    # pick the bidding hand
    hand = {"M": state.hand_M, "F": state.hand_F, "R": state.hand_R}[state.player_turn]
    game_vals = evaluate_game_type(hand)
    best_val = max(game_vals.values())

    # build score list
    scores = []
    for a in actions:
        if a == "pass":
            scores.append(1.0)
        else:

            scores.append(max(best_val - a, 0.1))
    arr = np.array(scores, dtype=np.float64)

    exp = np.exp(arr - arr.max())
    probs = exp / exp.sum()


    idx = np.random.choice(len(actions), p=probs)
    return actions[int(idx)]


def simulate_rollout(state):

    rollout_state = copy.deepcopy(state)
    while not rollout_state.is_terminal():
        action = rollout_policy(rollout_state)
        rollout_state.apply(action)
    return evaluate_state(rollout_state)



def evaluate_state(state):

    winner = state.players_in[0]
    sign = 1 if winner == "M" else -1


    hand = {"M":state.hand_M,"F":state.hand_F,"R":state.hand_R}[winner]
    game_vals = evaluate_game_type(hand)
    best_val = max(game_vals.values())


    overbid = max(state.current_bid - best_val, 0)

    return sign * (best_val - state.current_bid) - 0.5 * overbid



def mcts_bidding(state, iterations=2000):
    root = Node(state)
    for _ in range(iterations * 2):
        node = root
        sim_state = copy.deepcopy(state)

        # — Selection —
        while not sim_state.is_terminal() and node.is_fully_expanded():
            node = node.best_child(c_param=1.0)
            sim_state.apply(node.action)

        # — Expansion —
        if not sim_state.is_terminal():
            action = sim_state.get_untried_action()
            sim_state.apply(action)
            node = node.expand(action, copy.deepcopy(sim_state))

        # — Simulation via heuristic rollouts —
        reward = simulate_rollout(sim_state)

        # — Backpropagation —
        while node:
            node.update(reward)
            node = node.parent

    # final action (no exploration)
    return root.best_child(c_param=0).action

def interactive_bidding_phase(hand_F, hand_M, hand_R):

    print("=== Skat Bidding Phase ===\n")
    print("Your hand (Middlehand):")
    print(", ".join(str(card) for card in hand_M))
    print("\nStarting bid is: 18")

    # --- Round 1: Only Middlehand (M) and Forehand (F) bid ---
    state = BiddingState(current_bid=18, player_turn="M", players_in=["M", "F"],
                         hand_F=hand_F, hand_M=hand_M, hand_R=hand_R)
    print("\n--- Round 1: Middlehand (You) vs Forehand (F) ---")
    while "M" in state.players_in and "F" in state.players_in:
        if state.player_turn == "M":
            user_input = input(
                f"Your bid (current bid {state.current_bid}). Type a valid number or 'pass': ").strip().lower()
            if user_input == "pass":
                print("You passed.")
                state.apply("pass")
            else:
                try:
                    bid = int(user_input)
                except ValueError:
                    print("Invalid input. Please enter a valid number or 'pass'.")
                    continue
                if bid not in bid_values or bid <= state.current_bid:
                    print("Invalid bid; it must be greater than", state.current_bid)
                    continue
                state.apply(bid)
                print(f"You bid: {bid}")
        else:  # Forehand's turn
            action = mcts_bidding(state, iterations=2000)
            if action == "pass":
                print("Forehand passes.")
            else:
                print(f"Forehand raises the bid to: {action}")
            state.apply(action)
    round1_winner = state.players_in[0]
    print(f"\nRound 1 winner: {round1_winner}")
    print(f"Current bid after Round 1: {state.current_bid}")

    # --- Round 2: Winner vs. Rearhand (R) ---
    print("\n--- Round 2: Winner vs Rearhand (R) ---")
    state_round2 = BiddingState(
        current_bid=state.current_bid,
        player_turn=round1_winner,
        players_in=[round1_winner, "R"],
        hand_F=hand_F,
        hand_M=hand_M,
        hand_R=hand_R
    )
    while len(state_round2.players_in) > 1:
        if state_round2.player_turn == "M":
            user_input = input(
                f"Your bid (current bid {state_round2.current_bid}). Type a valid number or 'pass': ").strip().lower()
            if user_input == "pass":
                print("You passed.")
                state_round2.apply("pass")
            else:
                try:
                    bid = int(user_input)
                except ValueError:
                    print("Invalid input. Please enter a valid number or 'pass'.")
                    continue
                if bid not in bid_values or bid <= state_round2.current_bid:
                    print("Invalid bid; it must be greater than", state_round2.current_bid)
                    continue
                state_round2.apply(bid)
                print(f"You bid: {bid}")
        else:
            action = mcts_bidding(state_round2, iterations=2000)
            if action == "pass":
                print(f"{state_round2.player_turn} passes.")
            else:
                print(f"{state_round2.player_turn} raises the bid to: {action}")
            state_round2.apply(action)
    final_winner = state_round2.players_in[0]
    print(f"\nFinal winner of bidding: {final_winner}")
    print(f"Final bid: {state_round2.current_bid}\n")
    return final_winner, state_round2.current_bid
