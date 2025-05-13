# File: src/game/nn/generate_data.py
import os
import numpy as np
from tqdm import tqdm

from src.game.deck         import Deck
from src.game.trick_phase  import TrickTakingState, mcts_trick_phase
from src.game.card         import Suit, Rank
from src.game.nn.model     import card_to_index, encode_action


SUITS = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]

def random_contract_and_deal():

    deck = Deck()
    deck.shuffle()
    hF, hM, hR = deck.deal(num_hands=3, cards_per_hand=10)
    declarer = np.random.choice(["F", "M", "R"])
    game_type = np.random.choice(["suit", "grand", "null"])
    trump = None
    if game_type == "suit":
        trump = np.random.choice(list(Suit))
    return {"F": hF, "M": hM, "R": hR}, declarer, game_type, trump

def encode_state_custom(state: TrickTakingState,
                        declarer: str,
                        skat_picked: bool = False) -> np.ndarray:

    V = np.zeros(255, dtype=np.float32)
    o = 0

    # 1) Current player's hand
    cur = state.current_player
    hand_mask = np.zeros(32, dtype=np.float32)
    for c in state.hands[cur]:
        hand_mask[card_to_index(c)] = 1.0
    V[o:o+32] = hand_mask; o += 32

    # 2) Current trick slots
    for slot in range(3):
        slot_mask = np.zeros(32, dtype=np.float32)
        if slot < len(state.current_trick.cards):
            _, c = state.current_trick.cards[slot]
            slot_mask[card_to_index(c)] = 1.0
        V[o:o+32] = slot_mask; o += 32

    # 3) Previous-tricks played mask
    prev_mask = np.zeros(32, dtype=np.float32)
    for tr in state.completed_tricks:
        for _, c in tr.cards:
            prev_mask[card_to_index(c)] = 1.0
    V[o:o+32] = prev_mask; o += 32

    # 4) Trump suit one-hot (5 dims)
    trump_vec = np.zeros(5, dtype=np.float32)
    if state.game_type == "suit" and state.trump is not None:
        trump_vec[SUITS.index(state.trump)] = 1.0
    elif state.game_type == "null":
        trump_vec[4] = 1.0
    V[o:o+5] = trump_vec; o += 5

    # 5) Game type one-hot (3)
    gt_map = {"suit":0, "grand":1, "null":2}
    gt_vec = np.zeros(3, dtype=np.float32)
    gt_vec[gt_map[state.game_type]] = 1.0
    V[o:o+3] = gt_vec; o += 3

    # 6) Position in trick (3)
    pos = len(state.current_trick.cards)
    pos_vec = np.zeros(3, dtype=np.float32)
    pos_vec[pos] = 1.0
    V[o:o+3] = pos_vec; o += 3

    # 7) Tricks won by each (3, normalized by 10)
    wins = {"F":0, "M":0, "R":0}
    for tr in state.completed_tricks:
        wins[tr.winner()] += 1
    win_vec = np.array([wins[p] for p in ["F","M","R"]], dtype=np.float32)/10.0
    V[o:o+3] = win_vec; o += 3

    # 8) Has skat been picked up? (1)
    V[o] = 1.0 if skat_picked else 0.0; o += 1

    # 9) Declarer one-hot (3)
    dec_vec = np.zeros(3, dtype=np.float32)
    dec_vec[["F","M","R"].index(declarer)] = 1.0
    V[o:o+3] = dec_vec; o += 3

    # 10) Trump-cards-in-hand mask (32)
    trump_mask = np.zeros(32, dtype=np.float32)
    for c in state.hands[cur]:
        if state.game_type == "suit" and (c.rank == Rank.JACK or c.suit == state.trump):
            trump_mask[card_to_index(c)] = 1.0
    V[o:o+32] = trump_mask; o += 32

    # 11) Trick number one-hot (10)
    trick_no = len(state.completed_tricks) + 1
    idx = min(max(trick_no,1),10) - 1
    trick_vec = np.zeros(10, dtype=np.float32)
    trick_vec[idx] = 1.0
    V[o:o+10] = trick_vec; o += 10

    # 12) Legal-move mask (32)
    legal_mask = np.zeros(32, dtype=np.float32)
    for c in state.get_legal_actions():
        legal_mask[card_to_index(c)] = 1.0
    V[o:o+32] = legal_mask; o += 32

    # 13) Accumulated trick‐points by F/M/R (3, normalized by 120)
    pts = {"F":0, "M":0, "R":0}
    for tr in state.completed_tricks:
        pts[tr.winner()] += tr.points()
    pts_vec = np.array([pts[p] for p in ["F","M","R"]], dtype=np.float32)/120.0
    V[o:o+3] = pts_vec; o += 3

    assert o == 255, f"Encoded {o} dims; expected 255"
    return V


def main(num_deals: int = 6000,
         target_player: str = "R",
         out_npz: str = "data/skat_data_custom.npz",
         mcts_iterations: int = 1000):
    X_list, Y_list = [], []

    for deal_i in tqdm(range(1, num_deals + 1), desc="Generating deals"):
        hands, declarer, game_type, trump = random_contract_and_deal()
        state = TrickTakingState(hands, declarer, game_type, trump)

        # play out one deal
        while not state.is_terminal():
            card = mcts_trick_phase(state, iterations=mcts_iterations)

            if state.current_player == target_player:
                x = encode_state_custom(state, declarer, skat_picked=False)
                y = encode_action(card)
                X_list.append(x)
                Y_list.append(y)

            state.apply(card)

        if deal_i % 100 == 0:
            print(f" → {deal_i}/{num_deals} deals, {len(X_list)} examples collected")

    # save
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    X = np.stack(X_list, axis=0)  # now shape (N, 255)
    Y = np.array(Y_list, dtype=np.int64)
    np.savez(out_npz, X=X, Y=Y)
    print(f"\nSaved {X.shape[0]} examples to {out_npz}")


if __name__ == "__main__":
    main(num_deals=8000, target_player="R",
         out_npz="data/datask.npz",
         mcts_iterations=1000)
