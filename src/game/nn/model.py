# File: src/game/nn/model.py

import torch
import numpy as np

from src.game.trick_phase import TrickTakingState
from src.game.card import Card, Suit, Rank

SUITS = [Suit.CLUBS, Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS]
RANKS = [
    Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
    Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE
]

def card_to_index(card: Card) -> int:
    s = SUITS.index(card.suit)
    r = RANKS.index(card.rank)
    return s * len(RANKS) + r

def index_to_card(idx: int) -> Card:
    s, r = divmod(idx, len(RANKS))
    return Card(SUITS[s], RANKS[r])


ACTION_DIM = len(SUITS) * len(RANKS)    # 32


STATE_DIM = 255

GAME_TYPE_TO_IDX = {"suit": 0, "grand": 1, "null": 2}


# ——————————————————————————————————————————
# 3) Flat state encoder
# ——————————————————————————————————————————

def encode_state_flat(state: TrickTakingState) -> torch.Tensor:

    vec = np.zeros(STATE_DIM, dtype=np.float32)
    off = 0

    # 1) current hand
    cur = state.current_player
    for c in state.hands[cur]:
        vec[off + card_to_index(c)] = 1.0
    off += ACTION_DIM

    # 2) current trick slots
    for slot in range(3):
        if slot < len(state.current_trick.cards):
            _, c = state.current_trick.cards[slot]
            vec[off + card_to_index(c)] = 1.0
        off += ACTION_DIM

    # 3) previous-tricks mask
    for tr in state.completed_tricks:
        for _, c in tr.cards:
            vec[off + card_to_index(c)] = 1.0
    off += ACTION_DIM

    # 4) trump suit one-hot (5)
    trump_vec = np.zeros(5, dtype=np.float32)
    if state.game_type == "suit" and state.trump is not None:
        # note: uses SUITS order for clubs/spades/hearts/diamonds
        idx = SUITS.index(state.trump)
        trump_vec[idx] = 1.0
    elif state.game_type == "null":
        trump_vec[4] = 1.0
    vec[off:off+5] = trump_vec
    off += 5

    # 5) game type one-hot (3)
    gt = np.zeros(3, dtype=np.float32)
    gt[GAME_TYPE_TO_IDX[state.game_type]] = 1.0
    vec[off:off+3] = gt
    off += 3

    # 6) position in trick (3)
    pos = len(state.current_trick.cards)
    vec[off + pos] = 1.0
    off += 3

    # 7) tricks won by each (3, normalized by 10)
    wins = [state._sum_points(p)/10.0 for p in ("F","M","R")]
    vec[off:off+3] = wins
    off += 3

    # 8) skat-picked flag (1) — always 0 for trick‐phase
    off += 1

    # 9) declarer one-hot (3)
    dec = np.zeros(3, dtype=np.float32)
    dec[("F","M","R").index(state.declarer)] = 1.0
    vec[off:off+3] = dec
    off += 3

    # 10) trump-cards-in-hand mask (32)
    trump_mask = np.zeros(ACTION_DIM, dtype=np.float32)
    if state.game_type == "suit":
        for c in state.hands[cur]:
            if c.rank == Rank.JACK or c.suit == state.trump:
                trump_mask[card_to_index(c)] = 1.0
    vec[off:off+ACTION_DIM] = trump_mask
    off += ACTION_DIM

    # 11) trick-number one-hot (10)
    trick_no = len(state.completed_tricks) + 1
    idx = min(max(trick_no,1),10) - 1
    vec[off + idx] = 1.0
    off += 10

    # 12) legal-move mask (32)
    legal = np.zeros(ACTION_DIM, dtype=np.float32)
    for c in state.get_legal_actions():
        legal[card_to_index(c)] = 1.0
    vec[off:off+ACTION_DIM] = legal
    off += ACTION_DIM

    # 13) accumulated trick-points (3, normalized by 120)
    pts = [state._sum_points(p)/120.0 for p in ("F","M","R")]
    vec[off:off+3] = pts
    off += 3

    assert off == STATE_DIM, f"Encoded {off} dims; expected {STATE_DIM}"
    return torch.from_numpy(vec)



def encode_action(card: Card) -> int:
    return card_to_index(card)

def decode_action(idx: int) -> Card:
    return index_to_card(idx)



CHANNELS = 8
HEIGHT, WIDTH = len(RANKS), len(SUITS)   # 8 ranks × 4 suits

def encode_state_image(state: TrickTakingState) -> torch.Tensor:

    img = torch.zeros(CHANNELS, HEIGHT, WIDTH, dtype=torch.uint8)

    def mark(ch, cards):
        for c in cards:
            r = RANKS.index(c.rank)
            s = SUITS.index(c.suit)
            img[ch, r, s] = 1

    # hands
    mark(0, state.hands["F"])
    mark(1, state.hands["M"])
    mark(2, state.hands["R"])

    # current trick
    for i, (_, c) in enumerate(state.current_trick.cards):
        if i < 3:
            mark(3 + i, [c])

    # history
    hist = [c for t in state.completed_tricks for _, c in t.cards]
    mark(6, hist)

    # legal
    mark(7, state.get_legal_actions())

    return img
