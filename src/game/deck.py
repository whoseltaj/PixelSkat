import random
from .card import Card, Suit, Rank


class Deck:
    def __init__(self):

        self.cards = [Card(suit, rank) for suit in Suit for rank in Rank]


    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_hands: int, cards_per_hand: int):

        hands = []
        for i in range(num_hands):

            hand = self.cards[i * cards_per_hand: (i + 1) * cards_per_hand]
            hands.append(hand)

        self.cards = self.cards[num_hands * cards_per_hand:]
        return hands

    def __str__(self):

        return ", ".join(str(card) for card in self.cards)
