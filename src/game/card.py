# File: EngineSkat/src/game/card.py

import os
import pygame
from enum import Enum


class Suit(Enum):
    CLUBS = "clubs"
    SPADES = "spades"
    HEARTS = "hearts"
    DIAMONDS = "diamonds"


class Rank(Enum):
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank.value}{self.suit.value[0].upper()}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (
            isinstance(other, Card) and
            self.suit == other.suit and
            self.rank == other.rank
        )

    def __hash__(self):
        return hash((self.suit, self.rank))

    def load_image(self):
        mapping = {
            "A": "1",
            "J": "11",
            "Q": "12",
            "K": "13"
        }
        rank_str = mapping[self.rank.value] if self.rank.value in mapping else self.rank.value
        suit_str = self.suit.value[0].lower()
        filename = f"{rank_str}_{suit_str}.png"
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "images", "cards"))
        file_path = os.path.join(base_path, filename)
        try:
            return pygame.image.load(file_path).convert_alpha()
        except pygame.error as e:
            print(f"Image not found for {self}: {e}. Using placeholder.")
            return pygame.Surface((110, 130))

    @property
    def index(self) -> int:

        rank_order = {
            Rank.SEVEN: 0, Rank.EIGHT: 1, Rank.NINE: 2, Rank.JACK: 3,
            Rank.QUEEN: 4, Rank.KING: 5, Rank.TEN: 6, Rank.ACE: 7
        }
        suit_order = {
            Suit.CLUBS: 0, Suit.SPADES: 1,
            Suit.HEARTS: 2, Suit.DIAMONDS: 3
        }
        return suit_order[self.suit] * 8 + rank_order[self.rank]