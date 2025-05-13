import sys
import threading
import pygame

from src.game.deck import Deck
from src.game.state import interactive_bidding_phase
from src.game.declarer_phase import interactive_declarer_phase
from src.game.game_type_selection import select_game_type, select_game_type_entry
from src.game.state import evaluate_game_type, evaluate_suit_game
from src.game.interface.pygame_ui import run_ui


def splash_ui_thread():

    run_ui([], [], [], [])


def main():
    try:

        ui_thread = threading.Thread(target=splash_ui_thread)
        ui_thread.start()
        pygame.time.wait(500)


        deck = Deck()
        deck.shuffle()
        hands = deck.deal(num_hands=3, cards_per_hand=10)
        hand_F, hand_M, hand_R = hands
        skat_cards = deck.cards  # the two leftover cards


        winner, final_bid = interactive_bidding_phase(hand_F, hand_M, hand_R)
        if winner == "M":
            print("You are the Declarer!")
        elif winner == "F":
            print("Forehand (Player 1) is the Declarer!")
        else:
            print("Rearhand (Player 2) is the Declarer!")
        print("Final bid:", final_bid)


        if winner == "M":
            hand_M = interactive_declarer_phase(winner, hand_M, skat_cards)
            skat_cards = []
        elif winner == "F":
            hand_F = interactive_declarer_phase(winner, hand_F, skat_cards)
        else:
            hand_R = interactive_declarer_phase(winner, hand_R, skat_cards)


        pygame.event.post(pygame.event.Event(pygame.QUIT))
        ui_thread.join()


        if winner == "M":
            game_type, trump = select_game_type("M")
        else:
            ai_hand = hand_F if winner == "F" else hand_R
            game_type, trump = select_game_type_entry(winner, ai_hand)

        print("Selected game type:", game_type)
        if trump:
            print("Selected trump suit:", trump)


        if winner == "F":
            dec_hand = hand_F
        elif winner == "M":
            dec_hand = hand_M
        else:
            dec_hand = hand_R

        if game_type == "suit":
            base_value = evaluate_suit_game(dec_hand, trump)
        else:
            game_vals = evaluate_game_type(dec_hand)
            base_value = game_vals[game_type]

        is_hand = False
        is_ouvert = False
        contra = False
        re = False


        run_ui(
            hand_F,
            hand_M,
            hand_R,
            skat_cards,
            game_type=game_type,
            trump=trump,
            declarer=winner,
            base_value=base_value,
            is_hand=is_hand,
            is_ouvert=is_ouvert,
            contra=contra,
            re=re
        )

    except KeyboardInterrupt:
        print("\nExiting gracefully (KeyboardInterrupt).")
        sys.exit(0)


if __name__ == "__main__":
    main()
