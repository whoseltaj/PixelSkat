

# PLAYER 1- MCTS , PLAYER 2- MCTS+NN, PLAYER 3 - MCTS+NN : SELFPLAY F0R EXPERIMENT

import sys
import threading
import pygame
from src.game.deck import Deck
from src.game.experiments.config2.state_config2 import interactive_bidding_phase
from src.game.experiments.config2.declarer_phase_config2 import interactive_declarer_phase
from src.game.experiments.config2.game_type_selection_config2 import select_game_type_entry
from src.game.state import evaluate_game_type, evaluate_suit_game
from src.game.experiments.config2.pygame_ui_config2 import run_ui

def splash_ui_thread():
    run_ui([], [], [], [])

def main():
    try:
        ui_thread = threading.Thread(target=splash_ui_thread)
        ui_thread.start()
        pygame.time.wait(500)

        deck = Deck()
        deck.shuffle()
        hand_F, hand_M, hand_R = deck.deal(num_hands=3, cards_per_hand=10)
        skat_cards = deck.cards

        # Bidding
        winner, final_bid = interactive_bidding_phase(hand_F, hand_M, hand_R)
        print(f"Declarer: {winner}, Final bid: {final_bid}")

        # Skat pickup / hand game
        if winner == "M":
            hand_M = interactive_declarer_phase(winner, hand_M, skat_cards)
            skat_cards = []
        elif winner == "F":
            hand_F = interactive_declarer_phase(winner, hand_F, skat_cards)
        else:
            hand_R = interactive_declarer_phase(winner, hand_R, skat_cards)

        pygame.event.post(pygame.event.Event(pygame.QUIT))
        ui_thread.join()

        # Game‐type & trump
        ai_hand = {"F": hand_F, "M": hand_M, "R": hand_R}[winner]
        game_type, trump = select_game_type_entry(winner, ai_hand)
        print(f"Game type: {game_type}, Trump: {trump}")

        # Contract value
        dec_hand = ai_hand
        if game_type == "suit":
            base_value = evaluate_suit_game(dec_hand, trump)
        else:
            base_value = evaluate_game_type(dec_hand)[game_type]

        # Trick‐taking UI
        run_ui(
            hand_F, hand_M, hand_R, skat_cards,
            game_type=game_type,
            trump=trump,
            declarer=winner,
            base_value=base_value,
            is_hand=False,
            is_ouvert=False,
            contra=False,
            re=False
        )

    except KeyboardInterrupt:
        print("\nExiting gracefully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
