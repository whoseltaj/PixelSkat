# File: src/game/interface/pygame_ui.py

import os
import pygame
import numpy as np
from src.game.trick_phase import Trick, TrickTakingState, mcts_trick_phase
from tensorflow.keras.models import load_model
from src.game.card import Suit


def run_ui(hand_F, hand_M, hand_R, skat_cards,
           game_type=None, trump=None, declarer=None,
           base_value=1, is_hand=False, is_ouvert=False,
           contra=False, re=False):
    ui_hand_F = [c for c in hand_F if c not in skat_cards]
    ui_hand_M = [c for c in hand_M if c not in skat_cards]
    ui_hand_R = [c for c in hand_R if c not in skat_cards]

    pygame.init()
    screen_width, screen_height = 1400, 900
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pixel Skat")
    clock = pygame.time.Clock()

    # 2) Background
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    model_path = os.path.join(
        project_root, "src", "game", "nn", "3hidL",
        "three_hidden_layer_model.keras"
    )
    policy_net = load_model(model_path)
    bg_file = "bg.jpg" if (ui_hand_F or ui_hand_M or ui_hand_R) else "mainpg.png"
    bg_path = os.path.join(project_root, "images", bg_file)
    try:
        background = pygame.image.load(bg_path)
    except pygame.error:
        background = pygame.Surface((screen_width, screen_height))
        background.fill((0, 128, 0))
    background = pygame.transform.scale(background, (screen_width, screen_height))

    # 3) Splash‐only mode
    if not (ui_hand_F or ui_hand_M or ui_hand_R):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    return
            screen.blit(background, (0, 0))
            pygame.display.flip()
            clock.tick(30)

    # 4) Avatars
    perf_dir = os.path.join(project_root, "images", "performers")
    p1_img = pygame.transform.scale(
        pygame.image.load(os.path.join(perf_dir, "p1.png")), (80, 80)
    )
    p2_img = pygame.transform.scale(
        pygame.image.load(os.path.join(perf_dir, "p2.png")), (80, 80)
    )
    p3_img = pygame.transform.scale(
        pygame.image.load(os.path.join(perf_dir, "p3.png")), (80, 80)
    )

    p1_rect = p1_img.get_rect(topleft=(50, 70))
    p2_rect = p2_img.get_rect(topright=(screen_width - 50, 70))
    p3_rect = p3_img.get_rect(midbottom=(screen_width // 2, screen_height - 100))

    # 5) Name labels
    font = pygame.font.Font(None, 28)
    name_p1 = font.render("Player 1", True, (0, 0, 0))
    name_p2 = font.render("Player 2", True, (0, 0, 0))
    name_p3 = font.render("Player 3 (You)", True, (0, 0, 0))
    p1_name_rect = name_p1.get_rect(
        centerx=p1_rect.centerx + 190, top=p1_rect.bottom + 5
    )
    p2_name_rect = name_p2.get_rect(
        centerx=p2_rect.centerx - 190, top=p2_rect.bottom + 5
    )
    p3_name_rect = name_p3.get_rect(
        centerx=p3_rect.centerx, top=p3_rect.bottom + 5
    )

    # 6) Card layout
    card_w, card_h = 50, 70
    gap = 5

    def compute_positions(cx, y, count):
        total = count * card_w + (count - 1) * gap
        start = cx - total // 2
        return [(start + i * (card_w + gap), y) for i in range(count)]

    p1_positions = compute_positions(
        p1_name_rect.centerx, p1_name_rect.bottom + 40, len(ui_hand_F)
    )
    p2_positions = compute_positions(
        p2_name_rect.centerx, p2_name_rect.bottom + 40, len(ui_hand_R)
    )
    p3_positions = compute_positions(
        p3_rect.centerx, p3_rect.top - 40 - card_h, len(ui_hand_M)
    )

    # 7) Card back & image lists
    cards_dir = os.path.join(project_root, "images", "cards")
    back_img = pygame.transform.scale(
        pygame.image.load(os.path.join(cards_dir, "back_side.png")),
        (card_w, card_h),
    )
    p1_imgs = [back_img] * len(ui_hand_F)
    p2_imgs = [back_img] * len(ui_hand_R)
    p3_imgs = [
        pygame.transform.scale(c.load_image(), (card_w, card_h)) for c in ui_hand_M
    ]

    # 8) Center‐table slots
    cx0 = screen_width // 2 - card_w // 2
    cy0 = screen_height // 2 - card_h // 2
    center_slots = [
        (cx0, cy0),
        (cx0 - card_w - gap, cy0),
        (cx0 + card_w + gap, cy0),
    ]
    played = []

    # 9) Build MCTS state on UI copies
    if game_type:
        trump_suit = Suit[trump.upper()] if game_type == "suit" else None
        state = TrickTakingState(
            {"F": ui_hand_F, "M": ui_hand_M, "R": ui_hand_R},
            declarer,
            game_type,
            trump_suit,
            base_value=base_value,
            is_hand=is_hand,
            is_ouvert=is_ouvert,
            contra=contra,
            re=re
        )
        state.order = ["M", "F", "R"]
        state.current_player = "M"
        state.current_trick = Trick(game_type, trump_suit, leader="M")
    else:
        state = None

    softmax_log = []
    # 10) Main loop
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:

                played.clear()

                # Human plays
                for i, pos in enumerate(p3_positions):
                    if pygame.Rect(pos[0], pos[1], card_w, card_h).collidepoint(ev.pos):
                        img3 = p3_imgs.pop(i)
                        card3 = ui_hand_M.pop(i)
                        played.append((img3, center_slots[0]))
                        p3_positions = compute_positions(
                            p3_rect.centerx,
                            p3_rect.top - 40 - card_h,
                            len(ui_hand_M),
                        )

                        if state:
                            state.apply(card3)

                            # AI #1 plays
                            c1 = mcts_trick_phase(state, iterations=2000)
                            img1 = pygame.transform.scale(
                                c1.load_image(), (card_w, card_h)
                            )
                            played.append((img1, center_slots[1]))
                            ui_hand_F.remove(c1)
                            p1_imgs.pop(0)
                            p1_positions = compute_positions(
                                p1_name_rect.centerx,
                                p1_name_rect.bottom + 40,
                                len(ui_hand_F),
                            )
                            state.apply(c1)


                            feat = state.to_feature_vector()
                            logits = policy_net.predict(feat[None, :])[0]
                            legal_cards = state.get_legal_actions()
                            idxs = [c.index for c in legal_cards]
                            legal_logits = logits[idxs]
                            exp = np.exp(legal_logits - np.max(legal_logits))
                            probs = exp / exp.sum()


                            trick_softmax = [
                                (str(card), float(p)) for card, p in zip(legal_cards, probs)
                            ]
                            softmax_log.append(trick_softmax)


                            print("\n--- Player 2 Policy Priors ---")
                            for card, p in sorted(trick_softmax, key=lambda x: x[1], reverse=True):
                                print(f"  {card:3s} → {p:.3f}")
                            print("------------------------------\n")


                            c2 = mcts_trick_phase(state, iterations=2000, policy=policy_net)
                            img2 = pygame.transform.scale(
                                c2.load_image(), (card_w, card_h)
                            )
                            played.append((img2, center_slots[2]))
                            ui_hand_R.remove(c2)
                            p2_imgs.pop(0)
                            p2_positions = compute_positions(
                                p2_name_rect.centerx,
                                p2_name_rect.bottom + 40,
                                len(ui_hand_R),
                            )
                            state.apply(c2)

                            if state.is_terminal():
                                running = False


                                print("\n========== Softmax Output Summary Per Trick ==========")
                                for i, trick_probs in enumerate(softmax_log, 1):
                                    print(f"Trick {i}:")
                                    for card, prob in sorted(trick_probs, key=lambda x: x[1], reverse=True):
                                        print(f"  {card:3s} → {prob:.3f}")
                                    print("-----------------------------------------------------")
                                print("=====================================================\n")
                            else:

                                state.order = ["M", "F", "R"]
                                state.current_player = "M"
                                state.current_trick = Trick(
                                    game_type, trump_suit, leader="M"
                                )
                        break


        screen.blit(background, (0, 0))
        screen.blit(p1_img, p1_rect)
        screen.blit(name_p1, p1_name_rect)
        screen.blit(p2_img, p2_rect)
        screen.blit(name_p2, p2_name_rect)
        screen.blit(p3_img, p3_rect)
        screen.blit(name_p3, p3_name_rect)

        for pos, img in zip(p1_positions, p1_imgs):
            screen.blit(img, pos)
        for pos, img in zip(p2_positions, p2_imgs):
            screen.blit(img, pos)
        for pos, img in zip(p3_positions, p3_imgs):
            screen.blit(img, pos)

        # Show the current trick
        for img, pos in played:
            screen.blit(img, pos)

        pygame.display.flip()
        clock.tick(30)


    pygame.quit()


    if state:
        res = state.final_result()
        pts = {
            "F": state._sum_points("F"),
            "M": state._sum_points("M"),
            "R": state._sum_points("R")
        }
        # highest‐points player
        winner = max(pts, key=pts.get)
        print("\n=== Round Results ===")
        print(f"Player 1 (F): {pts['F']} points")
        print(f"Player 3 (M): {pts['M']} points")
        print(f"Player 2 (R): {pts['R']} points")
        print(f"Winner: Player { {'F':1,'M':3,'R':2}[winner] } with {pts[winner]} points")
        print(f"Declarer {'made' if res['made'] else 'failed'} the contract.")
        print(f"Contract Value: {res['contract_value']} (×{res['multiplier']})")
