
import os
import pygame
import time
import numpy as np
from src.game.experiments.config1.trick_phase_config import Trick, TrickTakingState, mcts_trick_phase
from tensorflow.keras.models import load_model
from src.game.card import Suit

def run_ui(hand_F, hand_M, hand_R, skat_cards,
           game_type=None, trump=None, declarer=None,
           base_value=1, is_hand=False, is_ouvert=False,
           contra=False, re=False):
    # filter out skat cards
    ui_hand_F = [c for c in hand_F if c not in skat_cards]
    ui_hand_M = [c for c in hand_M if c not in skat_cards]
    ui_hand_R = [c for c in hand_R if c not in skat_cards]

    pygame.init()
    screen_width, screen_height = 1400, 900
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pixel Skat")
    clock = pygame.time.Clock()

    # locate project root
    current_dir  = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))

    # load neural policy
    model_path = os.path.join(project_root, "src", "game", "nn", "3hidL",
                              "three_hidden_layer_model.keras")
    policy_net = load_model(model_path)

    # background
    bg_file    = "bg.jpg" if (ui_hand_F or ui_hand_M or ui_hand_R) else "mainpg.png"
    background = pygame.image.load(os.path.join(project_root, "images", bg_file))
    background = pygame.transform.scale(background, (screen_width, screen_height))

    # splash only?
    if not (ui_hand_F or ui_hand_M or ui_hand_R):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    return
            screen.blit(background, (0, 0))
            pygame.display.flip()
            clock.tick(30)

    # avatars & labels
    perf_dir     = os.path.join(project_root, "images", "performers")
    p1_img       = pygame.transform.scale(pygame.image.load(os.path.join(perf_dir, "p1.png")), (80,80))
    p2_img       = pygame.transform.scale(pygame.image.load(os.path.join(perf_dir, "p2.png")), (80,80))
    p3_img       = pygame.transform.scale(pygame.image.load(os.path.join(perf_dir, "p3.png")), (80,80))
    p1_rect      = p1_img.get_rect(topleft=(50,70))
    p2_rect      = p2_img.get_rect(topright=(screen_width-50,70))
    p3_rect      = p3_img.get_rect(midbottom=(screen_width//2, screen_height-100))
    font         = pygame.font.Font(None,28)
    name_p1      = font.render("Player 1", True, (0,0,0))
    name_p2      = font.render("Player 2", True, (0,0,0))
    name_p3      = font.render("Player 3 (AI)", True, (0,0,0))
    p1_name_rect = name_p1.get_rect(centerx=p1_rect.centerx+190, top=p1_rect.bottom+5)
    p2_name_rect = name_p2.get_rect(centerx=p2_rect.centerx-190, top=p2_rect.bottom+5)
    p3_name_rect = name_p3.get_rect(centerx=p3_rect.centerx,   top=p3_rect.bottom+5)

    # card layout
    card_w, card_h, gap = 50, 70, 5
    def compute_positions(cx,y,count):
        total = count*card_w + (count-1)*gap
        start = cx - total//2
        return [(start+i*(card_w+gap),y) for i in range(count)]

    p1_positions = compute_positions(p1_name_rect.centerx, p1_name_rect.bottom+40, len(ui_hand_F))
    p2_positions = compute_positions(p2_name_rect.centerx, p2_name_rect.bottom+40, len(ui_hand_R))
    p3_positions = compute_positions(p3_rect.centerx, p3_rect.top-40-card_h, len(ui_hand_M))

    cards_dir = os.path.join(project_root, "images", "cards")
    back_img   = pygame.transform.scale(pygame.image.load(os.path.join(cards_dir,"back_side.png")),
                                        (card_w,card_h))
    p1_imgs    = [back_img]*len(ui_hand_F)
    p2_imgs    = [back_img]*len(ui_hand_R)
    p3_imgs    = [pygame.transform.scale(c.load_image(), (card_w,card_h)) for c in ui_hand_M]

    cx0 = screen_width//2 - card_w//2
    cy0 = screen_height//2 - card_h//2
    center_slots = [(cx0,cy0), (cx0-card_w-gap,cy0), (cx0+card_w+gap,cy0)]
    played        = []

    # track timing & trick‐wins
    times = {"F":0.0, "M":0.0, "R":0.0}
    counts= {"F":0,   "M":0,   "R":0}

    # build initial state
    trump_suit = Suit[trump.upper()] if game_type=="suit" else None
    state = TrickTakingState(
        {"F":ui_hand_F, "M":ui_hand_M, "R":ui_hand_R},
        declarer, game_type, trump_suit,
        base_value, is_hand, is_ouvert, contra, re
    )
    state.order          = ["M","F","R"]
    state.current_player = "M"
    state.current_trick  = Trick(game_type, trump_suit, leader="M")

    def animate_card(img, start, end, duration=300):
        t0, sx, sy = pygame.time.get_ticks(), *start
        ex, ey      = end
        while True:
            t = min((pygame.time.get_ticks()-t0)/duration,1.0)
            x = sx+(ex-sx)*t; y=sy+(ey-sy)*t
            screen.blit(background,(0,0))
            # draw everything
            screen.blit(p1_img,p1_rect); screen.blit(name_p1,p1_name_rect)
            screen.blit(p2_img,p2_rect); screen.blit(name_p2,p2_name_rect)
            screen.blit(p3_img,p3_rect); screen.blit(name_p3,p3_name_rect)
            for pos,img_ in zip(p1_positions,p1_imgs): screen.blit(img_,pos)
            for pos,img_ in zip(p2_positions,p2_imgs): screen.blit(img_,pos)
            for pos,img_ in zip(p3_positions,p3_imgs): screen.blit(img_,pos)
            for img_,pos_ in played:                    screen.blit(img_,pos_)
            screen.blit(img,(x,y))
            pygame.display.flip()
            clock.tick(60)
            if t>=1.0: break

    softmax_log = []
    running     = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # AI‐3 turn
        if state.current_player == "M":
            feat = state.to_feature_vector()
            logits = policy_net.predict(feat[None, :])[0]
            legal_cards = state.get_legal_actions()
            idxs = [c.index for c in legal_cards]
            leg_logits = logits[idxs]
            exp_vals = np.exp(leg_logits - np.max(leg_logits))
            probs_p3 = exp_vals / exp_vals.sum()
            trick_softmax_p3 = [(str(c), float(p)) for c, p in zip(legal_cards, probs_p3)]

            print("\n--- P3 NN) ---")
            for card, p in sorted(trick_softmax_p3, key=lambda x: x[1], reverse=True):
                print(f"  {card:>3} → {p:.3f}")
            print("------------------------------\n")
            t0 = time.perf_counter()
            c3 = mcts_trick_phase(state, iterations=2000, policy=policy_net)
            dt = time.perf_counter() - t0
            times["M"] += dt; counts["M"] += 1

            idx3 = ui_hand_M.index(c3)
            img3 = pygame.transform.scale(c3.load_image(), (card_w,card_h))
            ui_hand_M.pop(idx3); p3_imgs.pop(idx3)
            animate_card(img3, p3_positions[idx3], center_slots[0])
            played.append((img3, center_slots[0]))
            p3_positions = compute_positions(p3_rect.centerx, p3_rect.top-40-card_h, len(ui_hand_M))
            state.apply(c3)

            # Player 1 (MCTS only)
            t0 = time.perf_counter()
            c1 = mcts_trick_phase(state, iterations=2000)
            dt = time.perf_counter() - t0
            times["F"] += dt; counts["F"] += 1

            img1 = pygame.transform.scale(c1.load_image(), (card_w,card_h))
            ui_hand_F.remove(c1); p1_imgs.pop(0)
            animate_card(img1, p1_positions[0], center_slots[1])
            played.append((img1, center_slots[1]))
            p1_positions = compute_positions(p1_name_rect.centerx, p1_name_rect.bottom+40, len(ui_hand_F))
            state.apply(c1)

            # Player 2 (MCTS+NN)
            feat        = state.to_feature_vector()
            logits      = policy_net.predict(feat[None,:])[0]
            legal_cards = state.get_legal_actions()
            idxs        = [c.index for c in legal_cards]
            leg_logits  = logits[idxs]
            exp_vals    = np.exp(leg_logits - np.max(leg_logits))
            probs       = exp_vals / exp_vals.sum()
            trick_softmax = [(str(c), float(p)) for c,p in zip(legal_cards,probs)]
            softmax_log.append(trick_softmax)

            print("\n--- P2 NN ---")
            for card, p in sorted(trick_softmax, key=lambda x: x[1], reverse=True):
                print(f"  {card:>3} → {p:.3f}")
            print("------------------------------\n")

            t0 = time.perf_counter()
            c2 = mcts_trick_phase(state, iterations=2000, policy=policy_net)
            dt = time.perf_counter() - t0
            times["R"] += dt; counts["R"] += 1

            img2 = pygame.transform.scale(c2.load_image(), (card_w,card_h))
            ui_hand_R.remove(c2); p2_imgs.pop(0)
            animate_card(img2, p2_positions[0], center_slots[2])
            played.append((img2, center_slots[2]))
            p2_positions = compute_positions(p2_name_rect.centerx, p2_name_rect.bottom+40, len(ui_hand_R))
            state.apply(c2)

            # next trick
            if state.is_terminal():
                running = False
            else:
                state.order          = ["M","F","R"]
                state.current_player = "M"
                state.current_trick  = Trick(game_type, trump_suit, leader="M")

        # redraw UI
        screen.blit(background,(0,0))
        screen.blit(p1_img,p1_rect); screen.blit(name_p1,p1_name_rect)
        screen.blit(p2_img,p2_rect); screen.blit(name_p2,p2_name_rect)
        screen.blit(p3_img,p3_rect); screen.blit(name_p3,p3_name_rect)
        for pos,img_ in zip(p1_positions,p1_imgs): screen.blit(img_,pos)
        for pos,img_ in zip(p2_positions,p2_imgs): screen.blit(img_,pos)
        for pos,img_ in zip(p3_positions,p3_imgs): screen.blit(img_,pos)
        for img_,pos_ in played:                     screen.blit(img_,pos_)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

    # end‐of‐game scoring & stats
    res = state.final_result()
    pts = {p: state._sum_points(p) for p in ["F","M","R"]}

    # trick win counts
    trick_wins = {"F":0, "M":0, "R":0}
    for trick in state.completed_tricks:
        trick_wins[trick.winner()] += 1

    print("\n=== Round Results ===")
    print(f"Player 1 (F): {pts['F']} points, tricks won: {trick_wins['F']}/10")
    print(f"Player 2 (R): {pts['R']} points, tricks won: {trick_wins['R']}/10")
    print(f"Player 3 (M): {pts['M']} points, tricks won: {trick_wins['M']}/10")
    print(f"Winner: Player {{'F':1,'M':3,'R':2}}[{max(pts, key=pts.get)}] with {max(pts.values())} points")
    print(f"Declarer {'made' if res['made'] else 'failed'} the contract.")
    print(f"Contract Value: {res['contract_value']} (×{res['multiplier']})\n")

    # average trick‐win rate & decision times
    print("=== Performance Metrics ===")
    for label,pid in [("Player 1","F"), ("Player 2","R"), ("Player 3","M")]:
        total_time = times[pid] + counts[pid] * 1.0  # add 1s per decision
        avg_time = total_time / counts[pid] if counts[pid] > 0 else 0
        avg_tricks = trick_wins[pid]
        print(f"{label}:  Trick Win Rate: {avg_tricks}/10,  Avg Decision Time: {avg_time:.1f}s")
