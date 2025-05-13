import os
import sys
import pygame
import math
import copy

from src.game.state import evaluate_game_type, evaluate_suit_game


class GameTypeState:
    def __init__(self, hand):
        self.hand = hand[:]
        self.selected = None

    def get_legal_actions(self):
        if self.selected is None:
            return ["suit", "grand", "null"]
        return []

    def apply(self, action):
        if self.selected is None:
            self.selected = action
        return self

    def is_terminal(self):
        return self.selected is not None

    def reward(self):
        values = evaluate_game_type(self.hand)
        if self.selected is None:
            return 0
        return values[self.selected]




class TrumpSelectionState:
    def __init__(self, hand):
        self.hand = hand[:]
        self.selected = None

    def get_legal_actions(self):
        if self.selected is None:
            return ["clubs", "spades", "hearts", "diamonds"]
        return []

    def apply(self, action):
        if self.selected is None:
            self.selected = action
        return self

    def is_terminal(self):
        return self.selected is not None

    def reward(self):
        return evaluate_suit_game(self.hand, self.selected)



def rollout_game_type_policy(state):
    best = None
    best_val = -float("inf")
    for a in state.get_legal_actions():
        tmp = copy.deepcopy(state)
        tmp.apply(a)
        val = tmp.reward()
        if val > best_val:
            best_val, best = val, a
    return best

def rollout_trump_policy(state):
    best = None
    best_val = -float("inf")
    for a in state.get_legal_actions():
        tmp = copy.deepcopy(state)
        tmp.apply(a)
        val = tmp.reward()
        if val > best_val:
            best_val, best = val, a
    return best

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.priors = {}

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def uct_score(self, c_puct=1.4):
        """PUCT score: Q + U."""
        if self.visits == 0:
            return float("inf")
        q = self.total_reward / self.visits
        u = c_puct * math.sqrt(math.log(self.parent.visits) / self.visits)
        return q + u

    def best_child(self, c_puct=1.4):
        return max(self.children, key=lambda n: n.uct_score(c_puct))

    def expand_one(self):
        tried = {c.action for c in self.children}
        for a in self.state.get_legal_actions():
            if a not in tried:
                new_state = copy.deepcopy(self.state)
                new_state.apply(a)
                child = Node(new_state, parent=self, action=a)
                self.children.append(child)
                return child
        raise RuntimeError("No untried actions")

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward


def mcts_search(initial_state, iterations=2000, c_puct=1.4):
    root = Node(initial_state)

    for _ in range(iterations * 2):
        # 1) SELECTION
        node = root
        sim_state = copy.deepcopy(initial_state)
        while node.children and not sim_state.is_terminal():
            node = node.best_child(c_puct)
            sim_state.apply(node.action)

        # 2) EXPANSION
        if not sim_state.is_terminal():
            node = node.expand_one()
            sim_state = copy.deepcopy(node.state)

        # 3) SIMULATION
        if isinstance(sim_state, GameTypeState):
            action = rollout_game_type_policy(sim_state)
            sim_state.apply(action)
            reward = sim_state.reward()
        else:  # TrumpSelectionState
            action = rollout_trump_policy(sim_state)
            sim_state.apply(action)
            reward = sim_state.reward()

        # 4) BACKPROPAGATION
        while node:
            node.update(reward)
            node = node.parent


    best = max(root.children, key=lambda n: n.visits)
    return best.action


def select_game_type(declarer):
    if declarer == "M":
        return _interactive_game_type()
    else:
        print("Error: For AI declarers, use select_game_type_entry(declarer, ai_hand).")
        sys.exit(1)


def select_game_type_entry(declarer, ai_hand):
    if declarer == "M":
        return _interactive_game_type()
    else:
        return select_game_type_ai(ai_hand)


def select_game_type_ai(ai_hand):
    selected_type = mcts_game_type_selection(ai_hand)
    if selected_type != "suit":
        return selected_type, None
    trump = mcts_trump_selection(ai_hand)
    return "suit", trump


def mcts_game_type_selection(ai_hand):
    state = GameTypeState(ai_hand)
    return mcts_search(state, iterations=2000)


def mcts_trump_selection(ai_hand):
    state = TrumpSelectionState(ai_hand)
    return mcts_search(state, iterations=2000)



def _interactive_game_type():

    screen_width, screen_height = 1400, 900
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Select Game Type")
    clock = pygame.time.Clock()

    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    game_type_bg_path = os.path.join(project_root, "images", "game_type.png")
    try:
        bg_img = pygame.image.load(game_type_bg_path)
    except Exception as e:
        print(f"Error loading game_type.png: {e}")
        bg_img = pygame.Surface((screen_width, screen_height))
        bg_img.fill((50, 50, 50))
    bg_img = pygame.transform.scale(bg_img, (screen_width, screen_height))

    main_bg_path = os.path.join(project_root, "images", "bg.jpg")
    try:
        main_bg_img = pygame.image.load(main_bg_path)
    except Exception as e:
        print(f"Error loading bg.jpg: {e}")
        main_bg_img = pygame.Surface((screen_width, screen_height))
        main_bg_img.fill((0, 0, 0))
    main_bg_img = pygame.transform.scale(main_bg_img, (screen_width, screen_height))

    button_width, button_height = 200, 50
    gap = 20
    total_buttons_width = 3 * button_width + 2 * gap
    start_x = (screen_width - total_buttons_width) // 2
    y = screen_height // 2 - button_height // 2
    buttons = {
        "Suit": pygame.Rect(start_x, y, button_width, button_height),
        "Grand": pygame.Rect(start_x + button_width + gap, y, button_width, button_height),
        "Null": pygame.Rect(start_x + 2 * (button_width + gap), y, button_width, button_height)
    }

    button_color = (160, 146, 180)
    text_color = (40, 40, 68)

    pixel_font_path = os.path.join(project_root, "images", "VCR_OSD_MONO_1.001.ttf")
    try:
        font = pygame.font.Font(pixel_font_path, 36)
    except Exception as e:
        print(f"Error loading pixel font: {e}")
        font = pygame.font.Font(None, 36)

    selected_game_type = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if buttons["Suit"].collidepoint(pos):
                    selected_game_type = "suit"
                    running = False
                elif buttons["Grand"].collidepoint(pos):
                    selected_game_type = "grand"
                    running = False
                elif buttons["Null"].collidepoint(pos):
                    selected_game_type = "null"
                    running = False
        screen.blit(bg_img, (0, 0))
        for label, rect in buttons.items():
            pygame.draw.rect(screen, button_color, rect)
            text_surf = font.render(label, True, text_color)
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect)
        pygame.display.flip()
        clock.tick(30)

    if selected_game_type != "suit":
        screen.blit(main_bg_img, (0, 0))
        pygame.display.flip()
        pygame.time.wait(1000)
        pygame.quit()
        return selected_game_type, None

    trump = _interactive_trump_selection(main_bg_img, project_root, screen_width, screen_height)
    return "suit", trump


def _interactive_trump_selection(main_bg_img, project_root, screen_width, screen_height):

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Select Trump Suit")
    clock = pygame.time.Clock()

    trump_bg_path = os.path.join(project_root, "images", "trump_suit.png")
    try:
        bg_img = pygame.image.load(trump_bg_path)
    except Exception as e:
        print(f"Error loading trump_suit.png: {e}")
        bg_img = pygame.Surface((screen_width, screen_height))
        bg_img.fill((70, 70, 70))
    bg_img = pygame.transform.scale(bg_img, (screen_width, screen_height))

    button_width, button_height = 200, 50
    gap = 20
    total_width = 4 * button_width + 3 * gap
    start_x = (screen_width - total_width) // 2
    y = screen_height // 2 - button_height // 2

    trump_buttons = {
        "clubs": pygame.Rect(start_x, y, button_width, button_height),
        "spades": pygame.Rect(start_x + (button_width + gap), y, button_width, button_height),
        "hearts": pygame.Rect(start_x + 2 * (button_width + gap), y, button_width, button_height),
        "diamonds": pygame.Rect(start_x + 3 * (button_width + gap), y, button_width, button_height)
    }

    button_color = (160, 146, 180)
    text_color = (40, 40, 68)

    pixel_font_path = os.path.join(project_root, "images", "VCR_OSD_MONO_1.001.ttf")
    try:
        font = pygame.font.Font(pixel_font_path, 36)
    except Exception as e:
        print(f"Error loading pixel font: {e}")
        font = pygame.font.Font(None, 36)

    selected_trump = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                for suit, rect in trump_buttons.items():
                    if rect.collidepoint(pos):
                        selected_trump = suit
                        running = False
                        break
        screen.blit(bg_img, (0, 0))
        for suit, rect in trump_buttons.items():
            pygame.draw.rect(screen, button_color, rect)
            label = font.render(suit.capitalize(), True, text_color)
            label_rect = label.get_rect(center=rect.center)
            screen.blit(label, label_rect)
        pygame.display.flip()
        clock.tick(30)

    # Briefly display main game background.
    main_bg_path = os.path.join(project_root, "images", "bg.jpg")
    try:
        main_bg_img = pygame.image.load(main_bg_path)
    except Exception as e:
        print(f"Error loading bg.jpg: {e}")
        main_bg_img = pygame.Surface((screen_width, screen_height))
        main_bg_img.fill((0, 0, 0))
    main_bg_img = pygame.transform.scale(main_bg_img, (screen_width, screen_height))
    screen.blit(main_bg_img, (0, 0))
    pygame.display.flip()
    pygame.time.wait(1000)
    pygame.quit()

    return selected_trump

