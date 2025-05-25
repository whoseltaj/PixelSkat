# Pixel Skat Engine 🤖

Welcome to **Pixel Skat**, a cutting-edge AI-driven Skat card game engine combining Monte Carlo Tree Search (MCTS) with neural-network-guided decision making.


## 📸 Game Flow Representation

https://github.com/user-attachments/assets/f5f72280-1269-46e1-9ff7-8897d01aeec9

---

## 🚀 Features

* **Full-game AI**

  * **Bidding**: Two-round interactive auction via pure MCTS.
  * **Skat Phase**: Declarer picks up or skips the skat using one-step MCTS rollouts.
  * **Game-Type & Trump Selection**: Sequential MCTS choices over {suit, grand, null} and, if suit, over {♣,♠,♥,♦}.
  * **Trick-Taking**: 255-dimensional game-state encoding + 3-layer neural network → softmax probability distribution → UCB1 + biased MCTS.

* **Three Play Configurations**

  1. Two players use MCTS+NN in trick-taking, one uses pure MCTS
  2. Two players use pure MCTS, two players use MCTS+NN in trick-taking
  3. Reconfigurable “config1” & “config2” folders for head-to-head AI experiments

* **Human-Compatible UI**

  * PyGame-based interface showing avatars, hand layouts, trick animation
  * Input modes for interactive bidding and skat decisions

* **Experimentation Toolkit**

  * `experiments/config1` & `config2` branches: swap human logic for AI, auto-run head-to-head matches
  * Built-in performance metrics: win counts, declarer points, trick-win rates, average decision time


## 🛠️ Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-org/pixel-skat.git
   cd pixel-skat
   ```

2. **Set up a Python 3.10+ virtualenv**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/macOS
   .venv\Scripts\activate         # Windows
   ```

3. **Fetch the pretrained NN model**
   Place `three_hidden_layer_model.keras` under `src/game/nn/3hidL/`.

---

## 🚩 Quick Start

### 1. Play interactively

```bash
python main.py
```

* F = Player 1 (Forehand)
* M = Player 3 (Middlehand)
* R = Player 2 (Rearhand)

### 2. Run experiment configurations

```bash
# config1: two MCTS+NN vs one pure-MCTS
python src/game/experiments/config1/main_config1.py

# config2: one MCTS+NN vs two pure-MCTS
python src/game/experiments/config2/main_config2.py
```

Results and in-game logs will print to your console, including softmax distribution and performance metrics.

---

## 🔍 Architecture Overview

Project Structure
```text
game/
├─ experiments/
│  ├─ config1/                # Code and configs for Experiment 1 (MCTS vs 2 MCTS+NN players)
│  └─ config2/                # Code and configs for Experiment 2 (1 MCTS vs 1 MCTS+NN players)
├─ interface/                 # Pygame GUI 
├─ nn/                        # Neural network architectures & training
│  ├─ 1hidL/, 3hidL/, 6hidL/
│  ├─ data/, logs/
│  ├─ data_split.py
│  ├─ generate_data.py       # Data Generation
│  └─ model.py
├─ card.py                   # Card and deck definitions
├─ deck.py                   # Deck shuffle & deal logic
├─ state.py                  # Bidding Phase 
├─ game_type_selection.py    # Game Type and Trump Suit Selection
├─ declarer_phase.py         # Skat pick-up vs hand decision
├─ trick_phase.py            # Trick-taking MCTS + NN integration
└─ main.py                   # Entry point for full game

```

1. **State Encodings**

   * **BiddingState**, **SkatPhaseState**, **GameTypeState**, **TrumpSelectionState**, **TrickTakingState**
   * Deep-copies + legal-action lists + reward evaluations for each phase.

2. **MCTS Core**

   * UCB1 selection, expansion, rollout, backpropagation

3. **Neural Network Integration**

   * **255→3-layer→32 logits** model
   * Softmax over legal actions → UCB1 + bias term
   * Guides early search, maintains exploration


## 📊 Experiments & Metrics

* **Games Won** (out of 120)
* **Total Declarer Points** (≥ 61 to make contract)
* **Average Trick-Win Rate** (tricks won / 10)
* **Failed Contracts** (declarer losses)
* **Avg Decision Time** (s per move, includes simulated NN eval)

## ✨ Contribute

* ⭐ Star this repo if you find it useful
* 🐞 Report issues or request features via GitHub Issues
* 📝 Submit PRs to improve algorithms, UIs, or experiment suites

## 🚩 Contact
* For questions or support, contact: eltaj0404@gmail.com
