# Pixel Skat AI Engine 🎴🤖

Welcome to **Pixel Skat**, a cutting-edge AI-driven Skat card game engine combining Monte Carlo Tree Search (MCTS) with neural-network-guided decision making. Whether you’re a researcher benchmarking game-AI techniques, a developer exploring hybrid search/learning methods, or just a Skat enthusiast, this project is for you!

---

## 🚀 Features

* **Full-game AI**

  * **Bidding**: Two-round interactive auction via pure MCTS.
  * **Skat Phase**: Declarer picks up or skips the skat using one-step MCTS rollouts.
  * **Game-Type & Trump Selection**: Sequential MCTS choices over {suit, grand, null} and, if suit, over {♣,♠,♥,♦}.
  * **Trick-Taking**: 255-dimensional game-state encoding + 3-layer neural network → softmax priors → PUCT-biased MCTS.

* **Three Play Configurations**

  1. All three players use pure MCTS
  2. Two players use MCTS+NN in trick-taking, one uses pure MCTS
  3. Reconfigurable “config1” & “config2” folders for head-to-head AI experiments

* **Human-Compatible UI**

  * PyGame-based interface showing avatars, hand layouts, trick animation
  * Input modes for interactive bidding and skat decisions
  * On-screen “policy priors” logging for neural-network guidance

* **Experimentation Toolkit**

  * `experiments/config1` & `config2` branches: swap human logic for AI, auto-run head-to-head matches
  * Built-in performance metrics: win counts, declarer points, trick-win rates, average decision time

---

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

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Fetch the pretrained NN model**
   Place `three_hidden_layer_model.keras` under `src/game/nn/3hidL/`.

---

## 🚩 Quick Start

### 1. Play interactively

```bash
python main.py
```

* F = Player 1 (Forehand)
* M = Player 3 (Middlehand, you)
* R = Player 2 (Rearhand)

### 2. Run experiment configurations

```bash
# config1: two MCTS+NN vs one pure-MCTS
python src/game/experiments/config1/main_config.py

# config2: one MCTS+NN vs two pure-MCTS
python src/game/experiments/config2/main_config.py
```

Results and in-game logs will print to your console, including softmax priors and performance metrics.

---

## 🔍 Architecture Overview

1. **State Encodings**

   * **BiddingState**, **SkatPhaseState**, **GameTypeState**, **TrumpSelectionState**, **TrickTakingState**
   * Deep-copies + legal-action lists + reward evaluations for each phase.

2. **MCTS Core**

   * UCB1/PUCT selection, expansion, rollout, backpropagation
   * Rollout heuristics:

     * Bidding: `max(handValue – bid, 0.1)` & fixed 1.0 for “pass”
     * Skat phase: “pickup” vs “skip” if handValue ↑ ≥ 3 points
     * Trick taking: lowest winning card or lowest discard

3. **Neural Network Integration**

   * **255→3-layer→32 logits** model
   * Softmax over legal actions → PUCT bias term
   * Guides early search, maintains exploration

---

## 📊 Experiments & Metrics

* **Games Won** (out of 120)
* **Total Declarer Points** (≥ 61 to make contract)
* **Average Trick-Win Rate** (tricks won / 10)
* **Failed Contracts** (declarer losses)
* **Avg Decision Time** (s per move, includes simulated NN eval)

Sample results in tables under `docs/` illustrate the boost from adding NN priors in the trick-taking phase.

---

## ✨ Contribute

* ⭐ Star this repo if you find it useful
* 🐞 Report issues or request features via GitHub Issues
* 📝 Submit PRs to improve algorithms, UIs, or experiment suites

Want to experiment with alternative network architectures or rollout policies? Simply drop in your own Keras model under `src/game/nn/`, update the config, and let the matches begin!

---

## 📝 License

This project is released under the MIT License. See `LICENSE` for details.

---

> “In MCTS we trust—until the neural net shows us the way.”
> Happy coding and may your contracts always be safe! 🚀
