# Forza-Style AI Racing Simulator (Python, Pygame, NumPy)

> A modular top–down racing game with real‑time physics, AI drivers, and a tiny from‑scratch behavior‑cloning system.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.x-brightgreen.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.x-orange.svg)

---

## ✨ Highlights

- **Two‑car Forza‑style top‑down racer** with lap timing and basic HUD.
- **Real‑time physics**: acceleration, braking, friction, and rotated‑rectangle collision.
- **Track system** using waypoints, segment lengths, and distance‑along‑track computation.
- **AI racing opponents**:
  - Rule‑based waypoint‑following driver with braking logic.
  - MLP (multi‑layer perceptron) policy trained via **behavior cloning** using in‑game telemetry.
- **Modular architecture** across multiple files (`car.py`, `physics.py`, `ai.py`, `track.py`, `game.py`, `settings.py`).
- **Data‑driven tuning** via a single `settings.py` file for all game parameters.

This project is designed to look and feel like something you’d see on a strong student / early‑career SWE / ML portfolio.

---

## 🧱 Project Structure

```text
forza_ai_racer/
├── ai.py              # Rule-based AI + MLP policy + behavior cloning trainer
├── car.py             # Car entity, physics integration, rendering
├── game.py            # Main game loop, orchestration, HUD
├── main.py            # Entry point (runs the game)
├── physics.py         # Vector math, friction, SAT collision, helpers
├── settings.py        # All tunable parameters (window, car, AI, etc.)
├── track.py           # Track representation, waypoints, lap logic
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── LICENSE            # MIT license
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/forza-ai-racer.git
cd forza-ai-racer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # on macOS / Linux
# .venv\Scripts\activate  # on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the game

```bash
python main.py
```

You should see a 1280×720 game window with a simple circuit track and two cars on the start line.

---

## 🎮 Controls

### Player 1 (Human – always enabled)

- **Throttle**: `W`
- **Brake / Reverse**: `S`
- **Steer Left**: `A`
- **Steer Right**: `D`

### Player 2

- By default, **AI‑controlled** (rule‑based + MLP hybrid).
- To make it human‑controlled instead, open `settings.py` and set:

```python
USE_AI_FOR_SECOND_CAR = False
```

Then use the **arrow keys**:

- **Throttle**: `↑`
- **Brake / Reverse**: `↓`
- **Steer Left**: `←`
- **Steer Right**: `→`

### Global Keys

- `SPACE` – Pause / resume
- `R` – Restart race
- `ESC` – Quit

---

## 🧠 AI & Behavior Cloning Overview

### Rule‑Based Agent (`RuleBasedAgent`)

- Looks ahead to the next few waypoints.
- Computes the signed angle between the car’s forward vector and the target.
- Uses a simple heuristic:
  - Small angle → full throttle, no brake.
  - Medium angle → partial throttle.
  - Large angle → low throttle, brake if going fast.

This gives you a reasonably clean, deterministic opponent without any ML.

### MLP Policy (`MLPPolicy`) + Behavior Cloning

- Input features (`extract_features` in `ai.py`):
  - `angle_to_next_wp` (normalized)
  - `speed_norm` (current speed / max speed)
  - `distance_to_wp_norm`
  - `opponent_distance_norm`
- Architecture:
  - `input_dim = 4`
  - `hidden_dim = 32` with `tanh`
  - `output_dim = 2` → `[steer, throttle]` (both in `[-1, 1]`)
- Training:
  - When Player 1 is actively steering/throttling, the game logs `(state, action)` pairs.
  - `BehaviorCloningTrainer` batches these samples and runs a simple gradient‑descent MSE loss.
  - The HUD shows:
    - Number of samples collected
    - Last training loss

The **AI driver used in the game** is a blend of the rule‑based agent and the learned MLP policy:

```python
control2 = 0.7 * rule_based_control + 0.3 * mlp_control
```

This makes the AI stable early on while still letting the learned policy influence behavior as it improves.

---

## ⚙️ Physics & Systems Design

### Car model (`car.py`)

- Maintains **position**, **velocity**, and **heading** in world space.
- Uses a simplified “force = mass × acceleration” model (mass = 1 for convenience).
- Simulates:
  - Engine force along the car’s forward vector.
  - Braking force opposite current velocity.
  - Rolling friction + air friction.
  - Steering angle scaled by speed (less twitchy when slow, responsive when fast).
- Collision shape is a **rotated rectangle** built from the car’s center + heading.

### Collision detection (`physics.py`)

- Car vs car collision uses **Separating Axis Theorem (SAT)** on the two rotated rectangles.
- World bounds are enforced with simple AABB checks + velocity damping “bounces.”

### Track & Lap Logic (`track.py`)

- Track is defined as a **closed list of waypoints**.
- Precomputes segment lengths and cumulative distances to treat the lap as a 1D loop.
- Each car’s world position is projected onto the nearest segment to get
  **distance‑along‑track**.
- Lap counting uses a circular **start/finish region** around the first waypoint.

---

## 🧪 Ideas for Extensions

Some natural next steps you (or a recruiter reading your repo) could imagine:

- Add **ghost laps** / best‑lap replay.
- Implement **checkpoints** and off‑track penalty detection.
- Train separate policies for **aggressive** vs **defensive** AI drivers.
- Add **camera follow** logic and simple **UI menus**.
- Export human driving logs to `.npz` and train offline with more advanced ML frameworks.

Documenting these ideas in the README shows that you’re thinking beyond the minimum.

---

## 📸 Screenshots / Demo GIF

You can add screenshots or gifs here after you record them, for example:

```markdown
![Gameplay GIF](docs/demo.gif)
```

This section is intentionally left as a placeholder for your visuals.

---

## 📜 License

This project is licensed under the **MIT License** – see [`LICENSE`](./LICENSE) for details.
