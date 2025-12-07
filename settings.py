"""
settings.py

Central place for tuning all high-level game parameters.
Changing values here should change game feel without touching core logic.
"""

import pygame

# --------------------------------------------------------------------------------------
# Display settings
# --------------------------------------------------------------------------------------

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60  # target FPS

WINDOW_TITLE = "Forza-Style AI Racer"

# --------------------------------------------------------------------------------------
# Colors (RGB)
# --------------------------------------------------------------------------------------

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (40, 40, 40)
LIGHT_GRAY = (120, 120, 120)
GREEN = (34, 177, 76)
RED = (200, 40, 40)
BLUE = (40, 80, 200)
YELLOW = (240, 230, 140)
ORANGE = (255, 160, 0)

# --------------------------------------------------------------------------------------
# Track & race settings
# --------------------------------------------------------------------------------------

# Waypoints for a simple circuit track (center line).
# These are tuned for a 1280x720 window.
TRACK_WAYPOINTS = [
    (260, 180),
    (1020, 180),
    (1120, 260),
    (1120, 460),
    (1020, 540),
    (260, 540),
    (160, 460),
    (160, 260),
]

# How thick the track line should be (in pixels).
TRACK_WIDTH = 80

# Number of laps to win.
TOTAL_LAPS = 3

# Radius around the first waypoint that counts as "start/finish line" for lap counting.
START_LINE_RADIUS = 80

# --------------------------------------------------------------------------------------
# Car settings
# --------------------------------------------------------------------------------------

# Common car dimensions (pixels). These loosely correspond to some physical scale.
CAR_LENGTH = 64
CAR_WIDTH = 32

# Car physics parameters
MAX_ENGINE_FORCE = 1600.0      # higher -> faster acceleration
MAX_BRAKE_FORCE = 2200.0       # higher -> faster braking
MAX_STEER_ANGLE = 35.0         # degrees (max steering wheel angle)
MAX_SPEED = 550.0              # pixels / second (top speed)
ROLLING_FRICTION = 2.8         # baseline friction
AIR_FRICTION = 0.5             # additional friction scaling with speed

COLLISION_DAMPING = 0.4        # how "bouncy" collisions are (0 = no bounce)

# --------------------------------------------------------------------------------------
# Input settings
# --------------------------------------------------------------------------------------

# For keyboard input, we map keys to conceptual actions.
PLAYER1_KEYS = {
    "throttle": pygame.K_w,
    "brake": pygame.K_s,
    "left": pygame.K_a,
    "right": pygame.K_d,
}

PLAYER2_KEYS = {
    "throttle": pygame.K_UP,
    "brake": pygame.K_DOWN,
    "left": pygame.K_LEFT,
    "right": pygame.K_RIGHT,
}

# --------------------------------------------------------------------------------------
# AI / ML settings
# --------------------------------------------------------------------------------------

# Whether the second car is AI-controlled.
USE_AI_FOR_SECOND_CAR = True

# Whether to log player driving data for behavior cloning.
ENABLE_BEHAVIOR_CLONING_LOG = True

# Maximum size of replay buffer (number of state-action pairs).
REPLAY_BUFFER_SIZE = 10_000

# MLP network architecture for behavior cloning (input, hidden, output sizes).
MLP_INPUT_DIM = 4   # [angle_to_wp, speed_norm, distance_norm, opponent_distance_norm]
MLP_HIDDEN_DIM = 32
MLP_OUTPUT_DIM = 2  # [steer, throttle]

# Learning rate for toy gradient-descent trainer.
MLP_LEARNING_RATE = 1e-3

# --------------------------------------------------------------------------------------
# HUD settings
# --------------------------------------------------------------------------------------

FONT_NAME = "consolas"
FONT_SIZE = 20

# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------

def world_to_screen_coords(x: float, y: float) -> tuple[int, int]:
    """
    Helper to convert world coordinates to screen coordinates.
    Currently 1:1 mapping, but this function exists so we could later
    implement camera systems without changing draw code everywhere.
    """
    return int(x), int(y)
