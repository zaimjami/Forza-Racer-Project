"""
game.py

High-level orchestration:
- Initialize Pygame, create window, clock, HUD font.
- Create Track, Cars, and AI/ML components.
- Main loop: input -> simulation -> AI/ML -> rendering.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygame

from car import Car, CarControl
from track import Track
from physics import polygons_intersect
from settings import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FPS,
    WINDOW_TITLE,
    WHITE,
    BLACK,
    RED,
    BLUE,
    YELLOW,
    FONT_NAME,
    FONT_SIZE,
    CAR_LENGTH,
    CAR_WIDTH,
    MAX_ENGINE_FORCE,
    MAX_BRAKE_FORCE,
    MAX_STEER_ANGLE,
    MAX_SPEED,
    ROLLING_FRICTION,
    AIR_FRICTION,
    PLAYER1_KEYS,
    PLAYER2_KEYS,
    USE_AI_FOR_SECOND_CAR,
    ENABLE_BEHAVIOR_CLONING_LOG,
    TOTAL_LAPS,
    TRACK_WAYPOINTS,
)
from ai import RuleBasedAgent, MLPPolicy, BehaviorCloningTrainer, MLPAgent, extract_features


class Game:
    """
    Main game controller.

    It owns:
    - window & clock
    - track
    - cars
    - AI agents and ML trainer
    - HUD rendering logic
    """

    def __init__(self) -> None:
        pygame.init()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

        # Create track.
        self.track = Track(waypoints=TRACK_WAYPOINTS)

        # Starting positions are at start line but offset sideways.
        start = self.track.start_pos
        self.player1_start_pos = (start.x - 20, start.y + 10)
        self.player2_start_pos = (start.x + 20, start.y - 10)

        # Create cars.
        self.car1 = Car(
            name="Player 1",
            color=BLUE,
            position=self.player1_start_pos,
            heading_degrees=0.0,
            length=CAR_LENGTH,
            width=CAR_WIDTH,
            max_engine_force=MAX_ENGINE_FORCE,
            max_brake_force=MAX_BRAKE_FORCE,
            max_steer_angle=MAX_STEER_ANGLE,
            max_speed=MAX_SPEED,
            rolling_friction=ROLLING_FRICTION,
            air_friction=AIR_FRICTION,
        )

        self.car2 = Car(
            name="Player 2/AI",
            color=RED,
            position=self.player2_start_pos,
            heading_degrees=0.0,
            length=CAR_LENGTH,
            width=CAR_WIDTH,
            max_engine_force=MAX_ENGINE_FORCE * 0.95,  # slightly different tuning
            max_brake_force=MAX_BRAKE_FORCE,
            max_steer_angle=MAX_STEER_ANGLE,
            max_speed=MAX_SPEED * 0.97,
            rolling_friction=ROLLING_FRICTION,
            air_friction=AIR_FRICTION,
        )

        # AI components.
        self.rule_based_agent = RuleBasedAgent()
        self.mlp_policy = MLPPolicy()
        self.mlp_agent = MLPAgent(policy=self.mlp_policy)
        self.bc_trainer = BehaviorCloningTrainer(policy=self.mlp_policy)

        # Game state flags.
        self.running = True
        self.paused = False
        self.time_scale = 1.0  # could be used for slow-motion debug.

        # For text debug of ML.
        self.last_bc_loss = 0.0

    # -------------------------------------------------------------------------
    # Input handling
    # -------------------------------------------------------------------------

    def _read_player_control(self, keys, mapping) -> CarControl:
        """
        Convert raw keyboard state into a CarControl object.
        This abstracts away specific key bindings.
        """
        throttle = 1.0 if keys[mapping["throttle"]] else 0.0
        brake = 1.0 if keys[mapping["brake"]] else 0.0

        steer = 0.0
        if keys[mapping["left"]]:
            steer -= 1.0
        if keys[mapping["right"]]:
            steer += 1.0

        return CarControl(throttle=throttle, brake=brake, steer=steer)

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """
        High-level main loop:
        - handle events
        - read input
        - compute controls (for AI and players)
        - step simulation
        - train ML (occasionally)
        - render
        """
        while self.running:
            dt_ms = self.clock.tick(FPS)
            dt = dt_ms / 1000.0 * self.time_scale

            self._handle_events()

            if not self.paused:
                self._step(dt)

            self._render()

        pygame.quit()

    def _handle_events(self) -> None:
        """
        Handle window and game-level events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Toggle pause for debugging.
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    # Reset race.
                    self._reset_race()

    # -------------------------------------------------------------------------
    # Simulation step
    # -------------------------------------------------------------------------

    def _step(self, dt: float) -> None:
        keys = pygame.key.get_pressed()

        # ---- Compute controls for both cars ----
        # Player 1 is always human-controlled.
        control1 = self._read_player_control(keys, PLAYER1_KEYS)

        # Player 2 is either human or AI.
        if USE_AI_FOR_SECOND_CAR:
            # For demonstration, we blend rule-based agent and MLP agent.
            rb_control = self.rule_based_agent.compute_control(self.car2, self.track, self.car1)
            ml_control = self.mlp_agent.compute_control(self.car2, self.track, self.car1)

            # Blend them 70% rule-based, 30% ML.
            control2 = CarControl(
                throttle=0.7 * rb_control.throttle + 0.3 * ml_control.throttle,
                brake=0.7 * rb_control.brake + 0.3 * ml_control.brake,
                steer=0.7 * rb_control.steer + 0.3 * ml_control.steer,
            )
        else:
            control2 = self._read_player_control(keys, PLAYER2_KEYS)

        # ---- Behavior cloning data logging ----
        if ENABLE_BEHAVIOR_CLONING_LOG and (control1.throttle > 0.0 or abs(control1.steer) > 0.0):
            # Only log when the human is actively doing something.
            state = extract_features(self.car1, self.track, self.car2)
            action = np.array([control1.steer, control1.throttle], dtype=np.float32)
            self.bc_trainer.add_sample(state, action)

        # ---- Update cars ----
        old_pos1 = self.car1.position.copy()
        old_pos2 = self.car2.position.copy()

        self.car1.update(control1, dt)
        self.car2.update(control2, dt)

        # Keep cars roughly within screen bounds (simple world boundary).
        self._clamp_car_to_world(self.car1)
        self._clamp_car_to_world(self.car2)

        # ---- Lap progress / lap counting ----
        self._update_lap_progress(self.car1, old_pos1)
        self._update_lap_progress(self.car2, old_pos2)

        # ---- Car-car collision ----
        self._handle_car_collision()

        # ---- Train ML occasionally ----
        self.last_bc_loss = self.bc_trainer.train_epoch(batch_size=256)

    def _clamp_car_to_world(self, car: Car) -> None:
        """
        Clamp car position to stay within the screen.
        If the car hits the boundary, we lightly bounce it by damping its velocity.
        """
        x, y = car.position.x, car.position.y
        bounced = False

        if x < 0:
            x = 0
            car.velocity.x *= -0.5
            bounced = True
        elif x > SCREEN_WIDTH:
            x = SCREEN_WIDTH
            car.velocity.x *= -0.5
            bounced = True

        if y < 0:
            y = 0
            car.velocity.y *= -0.5
            bounced = True
        elif y > SCREEN_HEIGHT:
            y = SCREEN_HEIGHT
            car.velocity.y *= -0.5
            bounced = True

        if bounced:
            car.position.x = x
            car.position.y = y

    def _update_lap_progress(self, car: Car, old_position: pygame.math.Vector2) -> None:
        """
        Update car's distance along track and handle lap crossing.
        """
        # Progress along track in [0, track.total_length).
        progress = self.track.project_position_onto_track(car.position)
        car.distance_along_track = progress

        # Lap crossing detection.
        if self.track.check_lap_crossing(old_position, car.position):
            # Completed a lap.
            if car.current_lap > 0:
                # Save best lap time.
                if car.best_lap_time is None or car.current_lap_time < car.best_lap_time:
                    car.best_lap_time = car.current_lap_time
            car.current_lap += 1
            car.current_lap_time = 0.0

            if car.current_lap > TOTAL_LAPS:
                car.finished = True

    def _handle_car_collision(self) -> None:
        """
        Check for collision between the two cars and apply simple separation.
        """
        poly1 = self.car1.get_polygon()
        poly2 = self.car2.get_polygon()

        if polygons_intersect(poly1, poly2):
            # Simple separation: move each car away from the midpoint between them.
            mid = (self.car1.position + self.car2.position) / 2
            dir1 = (self.car1.position - mid)
            dir2 = (self.car2.position - mid)

            if dir1.length_squared() > 1e-6:
                dir1 = dir1.normalize()
            if dir2.length_squared() > 1e-6:
                dir2 = dir2.normalize()

            separation_amount = 10.0
            self.car1.position += dir1 * separation_amount
            self.car2.position += dir2 * separation_amount

            # Dampen velocities to simulate impact.
            self.car1.velocity *= 0.7
            self.car2.velocity *= 0.7

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def _render(self) -> None:
        """
        Render the current frame:
        - track
        - cars
        - HUD
        """
        # Draw track + background.
        self.track.draw(self.screen)

        # Draw cars.
        self.car1.draw(self.screen)
        self.car2.draw(self.screen)

        # Draw HUD overlay.
        self._draw_hud()

        pygame.display.flip()

    def _draw_hud(self) -> None:
        """
        Draw heads-up display (laps, times, ML stats).
        """
        lines = []

        def fmt_time(t: Optional[float]) -> str:
            if t is None:
                return "--"
            return f"{t:5.2f}s"

        # Player 1 info
        lines.append(
            f"P1 Lap: {self.car1.current_lap}/{TOTAL_LAPS}   "
            f"Lap Time: {fmt_time(self.car1.current_lap_time)}   "
            f"Best: {fmt_time(self.car1.best_lap_time)}"
        )

        # Player 2 / AI info
        lines.append(
            f"P2 Lap: {self.car2.current_lap}/{TOTAL_LAPS}   "
            f"Lap Time: {fmt_time(self.car2.current_lap_time)}   "
            f"Best: {fmt_time(self.car2.best_lap_time)}"
        )

        # ML info
        lines.append(
            f"Behavior Cloning: samples={len(self.bc_trainer.states)} "
            f"last_loss={self.last_bc_loss:.4f}"
        )

        if self.paused:
            lines.append("[SPACE] Resume   [R] Restart   [ESC] Quit")
        else:
            lines.append("[SPACE] Pause    [R] Restart   [ESC] Quit")

        # Render each line.
        y = 8
        for text in lines:
            surf = self.font.render(text, True, YELLOW, BLACK)
            self.screen.blit(surf, (8, y))
            y += surf.get_height() + 2

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def _reset_race(self) -> None:
        """
        Reset cars to starting positions, laps/time to zero.
        """
        self.car1.reset(self.player1_start_pos, 0.0)
        self.car2.reset(self.player2_start_pos, 0.0)

        # Reset ML stats not strictly necessary, but we keep training data.
        self.last_bc_loss = 0.0
